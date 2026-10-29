import bisect
import hashlib
import logging
import os
import string
from argparse import Namespace
from collections.abc import Mapping

import pandas as pd
import torch
from PIL import Image
from vlmeval.dataset.image_mcq import CVBench, ImageMCQDataset
from vlmeval.dataset.image_yorn import ImageYORNDataset
from vlmeval.smp import LMUDataRoot
from vlmeval.vlm.base import BaseModel

from ablation import configure_ablation
from main import build_model
from model.ipmoe import IPMoE

# Instructions copied from the LLaVA-NeXT training questions so evaluation prompts match training.
MCQ_INSTRUCTION = "Answer with the option's letter from the given choices directly."
SHORT_ANSWER_INSTRUCTION = 'Answer the question using a single word or phrase.'

# Plain question-and-options datasets; subclasses with their own prompts (e.g. NaturalBench) keep them.
MCQ_DATASETS = set(ImageMCQDataset.supported_datasets()) | set(CVBench.supported_datasets())
YORN_DATASETS = set(ImageYORNDataset.supported_datasets())

# Test-time interventions on a trained checkpoint, chosen with IPMOE_INTERVENTION (see rebuttal.sh):
# swap builds the interaction map from another image of the same benchmark while the image tokens keep the true one,
# mean averages the shared map over patches as all_map_mean does, uniform replaces the learned weights with 1/3, and
# zero_<expert> sets that expert's learned weight to 0 without renormalizing the other two.
EXPERTS = ('self', 'cross', 'hybrid')
INTERVENTIONS = ('swap', 'mean', 'uniform', *(f'zero_{expert}' for expert in EXPERTS))
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

def _file_sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()

def _option_text(value):
    # Option columns with blanks load as floats, which would turn a count of 1 into "1.0".
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)

def mcq_prompt(line):
    """Same layout as training: hint, question, "A. option" lines, then the instruction."""
    hint = line['hint'] if 'hint' in line and not pd.isna(line['hint']) else None
    options = [f'{key}. {_option_text(line[key])}' for key in string.ascii_uppercase if key in line and not pd.isna(line[key])]

    return '\n'.join([*([hint] if hint else []), line['question'], *options, MCQ_INSTRUCTION])

class IPMoEVLMEval(BaseModel):
    INTERLEAVE = False
    VIDEO_LLM = False
    allowed_types = ['text', 'image']

    def __init__(self, checkpoint_path, max_new_tokens = 128, context_limit = None):
        super().__init__()

        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('max_new_tokens must be a positive integer')
        if context_limit is not None and (type(context_limit) is not int or context_limit <= max_new_tokens):
            raise ValueError('context_limit must be an integer greater than max_new_tokens')

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu', weights_only = True)
        args = Namespace(**checkpoint['args'])

        if not hasattr(args, 'num_aux_tokens'):
            raise ValueError('Checkpoint is missing num_aux_tokens')

        configure_ablation(args, checkpoint = True)
        args.gradient_checkpointing = False

        self.max_new_tokens = max_new_tokens

        ## Model Building
        processor, text_encoder, vision_encoder, interaction, moe, llm = build_model(self.device, args)
        # The checkpoint's context_limit is a training memory budget. Inference
        # keeps the complete prefix and question up to the decoder's capacity.
        config = llm.model.config
        text_config = config.get_text_config() if hasattr(config, 'get_text_config') else getattr(config, 'text_config', config)
        model_context_limit = text_config.max_position_embeddings
        self.context_limit = model_context_limit if context_limit is None else context_limit
        if not max_new_tokens < self.context_limit <= model_context_limit:
            raise ValueError(f'context_limit must be greater than max_new_tokens={max_new_tokens} '
                             f'and at most the decoder capacity={model_context_limit}; got {self.context_limit}')
        logging.getLogger(f'vlmeval.{__name__}').info('Context tokens: inference=%s, training=%s, decoder=%s, max_new_tokens=%s',
                                                    self.context_limit, args.context_limit, model_context_limit, max_new_tokens)
        self.processor = processor
        self.model = IPMoE(processor, text_encoder, vision_encoder, interaction, moe, llm,
                           args.embed_dim, args.patch_size, args.num_aux_tokens,
                           ablation = args.ablation_config).to(self.device)

        ## Load Trained Parameters
        parameters = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        saved = checkpoint['model']
        if not isinstance(saved, Mapping):
            raise TypeError('Checkpoint model must be a parameter mapping')
        expected, found = set(parameters), set(saved)
        if expected != found:
            missing = sorted(expected - found)
            unexpected = sorted(map(repr, found - expected))
            raise ValueError(f'Checkpoint keys differ: missing={missing}, unexpected={unexpected}')

        # Check supported source properties before modifying any parameter.
        for name, parameter in parameters.items():
            value = saved[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'Checkpoint parameter {name!r} must be a Tensor')
            if value.layout != torch.strided or value.is_meta or value.is_quantized:
                raise ValueError(f'Checkpoint parameter {name!r} must be a materialized dense non-quantized Tensor')
            if value.shape != parameter.shape:
                raise ValueError(f'Checkpoint shape mismatch for {name}: {tuple(value.shape)} != {tuple(parameter.shape)}')

        with torch.no_grad():
            for name, parameter in parameters.items():
                # Preserve the existing device/dtype conversion policy.
                parameter.copy_(saved[name])

        self.model.eval()

        self.intervention = os.environ.get('IPMOE_INTERVENTION') or None
        self._swap_images = {}
        if self.intervention is not None:
            self._apply_intervention(self.intervention)

    def _apply_intervention(self, name):
        if name not in INTERVENTIONS:
            raise ValueError(f'Unknown IPMOE_INTERVENTION {name!r}; choose from {INTERVENTIONS}')

        ablation, moe = self.model.ablation, self.model.moe
        if ablation.get('prefix', 'full') == 'image':
            raise ValueError('The image_only arm has no interaction map to intervene on')
        if name == 'mean':
            if ablation.get('shared_map', 'full') != 'full':
                raise ValueError('This arm already averages its shared map')
            # IPMoE.encode_prefix reads the configuration on every call.
            ablation['shared_map'] = 'spatial_mean'
        elif name != 'swap':
            if moe.router_mode != 'learned' or moe.expert_set != 'all':
                raise ValueError(f'{name} requires a learned router over all three experts')
            if name == 'uniform':
                moe.router_mode = 'uniform'
            else:
                column = EXPERTS.index(name[len('zero_'):])
                route = moe.router.forward

                def zeroed(*args, **kwargs):
                    weights = route(*args, **kwargs).clone()
                    weights[:, column] = 0

                    return weights

                moe.router.forward = zeroed

        logging.getLogger(f'vlmeval.{__name__}').warning('Test-time intervention: %s', name)

    def _swap_image(self, path):
        """A fixed image of the same benchmark whose content differs from this one; identical across ranks and runs."""
        images = os.path.join(LMUDataRoot(), 'images')
        path = os.path.abspath(path)
        relative = os.path.relpath(path, images)
        root = os.path.dirname(path) if relative.startswith('..') else os.path.join(images, relative.split(os.sep)[0])

        if root not in self._swap_images:
            # Benchmarks reuse images across questions (MMBench's circular copies, POPE, CV-Bench), so pair contents, not files.
            first = {}
            for directory, _, files in sorted(os.walk(root)):
                for file in sorted(files):
                    if file.lower().endswith(IMAGE_EXTENSIONS):
                        first.setdefault(_file_sha256(os.path.join(directory, file)), os.path.join(directory, file))
            if len(first) < 2:
                raise ValueError(f'swap needs at least two distinct images under {root}')
            self._swap_images[root] = (sorted(first), first)

        digests, first = self._swap_images[root]
        # Hash order is unrelated to content, so half-way round it lies an unrelated image, never this one.
        position = bisect.bisect_left(digests, _file_sha256(path))

        return first[digests[(position + len(digests) // 2) % len(digests)]]

    def use_custom_prompt(self, dataset):
        return dataset in MCQ_DATASETS or dataset in YORN_DATASETS

    def build_prompt(self, line, dataset = None):
        image_paths = self.dump_image(line, dataset)
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        if dataset in MCQ_DATASETS:
            prompt = mcq_prompt(line)

        else:
            prompt = f"{line['question']}\n{SHORT_ANSWER_INSTRUCTION}"

        return [dict(type = 'image', value = path) for path in image_paths] + [dict(type = 'text', value = prompt)]

    @torch.inference_mode()
    def generate_inner(self, message, dataset = None):
        question = '\n'.join(item['value'] for item in message if item['type'] == 'text')
        images = [item['value'] for item in message if item['type'] == 'image']
        if len(images) != 1:
            raise ValueError(f'IPMoE requires exactly one image per question; received {len(images)}')

        ## CLIP Inputs
        text_inputs = self.processor.tokenizer([question], return_tensors = 'pt', padding = True, truncation = False, verbose = False)
        with Image.open(images[0]) as image:
            image_inputs = self.processor.image_processor(images = [image.convert('RGB')], return_tensors = 'pt')

        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}

        map_image_inputs = None
        if self.intervention == 'swap':
            with Image.open(self._swap_image(images[0])) as image:
                map_image_inputs = self.processor.image_processor(images = [image.convert('RGB')], return_tensors = 'pt')
            map_image_inputs = {key: value.to(self.device) for key, value in map_image_inputs.items()}

        ## Generate Answer
        answers, _ = self.model.generate([question], text_inputs, image_inputs, self.context_limit,
                                         max_new_tokens = self.max_new_tokens, enforce_mcq_format = False,
                                         map_image_inputs = map_image_inputs)

        return answers[0]
