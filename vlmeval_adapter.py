from argparse import Namespace
from collections.abc import Mapping

import torch
from PIL import Image
from vlmeval.vlm.base import BaseModel

from main import build_model
from model.ipmoe import IPMoE

class IPMoEVLMEval(BaseModel):
    INTERLEAVE = False
    VIDEO_LLM = False
    allowed_types = ['text', 'image']

    def __init__(self, checkpoint_path, max_new_tokens = 128):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu', weights_only = True)
        args = Namespace(**checkpoint['args'])

        if not hasattr(args, 'num_aux_tokens'):
            raise ValueError('Checkpoint is missing num_aux_tokens')

        args.gradient_checkpointing = False

        self.context_limit = args.context_limit
        self.max_new_tokens = max_new_tokens

        ## Model Building
        processor, text_encoder, vision_encoder, interaction, moe, llm = build_model(self.device, args)
        self.processor = processor
        self.model = IPMoE(processor, text_encoder, vision_encoder, interaction, moe, llm,
                           args.embed_dim, args.patch_size, args.num_aux_tokens).to(self.device)

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

        ## Generate Answer
        answers, _ = self.model.generate([question], text_inputs, image_inputs, self.context_limit,
                                         max_new_tokens = self.max_new_tokens, enforce_mcq_format = False)

        return answers[0]
