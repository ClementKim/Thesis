import re
from functools import partial
from unittest import mock

import torch
from PIL import Image
from transformers import (AutoModelForVision2Seq, AutoProcessor, GenerationConfig, GenerationMixin, InstructBlipProcessor,
                          LlavaForConditionalGeneration, LlavaImageProcessor, LlavaNextForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration)
from vlmeval.dataset.image_mcq import CVBench
from vlmeval.vlm import InternVLChat, Qwen2VLChat
from vlmeval.vlm.base import BaseModel

from vlmeval_adapter import MCQ_INSTRUCTION, IPMoEVLMEval

# The model card's format prompt, appended to every question "to make the inference consistent with training".
SSRL_FORMAT_PROMPT = ('\n You FIRST think about the reasoning process as an internal monologue and then provide '
                      'the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. '
                      'The final answer MUST BE put in \\boxed{}.')

# Training-time conversation templates: LLaVA-1.5's vicuna_v1 and LLaVA-NeXT's llava_llama_3 (system prompt included;
# the Transformers chat templates of both conversions drop it).
LLAVA_V1_TEMPLATE = ("A chat between a curious human and an artificial intelligence assistant. The assistant gives "
                     "helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:")
LLAVA_LLAMA_3_TEMPLATE = ('<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. '
                          'You are able to understand the visual content that the user provides, and assist the user '
                          'with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user'
                          '<|end_header_id|>\n\n<image>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')


class BLIPFamily(BaseModel):
    """BLIP-2 and InstructBLIP through Transformers, since VLMEvalKit's InstructBLIP needs LAVIS and local Vicuna weights."""
    INTERLEAVE = False

    def __init__(self, model_path, prompt_template = '{}', max_new_tokens = 256, processor_class = AutoProcessor):
        super().__init__()
        self.processor = processor_class.from_pretrained(model_path)
        # bfloat16 rather than float16: Flan-T5 overflows in float16.
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype = torch.bfloat16).cuda().eval()
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens

    def generate_inner(self, message, dataset = None):
        prompt, image_path = self.message_to_promptimg(message, dataset = dataset)
        image = Image.open(image_path).convert('RGB')
        text = self.prompt_template.format(prompt)
        inputs = self.processor(images = image, text = text, return_tensors = 'pt')
        if 'qformer_input_ids' in inputs:
            # InstructBLIP's Q-Former has 512 positions, fewer than 12 MMBench prompts and 1 MMStar prompt need, so only
            # its copy of the prompt is truncated; the language model still reads all of it.
            limit = self.model.config.qformer_config.max_position_embeddings
            qformer = self.processor.qformer_tokenizer(text, truncation = True, max_length = limit, return_tensors = 'pt')
            inputs['qformer_input_ids'], inputs['qformer_attention_mask'] = qformer['input_ids'], qformer['attention_mask']
        inputs = inputs.to('cuda', torch.bfloat16)
        # At least one token, as LAVIS's min_length = 1 gives: InstructBLIP otherwise answers CV-Bench prompts with an
        # immediate </s>. Its language model config sets eos_token_id = 1 (<s>), so both checks need the tokenizer's </s>.
        output = self.model.generate(**inputs, do_sample = False, max_new_tokens = self.max_new_tokens, min_new_tokens = 1,
                                     eos_token_id = self.processor.tokenizer.eos_token_id)[0]
        # Depending on the Transformers version, decoder-only LLMs (Vicuna) echo the prompt; Flan-T5 never does.
        prompt_ids = inputs['input_ids'][0]
        if len(output) >= len(prompt_ids) and torch.equal(output[:len(prompt_ids)], prompt_ids):
            output = output[len(prompt_ids):]
        return self.processor.decode(output, skip_special_tokens = True).strip()


class MonoInternVL(InternVLChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Since Transformers 4.50, PreTrainedModel no longer inherits GenerationMixin, and this remote decoder predates that.
        language_model = self.model.language_model
        cls = type(language_model)
        # The decoder indexes its cache as tuples and uses its visual experts only while the cache is None, so generate()
        # must not hand it a DynamicCache.
        language_model.__class__ = type(cls.__name__, (cls, GenerationMixin),
                                        {'_supports_default_dynamic_cache': classmethod(lambda cls: False)})
        # Models that cannot generate when loaded get no generation config. The checkpoint sets output_hidden_states, which
        # from_model_config turns into return_dict_in_generate, but chat() decodes generate()'s output as token ids.
        generation_config = GenerationConfig.from_model_config(language_model.config)
        generation_config.output_hidden_states = generation_config.return_dict_in_generate = False
        language_model.generation_config = generation_config


class Qwen2VLChatBF16(Qwen2VLChat):
    def __init__(self, **kwargs):
        # Qwen2VLChat loads with torch_dtype = 'auto', i.e. the float32 these checkpoints are saved in, which
        # flash_attention_2 rejects. Converting afterwards would also round the RoPE inv_freq buffers, which a bfloat16
        # load keeps in float32.
        load = Qwen2_5_VLForConditionalGeneration.from_pretrained
        load_bf16 = lambda *args, **load_kwargs: load(*args, **{**load_kwargs, 'torch_dtype': torch.bfloat16})
        with mock.patch.object(Qwen2_5_VLForConditionalGeneration, 'from_pretrained', load_bf16):
            super().__init__(**kwargs)


class LLaVAHF(BaseModel):
    """LLaVA-1.5 and LLaVA-NeXT through Transformers: VLMEvalKit's LLaVA needs the original llava package, and its
    LLaVA_Next passes use_flash_attention_2, which Transformers 4.57 no longer accepts."""
    INTERLEAVE = False
    # Prompted exactly like IP-MoE; its "A. option" layout and instructions are LLaVA's own training format.
    use_custom_prompt = IPMoEVLMEval.use_custom_prompt

    def __init__(self, model_path, model_class, template, pad_to_square = False, cambrian_prompt = False,
                 max_new_tokens = 512):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path)
        if pad_to_square:
            # LLaVA-1.5 trains and evaluates on images padded to a square with the mean colour; the conversion's
            # CLIPImageProcessor center-crops instead, which cuts the sides off CV-Bench's 4:3 images.
            self.processor.image_processor = LlavaImageProcessor.from_pretrained(model_path, do_pad = True)
        # float16, the dtype both conversions are stored in.
        self.model = model_class.from_pretrained(model_path, torch_dtype = torch.float16,
                                                 attn_implementation = 'flash_attention_2').cuda().eval()
        # Llama-3 has no pad token, and generate() would say so on every question.
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        self.template = template
        self.cambrian_prompt = cambrian_prompt
        self.max_new_tokens = max_new_tokens

    def build_prompt(self, line, dataset = None):
        if not (self.cambrian_prompt and dataset in CVBench.supported_datasets()):
            return IPMoEVLMEval.build_prompt(self, line, dataset)

        # Cambrian-1's CV-Bench scripts send the dataset's prompt with "(A)" labels, which VLMEvalKit's TSV writes as
        # "A", followed by the MCQ instruction.
        prompt = re.sub(r'(?m)^([A-F]) ', r'(\1) ', line['prompt']) + '\n' + MCQ_INSTRUCTION
        image_paths = self.dump_image(line, dataset)
        return [dict(type = 'image', value = path) for path in image_paths] + [dict(type = 'text', value = prompt)]

    def generate_inner(self, message, dataset = None):
        prompt, image_path = self.message_to_promptimg(message, dataset = dataset)
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images = image, text = self.template.format(prompt), return_tensors = 'pt')
        inputs = inputs.to('cuda', torch.float16)
        output = self.model.generate(**inputs, do_sample = False, max_new_tokens = self.max_new_tokens)[0]
        return self.processor.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens = True).strip()


class SpatialSSRL(Qwen2VLChatBF16):
    def generate_inner(self, message, dataset = None):
        last_text = max(i for i, item in enumerate(message) if item['type'] == 'text')
        message = [dict(item) for item in message]
        message[last_text]['value'] += SSRL_FORMAT_PROMPT
        return super().generate_inner(message, dataset = dataset)


BASELINES = {
    'BLIP2-FlanT5-XL': partial(BLIPFamily, model_path = 'Salesforce/blip2-flan-t5-xl',
                               prompt_template = 'Question: {} Answer:'),
    # The repository's processor_config.json names InstructBlipVideoProcessor, which turns each image into a video.
    'InstructBLIP-Vicuna-7B': partial(BLIPFamily, model_path = 'Salesforce/instructblip-vicuna-7b',
                                      processor_class = InstructBlipProcessor),
    'Mono-InternVL-2B': partial(MonoInternVL, model_path = 'OpenGVLab/Mono-InternVL-2B', version = 'V2.0'),
    # Base model of the two Qwen fine-tunes below; its checkpoint is bfloat16, so the stock wrapper loads it.
    # VLMEvalKit's own entry sets min_pixels = 1280 * 28 * 28, which upsamples a 640x480 CV-Bench image from 391 to
    # 1302 visual tokens; the processor defaults keep native resolution, as for Spatial-SSRL-3B.
    'Qwen2.5-VL-3B-Instruct': partial(Qwen2VLChat, model_path = 'Qwen/Qwen2.5-VL-3B-Instruct', use_custom_prompt = False),
    # Same settings as VLMEvalKit's own SpatialLadder-3B entry. The checkpoint's processor still caps images at 100352
    # pixels (128 visual tokens) after min_pixels, as in the model card's own usage.
    'SpatialLadder-3B': partial(Qwen2VLChatBF16, model_path = 'hongxingli/SpatialLadder-3B',
                                min_pixels = 1280 * 28 * 28, max_pixels = 16384 * 28 * 28, use_custom_prompt = False),
    # post_process keeps only the last \boxed{} answer; 4096 tokens as in the model card.
    'Spatial-SSRL-3B': partial(SpatialSSRL, model_path = 'internlm/Spatial-SSRL-3B', use_custom_prompt = False,
                               post_process = True, max_new_tokens = 4096),
    # CLIP-336 patches through a 2-layer MLP, 576 visual tokens, as in IP-MoE's image path.
    'LLaVA-1.5-7B': partial(LLaVAHF, model_path = 'llava-hf/llava-1.5-7b-hf', model_class = LlavaForConditionalGeneration,
                            template = LLAVA_V1_TEMPLATE, pad_to_square = True),
    # Fine-tuned on LLaVA-NeXT-Data, IP-MoE's training set, with AnyRes (up to 2880 visual tokens). Cambrian-1 reports
    # CV-Bench 2D 62.2 / 3D 65.3 for it.
    'LLaVA-NeXT-8B': partial(LLaVAHF, model_path = 'llava-hf/llama3-llava-next-8b-hf',
                             model_class = LlavaNextForConditionalGeneration, template = LLAVA_LLAMA_3_TEMPLATE),
}
