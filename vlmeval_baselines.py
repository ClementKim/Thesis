from functools import partial

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from vlmeval.vlm import InternVLChat, Qwen2VLChat
from vlmeval.vlm.base import BaseModel

# The model card's format prompt, appended to every question "to make the inference consistent with training".
SSRL_FORMAT_PROMPT = ('\n You FIRST think about the reasoning process as an internal monologue and then provide '
                      'the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. '
                      'The final answer MUST BE put in \\boxed{}.')


class BLIPFamily(BaseModel):
    """BLIP-2 and InstructBLIP through Transformers, since VLMEvalKit's InstructBLIP needs LAVIS and local Vicuna weights."""
    INTERLEAVE = False

    def __init__(self, model_path, prompt_template = '{}', max_new_tokens = 256):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path)
        # bfloat16 rather than float16: Flan-T5 overflows in float16.
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype = torch.bfloat16).cuda().eval()
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens

    def generate_inner(self, message, dataset = None):
        prompt, image_path = self.message_to_promptimg(message, dataset = dataset)
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images = image, text = self.prompt_template.format(prompt), return_tensors = 'pt')
        inputs = inputs.to('cuda', torch.bfloat16)
        output = self.model.generate(**inputs, do_sample = False, max_new_tokens = self.max_new_tokens)[0]
        # Depending on the Transformers version, decoder-only LLMs (Vicuna) echo the prompt; Flan-T5 never does.
        prompt_ids = inputs['input_ids'][0]
        if len(output) >= len(prompt_ids) and torch.equal(output[:len(prompt_ids)], prompt_ids):
            output = output[len(prompt_ids):]
        return self.processor.decode(output, skip_special_tokens = True).strip()


class SpatialSSRL(Qwen2VLChat):
    def generate_inner(self, message, dataset = None):
        last_text = max(i for i, item in enumerate(message) if item['type'] == 'text')
        message = [dict(item) for item in message]
        message[last_text]['value'] += SSRL_FORMAT_PROMPT
        return super().generate_inner(message, dataset = dataset)


BASELINES = {
    'BLIP2-FlanT5-XL': partial(BLIPFamily, model_path = 'Salesforce/blip2-flan-t5-xl',
                               prompt_template = 'Question: {} Answer:'),
    'InstructBLIP-Vicuna-7B': partial(BLIPFamily, model_path = 'Salesforce/instructblip-vicuna-7b'),
    'Mono-InternVL-2B': partial(InternVLChat, model_path = 'OpenGVLab/Mono-InternVL-2B', version = 'V2.0'),
    # Same settings as VLMEvalKit's own SpatialLadder-3B entry.
    'SpatialLadder-3B': partial(Qwen2VLChat, model_path = 'hongxingli/SpatialLadder-3B',
                                min_pixels = 1280 * 28 * 28, max_pixels = 16384 * 28 * 28, use_custom_prompt = False),
    # post_process keeps only the last \boxed{} answer; 4096 tokens as in the model card.
    'Spatial-SSRL-3B': partial(SpatialSSRL, model_path = 'internlm/Spatial-SSRL-3B', use_custom_prompt = False,
                               post_process = True, max_new_tokens = 4096),
}
