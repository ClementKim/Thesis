"""CLIP precision boundaries with tiny local models and no pretrained downloads."""

import unittest
from types import SimpleNamespace

import torch
from transformers import CLIPConfig, CLIPModel

from model.encoder import TextEncoder, VisionEncoder


class EncoderPrecisionTests(unittest.TestCase):
    def check_precision_and_gradients(self, device, clip_dtype):
        torch.manual_seed(7)
        config = CLIPConfig(
            projection_dim = 8,
            text_config = dict(vocab_size = 32, hidden_size = 16, intermediate_size = 32,
                               num_hidden_layers = 2, num_attention_heads = 2,
                               max_position_embeddings = 6, bos_token_id = 1,
                               eos_token_id = 2, pad_token_id = 0),
            vision_config = dict(hidden_size = 16, intermediate_size = 32,
                                 num_hidden_layers = 2, num_attention_heads = 2,
                                 image_size = 8, patch_size = 4),
        )
        clip = CLIPModel(config).to(device).requires_grad_(False).eval()
        wrapper = SimpleNamespace(model = clip)
        text_encoder = TextEncoder(wrapper, drop = 0, out_dim = 6, chunk_batch_size = 2).to(device).train()
        vision_encoder = VisionEncoder(wrapper, drop = 0, out_dim = 6).to(device).train()
        observed = {'text': [], 'vision': []}

        def record_dtype(branch):
            def hook(module, inputs, output):
                observed[branch].append(output.dtype)
            return hook

        handles = [clip.text_projection.register_forward_hook(record_dtype('text')),
                   clip.vision_model.embeddings.patch_embedding.register_forward_hook(record_dtype('vision'))]
        # The first question spans two CLIP chunks; the second needs one.
        ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 10, 2],
                            [1, 11, 12, 2, 0, 0, 0, 0, 0, 0]], device = device)
        mask = ids.ne(0).long()
        pixels = torch.randn(2, 3, 8, 8, device = device, requires_grad = True)
        try:
            text, text_mask = text_encoder(ids, mask)
            image = vision_encoder(pixels)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(observed['text'], [clip_dtype, clip_dtype])
        self.assertEqual(observed['vision'], [clip_dtype])
        self.assertEqual(tuple(text.shape), (2, 8, 6))
        self.assertEqual(tuple(image.shape), (2, 4, 6))
        self.assertEqual(text_mask.sum(dim = 1).tolist(), [8, 2])
        self.assertEqual(text[1, 2:].count_nonzero().item(), 0)
        self.assertEqual(ids.dtype, torch.long)
        for features in (text, image):
            self.assertEqual(features.dtype, torch.float32)
            self.assertTrue(torch.isfinite(features).all().item())

        (text.square().mean() + image.square().mean()).backward()
        for encoder in (text_encoder, vision_encoder):
            for parameter in encoder.mlp.parameters():
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in encoder.mlp.parameters()), 0)
        self.assertIsNone(pixels.grad)
        self.assertFalse(clip.training)
        self.assertTrue(all(parameter.grad is None for parameter in clip.parameters()))

    def test_cpu_keeps_float32_and_projection_gradients(self):
        self.check_precision_and_gradients('cpu', torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required to exercise FP16 CLIP')
    def test_cuda_uses_fp16_clip_and_float32_projection_gradients(self):
        self.check_precision_and_gradients('cuda', torch.float16)


if __name__ == '__main__':
    unittest.main()
