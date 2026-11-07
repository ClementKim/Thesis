"""Ablation smoke tests with synthetic encoders and local CPU decoders."""

import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from configuration import ABLATIONS
from main import train_step
from model.ipmoe import IPMoE
from model.llm import MM_MARKER, NoValidTargetsError
from smoke_tests import test_llm
from smoke_tests.test_training import TinyLLM, TinyTextEncoder, TinyVisionEncoder


class AblationSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def make_model(self, ablation = None, image_size = 8, patch_size = 2, svd_rank = 4,
                   llm = None, trainable_encoders = False):
        text_encoder = TinyTextEncoder()
        vision_encoder = TinyVisionEncoder()
        vision_encoder.clip.config.vision_config.image_size = image_size
        text_encoder.embedding.requires_grad_(trainable_encoders)
        vision_encoder.projection.requires_grad_(trainable_encoders)
        options = dict(embed_dim = 8, patch_size = patch_size, svd_rank = svd_rank,
                       evidence_dim = 8, position_dim = 4, selector_dim = 4)
        if ablation is not None:
            options['ablation'] = ablation
        return IPMoE(text_encoder, vision_encoder, TinyLLM() if llm is None else llm, **options)

    def make_inputs(self, lengths, image_size = 8):
        rows = [torch.tensor([1] + [3 + index % 13 for index in range(length)] + [2]) for length in lengths]
        ids = pad_sequence(rows, batch_first = True)
        text_inputs = {'input_ids': ids, 'attention_mask': (ids != 0).long()}
        image_inputs = {'pixel_values': torch.randn(len(lengths), 3, image_size, image_size)}
        return text_inputs, image_inputs

    def test_default_full_matches_explicit_full_and_checkpoint_keys(self):
        torch.manual_seed(19)
        default = self.make_model().eval()
        torch.manual_seed(19)
        explicit = self.make_model('full').eval()
        default_state, explicit_state = default.state_dict(), explicit.state_dict()
        self.assertEqual(set(default_state), set(explicit_state))
        self.assertEqual({key.split('.')[0] for key in default_state},
                         {'text_encoder', 'vision_encoder', 'llm', 'interaction_adapter', 'evidence_selector',
                          'interaction_head', 'evidence_head', 'text_selector', 'image_selector'})
        for key in default_state:
            torch.testing.assert_close(default_state[key], explicit_state[key], rtol = 0, atol = 0)
        explicit.load_state_dict(default_state, strict = True)
        text_inputs, image_inputs = self.make_inputs([1, 5])
        with torch.no_grad():
            default_prefix, default_mask = default.encode_prefix(text_inputs, image_inputs)
            explicit_prefix, explicit_mask = explicit.encode_prefix(text_inputs, image_inputs)
        torch.testing.assert_close(default_prefix, explicit_prefix, rtol = 0, atol = 0)
        torch.testing.assert_close(default_mask, explicit_mask)

    def test_variant_prefix_lengths_match_predictions_for_variable_text_lengths(self):
        text_inputs, image_inputs = self.make_inputs([1, 3, 5, 7])
        expected = {'full': [6, 6, 10, 10], 'spectral_only': [4, 4, 8, 8],
                    'evidence_only': [2, 2, 2, 2], 'random_selection': [6, 6, 10, 10],
                    'direct_image': [6, 6, 10, 10]}
        for ablation in ABLATIONS:
            with self.subTest(ablation = ablation):
                model = self.make_model(ablation).eval()
                with torch.no_grad():
                    prefix, mask = model.encode_prefix(text_inputs, image_inputs)
                self.assertEqual(model._prefix_lengths(text_inputs, image_inputs), expected[ablation])
                self.assertEqual(mask.sum(dim = 1).tolist(), expected[ablation])
                self.assertEqual(tuple(prefix.shape[:2]), tuple(mask.shape))
                self.assertTrue(torch.isfinite(prefix).all().item())
                self.assertEqual(prefix[~mask].count_nonzero().item(), 0)

    def test_full_and_direct_image_share_default_twenty_token_budget(self):
        text_inputs, image_inputs = self.make_inputs([1, 16], image_size = 48)
        for ablation in ('full', 'direct_image'):
            with self.subTest(ablation = ablation):
                # 24 x 24 image patches, half retained, rank <= 8, interaction patches of 16, plus two evidence tokens.
                model = self.make_model(ablation, image_size = 48, patch_size = 16, svd_rank = 8).eval()
                with torch.no_grad():
                    prefix, mask = model.encode_prefix(text_inputs, image_inputs)
                self.assertEqual(model._prefix_lengths(text_inputs, image_inputs), [20, 20])
                self.assertEqual(mask.sum(dim = 1).tolist(), [20, 20])
                self.assertEqual(tuple(prefix.shape), (2, 20, 8))

    def test_variants_only_register_active_connector_parameters(self):
        spectral = self.make_model('spectral_only')
        evidence = self.make_model('evidence_only')
        random = self.make_model('random_selection')
        direct = self.make_model('direct_image', trainable_encoders = True)
        self.assertFalse(any(name.startswith(('evidence_selector.', 'evidence_head.')) for name in spectral.state_dict()))
        self.assertFalse(any(name.startswith('interaction_head.') for name in evidence.state_dict()))
        self.assertTrue(any(name.startswith('interaction_adapter.') for name in evidence.state_dict()))
        self.assertFalse(any(name.startswith(('text_selector.', 'image_selector.')) for name, _ in random.named_parameters()))
        self.assertFalse(any(name.startswith(('interaction_adapter.', 'evidence_selector.', 'interaction_head.',
                                              'evidence_head.', 'text_selector.', 'image_selector.')) for name in direct.state_dict()))
        self.assertTrue(any(name.startswith('image_head.') for name, _ in direct.named_parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in direct.text_encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in direct.vision_encoder.parameters()))

    def test_direct_image_pools_original_patches_without_text_or_interactions(self):
        model = self.make_model('direct_image').eval()
        text_inputs, image_inputs = self.make_inputs([1, 5])
        changed_text = {key: value.clone() for key, value in text_inputs.items()}
        ids = changed_text['input_ids']
        changed_text['input_ids'] = torch.where(ids > 2, 3 + (ids - 3 + 5) % 13, ids)
        self.assertFalse(torch.equal(text_inputs['input_ids'], changed_text['input_ids']))

        with torch.no_grad(), patch.object(model.text_encoder, 'forward', side_effect = AssertionError('CLIP text executed')), \
                patch('torch.bmm', side_effect = AssertionError('Interaction matrix executed')), \
                patch('model.interaction.compressed_svd', side_effect = AssertionError('SVD executed')):
            prefix, mask = model.encode_prefix(text_inputs, image_inputs)
            changed_prefix, changed_mask = model.encode_prefix(changed_text, image_inputs)
            features = model.vision_encoder(**image_inputs)
            for index, length in enumerate(mask.sum(dim = 1).tolist()):
                pooled = F.adaptive_avg_pool1d(features[index].T[None], length)[0].T
                torch.testing.assert_close(prefix[index, mask[index]], model.image_head(pooled))
        torch.testing.assert_close(prefix, changed_prefix, rtol = 0, atol = 0)
        torch.testing.assert_close(mask, changed_mask)
        self.assertEqual(model.text_encoder.batch_sizes, [])

    def test_each_variant_takes_an_answer_loss_step_with_no_unused_trainable_parameters(self):
        text_inputs, image_inputs = self.make_inputs([1, 3])
        batch = (['synthetic'] * 2, ['question'] * 2, text_inputs, image_inputs,
                 ['answer', ' '.join(['answer'] * 30)])
        for ablation in ABLATIONS:
            with self.subTest(ablation = ablation):
                model = self.make_model(ablation, trainable_encoders = True)
                active = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
                before = {name: parameter.detach().clone() for name, parameter in active.items()}
                optimizer = torch.optim.AdamW(active.values(), lr = 0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1)
                initial_epoch = scheduler.last_epoch

                loss, retained, grad_norm = train_step(model, batch, optimizer, scheduler, torch.device('cpu'), 12, 1.0)

                self.assertTrue(math.isfinite(loss))
                self.assertEqual(retained, 1)
                self.assertTrue(math.isfinite(grad_norm))
                self.assertGreater(grad_norm, 0)
                self.assertEqual(scheduler.last_epoch, initial_epoch + 1)
                self.assertEqual(model.vision_encoder.batch_sizes, [1])
                self.assertEqual(model.text_encoder.batch_sizes, [] if ablation == 'direct_image' else [1])
                for name, parameter in active.items():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in active.items()))

    def test_direct_pooling_matches_adaptive_average_values_and_gradients(self):
        for patches, length in ((576, 20), (17, 5), (3, 7)):
            with self.subTest(patches = patches, length = length):
                features = torch.randn(patches, 8, dtype = torch.float64, requires_grad = True)
                reference = features.detach().clone().requires_grad_()
                actual = IPMoE._pool_image_features(features, length)
                expected = F.adaptive_avg_pool1d(reference.T[None], length)[0].T
                torch.testing.assert_close(actual, expected)
                upstream = torch.randn_like(actual)
                actual.backward(upstream)
                expected.backward(upstream)
                torch.testing.assert_close(features.grad, reference.grad)

    def test_each_variant_handles_skipped_and_allowed_empty_batches(self):
        text_inputs, image_inputs = self.make_inputs([1, 3])
        questions = ['question'] * 2
        answers = [' '.join(['answer'] * 30)] * 2
        batch = (['synthetic'] * 2, questions, text_inputs, image_inputs, answers)
        for ablation in ABLATIONS:
            with self.subTest(ablation = ablation):
                model = self.make_model(ablation, trainable_encoders = True)
                optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr = 0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1)
                initial_epoch = scheduler.last_epoch
                result = train_step(model, batch, optimizer, scheduler, torch.device('cpu'), 12, 1.0)
                self.assertEqual(result, (None, 0, None))
                self.assertFalse(optimizer.state)
                self.assertEqual(scheduler.last_epoch, initial_epoch)
                with self.assertRaises(NoValidTargetsError):
                    model(questions, text_inputs, image_inputs, answers, context_limit = 12)

                loss, retained, tokens = model(questions, text_inputs, image_inputs, answers,
                                               context_limit = 12, allow_empty = True)
                self.assertEqual((loss.item(), retained, tokens), (0, 0, 0))
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertEqual(parameter.grad.count_nonzero().item(), 0, name)
                self.assertEqual(model.text_encoder.batch_sizes, [])
                self.assertEqual(model.vision_encoder.batch_sizes, [])

    def test_all_variants_generate_with_real_tiny_decoder_and_original_questions(self):
        text_inputs, image_inputs = self.make_inputs([1, 5])
        questions = ['Which animal?\nA. cat\nB. dog', 'What color?']
        original_questions = list(questions)
        for ablation in ABLATIONS:
            with self.subTest(ablation = ablation):
                fixture = test_llm.PrefixLLMSmokeTests('test_generation_preserves_the_original_question_and_enforces_budget')
                fixture.setUp()
                llm = fixture.llm
                llm.model.generation_config.eos_token_id = None
                model = self.make_model(ablation, llm = llm).eval()

                answers = model.generate(questions, text_inputs, image_inputs, context_limit = 96, max_new_tokens = 2)

                self.assertIsInstance(answers, list)
                self.assertEqual(len(answers), len(questions))
                self.assertTrue(all(isinstance(answer, str) for answer in answers))
                self.assertEqual(tuple(llm.tokenizer.decoded_ids.shape), (2, 2))
                self.assertEqual(questions, original_questions)
                self.assertEqual(llm.tokenizer.calls, [[{'role': 'user', 'content': MM_MARKER + '\n' + question}]
                                                       for question in questions])


if __name__ == '__main__':
    unittest.main()
