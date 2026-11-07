"""CPU training smoke tests with synthetic inputs and no pretrained downloads."""

import io
import math
import os
import sys
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from configuration import ABLATIONS, MODEL_DEFAULTS, configure_model
from main import batch_loss, evaluate, parse_args, train_step
from model.ipmoe import IPMoE
from model.llm import NoValidTargetsError


class TinyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = SimpleNamespace(config = SimpleNamespace(
            text_config = SimpleNamespace(bos_token_id = 1, eos_token_id = 2)))
        self.embedding = nn.Embedding(16, 8).requires_grad_(False)
        self.batch_sizes = []

    def forward(self, input_ids, attention_mask):
        self.batch_sizes.append(input_ids.size(0))
        rows = [ids[mask.bool()][1:-1] for ids, mask in zip(input_ids, attention_mask)]
        lengths = torch.tensor([len(row) for row in rows], device = input_ids.device)
        features = self.embedding(pad_sequence(rows, batch_first = True))
        mask = torch.arange(features.size(1), device = input_ids.device)[None] < lengths[:, None]
        return features, mask


class TinyVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = SimpleNamespace(config = SimpleNamespace(
            vision_config = SimpleNamespace(image_size = 4, patch_size = 2)))
        self.projection = nn.Conv2d(3, 8, kernel_size = 2, stride = 2).requires_grad_(False)
        self.batch_sizes = []

    def forward(self, pixel_values):
        self.batch_sizes.append(pixel_values.size(0))
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.readout = nn.Linear(8, 16)

    def get_input_embeddings(self):
        return self.embedding


class TinyLLM(nn.Module):
    """A prefix-sensitive causal loss; one whitespace word is one token."""

    def __init__(self):
        super().__init__()
        self.model = TinyDecoder()
        self.grad_modes = []

    def tokenize_training_chats(self, questions, answers, max_total_tokens):
        token_rows, label_rows, kept_rows = [], [], []
        for index, (question, answer, budget) in enumerate(zip(questions, answers, max_total_tokens)):
            prompt = [1] + [3] * len(question.split())
            target = [4] * len(answer.split()) + [2]
            if len(prompt) + len(target) > budget:
                continue
            token_rows.append(prompt + target)
            label_rows.append([-100] * len(prompt) + target)
            kept_rows.append(index)
        if not kept_rows:
            raise NoValidTargetsError('No synthetic answer fits the context budget')
        return token_rows, label_rows, kept_rows, [0] * len(kept_rows)

    def build_inputs(self, token_rows, positions, prefix, label_rows, prefix_mask):
        embeddings, labels = [], []
        for tokens, target, features, valid in zip(token_rows, label_rows, prefix, prefix_mask):
            features = features[valid]
            ids = torch.tensor(tokens, device = prefix.device)
            embeddings.append(torch.cat((features, self.model.embedding(ids))))
            labels.append(torch.tensor([-100] * len(features) + target, device = prefix.device))
        lengths = torch.tensor([len(row) for row in embeddings], device = prefix.device)
        embeddings = pad_sequence(embeddings, batch_first = True)
        labels = pad_sequence(labels, batch_first = True, padding_value = -100)
        mask = torch.arange(embeddings.size(1), device = prefix.device)[None] < lengths[:, None]
        position_ids = (mask.long().cumsum(dim = 1) - 1).clamp_min(0)
        return embeddings, mask, position_ids, labels

    def forward(self, embeds, mask, position_ids, labels):
        self.grad_modes.append(torch.is_grad_enabled())
        logits = self.model.readout(embeds.cumsum(dim = 1))
        return F.cross_entropy(logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1), ignore_index = -100)


def make_batch(answers):
    count = len(answers)
    text_inputs = {
        'input_ids': torch.tensor([[1, 3, 4, 2]] * count),
        'attention_mask': torch.ones(count, 4, dtype = torch.long),
    }
    image_inputs = {'pixel_values': torch.randn(count, 3, 4, 4)}
    return ['synthetic'] * count, ['question'] * count, text_inputs, image_inputs, answers


class TrainingSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = IPMoE(TinyTextEncoder(), TinyVisionEncoder(), TinyLLM(),
                           embed_dim = 8, patch_size = 2, svd_rank = 2,
                           evidence_dim = 8, position_dim = 4, selector_dim = 4)
        self.device = torch.device('cpu')
        self.context_limit = 12
        self.too_long = ' '.join(['answer'] * 20)

    def optimizer_and_scheduler(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.5)
        return optimizer, scheduler

    def test_step_updates_parameters_once_and_excludes_skipped_rows(self):
        batch = make_batch(['answer', self.too_long])
        optimizer, scheduler = self.optimizer_and_scheduler()
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        initial_epoch = scheduler.last_epoch
        self.model.eval()

        loss, retained, grad_norm = train_step(self.model, batch, optimizer, scheduler,
                                              self.device, self.context_limit, 1.0)

        self.assertTrue(math.isfinite(loss))
        self.assertEqual(retained, 1)
        self.assertTrue(math.isfinite(grad_norm))
        self.assertGreater(grad_norm, 0)
        self.assertTrue(self.model.training)
        self.assertEqual(self.model.llm.grad_modes, [True])
        self.assertEqual(self.model.text_encoder.batch_sizes, [1])
        self.assertEqual(self.model.vision_encoder.batch_sizes, [1])
        self.assertEqual(scheduler.last_epoch, initial_epoch + 1)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.0005)
        self.assertTrue(optimizer.state)
        self.assertTrue(all(state['step'].item() == 1 for state in optimizer.state.values()))
        for name in ('interaction_head.weight', 'evidence_head.weight'):
            self.assertFalse(torch.equal(before[name], dict(self.model.named_parameters())[name]), name)

        with torch.no_grad():
            _, retained, tokens = batch_loss(self.model, batch, self.device, self.context_limit)
        self.assertEqual((retained, tokens), (1, 2))  # Answer plus EOS; no prompt or prefix labels.

    def test_all_skipped_step_does_not_advance_optimizer_or_scheduler(self):
        batch = make_batch([self.too_long, self.too_long])
        optimizer, scheduler = self.optimizer_and_scheduler()
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        initial_epoch = scheduler.last_epoch

        result = train_step(self.model, batch, optimizer, scheduler, self.device, self.context_limit, 1.0)

        self.assertEqual(result, (None, 0, None))
        self.assertEqual(scheduler.last_epoch, initial_epoch)
        self.assertFalse(optimizer.state)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, self.model.parameters())))
        self.assertEqual(self.model.text_encoder.batch_sizes, [])
        self.assertEqual(self.model.vision_encoder.batch_sizes, [])

    def test_allowed_empty_batch_has_zero_gradient_for_every_trainable_parameter(self):
        batch = make_batch([self.too_long])
        with self.assertRaises(NoValidTargetsError):
            batch_loss(self.model, batch, self.device, self.context_limit)

        loss, retained, tokens = batch_loss(self.model, batch, self.device, self.context_limit, allow_empty = True)

        self.assertEqual((retained, tokens), (0, 0))
        self.assertEqual(loss.item(), 0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, name)
        self.assertEqual(self.model.text_encoder.batch_sizes, [])
        self.assertEqual(self.model.vision_encoder.batch_sizes, [])

    def test_evaluation_uses_answer_token_weights_and_counts_skipped_rows(self):
        batches = [make_batch(['answer', 'answer answer answer', self.too_long]),
                   make_batch(['answer answer']), make_batch([self.too_long, self.too_long])]
        self.model.eval()
        with torch.no_grad():
            first_loss, first_retained, first_tokens = batch_loss(self.model, batches[0], self.device, self.context_limit)
            second_loss, second_retained, second_tokens = batch_loss(self.model, batches[1], self.device, self.context_limit)
        self.assertEqual((first_retained, first_tokens, second_retained, second_tokens), (2, 6, 1, 3))
        expected = (first_loss.item() * 6 + second_loss.item() * 3) / 9
        self.model.llm.grad_modes.clear()
        self.model.train()

        mean_loss, skipped, seen = evaluate(self.model, batches, self.device, self.context_limit)

        self.assertAlmostEqual(mean_loss, expected, places = 6)
        self.assertEqual((skipped, seen), (3, 6))
        self.assertFalse(self.model.training)
        self.assertEqual(self.model.llm.grad_modes, [False, False])
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_evaluation_handles_empty_and_fully_skipped_shards(self):
        self.assertEqual(evaluate(self.model, [], self.device, self.context_limit), (None, 0, 0))
        batch = make_batch([self.too_long, self.too_long])
        self.assertEqual(evaluate(self.model, [batch], self.device, self.context_limit), (None, 2, 2))


class ConfigurationSmokeTests(unittest.TestCase):
    def test_configuration_round_trip_preserves_complete_checkpoint(self):
        args = Namespace()
        config = configure_model(args)
        self.assertEqual(config, MODEL_DEFAULTS)
        restored = Namespace(**vars(args))
        self.assertEqual(configure_model(restored, checkpoint = True), config)

    def test_saved_checkpoint_round_trip_preserves_each_ablation_and_seed(self):
        for ablation in ABLATIONS:
            with self.subTest(ablation = ablation):
                args = Namespace(ablation = ablation, selection_seed = 17)
                expected = configure_model(args)
                checkpoint = io.BytesIO()
                torch.save({'args': vars(args)}, checkpoint)
                checkpoint.seek(0)
                saved = torch.load(checkpoint, map_location = 'cpu', weights_only = True)
                restored = Namespace(**saved['args'])
                self.assertEqual(configure_model(restored, checkpoint = True), expected)
                self.assertEqual((restored.ablation, restored.selection_seed), (ablation, 17))

    def test_legacy_checkpoint_defaults_only_when_both_variant_fields_are_absent(self):
        legacy = {field: value for field, value in MODEL_DEFAULTS.items() if field not in ('ablation', 'selection_seed')}
        restored = Namespace(**legacy)
        self.assertEqual(configure_model(restored, checkpoint = True), MODEL_DEFAULTS)
        self.assertEqual((restored.ablation, restored.selection_seed), ('full', 42))
        for field in legacy:
            with self.subTest(missing = field):
                values = dict(legacy)
                del values[field]
                with self.assertRaises(ValueError):
                    configure_model(Namespace(**values), checkpoint = True)
        for field, value in (('ablation', 'full'), ('selection_seed', 42)):
            with self.subTest(partial_variant = field):
                with self.assertRaisesRegex(ValueError, 'incomplete model configuration'):
                    configure_model(Namespace(**dict(legacy, **{field: value})), checkpoint = True)

    def test_checkpoint_requires_architecture_and_every_model_field(self):
        for field in MODEL_DEFAULTS:
            with self.subTest(missing = field):
                values = dict(MODEL_DEFAULTS)
                del values[field]
                with self.assertRaises(ValueError):
                    configure_model(Namespace(**values), checkpoint = True)
        for field, value in (('architecture', 'interaction_vit'), ('interaction_type', 'cosine'), ('ablation', None),
                             ('ablation_version', 1), ('ablation_config', {})):
            with self.subTest(field = field, value = value):
                values = dict(MODEL_DEFAULTS, **{field: value})
                with self.assertRaises(ValueError):
                    configure_model(Namespace(**values), checkpoint = True)

    def test_configuration_rejects_unsupported_ablation_and_invalid_selection_seeds(self):
        invalid = [('ablation', 'unknown'), ('ablation', ''), ('ablation', ['full']),
                   ('selection_seed', -1), ('selection_seed', True), ('selection_seed', 1.5), ('selection_seed', '42'),
                   ('selection_seed', 2 ** 63 - 1), ('selection_seed', 2 ** 63)]
        for field, value in invalid:
            with self.subTest(field = field, value = value):
                with self.assertRaises(ValueError):
                    configure_model(Namespace(**{field: value}))
        for seed in (0, 2 ** 63 - 2):
            with self.subTest(valid_seed = seed):
                self.assertEqual(configure_model(Namespace(selection_seed = seed))['selection_seed'], seed)

    def test_configuration_rejects_invalid_dimensions_and_nonfinite_controls(self):
        invalid = [('embed_dim', True), ('svd_rank', 0), ('patch_size', -1), ('position_dim', 1),
                   ('selector_keep_ratio', 0), ('image_keep_ratio', 1.1), ('selector_temperature', 0),
                   ('drop', 1), ('drop', float('nan')), ('selector_temperature', float('inf'))]
        for field, value in invalid:
            with self.subTest(field = field, value = value):
                with self.assertRaises((TypeError, ValueError)):
                    configure_model(Namespace(**{field: value}))


class CommandLineSmokeTests(unittest.TestCase):
    def test_default_path_preserves_full_and_separates_ablation_runs(self):
        with patch.object(sys, 'argv', ['main.py']):
            args = parse_args()
        self.assertEqual((args.ablation, args.selection_seed), ('full', 42))
        self.assertEqual(args.output_dir, os.path.join('checkpoint', 'svd_evidence_prefix', 'seed_42'))

        for ablation in ABLATIONS:
            with self.subTest(ablation = ablation):
                with patch.object(sys, 'argv', ['main.py', '--ablation', ablation, '--seed', '7', '--selection-seed', '13']):
                    args = parse_args()
                directory = os.path.join('checkpoint', 'svd_evidence_prefix')
                if ablation != 'full':
                    directory = os.path.join(directory, ablation)
                self.assertEqual(args.output_dir, os.path.join(directory, 'seed_7'))
                self.assertEqual((args.ablation, args.selection_seed), (ablation, 13))

    def test_explicit_output_path_is_preserved(self):
        with patch.object(sys, 'argv', ['main.py', '--ablation', 'random_selection', '--output-dir', 'custom-checkpoint']):
            args = parse_args()
        self.assertEqual(args.output_dir, 'custom-checkpoint')

    def test_cli_rejects_unknown_variant_and_out_of_range_selection_seed(self):
        with patch.object(sys, 'argv', ['main.py', '--ablation', 'unknown']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args()
        self.assertEqual(raised.exception.code, 2)
        for seed in (-1, 2 ** 63 - 1):
            with self.subTest(seed = seed):
                with patch.object(sys, 'argv', ['main.py', '--selection-seed', str(seed)]):
                    with self.assertRaisesRegex(ValueError, 'selection_seed'):
                        parse_args()


if __name__ == '__main__':
    unittest.main()
