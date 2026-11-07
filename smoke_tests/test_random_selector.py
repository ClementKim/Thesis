import unittest
from unittest.mock import patch

import torch

from model.selector import TokenSelector


class RandomSelectorSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def make_inputs(self):
        token_mask = torch.tensor([[True] * 7, [False, True, False, True, False, True, False],
                                   [False, False, True, False, False, False, False]])
        context_mask = torch.tensor([[True] * 4, [True, False, True, False], [True, False, False, False]])
        tokens = torch.randn(3, 7, 6).masked_fill(~token_mask[:, :, None], float('nan'))
        context = torch.randn(3, 4, 6).masked_fill(~context_mask[:, :, None], float('nan'))

        return tokens, token_mask, context, context_mask

    def test_random_selection_has_no_scoring_parameters_or_calls(self):
        selector = TokenSelector(6, selection = 'random')
        self.assertEqual(list(selector.parameters()), [])
        self.assertEqual(dict(selector.state_dict()), {})
        with patch.object(selector, 'score_tokens', side_effect = AssertionError('Random selection computed scores')):
            for training in (True, False):
                selector.train(training)
                selector(*self.make_inputs())
        with self.assertRaisesRegex(RuntimeError, 'does not compute token scores'):
            selector.score_tokens(*self.make_inputs())

    def test_random_budgets_source_order_and_padding_match_learned(self):
        tokens, token_mask, context, context_mask = self.make_inputs()
        random = TokenSelector(6, keep_ratio = 0.5, selection = 'random').eval()
        learned = TokenSelector(6, keep_ratio = 0.5).eval()
        selected, mask, indices = random(tokens, token_mask, context, context_mask)
        _, learned_mask, _ = learned(tokens, token_mask, context, context_mask)

        self.assertEqual(mask.sum(dim = 1).tolist(), [4, 2, 1])
        torch.testing.assert_close(mask, learned_mask)
        self.assertTrue(torch.isfinite(selected).all().item())
        self.assertEqual(selected[~mask].count_nonzero().item(), 0)
        self.assertTrue((indices[~mask] == -1).all().item())
        for sample in range(3):
            source = indices[sample, mask[sample]]
            self.assertEqual(source.tolist(), sorted(set(source.tolist())))
            self.assertTrue(token_mask[sample, source].all().item())
            torch.testing.assert_close(selected[sample, mask[sample]], tokens[sample, source], rtol = 0, atol = 0)

    def test_random_backward_only_reaches_selected_original_features(self):
        tokens, token_mask, context, context_mask = self.make_inputs()
        tokens.requires_grad_()
        context.requires_grad_()
        selector = TokenSelector(6, keep_ratio = 0.5, selection = 'random').train()

        selected, mask, indices = selector(tokens, token_mask, context, context_mask)
        selected.sum().backward()

        expected = torch.zeros_like(tokens)
        for sample in range(3):
            expected[sample, indices[sample, mask[sample]]] = 1
        torch.testing.assert_close(tokens.grad, expected, rtol = 0, atol = 0)
        self.assertIsNone(context.grad)

    def test_random_eval_is_reproducible_and_batch_independent(self):
        selector = TokenSelector(6, keep_ratio = 0.5, selection = 'random', seed = 123).eval()
        tokens, token_mask, context, context_mask = self.make_inputs()
        rng_before = torch.random.get_rng_state().clone()
        selected, mask, indices = selector(tokens, token_mask, context, context_mask)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol = 0, atol = 0)

        order = torch.tensor([2, 0, 1])
        reordered = selector(tokens[order], token_mask[order], context[order], context_mask[order])
        for original, actual in zip((selected, mask, indices), reordered):
            torch.testing.assert_close(actual, original[order], rtol = 0, atol = 0)

        for sample in range(3):
            token_padding = torch.full((1, 5, 6), float('nan'))
            context_padding = torch.full((1, 3, 6), float('nan'))
            single = selector(torch.cat((tokens[sample:sample + 1], token_padding), dim = 1),
                              torch.cat((token_mask[sample:sample + 1], torch.zeros(1, 5, dtype = torch.bool)), dim = 1),
                              torch.cat((context[sample:sample + 1], context_padding), dim = 1),
                              torch.cat((context_mask[sample:sample + 1], torch.zeros(1, 3, dtype = torch.bool)), dim = 1))
            torch.testing.assert_close(single[0][0], selected[sample, mask[sample]], rtol = 0, atol = 0)
            torch.testing.assert_close(single[2][0], indices[sample, mask[sample]], rtol = 0, atol = 0)

        repeated = TokenSelector(6, selection = 'random', seed = 123).eval()(tokens, token_mask, context, context_mask)
        torch.testing.assert_close(repeated[2], indices, rtol = 0, atol = 0)

    def test_random_eval_uses_seed_and_length_without_feature_content(self):
        tokens = torch.randn(2, 64, 6)
        context = torch.randn(2, 8, 6)
        token_mask = torch.ones(2, 64, dtype = torch.bool)
        context_mask = torch.ones(2, 8, dtype = torch.bool)
        first = TokenSelector(6, selection = 'random', seed = 123).eval()
        second = TokenSelector(6, selection = 'random', seed = 456).eval()

        _, _, indices = first(tokens, token_mask, context, context_mask)
        _, _, changed = second(tokens, token_mask, context, context_mask)
        _, _, other_features = first(tokens * 20, token_mask, -context, context_mask)

        torch.testing.assert_close(indices[0], indices[1], rtol = 0, atol = 0)
        torch.testing.assert_close(indices, other_features, rtol = 0, atol = 0)
        self.assertFalse(torch.equal(indices, changed))

    def test_random_training_resamples_and_repeats_with_global_seed(self):
        selector = TokenSelector(6, selection = 'random', seed = 123).train()
        tokens = torch.randn(1, 64, 6)
        context = torch.randn(1, 8, 6)
        token_mask = torch.ones(1, 64, dtype = torch.bool)
        context_mask = torch.ones(1, 8, dtype = torch.bool)

        torch.manual_seed(37)
        first = selector(tokens, token_mask, context, context_mask)[2]
        next_sample = selector(tokens, token_mask, context, context_mask)[2]
        torch.manual_seed(37)
        repeated = selector(tokens, token_mask, context, context_mask)[2]

        torch.testing.assert_close(first, repeated, rtol = 0, atol = 0)
        self.assertFalse(torch.equal(first, next_sample))

    def test_random_selection_retains_shape_and_nonempty_validation(self):
        selector = TokenSelector(6, selection = 'random')
        tokens, token_mask, context, context_mask = self.make_inputs()
        invalid = [(tokens[:, :, :5], token_mask, context, context_mask),
                   (tokens, token_mask[:, :6], context, context_mask),
                   (tokens, token_mask, context[:1], context_mask[:1]),
                   (tokens, torch.zeros_like(token_mask), context, context_mask),
                   (tokens, token_mask, context, torch.zeros_like(context_mask))]
        for inputs in invalid:
            with self.subTest(shapes = [tuple(value.shape) for value in inputs]):
                with self.assertRaises(ValueError):
                    selector(*inputs)


if __name__ == '__main__':
    unittest.main()
