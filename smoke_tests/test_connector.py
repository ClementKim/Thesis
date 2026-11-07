import math
import unittest

import torch

from model.common import PatchEmbeddingLayer
from model.evidence import CorrespondenceSelector
from model.interaction import SVDInteractionAdapter
from model.selector import TokenSelector
from model.spectral import compressed_svd


class ConnectorSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def assert_finite_gradient(self, tensor, nonzero = True):
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(torch.isfinite(tensor.grad).all().item())
        if nonzero:
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_selector_keeps_source_order_and_masks_variable_lengths(self):
        selector = TokenSelector(6, keep_ratio = 0.5, selector_dim = 4).eval()
        # Equal learned scores must favor earlier valid source positions.
        with torch.no_grad():
            selector.score[-1].weight.zero_()
        token_mask = torch.tensor([[True, True, True, True, True], [False, True, False, True, False]])
        context_mask = torch.tensor([[True, True, True], [True, False, False]])
        tokens = torch.randn(2, 5, 6).masked_fill(~token_mask[:, :, None], float('nan'))
        context = torch.randn(2, 3, 6).masked_fill(~context_mask[:, :, None], float('nan'))

        selected, mask, indices = selector(tokens, token_mask, context, context_mask)

        self.assertEqual(indices.tolist(), [[0, 1, 2], [1, -1, -1]])
        self.assertEqual(mask.tolist(), [[True, True, True], [True, False, False]])
        torch.testing.assert_close(selected[0], tokens[0, :3])
        torch.testing.assert_close(selected[1, 0], tokens[1, 1])
        self.assertEqual(selected[1, 1:].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(selected).all().item())

    def test_selector_surrogate_gradients_preserve_gathered_features(self):
        selector = TokenSelector(6, keep_ratio = 0.5, selector_dim = 5).train()
        tokens = torch.randn(2, 5, 6, requires_grad = True)
        context = torch.randn(2, 4, 6, requires_grad = True)
        token_mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
        context_mask = torch.tensor([[True] * 4, [True, False, True, False]])

        selected, mask, indices = selector(tokens, token_mask, context, context_mask)
        for sample in range(2):
            source = indices[sample, mask[sample]]
            self.assertEqual(source.tolist(), sorted(set(source.tolist())))
            torch.testing.assert_close(selected[sample, mask[sample]], tokens[sample, source], rtol = 0, atol = 0)
        selected.square().sum().backward()

        for tensor in (tokens, context, selector.score[-1].weight, selector.token_projection.weight,
                       selector.target_projection.weight, selector.context_projection.weight):
            self.assert_finite_gradient(tensor)
        self.assertEqual(tokens.grad[~token_mask].count_nonzero().item(), 0)
        self.assertEqual(context.grad[~context_mask].count_nonzero().item(), 0)

    def test_svd_exact_coefficients_sign_convention_and_rank_limit(self):
        matrix = torch.tensor([[-5., 0., 0., 0.], [0., 3., 0., 0.], [0., 0., -1., 0.]], dtype = torch.float64)
        expected = torch.tensor([[5., 0., 0., 0.], [0., 3., 0., 0.]], dtype = torch.float64)

        torch.testing.assert_close(compressed_svd(matrix, 2), expected)
        full = compressed_svd(matrix, 20)
        self.assertEqual(tuple(full.shape), (3, 4))
        self.assertEqual(full.dtype, torch.float64)
        torch.testing.assert_close(full.T @ full, matrix.T @ matrix)
        promoted = compressed_svd(matrix.to(torch.float16), 2)
        self.assertEqual(promoted.dtype, torch.float32)
        torch.testing.assert_close(promoted, expected.float())

    def test_svd_backward_is_exact_away_from_ties_and_finite_at_degeneracy(self):
        matrix = torch.randn(4, 3, dtype = torch.float64, requires_grad = True)
        self.assertTrue(torch.autograd.gradcheck(lambda value: compressed_svd(value, 2), (matrix,),
                                                eps = 1e-6, atol = 1e-5, rtol = 1e-3))
        for values in ([0., 0., 0.], [1., 1., 1.], [1., 1. + 1e-10, 0.], [1e-20, 1e-20, 0.], [1e20, 1e20, 0.]):
            with self.subTest(values = values):
                matrix = torch.diag(torch.tensor(values, dtype = torch.float64)).requires_grad_()
                coefficients = compressed_svd(matrix, 2)
                coefficients.backward(torch.randn_like(coefficients))
                self.assert_finite_gradient(matrix, nonzero = False)

    def test_patch_padding_preserves_boundary_cells_and_source_columns(self):
        layer = PatchEmbeddingLayer(1, 7, 2)
        # Isolate position signals from learned patch content.
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.zero_()
        mask = torch.ones(2, 3, 5, dtype = torch.bool)
        mask[1] = False
        mask[1, 0, :2] = True
        maps = torch.randn(2, 1, 3, 5).masked_fill(~mask[:, None], float('nan'))
        columns = torch.tensor([[0, 1, 4, 5, 8], [0, 1, -1, -1, -1]])

        tokens, token_mask = layer(maps, mask, column_indices = columns)

        self.assertEqual(tuple(tokens.shape), (2, 6, 7))
        self.assertEqual(token_mask.sum(dim = 1).tolist(), [6, 1])
        self.assertTrue(torch.isfinite(tokens).all().item())
        self.assertEqual(tokens[1, 1:].count_nonzero().item(), 0)
        # The last partial column contains source cell 8, whose source block is 4.
        expected = torch.tensor([[0., 1., 0., 1.],
                                 [math.sin(2), math.cos(2), math.sin(0.02), math.cos(0.02)],
                                 [math.sin(4), math.cos(4), math.sin(0.04), math.cos(0.04)]])
        torch.testing.assert_close(tokens[0, :3, 3:], expected)

    def test_spectral_adapter_has_default_18_tokens_and_ignores_padding(self):
        adapter = SVDInteractionAdapter(8, patch_size = 16, svd_rank = 8)
        text_mask = torch.tensor([[True] * 9, [True] * 3 + [False] * 6])
        image_mask = torch.tensor([[True] * 288, [True] * 17 + [False] * 271])
        valid = text_mask[:, :, None] & image_mask[:, None, :]
        similarity = torch.randn(2, 9, 288).masked_fill(~valid, float('nan')).requires_grad_()
        columns = torch.arange(0, 576, 2).expand(2, -1).masked_fill(~image_mask, -1)

        coefficients, components = adapter.compress(similarity, text_mask, image_mask)
        tokens, mask = adapter(similarity, text_mask, image_mask, column_indices = columns)

        self.assertEqual(tuple(coefficients.shape), (2, 8, 288))
        self.assertEqual(components.sum(dim = 1).tolist(), [8, 3])
        self.assertEqual(tuple(tokens.shape), (2, 18, 8))
        self.assertEqual(mask.sum(dim = 1).tolist(), [18, 2])
        self.assertEqual(tokens[~mask].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(tokens).all().item())
        # Padding and other batch members cannot change an example's valid prefix.
        single, single_mask = adapter(similarity[1:2, :3, :17], torch.ones(1, 3, dtype = torch.bool),
                                      column_indices = columns[1:2, :17])
        torch.testing.assert_close(tokens[1, mask[1]], single[0, single_mask[0]])
        tokens.square().sum().backward()
        self.assert_finite_gradient(similarity)
        self.assertEqual(similarity.grad[~valid].count_nonzero().item(), 0)

    def test_evidence_ignores_padding_and_uses_original_positions(self):
        evidence = CorrespondenceSelector(8, evidence_dim = 6, position_dim = 5)
        text_mask = torch.tensor([[True, True, False]])
        image_mask = torch.tensor([[True, True, False]])
        context_mask = torch.tensor([[True, True, False]])
        text_indices = torch.tensor([[1, 7, -1]])
        image_indices = torch.tensor([[0, 8, -1]])
        similarity = torch.randn(1, 3, 3)
        text, question, spectral = (torch.randn(1, 3, 8) for _ in range(3))

        def summarize(scores, words, question_features, spectral_features, positions):
            return evidence(scores, words, text_mask, image_mask, positions, image_indices,
                            question_features, context_mask, spectral_features, context_mask, grid_size = 3)

        expected = summarize(similarity, text, question, spectral, text_indices)
        actual = summarize(similarity.masked_fill(~(text_mask[:, :, None] & image_mask[:, None, :]), float('nan')),
                           text.masked_fill(~text_mask[:, :, None], float('nan')),
                           question.masked_fill(~context_mask[:, :, None], float('nan')),
                           spectral.masked_fill(~context_mask[:, :, None], float('nan')), text_indices)
        self.assertEqual(tuple(actual.shape), (1, 2, 6))
        torch.testing.assert_close(actual, expected)
        shifted = summarize(similarity, text, question, spectral, torch.tensor([[2, 9, -1]]))
        self.assertFalse(torch.allclose(expected, shifted))

    def test_gradients_connect_both_selectors_spectral_and_evidence(self):
        text_selector = TokenSelector(8, keep_ratio = 0.6, selector_dim = 5)
        image_selector = TokenSelector(8, keep_ratio = 0.5, selector_dim = 5)
        adapter = SVDInteractionAdapter(8, patch_size = 2, svd_rank = 3)
        evidence = CorrespondenceSelector(8, evidence_dim = 6, position_dim = 5)
        text = torch.randn(2, 5, 8, requires_grad = True)
        image = torch.randn(2, 9, 8, requires_grad = True)
        text_mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
        image_mask = torch.tensor([[True] * 9, [True] * 4 + [False] * 5])

        words, word_mask, word_indices = text_selector(text, text_mask, image, image_mask)
        patches, patch_mask, patch_indices = image_selector(image, image_mask, text, text_mask)
        similarity = torch.bmm(words, patches.transpose(1, 2))
        spectral, spectral_mask = adapter(similarity, word_mask, patch_mask, column_indices = patch_indices)
        summaries = evidence(similarity, words, word_mask, patch_mask, word_indices, patch_indices,
                             text, text_mask, spectral, spectral_mask, grid_size = 3)
        (spectral.square().mean() + summaries.square().mean()).backward()

        self.assertEqual(tuple(summaries.shape), (2, 2, 6))
        for tensor in (text, image, text_selector.score[-1].weight, image_selector.score[-1].weight,
                       adapter.patch_embed.patchify.weight, evidence.record_encoder[0].weight,
                       evidence.question_projection.weight, evidence.spectral_projection.weight,
                       evidence.previous_projection.weight):
            self.assert_finite_gradient(tensor)
        self.assertEqual(text.grad[~text_mask].count_nonzero().item(), 0)
        self.assertEqual(image.grad[~image_mask].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
