import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PatchEmbeddingLayer
from .spectral import compressed_svd

class SVDInteractionAdapter(nn.Module):
    def __init__(self, embed_dim, patch_size, svd_rank = 8):
        super().__init__()

        for name, value in (('embed_dim', embed_dim), ('patch_size', patch_size), ('svd_rank', svd_rank)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')

        self.svd_rank = svd_rank
        self.patch_embed = PatchEmbeddingLayer(1, embed_dim, patch_size)

    def compress(self, similarity, text_mask, image_mask):
        if similarity.ndim != 3:
            raise ValueError('Interaction matrices must have shape batch by text by image')
        if tuple(text_mask.shape) != tuple(similarity.shape[:2]) or tuple(image_mask.shape) != (similarity.size(0), similarity.size(2)):
            raise ValueError('Interaction masks must match the text and image axes')

        text_mask = text_mask.to(device = similarity.device, dtype = torch.bool)
        image_mask = image_mask.to(device = similarity.device, dtype = torch.bool)
        if not text_mask.any(dim = 1).all() or not image_mask.any(dim = 1).all():
            raise ValueError('Each interaction matrix must contain valid text and image tokens')

        matrices, ranks = [], []
        # Padding is not part of the SVD: it would create artificial zero singular values.
        with torch.autocast(device_type = similarity.device.type, enabled = False):
            dtype = torch.float64 if similarity.dtype == torch.float64 else torch.float32
            for matrix, words, patches in zip(similarity, text_mask, image_mask):
                columns = patches.nonzero(as_tuple = False).flatten()
                valid = matrix[words].index_select(1, columns).to(dtype)
                rank = min(self.svd_rank, valid.size(0), valid.size(1))
                coefficients = compressed_svd(valid, rank)
                # Restore only the image axis; rows now contain SVD components.
                coefficients = coefficients.new_zeros(rank, similarity.size(2)).index_copy(1, columns, coefficients)
                matrices.append(coefficients)
                ranks.append(rank)

        rank = max(ranks)
        coefficients = torch.stack([F.pad(matrix, (0, 0, 0, rank - matrix.size(0))) for matrix in matrices])
        component_mask = torch.arange(rank, device = similarity.device)[None] < torch.tensor(ranks, device = similarity.device)[:, None]

        return coefficients, component_mask

    def forward(self, similarity, text_mask, image_mask = None, column_indices = None):
        if image_mask is None:
            image_mask = torch.ones((similarity.size(0), similarity.size(2)), dtype = torch.bool, device = similarity.device)
        image_mask = image_mask.to(device = similarity.device, dtype = torch.bool)
        coefficients, component_mask = self.compress(similarity, text_mask, image_mask)
        map_mask = component_mask[:, :, None] & image_mask[:, None, :]
        interaction_map = coefficients.unsqueeze(1)
        # Rows are spectral component groups, not original question positions.
        tokens, token_mask = self.patch_embed(interaction_map.to(self.patch_embed.patchify.weight.dtype), mask = map_mask,
                                             column_indices = column_indices)

        return tokens, token_mask
