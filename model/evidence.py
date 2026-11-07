import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrespondenceSelector(nn.Module):
    num_tokens = 2

    def __init__(self, embed_dim, evidence_dim = 64, position_dim = 16):
        super().__init__()

        for name, value in (('embed_dim', embed_dim), ('evidence_dim', evidence_dim), ('position_dim', position_dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if position_dim < 2:
            raise ValueError('position_dim must encode both image coordinates')

        self.embed_dim = embed_dim
        self.evidence_dim = evidence_dim
        self.position_dim = position_dim
        self.scale = evidence_dim ** -0.5

        self.record_encoder = nn.Sequential(nn.Linear(embed_dim + 2 * position_dim + 1, evidence_dim), nn.GELU(),
                                            nn.Linear(evidence_dim, evidence_dim))
        self.question_projection = nn.Linear(embed_dim, evidence_dim)
        self.spectral_projection = nn.Linear(embed_dim, evidence_dim)
        self.keys = nn.Linear(evidence_dim, evidence_dim)
        self.values = nn.Linear(evidence_dim, evidence_dim)
        self.slots = nn.Parameter(torch.randn(self.num_tokens, evidence_dim) * 0.02)
        self.query = nn.Sequential(nn.Linear(evidence_dim, evidence_dim), nn.GELU(), nn.Linear(evidence_dim, evidence_dim))
        self.previous_projection = nn.Linear(evidence_dim, evidence_dim)

    @staticmethod
    def position_encoding(indices, dim, dtype):
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, device = indices.device).float() / max(dim, 1))
        angles = indices.clamp_min(0).float()[..., None] * frequencies
        positions = torch.zeros(*indices.shape, dim, device = indices.device, dtype = torch.float32)
        positions[..., 0::2] = angles.sin()
        positions[..., 1::2] = angles[..., :dim // 2].cos()

        return positions.to(dtype)

    @staticmethod
    def masked_mean(features, mask):
        mask = mask.to(device = features.device, dtype = torch.bool)
        if features.ndim != 3 or tuple(mask.shape) != tuple(features.shape[:2]) or not mask.any(dim = 1).all():
            raise ValueError('Context features require at least one valid token per example')

        return features.masked_fill(~mask[:, :, None], 0).sum(dim = 1) / mask.sum(dim = 1, keepdim = True)

    def encode_records(self, similarity, text, text_mask, image_mask, text_indices, image_indices, grid_size):
        if type(grid_size) is not int or grid_size <= 0:
            raise ValueError('grid_size must be a positive integer')
        if similarity.ndim != 3 or tuple(text.shape) != (*similarity.shape[:2], self.embed_dim):
            raise ValueError('Selected text features must match the interaction rows')
        if tuple(text_mask.shape) != tuple(similarity.shape[:2]) or tuple(image_mask.shape) != (similarity.size(0), similarity.size(2)):
            raise ValueError('Correspondence masks must match the interaction matrix')
        if tuple(text_indices.shape) != tuple(text_mask.shape) or tuple(image_indices.shape) != tuple(image_mask.shape):
            raise ValueError('Original positions must match the selected token masks')

        text_mask = text_mask.to(device = similarity.device, dtype = torch.bool)
        image_mask = image_mask.to(device = similarity.device, dtype = torch.bool)
        text_indices = text_indices.to(device = similarity.device)
        image_indices = image_indices.to(device = similarity.device)
        if not text_mask.any(dim = 1).all() or not image_mask.any(dim = 1).all():
            raise ValueError('Each example requires valid word-patch records')
        if (text_indices[text_mask] < 0).any() or (image_indices[image_mask] < 0).any() or (image_indices[image_mask] >= grid_size ** 2).any():
            raise ValueError('Selected positions must refer to original text and image tokens')

        record_mask = text_mask[:, :, None] & image_mask[:, None, :]
        text = text.masked_fill(~text_mask[:, :, None], 0)
        similarity = similarity.masked_fill(~record_mask, 0)
        text_positions = self.position_encoding(text_indices, self.position_dim, text.dtype)
        row_dim = self.position_dim // 2
        image_positions = torch.cat((self.position_encoding(image_indices.clamp_min(0) // grid_size, row_dim, text.dtype),
                                     self.position_encoding(image_indices.clamp_min(0) % grid_size, self.position_dim - row_dim, text.dtype)), dim = -1)

        # Apply the record encoder's first linear layer without a batch x text x image x 289 allocation.
        first = self.record_encoder[0]
        words = F.linear(text, first.weight[:, :self.embed_dim], first.bias)
        words = words + F.linear(text_positions, first.weight[:, self.embed_dim:self.embed_dim + self.position_dim])
        patches = F.linear(image_positions, first.weight[:, self.embed_dim + self.position_dim:-1])
        scores = F.linear(similarity[:, :, :, None], first.weight[:, -1:])
        records = words[:, :, None, :] + patches[:, None, :, :] + scores
        records = self.record_encoder[2](self.record_encoder[1](records))
        records = records.masked_fill(~record_mask[:, :, :, None], 0).flatten(1, 2)

        return records, record_mask.flatten(1)

    def forward(self, similarity, selected_text, text_mask, image_mask, text_indices, image_indices,
                question, question_mask, spectral, spectral_mask, grid_size):
        records, record_mask = self.encode_records(similarity, selected_text, text_mask, image_mask,
                                                  text_indices, image_indices, grid_size)
        context = self.question_projection(self.masked_mean(question, question_mask))
        context = context + self.spectral_projection(self.masked_mean(spectral, spectral_mask))
        keys, values = self.keys(records), self.values(records)

        summaries, weights = [], []
        for index in range(self.num_tokens):
            features = context + self.slots[index]
            if summaries:
                features = features + self.previous_projection(summaries[-1])
            query = self.query(features)
            scores = torch.bmm(query[:, None, :], keys.transpose(1, 2)).squeeze(1) * self.scale
            attention = torch.softmax(scores.float().masked_fill(~record_mask, -torch.inf), dim = -1)
            summaries.append(torch.bmm(attention.to(values.dtype)[:, None, :], values).squeeze(1))
            weights.append(attention)

        return torch.stack(summaries, dim = 1), torch.stack(weights, dim = 1)
