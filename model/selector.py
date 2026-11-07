import math

import torch
import torch.nn as nn


class TokenSelector(nn.Module):
    def __init__(self, embed_dim, keep_ratio = 0.5, selector_dim = 64, temperature = 1.0):
        super().__init__()

        for name, value in (('embed_dim', embed_dim), ('selector_dim', selector_dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in (('keep_ratio', keep_ratio), ('temperature', temperature)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number')
        if keep_ratio > 1:
            raise ValueError('keep_ratio must not exceed one')

        self.embed_dim = embed_dim
        self.keep_ratio = float(keep_ratio)
        self.temperature = float(temperature)

        self.norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.token_projection = nn.Linear(embed_dim, selector_dim)
        self.target_projection = nn.Linear(embed_dim, selector_dim)
        self.context_projection = nn.Linear(embed_dim, selector_dim)
        self.score = nn.Sequential(nn.GELU(), nn.Linear(selector_dim, 1, bias = False))

    def selected_count(self, length):
        if type(length) is not int or length <= 0:
            raise ValueError('Selection requires at least one valid token')

        return max(1, math.ceil(length * self.keep_ratio))

    def score_tokens(self, tokens, token_mask, context_tokens, context_mask):
        for features, mask in ((tokens, token_mask), (context_tokens, context_mask)):
            if features.ndim != 3 or features.size(-1) != self.embed_dim:
                raise ValueError('Selector features must have shape batch by tokens by embed_dim')
            if tuple(mask.shape) != tuple(features.shape[:2]):
                raise ValueError('Selector masks must match the feature batch and token dimensions')
        if tokens.size(0) != context_tokens.size(0):
            raise ValueError('Both selector modalities must have the same batch size')

        token_mask = token_mask.to(device = tokens.device, dtype = torch.bool)
        context_mask = context_mask.to(device = context_tokens.device, dtype = torch.bool)
        if not token_mask.any(dim = 1).all() or not context_mask.any(dim = 1).all():
            raise ValueError('Both selector modalities require at least one valid token per example')

        # Clear padding before normalization so padded NaNs cannot affect either context.
        target = self.norm(tokens.masked_fill(~token_mask[:, :, None], 0))
        context = self.context_norm(context_tokens.masked_fill(~context_mask[:, :, None], 0))
        target = target.masked_fill(~token_mask[:, :, None], 0)
        context = context.masked_fill(~context_mask[:, :, None], 0)
        target_mean = target.sum(dim = 1) / token_mask.sum(dim = 1, keepdim = True)
        context_mean = context.sum(dim = 1) / context_mask.sum(dim = 1, keepdim = True)

        # Nonlinear scoring lets both global contexts change the ordering of local tokens.
        features = self.token_projection(target)
        features = features + self.target_projection(target_mean)[:, None] + self.context_projection(context_mean)[:, None]
        scores = self.score(features).squeeze(-1)

        return scores.masked_fill(~token_mask, -torch.inf)

    def forward(self, tokens, token_mask, context_tokens, context_mask):
        scores = self.score_tokens(tokens, token_mask, context_tokens, context_mask)
        token_mask = token_mask.to(device = tokens.device, dtype = torch.bool)
        counts = [self.selected_count(length) for length in token_mask.sum(dim = 1).tolist()]
        count = max(counts)

        # Stable ties favor the earlier token; restore source order after choosing top-k.
        indices = torch.argsort(scores, dim = 1, descending = True, stable = True)[:, :count]
        selected_mask = torch.arange(count, device = tokens.device)[None] < torch.tensor(counts, device = tokens.device)[:, None]
        indices = indices.masked_fill(~selected_mask, tokens.size(1)).sort(dim = 1).values
        selected_mask = indices < tokens.size(1)
        gather_indices = indices.clamp_max(tokens.size(1) - 1)
        selected = tokens.masked_fill(~token_mask[:, :, None], 0).gather(
            1, gather_indices[:, :, None].expand(-1, -1, tokens.size(-1)))

        if self.training:
            probabilities = torch.softmax(scores.float() / self.temperature, dim = 1)
            probabilities = probabilities.gather(1, gather_indices)
            # Hard indices are nondifferentiable. This unit-valued straight-through gate
            # preserves gathered features while supplying a surrogate gradient to scores.
            gate = 1 + (probabilities - probabilities.detach())
            selected = selected * gate.to(selected.dtype)[:, :, None]

        selected = selected.masked_fill(~selected_mask[:, :, None], 0)

        return selected, selected_mask, indices.masked_fill(~selected_mask, -1)
