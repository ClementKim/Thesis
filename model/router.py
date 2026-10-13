import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicRouter(nn.Module):
    """Give every expert a learned weight for each input feature map."""

    def __init__(self, in_c, num_experts):
        super().__init__()

        self.gate = nn.Linear(in_c, num_experts)

    def forward(self, x):
        # Summarize each channel: (B, C, H, W) -> (B, C).
        features = x.mean(dim = (2, 3))
        logits = self.gate(features)
        weights = F.softmax(logits, dim = -1)

        return weights

class DenseMoE(nn.Module):
    """Combine SelfExpert, CrossExpert, and Hybrid with one weight per input."""

    def __init__(self, in_c, self_expert, cross_expert, hybrid_expert):
        super().__init__()

        self.router = DynamicRouter(in_c, num_experts = 3)
        self.self_expert = self_expert
        self.cross_expert = cross_expert
        self.hybrid_expert = hybrid_expert

    def forward(self, x, auxiliary):
        """Return (B, N, D) features and (B, 3) weights in Self, Cross, Hybrid order."""
        weights = self.router(x)

        self_output = self.self_expert(x)
        cross_output = self.cross_expert(x, auxiliary)
        hybrid_output = self.hybrid_expert(x, auxiliary)

        # All experts must produce the same (B, N, D) shape.
        expert_outputs = torch.stack([self_output, cross_output, hybrid_output], dim = 1)

        # Apply each sample's expert weights to all its tokens and features.
        out = expert_outputs * weights[:, :, None, None]
        out = out.sum(dim = 1)

        return out, weights

class TopKRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k = 2):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, num_experts)

    def load_balancing_loss(self, probs, indices):
        # Switch Transformer loss: small when tokens are spread evenly over experts.
        probs = probs.reshape(-1, self.num_experts)
        indices = indices.reshape(-1, self.top_k)

        # Fraction of tokens sent to each expert.
        dispatch = F.one_hot(indices, self.num_experts).float().sum(dim = 1)
        tokens_per_expert = dispatch.mean(dim = 0) / self.top_k

        # Average router probability for each expert.
        prob_per_expert = probs.mean(dim = 0)

        return self.num_experts * (tokens_per_expert * prob_per_expert).sum()

    def forward(self, x):
        B, N, D = x.shape

        logits = self.gate(x)
        probs = F.softmax(logits, dim = -1)

        # Keep the k best experts per token, then renormalize their weights to sum to 1.
        weights, indices = probs.topk(self.top_k, dim = -1)
        weights = weights / weights.sum(dim = -1, keepdim = True)

        aux_loss = self.load_balancing_loss(probs, indices)

        return weights, indices, aux_loss

class SparseMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k = 2, hidden_ratio = 4):
        super().__init__()

        self.top_k = top_k
        self.router = TopKRouter(dim, num_experts, top_k)

        # Each expert is a plain FFN. Swap in your own expert module here.
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * hidden_ratio), nn.GELU(), nn.Linear(dim * hidden_ratio, dim))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, N, D = x.shape

        weights, indices, aux_loss = self.router(x)

        x_flat = x.reshape(-1, D)
        weights = weights.reshape(-1, self.top_k)
        indices = indices.reshape(-1, self.top_k)

        out = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            # Find the tokens (and which of their top-k slots) that picked expert e.
            token_idx, slot = (indices == e).nonzero(as_tuple = True)
            if token_idx.numel() == 0:
                continue
            out.index_add_(0, token_idx, expert(x_flat[token_idx]) * weights[token_idx, slot, None])

        return out.reshape(B, N, D), aux_loss
