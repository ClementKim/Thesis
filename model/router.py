import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicRouter(nn.Module):
    """Give every expert a learned weight for each input feature map."""

    def __init__(self, in_c, num_experts):
        super().__init__()

        self.gate = nn.Linear(in_c, num_experts)

    def forward(self, x, mask = None):
        # Summarize each channel: (B, C, H, W) -> (B, C).
        if mask is None:
            features = x.mean(dim = (2, 3))
        else:
            mask = mask.to(device = x.device, dtype = torch.bool)
            features = x.masked_fill(~mask[:, None], 0).sum(dim = (2, 3), dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype)
            features = (features / mask.sum(dim = (1, 2))[:, None]).to(x.dtype)
        logits = self.gate(features)
        weights = F.softmax(logits, dim = -1)

        return weights

class DenseMoE(nn.Module):
    """Keep each expert's output with its own weight for each input."""

    def __init__(self, in_c, self_expert, cross_expert, hybrid_expert):
        super().__init__()

        self.router = DynamicRouter(in_c, num_experts = 3)
        self.self_expert = self_expert
        self.cross_expert = cross_expert
        self.hybrid_expert = hybrid_expert

    def forward(self, x, auxiliary, mask = None):
        """Return (B, 3, D) features and (B, 3) weights in Self, Cross, Hybrid order."""
        weights = self.router(x, mask)

        self_output = self.self_expert(x, mask)
        cross_output = self.cross_expert(x, auxiliary, mask)
        hybrid_output = self.hybrid_expert(x, auxiliary, mask)

        # All experts must produce the same (B, 1, D) shape.
        expert_outputs = torch.stack([self_output, cross_output, hybrid_output], dim = 1)

        # Apply each sample's expert weights to all its tokens and features.
        out = expert_outputs * weights[:, :, None, None]
        # Keep the weighted outputs as separate tokens for the LLM.
        out = out.flatten(1, 2)

        return out, weights
