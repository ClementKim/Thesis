import torch
import torch.nn as nn

from .common import PatchEmbeddingLayer

class Interaction(nn.Module):
    def __init__(self, interaction_type):
        super().__init__()

        self.interaction_type = interaction_type

    def forward(self, text, vision):
        if self.interaction_type == "bilinear":
            return torch.bmm(text, vision.transpose(1, 2))

class InteractionAdapter(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()

        self.patch_embed = PatchEmbeddingLayer(1, embed_dim, patch_size, is_cls_token = False)

    def forward(self, similarity, text_mask):
        map_mask = text_mask.bool().unsqueeze(-1).expand_as(similarity)
        interaction_map = similarity.unsqueeze(1)
        interaction_tokens, token_mask = self.patch_embed(interaction_map, mask = map_mask)

        return interaction_tokens, interaction_map, map_mask, token_mask
