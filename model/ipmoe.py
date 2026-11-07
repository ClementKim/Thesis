import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from configuration import ABLATIONS

from .interaction import SVDInteractionAdapter
from .evidence import CorrespondenceSelector
from .llm import NoValidTargetsError
from .selector import TokenSelector

class IPMoE(nn.Module):
    def __init__(self, text_encoder, vision_encoder, llm, embed_dim, patch_size,
                 svd_rank = 8, evidence_dim = 64, position_dim = 16,
                 selector_keep_ratio = 0.5, image_keep_ratio = 0.5, selector_dim = 64, selector_temperature = 1.0,
                 ablation = 'full', selection_seed = 42):
        super().__init__()

        if ablation not in ABLATIONS:
            raise ValueError(f'Unknown ablation: {ablation!r}')
        for name, value in (('embed_dim', embed_dim), ('patch_size', patch_size), ('svd_rank', svd_rank)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for ratio in (selector_keep_ratio, image_keep_ratio):
            if type(ratio) not in (int, float) or not 0 < ratio <= 1:
                raise ValueError('Selector keep ratios must be in (0, 1]')
        if type(selection_seed) is not int or not 0 <= selection_seed < 2 ** 63 - 1:
            raise ValueError('selection_seed must be an integer in [0, 2**63 - 1)')

        self.ablation = ablation
        self.patch_size = patch_size
        self.svd_rank = svd_rank
        self.selector_keep_ratio = selector_keep_ratio
        self.image_keep_ratio = image_keep_ratio
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.llm = llm
        llm_width = self.llm.model.get_input_embeddings().embedding_dim

        if ablation == 'direct_image':
            # Reuse the same vision MLP and LLM; match the full model's prefix budget.
            # The unused text MLP must not enter the optimizer or DDP reductions.
            self.text_encoder.requires_grad_(False)
            self.image_head = nn.Linear(embed_dim, llm_width)
            return

        ## Compressed Interaction Prefix and Two Soft Evidence Tokens
        self.interaction_adapter = SVDInteractionAdapter(embed_dim, patch_size, svd_rank)
        if ablation != 'spectral_only':
            self.evidence_selector = CorrespondenceSelector(embed_dim, evidence_dim, position_dim)
        if ablation != 'evidence_only':
            self.interaction_head = nn.Linear(embed_dim, llm_width)
        if ablation != 'spectral_only':
            self.evidence_head = nn.Linear(evidence_dim, llm_width)

        ## Both Selectors See the Complete Text and Image Features
        selection = 'random' if ablation == 'random_selection' else 'learned'
        self.text_selector = TokenSelector(embed_dim, selector_keep_ratio, selector_dim, selector_temperature,
                                           selection = selection, seed = selection_seed)
        self.image_selector = TokenSelector(embed_dim, image_keep_ratio, selector_dim, selector_temperature,
                                            selection = selection, seed = selection_seed + 1)

    @staticmethod
    def _pool_image_features(features, length):
        # Adaptive average bins expressed as a matrix multiply: the native pooling
        # backward is incompatible with strict deterministic training on CUDA.
        positions = torch.arange(features.size(0), device = features.device)
        bins = torch.arange(length, device = features.device)
        starts = bins * features.size(0) // length
        ends = ((bins + 1) * features.size(0) + length - 1) // length
        weights = ((positions[None] >= starts[:, None]) & (positions[None] < ends[:, None])).to(features.dtype)
        weights = weights / (ends - starts)[:, None].to(features.dtype)

        return weights @ features

    def encode_prefix(self, text_inputs, image_inputs):
        image_features = self.vision_encoder(**image_inputs)
        if self.ablation == 'direct_image':
            # Pool raster-ordered CLIP patches without text-conditioned selection.
            # Only the token count depends on question length, to match the full model.
            lengths = self._prefix_lengths(text_inputs, image_inputs)
            pooled = [self._pool_image_features(features, length) for features, length in zip(image_features, lengths)]
            prefix = self.image_head(pad_sequence(pooled, batch_first = True))
            prefix_mask = torch.arange(prefix.size(1), device = prefix.device)[None] < torch.tensor(lengths, device = prefix.device)[:, None]
            return prefix.masked_fill(~prefix_mask[:, :, None], 0), prefix_mask

        image_mask = torch.ones(image_features.shape[:2], dtype = torch.bool, device = image_features.device)
        text_features, text_mask = self.text_encoder(**text_inputs)

        selected_text, selected_text_mask, text_indices = self.text_selector(text_features, text_mask, image_features, image_mask)
        selected_image, selected_image_mask, image_indices = self.image_selector(image_features, image_mask, text_features, text_mask)

        # Select both subsets before constructing their text-by-image map.
        similarity = torch.bmm(selected_text, selected_image.transpose(1, 2))
        interaction_tokens, token_mask = self.interaction_adapter(similarity, selected_text_mask, selected_image_mask,
                                                                  column_indices = image_indices)

        if self.ablation == 'spectral_only':
            prefix = self.interaction_head(interaction_tokens)
            return prefix.masked_fill(~token_mask[:, :, None], 0), token_mask

        vision = self.vision_encoder.clip.config.vision_config
        grid_size = int(vision.image_size) // int(vision.patch_size)
        evidence = self.evidence_selector(similarity, selected_text, selected_text_mask, selected_image_mask,
                                          text_indices, image_indices, text_features, text_mask,
                                          interaction_tokens, token_mask, grid_size)

        if self.ablation == 'evidence_only':
            # The evidence selector still uses spectral conditioning internally.
            prefix = self.evidence_head(evidence)
            return prefix, torch.ones(prefix.shape[:2], dtype = torch.bool, device = prefix.device)

        # The original question is tokenized separately by the LLM, unchanged.
        prefix = torch.cat((self.interaction_head(interaction_tokens), self.evidence_head(evidence)), dim = 1)
        prefix_mask = torch.cat((token_mask, torch.ones(evidence.shape[:2], dtype = torch.bool, device = token_mask.device)), dim = 1)
        prefix = prefix.masked_fill(~prefix_mask[:, :, None], 0)

        return prefix, prefix_mask

    def _prefix_lengths(self, text_inputs, image_inputs):
        vision = self.vision_encoder.clip.config.vision_config

        image_size, patch_size = int(vision.image_size), int(vision.patch_size)
        if image_size % patch_size:
            raise ValueError('CLIP image size must be divisible by its patch size')
        
        if tuple(image_inputs['pixel_values'].shape[-2:]) != (image_size, image_size):
            raise ValueError('Prefix prediction requires the configured square CLIP image size')
        
        image_tokens = (image_size // patch_size) ** 2
        
        p = self.patch_size
        
        if p <= 0:
            raise ValueError('Interaction patch size must be positive')
        
        cfg = self.text_encoder.clip.config.text_config
        
        if cfg.bos_token_id is None or cfg.eos_token_id is None:
            raise ValueError('CLIP text configuration must define BOS and EOS token IDs')
        
        lengths = []
        for ids, mask in zip(text_inputs['input_ids'], text_inputs['attention_mask']):
            content = ids[mask.bool()]
            if content.numel() and content[0].item() == cfg.bos_token_id:
                content = content[1:]
        
            if content.numel() and content[-1].item() == cfg.eos_token_id:
                content = content[:-1]
        
            text_tokens = max(content.numel(), 1)  # Empty chunk retains its EOS.
            selected_text = max(1, math.ceil(text_tokens * self.selector_keep_ratio))
            selected_image = max(1, math.ceil(image_tokens * self.image_keep_ratio))
            rank = min(self.svd_rank, selected_text, selected_image)
            interaction_tokens = ((rank + p - 1) // p) * ((selected_image + p - 1) // p)
            if self.ablation == 'spectral_only':
                lengths.append(interaction_tokens)
            elif self.ablation == 'evidence_only':
                lengths.append(CorrespondenceSelector.num_tokens)
            else:
                lengths.append(interaction_tokens + CorrespondenceSelector.num_tokens)
        
        return lengths

    def forward(self, questions, text_inputs, image_inputs, answers, context_limit, allow_empty = False):
        prefix_lengths = self._prefix_lengths(text_inputs, image_inputs)
        text_budget = [context_limit - length for length in prefix_lengths]

        # The LLM receives the original question and supervises only its answer.
        try:
            token_rows, label_rows, kept_rows, positions = self.llm.tokenize_training_chats(questions, answers, max_total_tokens = text_budget)
        
        except NoValidTargetsError:
            if not allow_empty:
                raise

            # Empty DDP ranks must still participate in every gradient reduction.
            loss = sum(parameter.reshape(-1)[0].float() * 0 for parameter in self.parameters() if parameter.requires_grad)
            return loss, 0, 0

        # The tokenizer raises NoValidTargetsError before encoding an empty selection.
        def select(inputs):
            return {key: value.index_select(0, torch.tensor(kept_rows, device = value.device, dtype = torch.long)) for key, value in inputs.items()}

        prefix, prefix_mask = self.encode_prefix(select(text_inputs), select(image_inputs))
        expected = [prefix_lengths[index] for index in kept_rows]
        if prefix_mask.sum(dim = 1).tolist() != expected:
            raise RuntimeError('Predicted prefix lengths disagree with encoded prefix lengths')

        embeds, mask, position_ids, labels = self.llm.build_inputs(token_rows, positions, prefix, label_rows, prefix_mask = prefix_mask)
        loss = self.llm(embeds, mask, position_ids, labels)

        return loss, len(kept_rows), (labels[:, 1:] != -100).sum().item()

    @torch.no_grad()
    def generate(self, questions, text_inputs, image_inputs, context_limit, max_new_tokens = 32):
        if self.training:
            raise RuntimeError('Call model.eval() before generation')
        
        prefix, prefix_mask = self.encode_prefix(text_inputs, image_inputs)
        text_budget = [context_limit - length - max_new_tokens for length in prefix_mask.sum(dim = 1).tolist()]

        token_rows, positions = self.llm.tokenize_generation_chats(questions, max_prompt_tokens = text_budget)
        embeds, mask, _, _ = self.llm.build_inputs(token_rows, positions, prefix, prefix_mask = prefix_mask)
        answers = self.llm.generate(embeds, mask, max_new_tokens = max_new_tokens)

        return answers
