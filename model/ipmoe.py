import torch
import torch.nn as nn

from .interaction import InteractionAdapter

class IPMoE(nn.Module):
    def __init__(self, processor, text_encoder, vision_encoder, interaction, moe, llm, embed_dim, patch_size):
        super().__init__()

        self.processor = processor
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.interaction = interaction
        self.moe = moe
        self.llm = llm

        ## Auxiliary Token
        self.aux = nn.Parameter(torch.randn(1, 1, embed_dim))

        ## Interaction Map and LLM Projection Heads
        self.interaction_adapter = InteractionAdapter(embed_dim, patch_size)
        llm_width = self.llm.model.get_input_embeddings().embedding_dim
        self.image_head = nn.Linear(embed_dim, llm_width)
        self.interaction_head = nn.Linear(embed_dim, llm_width)
        self.expert_head = nn.Linear(embed_dim, llm_width)

    def encode_prefix(self, text_inputs, image_inputs):
        # Process the image and question.
        text_features, text_mask = self.text_encoder(**text_inputs)
        image_features = self.vision_encoder(**image_inputs)

        # Interaction between text and vision features.
        similarity = self.interaction(text_features, image_features)
        interaction_tokens, interaction_map, map_mask, token_mask = self.interaction_adapter(similarity, text_mask)

        # Route to one self, one cross, and one hybrid expert.
        expert_outputs, weights = self.moe(interaction_map, self.aux, mask = map_mask)

        # Pass image, interaction, and each weighted expert output to the LLM.
        prefix = torch.cat([self.image_head(image_features), self.interaction_head(interaction_tokens),
                            self.expert_head(expert_outputs)], dim = 1)
        image_mask = torch.ones(image_features.shape[:2], dtype = torch.bool, device = prefix.device)
        expert_mask = torch.ones(expert_outputs.shape[:2], dtype = torch.bool, device = prefix.device)
        prefix_mask = torch.cat([image_mask, token_mask, expert_mask], dim = 1)

        return prefix, prefix_mask, weights

    def forward(self, questions, text_inputs, image_inputs, answers, context_limit, return_stats = False):
        prefix, prefix_mask, weights = self.encode_prefix(text_inputs, image_inputs)
        text_budget = [context_limit - length for length in prefix_mask.sum(dim = 1).tolist()]

        # The LLM receives the original question and supervises only its answer.
        token_rows, label_rows, kept_rows, positions = self.llm.tokenize_training_chats(questions, answers, max_total_tokens = text_budget)
        kept = torch.tensor(kept_rows, dtype = torch.long, device = prefix.device)
        prefix = prefix.index_select(0, kept)
        prefix_mask = prefix_mask.index_select(0, kept)
        weights = weights.index_select(0, kept)

        embeds, mask, position_ids, labels = self.llm.build_inputs(token_rows, positions, prefix, label_rows, prefix_mask = prefix_mask)
        loss = self.llm(embeds, mask, position_ids, labels)

        if return_stats:
            return loss, weights, (labels[:, 1:] != -100).sum().item()

        return loss, weights

    @torch.no_grad()
    def generate(self, questions, text_inputs, image_inputs, context_limit, max_new_tokens = 32, enforce_mcq_format = True):
        prefix, prefix_mask, weights = self.encode_prefix(text_inputs, image_inputs)
        text_budget = [context_limit - length - max_new_tokens for length in prefix_mask.sum(dim = 1).tolist()]

        token_rows, positions = self.llm.tokenize_generation_chats(questions, max_prompt_tokens = text_budget, enforce_mcq_format = enforce_mcq_format)
        embeds, mask, _, _ = self.llm.build_inputs(token_rows, positions, prefix, prefix_mask = prefix_mask)
        answers = self.llm.generate(embeds, mask, max_new_tokens = max_new_tokens)

        return answers, weights
