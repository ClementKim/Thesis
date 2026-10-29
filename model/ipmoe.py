import torch
import torch.nn as nn

from ablation import ABLATIONS, select_interaction_map

from .interaction import InteractionAdapter
from .llm import NoValidTargetsError

class IPMoE(nn.Module):
    def __init__(self, processor, text_encoder, vision_encoder, interaction, moe, llm, embed_dim, patch_size, num_aux_tokens, ablation = None):
        super().__init__()

        if num_aux_tokens < 2:
            raise ValueError('num_aux_tokens must be at least 2')

        self.processor = processor
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.interaction = interaction
        self.moe = moe
        self.llm = llm

        self.ablation = dict(ABLATIONS['full'] if ablation is None else ablation)
        if self.ablation not in ABLATIONS.values():
            raise ValueError('Unsupported ablation configuration')
        if self.moe.router_mode != self.ablation['router_mode']:
            raise ValueError('Router mode differs from the ablation configuration')
        if self.moe.expert_set != self.ablation.get('expert_set', 'all'):
            raise ValueError('Expert set differs from the ablation configuration')

        ## Auxiliary Tokens
        self.aux = nn.Parameter(torch.randn(1, num_aux_tokens, embed_dim))

        ## Interaction Map and LLM Projection Heads
        self.interaction_adapter = InteractionAdapter(embed_dim, patch_size)

        llm_width = self.llm.model.get_input_embeddings().embedding_dim
        self.image_head = nn.Linear(embed_dim, llm_width)
        self.interaction_head = nn.Linear(embed_dim, llm_width)
        self.expert_head = nn.Linear(embed_dim, llm_width)

        # Unused modules get no gradients; DDP would otherwise fail on unused parameters.
        if self.ablation.get('prefix', 'full') == 'image':
            for module in (self.text_encoder, self.interaction, self.interaction_adapter, self.moe, self.interaction_head, self.expert_head):
                module.requires_grad_(False)
            self.aux.requires_grad_(False)

        if self.ablation.get('expert_set', 'all') == 'self':
            self.aux.requires_grad_(False)

    def encode_prefix(self, text_inputs, image_inputs, map_image_inputs = None):
        """map_image_inputs, if given, builds the interaction map from another image while the image tokens keep the true one."""
        image_features = self.vision_encoder(**image_inputs)
        image_mask = torch.ones(image_features.shape[:2], dtype = torch.bool, device = image_features.device)

        # The image-only baseline has no router, so it returns zero-width weights.
        if self.ablation.get('prefix', 'full') == 'image':
            return self.image_head(image_features), image_mask, image_features.new_empty((image_features.size(0), 0))

        # Process the question.
        text_features, text_mask = self.text_encoder(**text_inputs)

        # Interaction between text and vision features.
        map_features = image_features if map_image_inputs is None else self.vision_encoder(**map_image_inputs)
        similarity = self.interaction(text_features, map_features)
        similarity = select_interaction_map(similarity, self.ablation.get('shared_map', 'full'))
        interaction_tokens, interaction_map, map_mask, token_mask = self.interaction_adapter(similarity, text_mask)

        # Select the expert and router inputs from the shared map independently.
        expert_map = select_interaction_map(interaction_map, self.ablation['expert_map'])
        router_map = select_interaction_map(interaction_map, self.ablation['router_map'])
        expert_outputs, expert_mask, weights = self.moe(expert_map, self.aux, map_mask, router_x = router_map)

        # Pass image, interaction, and each weighted expert output to the LLM.
        prefix = torch.cat([self.image_head(image_features), self.interaction_head(interaction_tokens), self.expert_head(expert_outputs)], dim = 1)
        prefix_mask = torch.cat([image_mask, token_mask, expert_mask], dim = 1)

        return prefix, prefix_mask, weights

    def _prefix_lengths(self, text_inputs, image_inputs):
        vision = self.vision_encoder.clip.config.vision_config

        image_size, patch_size = int(vision.image_size), int(vision.patch_size)
        if image_size % patch_size:
            raise ValueError('CLIP image size must be divisible by its patch size')
        
        if tuple(image_inputs['pixel_values'].shape[-2:]) != (image_size, image_size):
            raise ValueError('Prefix prediction requires the configured square CLIP image size')
        
        image_tokens = (image_size // patch_size) ** 2
        
        p = self.interaction_adapter.patch_embed.patch_size
        
        if p <= 0:
            raise ValueError('Interaction patch size must be positive')
        
        cfg = self.text_encoder.clip.config.text_config
        
        if cfg.bos_token_id is None or cfg.eos_token_id is None:
            raise ValueError('CLIP text configuration must define BOS and EOS token IDs')
        
        num_experts = 3 if self.ablation.get('expert_set', 'all') == 'all' else 1
        lengths = []
        for ids, mask in zip(text_inputs['input_ids'], text_inputs['attention_mask']):
            content = ids[mask.bool()]
            if content.numel() and content[0].item() == cfg.bos_token_id:
                content = content[1:]
        
            if content.numel() and content[-1].item() == cfg.eos_token_id:
                content = content[:-1]
        
            text_tokens = max(content.numel(), 1)  # Empty chunk retains its EOS.
            interaction_tokens = ((text_tokens + p - 1) // p) * ((image_tokens + p - 1) // p)
            if self.ablation.get('prefix', 'full') == 'image':
                interaction_tokens = 0

            lengths.append(image_tokens + (1 + num_experts) * interaction_tokens)
        
        return lengths

    def forward(self, questions, text_inputs, image_inputs, answers, context_limit, return_stats = False, allow_empty = False):
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
            weights = self.aux.new_empty((0, 3))
            return (loss, weights, 0) if return_stats else (loss, weights)

        # The tokenizer raises NoValidTargetsError before encoding an empty selection.
        def select(inputs):
            return {key: value.index_select(0, torch.tensor(kept_rows, device = value.device, dtype = torch.long)) for key, value in inputs.items()}

        prefix, prefix_mask, weights = self.encode_prefix(select(text_inputs), select(image_inputs))
        expected = [prefix_lengths[index] for index in kept_rows]
        if prefix_mask.sum(dim = 1).tolist() != expected:
            raise RuntimeError('Predicted prefix lengths disagree with encoded prefix lengths')

        embeds, mask, position_ids, labels = self.llm.build_inputs(token_rows, positions, prefix, label_rows, prefix_mask = prefix_mask)
        loss = self.llm(embeds, mask, position_ids, labels)

        if return_stats:
            return loss, weights, (labels[:, 1:] != -100).sum().item()

        return loss, weights

    @torch.no_grad()
    def generate(self, questions, text_inputs, image_inputs, context_limit, max_new_tokens = 32, enforce_mcq_format = True,
                 map_image_inputs = None):
        if self.training:
            raise RuntimeError('Call model.eval() before generation')
        
        prefix, prefix_mask, weights = self.encode_prefix(text_inputs, image_inputs, map_image_inputs)
        text_budget = [context_limit - length - max_new_tokens for length in prefix_mask.sum(dim = 1).tolist()]

        token_rows, positions = self.llm.tokenize_generation_chats(questions, max_prompt_tokens = text_budget, enforce_mcq_format = enforce_mcq_format)
        embeds, mask, _, _ = self.llm.build_inputs(token_rows, positions, prefix, prefix_mask = prefix_mask)
        answers = self.llm.generate(embeds, mask, max_new_tokens = max_new_tokens)

        return answers, weights
