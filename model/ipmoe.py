import torch
import torch.nn as nn

class IPMoE(nn.Module):
    def __init__(self, processor, text_encoder, vision_encoder, interaction, self_experts, cross_experts, hybrid_experts, router, llm, embed_dim):
        super().__init__()

        self.processor = processor
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.interaction = interaction
        self.self_experts = self_experts
        self.cross_experts = cross_experts
        self.hybrid_experts = hybrid_experts
        self.router = router
        self.llm = llm

        ## Auxiliary Token
        self.aux = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, question, image):
        # Process the image and question
        image_features = self.vision_encoder(image)
        text_features = self.text_encoder(question)

        # Interaction between text and vision features
        interaction_output = self.interaction(text_features, image_features)

        # Routing to experts
        expert_outputs = []
        for i in range(len(self.self_experts)):
            self_output = self.self_experts[i](interaction_output)
            cross_output = self.cross_experts[i](interaction_output, self.aux)
            hybrid_output = self.hybrid_experts[i](interaction_output, self.aux)
            expert_outputs.append((self_output, cross_output, hybrid_output))

        # Combine expert outputs (this is a placeholder; actual combination logic may vary)
        combined_output = torch.mean(torch.stack([torch.mean(torch.stack(outputs), dim=0) for outputs in expert_outputs]), dim=0)

        # Pass through the LLM
        llm_output = self.llm(combined_output)

        return llm_output