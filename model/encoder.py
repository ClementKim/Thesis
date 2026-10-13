import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

# The 336 px model produces a 24 x 24 patch grid, checked by build_model.
MODEL_NAME = 'openai/clip-vit-large-patch14-336'

class CLIPEncoder(nn.Module):
    """One frozen backbone shared by the text and image encoders."""

    def __init__(self):
        super().__init__()

        self.model = CLIPModel.from_pretrained(MODEL_NAME)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def config(self):
        return self.model.config

class TextEncoder(nn.Module):
    """Split long questions into CLIP chunks, then join their content tokens."""

    def __init__(self, CLIP, drop, out_dim = None, chunk_batch_size = 64, add_chunk_position = True):
        super().__init__()

        self.clip = CLIP.model
        projection_dim = self.clip.config.projection_dim
        if out_dim is None:
            out_dim = projection_dim

        self.out_dim = out_dim
        self.chunk_batch_size = chunk_batch_size
        self.add_chunk_position = add_chunk_position

        if add_chunk_position:
            self.chunk_position_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter('chunk_position_scale', None)

        self.mlp = nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.GELU(), nn.Dropout(drop), nn.Linear(projection_dim, out_dim))

    def train(self, mode = True):
        super().train(mode)
        self.clip.eval()

        return self

    @staticmethod
    def _chunk_position_encoding(num_chunks, dim, device, dtype):
        """Sinusoidal positions identify which chunk each token came from."""
        positions = torch.arange(num_chunks, device = device, dtype = torch.float32).unsqueeze(1)
        frequency_indices = torch.arange(0, dim, 2, device = device, dtype = torch.float32)
        frequencies = torch.exp(-math.log(10000.0) * frequency_indices / max(dim, 1))
        angles = positions * frequencies.unsqueeze(0)

        encoding = torch.zeros(num_chunks, dim, device = device, dtype = torch.float32)
        encoding[:, 0::2] = torch.sin(angles)

        # Odd widths have one fewer cosine slot.
        cosine_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(angles[:, :cosine_width])

        return encoding.to(dtype = dtype)

    def _make_chunks(self, input_ids, attention_mask):
        """Build fixed-length chunks and record (sample, length, chunk index)."""
        config = self.clip.config.text_config
        max_length = config.max_position_embeddings
        content_window = max_length - 2  # Reserve BOS and EOS in each chunk.
        bos_token_id = config.bos_token_id
        eos_token_id = config.eos_token_id
        pad_token_id = config.pad_token_id

        if bos_token_id is None or eos_token_id is None:
            raise ValueError('CLIP text configuration must define BOS and EOS token IDs')
        if pad_token_id is None:
            pad_token_id = 0

        chunk_id_rows = []
        chunk_mask_rows = []
        chunk_info = []

        for sample_index in range(input_ids.size(0)):
            content_ids = input_ids[sample_index][attention_mask[sample_index].bool()]
            # Remove the tokenizer's BOS/EOS; each chunk gets its own pair.
            if content_ids.numel() > 0 and content_ids[0].item() == bos_token_id:
                content_ids = content_ids[1:]
            if content_ids.numel() > 0 and content_ids[-1].item() == eos_token_id:
                content_ids = content_ids[:-1]

            if content_ids.numel():
                chunks = list(torch.split(content_ids, content_window))
            else:
                chunks = [content_ids]

            for chunk_index, chunk in enumerate(chunks):
                content_length = int(chunk.numel())
                chunk_ids = torch.full((max_length,), pad_token_id, dtype = input_ids.dtype, device = input_ids.device)
                chunk_mask = torch.zeros(max_length, dtype = attention_mask.dtype, device = attention_mask.device)

                # [BOS] content... [EOS] padding...
                chunk_ids[0] = bos_token_id
                chunk_ids[1:1 + content_length] = chunk
                eos_position = 1 + content_length
                chunk_ids[eos_position] = eos_token_id
                chunk_mask[:eos_position + 1] = 1

                chunk_id_rows.append(chunk_ids)
                chunk_mask_rows.append(chunk_mask)
                chunk_info.append((sample_index, content_length, chunk_index))

        return torch.stack(chunk_id_rows), torch.stack(chunk_mask_rows), chunk_info

    def _encode_chunks(self, chunk_ids, chunk_mask):
        """Limit CLIP activation memory by encoding a few chunks at a time."""
        encoded_batches = []

        for start in range(0, chunk_ids.size(0), self.chunk_batch_size):
            end = min(start + self.chunk_batch_size, chunk_ids.size(0))
            with torch.no_grad():
                hidden = self.clip.text_model(input_ids = chunk_ids[start:end], attention_mask = chunk_mask[start:end]).last_hidden_state
                projected = self.clip.text_projection(hidden)
            encoded_batches.append(self.mlp(projected))

        return torch.cat(encoded_batches, dim = 0)

    def _join_chunks(self, encoded_chunks, chunk_info, batch_size, mask_dtype):
        """Restore each question's token sequence and pad the batch."""
        if self.add_chunk_position:
            num_positions = max(chunk_index for _, _, chunk_index in chunk_info) + 1
            chunk_positions = self._chunk_position_encoding(num_chunks = num_positions, dim = self.out_dim,
                                                            device = encoded_chunks.device, dtype = encoded_chunks.dtype)

        sample_chunks = [[] for _ in range(batch_size)]

        for row_index, (sample_index, content_length, chunk_index) in enumerate(chunk_info):
            # Empty questions retain their EOS token as a placeholder.
            token_count = max(content_length, 1)
            features = encoded_chunks[row_index, 1:1 + token_count]
            if self.add_chunk_position:
                position = chunk_positions[chunk_index].unsqueeze(0)
                features = features + self.chunk_position_scale * position
            sample_chunks[sample_index].append(features)

        sample_features = [torch.cat(chunks, dim = 0) for chunks in sample_chunks]
        sample_masks = [torch.ones(features.size(0), dtype = mask_dtype, device = features.device) for features in sample_features]
        token_features = pad_sequence(sample_features, batch_first = True, padding_value = 0.0)
        token_mask = pad_sequence(sample_masks, batch_first = True, padding_value = 0)

        return token_features, token_mask

    def forward(self, input_ids, attention_mask, **kwargs):
        chunk_ids, chunk_mask, chunk_info = self._make_chunks(input_ids, attention_mask)
        encoded_chunks = self._encode_chunks(chunk_ids, chunk_mask)

        return self._join_chunks(encoded_chunks, chunk_info, input_ids.size(0), attention_mask.dtype)

class VisionEncoder(nn.Module):
    """Project CLIP patch tokens while preserving the image's spatial layout."""

    def __init__(self, CLIP, drop, out_dim = None):
        super().__init__()

        self.clip = CLIP.model
        feature_dim = self.clip.config.vision_config.hidden_size
        projection_dim = self.clip.config.projection_dim
        if out_dim is None:
            out_dim = projection_dim

        self.mlp = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim),
                                 nn.GELU(), nn.Dropout(drop), nn.Linear(feature_dim, out_dim))

    def train(self, mode = True):
        super().train(mode)
        self.clip.eval()

        return self

    def forward(self, pixel_values, **kwargs):
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values = pixel_values, output_hidden_states = True)
            # Use the penultimate block's patch tokens, excluding the class token.
            patches = outputs.hidden_states[-2][:, 1:]

        return self.mlp(patches)

def encoding(drop, device, out_dim = None):
    """Build the processor and both encoders with one shared CLIP backbone."""
    image_size = int(CLIPConfig.from_pretrained(MODEL_NAME).vision_config.image_size)
    # Keep the whole image; center cropping could remove regions in the question.
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast = True, size = {'height': image_size, 'width': image_size}, do_center_crop = False)
    clip = CLIPEncoder().to(device)
    text_encoder = TextEncoder(clip, drop, out_dim).to(device)
    vision_encoder = VisionEncoder(clip, drop, out_dim).to(device)

    return processor, text_encoder, vision_encoder
