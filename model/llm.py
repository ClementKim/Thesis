import inspect
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

MODEL_NAMES = {
    1: 'meta-llama/Llama-3.2-1B-Instruct',
    3: 'meta-llama/Llama-3.2-3B-Instruct',
}

QWEN_MODEL_NAMES = {
    3: 'Qwen/Qwen2.5-VL-3B-Instruct',
    7: 'Qwen/Qwen2.5-VL-7B-Instruct',
}

# Placeholder token that marks where the image features go.
MM_MARKER = '<|ipmoe_mm|>'

# Fixed date so the chat template always gives the same tokens.
FIXED_CHAT_DATE = '13 OCT 2026'

# Matches option lines like "A. cat", "A) cat", "(A) cat" or "A cat".
OPTION_LINE_RE = re.compile(r'^\s*(?:\(([A-H])\)|([A-H])(?:[\.)\:]|\s))\s*', flags = re.IGNORECASE)

def extract_mcq_labels(question):
    """Read consecutive option letters, or return [] for a regular question."""
    labels = []

    for line in str(question).splitlines():
        match = OPTION_LINE_RE.match(line)
        if match is None:
            continue

        label = (match.group(1) or match.group(2)).upper()
        if label not in labels:
            labels.append(label)

    # Require A, B, ... in order so regular text is not mistaken for options.
    expected_labels = [chr(ord('A') + index) for index in range(len(labels))]
    if len(labels) < 2 or labels != expected_labels:
        return []

    return labels

def _format_question(question, enforce_mcq_format):
    """Ask for one option letter when the question has multiple choices."""
    question = str(question)
    if not enforce_mcq_format:
        return question

    labels = extract_mcq_labels(question)
    if not labels:
        return question

    options = ', '.join(labels)
    instruction = f'Answer with exactly one option letter ({options}). Do not provide an explanation.'

    return f'{question}\n\n{instruction}'

class NoValidTargetsError(ValueError):
    pass

class _ImagePrefixLLM(nn.Module):
    """Shared prompt preparation and LoRA training for both language models."""

    def __init__(self, model_name, model_class, lora_rank, attn_implementation, gradient_checkpointing, target_modules, torch_dtype = torch.float32, revision = None):
        super().__init__()

        if lora_rank < 1:
            raise ValueError(f'lora_rank must be positive, got {lora_rank}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision = revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = 'left'
        self.marker_id = self._add_marker()

        model = model_class.from_pretrained(model_name, dtype = torch_dtype, revision = revision, low_cpu_mem_usage = True,
                                            attn_implementation = attn_implementation)

        # Cache off for training. generate() turns it back on.
        model.config.use_cache = False
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {'use_reentrant': False})

        lora_config = LoraConfig(task_type = TaskType.CAUSAL_LM, inference_mode = False,
                                 r = lora_rank, lora_alpha = 2 * lora_rank, lora_dropout = 0.05,
                                 target_modules = target_modules)
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()

        self.logits_to_keep_supported = self._supports_logits_to_keep()

    @property
    def device(self):
        return self.model.device

    def _add_marker(self):
        """Add the marker as a single token and return its ID."""
        self.tokenizer.add_special_tokens({'additional_special_tokens': [MM_MARKER]})
        marker_id = self.tokenizer.convert_tokens_to_ids(MM_MARKER)

        if marker_id is None:
            raise RuntimeError('Failed to register the multimodal marker')
        if self.tokenizer.unk_token_id is not None and marker_id == self.tokenizer.unk_token_id:
            raise RuntimeError('The multimodal marker became UNK')

        return int(marker_id)

    def _supports_logits_to_keep(self):
        """Check if the model accepts logits_to_keep."""
        try:
            model = self.model.get_base_model()
            parameters = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

        return 'logits_to_keep' in parameters

    def _tokenize_chat(self, messages, add_generation_prompt):
        """Tokenize the full chat, then remove the marker and record its position."""
        token_ids = self.tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = add_generation_prompt,
                                                       date_string = FIXED_CHAT_DATE, return_dict = False)
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Some tokenizer versions return [[ids]] instead of [ids].
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise RuntimeError('Unexpected batched output from apply_chat_template')
            token_ids = token_ids[0]

        token_ids = [int(token_id) for token_id in token_ids]

        marker_positions = [index for index, token_id in enumerate(token_ids) if token_id == self.marker_id]
        if len(marker_positions) != 1:
            raise RuntimeError(f'The multimodal marker must appear exactly once after tokenization. Found {len(marker_positions)}')

        insert_position = marker_positions[0]
        token_ids = token_ids[:insert_position] + token_ids[insert_position + 1:]

        return token_ids, insert_position

    def tokenize_training_chats(self, questions, answers, max_total_tokens = 1024):
        """Prepare answer labels and record the samples whose prompts fit the limit."""
        if len(questions) != len(answers):
            raise ValueError('questions and answers must have the same length')

        token_rows = []
        label_rows = []
        kept_rows = []
        insert_positions = []

        for row_index, (question, answer) in enumerate(zip(questions, answers)):
            if MM_MARKER in str(question) or MM_MARKER in str(answer):
                raise ValueError(f'Reserved multimodal marker in training text: sample={row_index}')
            text_budget = max_total_tokens if isinstance(max_total_tokens, int) else max_total_tokens[row_index]
            user_message = {'role': 'user', 'content': f'{MM_MARKER}\n{question}'}
            assistant_message = {'role': 'assistant', 'content': str(answer)}

            prompt_ids, prompt_insert_position = self._tokenize_chat([user_message], add_generation_prompt = True)
            full_ids, full_insert_position = self._tokenize_chat([user_message, assistant_message], add_generation_prompt = False)

            if prompt_insert_position != full_insert_position:
                raise RuntimeError('Multimodal insertion position changed between prompt and training target')

            # The full chat must start with the prompt so answer labels align.
            prompt_length = len(prompt_ids)
            if full_ids[:prompt_length] != prompt_ids:
                raise RuntimeError('Assistant target sequence is not prefix-aligned with the prompt')
            # Keep the complete answer and its model-specific termination suffix.
            if len(full_ids) > text_budget:
                continue

            # Ignore prompt tokens when computing the loss.
            labels = [-100] * prompt_length + full_ids[prompt_length:]
            token_row = torch.tensor(full_ids, dtype = torch.long, device = self.device)
            label_row = torch.tensor(labels, dtype = torch.long, device = self.device)

            token_rows.append(token_row)
            label_rows.append(label_row)
            kept_rows.append(row_index)
            insert_positions.append(prompt_insert_position)

        if not token_rows:
            raise NoValidTargetsError('No complete training answers fit the text budget')

        return token_rows, label_rows, kept_rows, insert_positions

    def tokenize_generation_chats(self, questions, max_prompt_tokens = 2048, enforce_mcq_format = True):
        """Prepare question prompts for generation."""
        token_rows = []
        insert_positions = []

        for row_index, question in enumerate(questions):
            if MM_MARKER in str(question):
                raise ValueError(f'Reserved multimodal marker in generation question: sample={row_index}')
            text_budget = max_prompt_tokens if isinstance(max_prompt_tokens, int) else max_prompt_tokens[row_index]
            question = _format_question(question, enforce_mcq_format)
            user_message = {'role': 'user', 'content': f'{MM_MARKER}\n{question}'}
            prompt_ids, insert_position = self._tokenize_chat([user_message], add_generation_prompt = True)

            if len(prompt_ids) > text_budget:
                raise ValueError(f'Generation prompt too long: sample={row_index}, tokens={len(prompt_ids)}, limit={text_budget}')

            token_row = torch.tensor(prompt_ids, dtype = torch.long, device = self.device)
            token_rows.append(token_row)
            insert_positions.append(insert_position)

        return token_rows, insert_positions

    def _insert_prefix(self, token_rows, insert_positions, prefix, label_rows, prefix_mask = None):
        """Insert each image's features into its text embeddings and label sequence."""
        B, P, H = prefix.shape
        embed = self.model.get_input_embeddings()
        input_rows = []
        merged_label_rows = [] if label_rows is not None else None

        for row_index in range(B):
            token_row = token_rows[row_index]
            insert_position = int(insert_positions[row_index])
            if not (0 <= insert_position <= token_row.numel()):
                raise ValueError(f'Invalid insertion position {insert_position} for sequence length {token_row.numel()}')

            token_embeddings = embed(token_row)
            text_before = token_embeddings[:insert_position]
            image_features = prefix[row_index]
            if prefix_mask is not None:
                image_features = image_features[prefix_mask[row_index].bool()]
            image_features = image_features.to(token_embeddings.dtype)
            text_after = token_embeddings[insert_position:]

            input_row = torch.cat([text_before, image_features, text_after], dim = 0)
            input_rows.append(input_row)

            if label_rows is not None:
                text_labels = label_rows[row_index]
                labels_before = text_labels[:insert_position]
                labels_after = text_labels[insert_position:]

                # Image features are inputs, so they have no target token labels.
                image_labels = torch.full((image_features.size(0),), -100, dtype = text_labels.dtype, device = text_labels.device)
                label_row = torch.cat([labels_before, image_labels, labels_after], dim = 0)
                merged_label_rows.append(label_row)

        return input_rows, merged_label_rows

    def _pad_inputs(self, input_rows, label_rows):
        """Pad on the left and build the attention mask, positions, and optional labels."""
        max_length = max(row.size(0) for row in input_rows)
        padded_inputs = []
        padded_masks = []
        padded_labels = []

        for row_index, input_row in enumerate(input_rows):
            input_length = input_row.size(0)
            pad_length = max_length - input_length

            padded_input = F.pad(input_row, (0, 0, pad_length, 0), value = 0.0)
            padded_inputs.append(padded_input)

            mask = torch.ones(input_length, dtype = torch.long, device = input_row.device)
            padded_mask = F.pad(mask, (pad_length, 0), value = 0)
            padded_masks.append(padded_mask)

            if label_rows is not None:
                padded_label = F.pad(label_rows[row_index], (pad_length, 0), value = -100)
                padded_labels.append(padded_label)

        input_embeds = torch.stack(padded_inputs, dim = 0)
        attention_mask = torch.stack(padded_masks, dim = 0)

        # Each sample's first unpadded token gets position 0.
        position_ids = attention_mask.cumsum(dim = -1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        if label_rows is not None:
            labels = torch.stack(padded_labels, dim = 0)
        else:
            labels = None

        return input_embeds, attention_mask, position_ids, labels

    def build_inputs(self, token_rows, insert_positions, prefix, label_rows = None, prefix_mask = None):
        """Insert image features (B, P, H), then return padded embeddings, masks, positions, and labels."""
        if prefix.ndim != 3:
            raise ValueError(f'prefix must have shape (B, P, H), got {tuple(prefix.shape)}')

        B, P, H = prefix.shape
        if len(token_rows) != B:
            raise ValueError('token_rows batch does not match prefix batch')
        if len(insert_positions) != B:
            raise ValueError('insert_positions batch does not match prefix batch')
        if label_rows is not None and len(label_rows) != B:
            raise ValueError('label_rows batch does not match prefix batch')

        input_rows, merged_label_rows = self._insert_prefix(token_rows, insert_positions, prefix, label_rows, prefix_mask)

        return self._pad_inputs(input_rows, merged_label_rows)

    def forward(self, input_embeds, attention_mask, position_ids, labels, optimize = True):
        """Compute answer-token loss, optionally calculating only the logits we need."""
        model_inputs = {
            'inputs_embeds': input_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'use_cache': False,
        }

        if not optimize or not self.logits_to_keep_supported:
            output = self.model(**model_inputs, labels = labels)
            return output.loss

        # Each position predicts the next token. -100 labels do not contribute.
        next_token_labels = labels[:, 1:]
        answer_mask = next_token_labels != -100

        # Keep a position if at least one sample has an answer token there.
        active_positions = answer_mask.any(dim = 0)
        prediction_positions = active_positions.nonzero(as_tuple = True)[0]
        if prediction_positions.numel() == 0:
            output = self.model(**model_inputs, labels = labels)
            return output.loss

        output = self.model(**model_inputs, logits_to_keep = prediction_positions)
        answer_labels = next_token_labels.index_select(1, prediction_positions)

        logits = output.logits.float()
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        answer_labels = answer_labels.reshape(-1)

        loss = F.cross_entropy(logits, answer_labels, ignore_index = -100)

        return loss

    def generate(self, input_embeds, attention_mask, max_new_tokens = 32):
        """Generate and decode only the new answer tokens."""
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')

        # Keep all EOS IDs from the checkpoint's generation configuration.
        # generate() updates positions from the mask as it adds tokens.
        generated_ids = self.model.generate(inputs_embeds = input_embeds, attention_mask = attention_mask, max_new_tokens = max_new_tokens,
                                            do_sample = False, use_cache = True, pad_token_id = self.tokenizer.pad_token_id)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens = True)

        return [prediction.strip() for prediction in decoded]

class LlamaLLM(_ImagePrefixLLM):
    """Frozen Llama with trainable LoRA adapters and its tokenizer."""

    def __init__(self, llm_size = 1, lora_rank = 64, attn_implementation = 'sdpa', gradient_checkpointing = True, torch_dtype = torch.float32, revision = None):
        if llm_size not in MODEL_NAMES:
            raise ValueError(f'Invalid llm_size={llm_size}. Expected one of {sorted(MODEL_NAMES)}')

        model_name = MODEL_NAMES[llm_size]
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

        super().__init__(model_name, AutoModelForCausalLM, lora_rank, attn_implementation,
                         gradient_checkpointing, target_modules, torch_dtype, revision)

class Qwen25VLLLM(_ImagePrefixLLM):
    """Qwen2.5-VL with decoder LoRA. Image features must already match its embedding width."""

    def __init__(self, llm_size = 3, lora_rank = 64, attn_implementation = 'sdpa', gradient_checkpointing = True, torch_dtype = torch.float32, revision = None):
        if llm_size not in QWEN_MODEL_NAMES:
            raise ValueError(f'Invalid llm_size={llm_size}. Expected one of {sorted(QWEN_MODEL_NAMES)}')

        model_name = QWEN_MODEL_NAMES[llm_size]

        # Match decoder attention and MLP layers, excluding the unused vision encoder.
        decoder_layers = r'model\.(language_model\.)?layers\.\d+\.'
        attention_layers = r'self_attn\.(q_proj|k_proj|v_proj|o_proj)'
        mlp_layers = r'mlp\.(gate_proj|up_proj|down_proj)'
        target_modules = decoder_layers + '(' + attention_layers + '|' + mlp_layers + ')'

        # Our image features use sequence positions on all three Qwen MRoPE axes.
        super().__init__(model_name, Qwen2_5_VLForConditionalGeneration, lora_rank, attn_implementation,
                         gradient_checkpointing, target_modules, torch_dtype, revision)

def llamaAPI(device, llm_size = 1, lora_rank = 64, attn_implementation = 'sdpa', gradient_checkpointing = True, torch_dtype = torch.float32, revision = None):
    """Build the Llama model with LoRA adapters on the given device."""
    return LlamaLLM(llm_size, lora_rank, attn_implementation, gradient_checkpointing, torch_dtype, revision).to(device)

def qwenAPI(device, llm_size = 3, lora_rank = 64, attn_implementation = 'sdpa', gradient_checkpointing = True, torch_dtype = torch.float32, revision = None):
    """Build Qwen2.5-VL with LoRA adapters for the same image-prefix interface as Llama."""
    return Qwen25VLLLM(llm_size, lora_rank, attn_implementation, gradient_checkpointing, torch_dtype, revision).to(device)
