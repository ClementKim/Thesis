"""Offline smoke tests with small random CLIP/Llama weights and synthetic data.

Run with the dependencies in requirements.txt installed. This checks execution,
masking, and gradients; it does not measure answer quality or trained-model rank.
"""

import os
import argparse
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

# Every model and tokenizer used below is created locally for this test.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import CLIPConfig, CLIPModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from model.common import SelfAttentionBlock, CrossAttentionBlock
    from model.encoder import CLIPEncoder, TextEncoder, VisionEncoder
    from model.interaction import Interaction, InteractionAdapter
    from model.experts import SelfExpert, CrossExpert, HybridExpert
    from model.router import MoE
    from model.llm import LlamaLLM, NoValidTargetsError
    from model.ipmoe import IPMoE
except ImportError as exc:
    raise SystemExit(f'Missing or incompatible dependency: {exc}\n'
                     'Use an environment with requirements.txt installed, or set PYTHON_BIN for smoke_test.sh.') from exc


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_error(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f'Expected {error_type.__name__} from {function.__name__}')


def build_fixture(directory, device):
    """Exercise the real loaders using temporary, randomly initialized weights."""
    words = ['<pad>', '<bos>', '<eos>', '<unk>', '<user>', '<assistant>',
             'what', 'color', 'is', 'the', 'square', 'red', 'blue', 'describe', 'image']
    vocab = {word: index for index, word in enumerate(words)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token = '<unk>'))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = tokenizer, pad_token = '<pad>',
                                        bos_token = '<bos>', eos_token = '<eos>', unk_token = '<unk>')
    tokenizer.chat_template = (
        '{{ bos_token }} '
        '{% for message in messages %}'
        '{{ "<" + message["role"] + ">" }} {{ message["content"] }} '
        '{% if message["role"] == "assistant" %}{{ eos_token }} {% endif %}'
        '{% endfor %}'
        '{% if add_generation_prompt %}<assistant> {% endif %}'
    )

    llm_directory = Path(directory) / 'llama'
    config = LlamaConfig(vocab_size = len(vocab), hidden_size = 32, intermediate_size = 64,
                         num_hidden_layers = 1, num_attention_heads = 4, num_key_value_heads = 2,
                         max_position_embeddings = 2048, attention_dropout = 0.0,
                         pad_token_id = 0, bos_token_id = 1, eos_token_id = 2)
    LlamaForCausalLM(config).save_pretrained(llm_directory)
    tokenizer.save_pretrained(llm_directory)

    clip_directory = Path(directory) / 'clip'
    config = CLIPConfig(
        text_config = {'vocab_size': len(vocab), 'hidden_size': 32, 'intermediate_size': 64,
                       'num_hidden_layers': 1, 'num_attention_heads': 4, 'max_position_embeddings': 77,
                       'pad_token_id': 0, 'bos_token_id': 1, 'eos_token_id': 2},
        vision_config = {'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2,
                         'num_attention_heads': 4, 'image_size': 336, 'patch_size': 14},
        projection_dim = 32,
    )
    CLIPModel(config).save_pretrained(clip_directory)

    # Redirect only model locations; forward methods and LoRA setup remain unchanged.
    with patch('model.encoder.MODEL_NAME', str(clip_directory)):
        clip = CLIPEncoder()
    with patch.dict('model.llm.MODEL_NAMES', {3: str(llm_directory)}):
        llm = LlamaLLM(3, lora_rank = 2, gradient_checkpointing = True)

    text_encoder = TextEncoder(clip, 0.0, out_dim = 32, chunk_batch_size = 2)
    vision_encoder = VisionEncoder(clip, 0.0, out_dim = 32)
    self_expert = SelfExpert(None, 16, 1, 32, 4, 2.0, 0.0, 0.0, True)
    cross_expert = CrossExpert(None, 16, 1, 32, 4, 2.0, 0.0, 0.0, False)
    hybrid_expert = HybridExpert(None, 16, 1, 32, 4, 2.0, 0.0, 0.0, False)
    moe = MoE(1, self_expert, cross_expert, hybrid_expert)
    model = IPMoE(None, text_encoder, vision_encoder, Interaction('bilinear'), moe, llm, 32, 16).to(device)

    # The second question crosses CLIP's 75-content-token boundary.
    questions = ['what color is the square', ' '.join(['image'] * 76)]
    answers = ['red', 'the square is blue']
    rows = [[1] + [vocab[word] for word in question.split()] + [2] for question in questions]
    input_ids = torch.zeros((len(rows), max(map(len, rows))), dtype = torch.long, device = device)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, :len(row)] = torch.tensor(row, device = device)
        attention_mask[index, :len(row)] = 1
    text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    image_inputs = {'pixel_values': torch.randn(2, 3, 336, 336, device = device)}

    return model, questions, answers, text_inputs, image_inputs


def test_interaction():
    """Check actual adapter algebra and recover a padded map in float64 on CPU."""
    text = torch.randn(2, 5, 32, dtype = torch.float64)
    vision = torch.randn(2, 11, 32, dtype = torch.float64)
    similarity = Interaction('bilinear')(text, vision)
    torch.testing.assert_close(similarity, torch.einsum('bld,bnd->bln', text, vision))

    # This independent fixture has enough width to retain a 4 x 4 window.
    adapter = InteractionAdapter(32, 4).double()
    head = nn.Linear(32, 48).double()
    text_mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    with torch.no_grad():
        tokens, _, map_mask, token_mask = adapter(similarity, text_mask)
        zeros, _, _, _ = adapter(torch.zeros_like(similarity), text_mask)
        require(token_mask.tolist() == [[True] * 6, [True] * 3 + [False] * 3], 'Wrong patch mask positions')

        layer = adapter.patch_embed
        matrix = head.weight @ layer.linear.weight @ layer.patchify.weight.flatten(1)
        require(torch.linalg.matrix_rank(matrix).item() == 16, 'Random test projection is rank deficient')
        target = F.pad(similarity.masked_fill(~map_mask, 0)[:, None], (0, 1, 0, 3))
        windows = F.unfold(target, kernel_size = 4, stride = 4).transpose(1, 2)
        encoded = head(tokens) - head(zeros)
        torch.testing.assert_close(encoded, windows @ matrix.T, rtol = 1e-7, atol = 1e-8)
        recovered = encoded @ torch.linalg.pinv(matrix).T
        recovered = F.fold(recovered.transpose(1, 2), (8, 12), kernel_size = 4, stride = 4)
        torch.testing.assert_close(recovered, target, rtol = 1e-7, atol = 1e-8)

        changed = similarity.masked_fill(~map_mask, 10000.0)
        changed_tokens = adapter(changed, text_mask)[0]
        torch.testing.assert_close(changed_tokens, tokens)


def test_attention(device):
    mask = torch.ones(2, 5, dtype = torch.bool, device = device)
    padded_mask = F.pad(mask, (0, 3), value = False)
    context = torch.randn(2, 5, 32, device = device)
    padded = torch.cat([context, torch.randn(2, 3, 32, device = device) * 100], dim = 1)
    self_attention = SelfAttentionBlock(32, 4).to(device).eval()
    cross_attention = CrossAttentionBlock(32, 4).to(device).eval()
    query = torch.randn(2, 1, 32, device = device)

    with torch.no_grad():
        output = self_attention(padded, padded_mask)
        torch.testing.assert_close(output[:, :5], self_attention(context, mask), rtol = 1e-4, atol = 1e-5)
        require(torch.count_nonzero(output[:, 5:]).item() == 0, 'Masked queries produced nonzero outputs')
        torch.testing.assert_close(cross_attention(query, padded, padded_mask),
                                   cross_attention(query, context, mask), rtol = 1e-4, atol = 1e-5)


def test_encoders_and_prefix(model, text_inputs, image_inputs):
    model.eval()
    encoder = model.text_encoder
    chunk_ids, _, chunk_info = encoder._make_chunks(**text_inputs)
    require(chunk_info == [(0, 5, 0), (1, 75, 0), (1, 1, 1)], 'Incorrect CLIP chunk boundaries')
    for sample_index in range(2):
        content = torch.cat([chunk_ids[row, 1:1 + length]
                             for row, (sample, length, _) in enumerate(chunk_info) if sample == sample_index])
        original = text_inputs['input_ids'][sample_index][text_inputs['attention_mask'][sample_index].bool()][1:-1]
        torch.testing.assert_close(content, original)

    with torch.no_grad():
        prefix, mask, weights = model.encode_prefix(text_inputs, image_inputs)
        require(prefix.shape == (2, 759, 32), f'Unexpected prefix shape: {tuple(prefix.shape)}')
        expected = torch.ones_like(mask)
        expected[0, 576 + 36:576 + 180] = False
        require(torch.equal(mask, expected), 'Prefix mask misplaced image, interaction, or expert tokens')
        require(model._prefix_lengths(text_inputs, image_inputs) == [615, 759], 'Wrong predicted prefix lengths')
        require(mask.sum(dim = 1).tolist() == [615, 759], 'Wrong encoded prefix lengths')
        require(torch.isfinite(prefix).all().item() and torch.isfinite(weights).all().item(), 'Nonfinite prefix or routing')
        require((weights > 0).all().item(), 'Routing weights must be positive')
        torch.testing.assert_close(weights.sum(dim = 1), torch.ones(2, device = prefix.device))

        short_text = {key: value[:1, :7] for key, value in text_inputs.items()}
        short_image = {key: value[:1] for key, value in image_inputs.items()}
        short_prefix, short_mask, short_weights = model.encode_prefix(short_text, short_image)
        torch.testing.assert_close(prefix[0, mask[0]], short_prefix[0, short_mask[0]], rtol = 1e-4, atol = 1e-5)
        torch.testing.assert_close(weights[:1], short_weights, rtol = 1e-4, atol = 1e-5)

        empty = {'input_ids': text_inputs['input_ids'].new_tensor([[1, 2]]),
                 'attention_mask': text_inputs['attention_mask'].new_ones((1, 2))}
        features, empty_mask = encoder(**empty)
        require(features.shape == (1, 1, 32) and empty_mask.sum().item() == 1, 'Empty text lost its EOS placeholder')

        norm = model.vision_encoder.mlp[0]
        patches = torch.randn(2, 4, 32, device = prefix.device)
        torch.testing.assert_close(norm(patches), norm(patches + 3), rtol = 1e-4, atol = 1e-5)


def test_llm(model, questions, answers):
    llm = model.llm.eval()
    rows, labels, kept, positions = llm.tokenize_training_chats(questions, answers, max_total_tokens = 2048)
    require(kept == [0, 1], 'Training tokenizer unexpectedly dropped a sample')
    for row, label, answer in zip(rows, labels, answers):
        expected = llm.tokenizer.encode(answer, add_special_tokens = False) + [llm.tokenizer.eos_token_id]
        require(label[label != -100].tolist() == expected, 'Answer or EOS was removed from supervision')
        require(llm.marker_id not in row.tolist(), 'Multimodal marker was not removed')

    selected = llm.tokenize_training_chats(questions, answers, max_total_tokens = [rows[0].numel(), rows[1].numel() - 1])
    require(selected[2] == [0], 'Context boundary retained the wrong samples')
    torch.testing.assert_close(selected[0][0], rows[0])
    torch.testing.assert_close(selected[1][0], labels[0])
    expect_error(NoValidTargetsError, llm.tokenize_training_chats, questions, answers, max_total_tokens = 0)

    prefix = torch.randn(2, 4, 32, device = llm.device, requires_grad = True)
    prefix_mask = torch.tensor([[True, False, True, False], [False, True, True, True]], device = llm.device)
    inputs = llm.build_inputs(rows, positions, prefix, labels, prefix_mask = prefix_mask)
    embeds, mask, position_ids, merged_labels = inputs
    for index in range(2):
        valid = mask[index].bool()
        expected_length = rows[index].numel() + prefix_mask[index].sum().item()
        require(valid.sum().item() == expected_length, 'Wrong merged input length')
        torch.testing.assert_close(position_ids[index, valid], torch.arange(expected_length, device = llm.device))
        torch.testing.assert_close(merged_labels[index][merged_labels[index] != -100], labels[index][labels[index] != -100])
        insertion = positions[index]
        prefix_length = int(prefix_mask[index].sum().item())
        torch.testing.assert_close(embeds[index, valid][insertion:insertion + prefix_length], prefix[index, prefix_mask[index]])
        require((merged_labels[index, valid][insertion:insertion + prefix_length] == -100).all().item(), 'Prefix has labels')
        require((merged_labels[index, ~valid] == -100).all().item(), 'Padding has labels')

    require(llm.logits_to_keep_supported, 'Tiny Llama must support the optimized loss path')
    parameters = [parameter for parameter in llm.parameters() if parameter.requires_grad]
    full_loss = llm(*inputs, optimize = False)
    full_gradients = torch.autograd.grad(full_loss, [prefix] + parameters, retain_graph = True)
    optimized_loss = llm(*inputs, optimize = True)
    optimized_gradients = torch.autograd.grad(optimized_loss, [prefix] + parameters)
    torch.testing.assert_close(optimized_loss, full_loss, rtol = 2e-5, atol = 2e-6)
    for actual, expected in zip(optimized_gradients, full_gradients):
        require(torch.isfinite(actual).all().item(), 'Nonfinite loss gradient')
        torch.testing.assert_close(actual, expected, rtol = 1e-4, atol = 2e-6)
    require(torch.count_nonzero(optimized_gradients[0][~prefix_mask]).item() == 0, 'Masked prefix received gradients')
    require(optimized_gradients[0][prefix_mask].abs().sum().item() > 0, 'Retained prefix received no gradient')
    require(any(gradient.abs().sum().item() > 0 for gradient in optimized_gradients[1:]), 'LoRA received no gradient')


def test_training(model, questions, answers, text_inputs, image_inputs, steps):
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr = 1e-3)
    model.train()
    require(not model.text_encoder.clip.training, 'Frozen CLIP entered training mode')
    require(all(not parameter.requires_grad for parameter in model.text_encoder.clip.parameters()), 'CLIP parameters are not frozen')
    require(all('lora_' in name for name, parameter in model.llm.named_parameters() if parameter.requires_grad),
            'The language model has trainable parameters outside LoRA')
    initial = model.image_head.weight.detach().clone()
    groups = {
        'text projector': model.text_encoder.mlp, 'vision projector': model.vision_encoder.mlp,
        'interaction adapter': model.interaction_adapter, 'router': model.moe.router,
        'self expert': model.moe.self_expert, 'cross expert': model.moe.cross_expert,
        'hybrid expert': model.moe.hybrid_expert, 'image head': model.image_head,
        'interaction head': model.interaction_head, 'expert head': model.expert_head, 'LoRA': model.llm,
    }
    for step in range(steps):
        optimizer.zero_grad(set_to_none = True)
        loss, weights, tokens = model(questions, text_inputs, image_inputs, answers, 1024, return_stats = True)
        require(torch.isfinite(loss).item() and tokens == 7 and weights.shape == (2, 3), 'Invalid training result')
        loss.backward()
        for name, module in groups.items():
            gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
            require(gradients and all(gradient is not None and torch.isfinite(gradient).all().item()
                                     for gradient in gradients), f'Missing or nonfinite gradient in {name}')
            require(any(gradient.abs().sum().item() > 0 for gradient in gradients), f'Zero gradients in {name}')
        require(model.aux.grad is not None and torch.isfinite(model.aux.grad).all().item()
                and model.aux.grad.abs().sum().item() > 0, 'Auxiliary token received no finite gradient')
        require(all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad),
                'A frozen parameter received a gradient')
        optimizer.step()
        require(all(torch.isfinite(parameter).all().item() for parameter in model.parameters() if parameter.requires_grad),
                'Optimizer produced nonfinite parameters')
        print(f'  step {step + 1}: loss={loss.item():.6f}, supervised_tokens={tokens}', flush = True)
    require(not torch.equal(initial, model.image_head.weight), 'Optimizer did not update the image head')

    model.eval()
    with torch.no_grad():
        loss, weights, tokens = model(questions, text_inputs, image_inputs, answers, 650, return_stats = True)
        require(torch.isfinite(loss).item() and weights.shape == (1, 3) and tokens == 2, 'Mixed-budget batch failed')
        expect_error(NoValidTargetsError, model.forward, questions, text_inputs, image_inputs, answers, 1)
        predictions, weights = model.generate(questions, text_inputs, image_inputs, 1024,
                                              max_new_tokens = 2, enforce_mcq_format = False)
        require(len(predictions) == 2 and all(isinstance(value, str) for value in predictions), 'Generation returned invalid answers')
        require(weights.shape == (2, 3), 'Generation returned invalid routing weights')


def parse_args():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--device', choices = ['cpu', 'cuda', 'mps'], default = 'cpu')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--steps', type = int, default = 2)
    parser.add_argument('--threads', type = int, default = 1)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error('steps and threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is not available')
    return args


def main(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    print(f'Offline smoke test: device={args.device}, seed={args.seed}, random small backbones', flush = True)
    with tempfile.TemporaryDirectory(prefix = 'ipmoe-smoke-') as directory:
        model, questions, answers, text_inputs, image_inputs = build_fixture(directory, args.device)
        checks = [
            ('interaction algebra, masks, and reconstruction', lambda: test_interaction()),
            ('attention padding invariance', lambda: test_attention(args.device)),
            ('CLIP chunks and complete prefix', lambda: test_encoders_and_prefix(model, text_inputs, image_inputs)),
            ('answer labels and optimized loss/gradient equivalence', lambda: test_llm(model, questions, answers)),
            ('training, budget filtering, and generation', lambda: test_training(model, questions, answers, text_inputs, image_inputs, args.steps)),
        ]
        for name, check in checks:
            start = time.perf_counter()
            check()
            print(f'PASS: {name} ({time.perf_counter() - start:.2f}s)', flush = True)
    print('All 5 checks passed. Pretrained weights, the dataset, and answer quality were not tested.', flush = True)


if __name__ == '__main__':
    main(parse_args())
