import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from model.ipmoe import IPMoE
from model.llm import MM_MARKER, NoValidTargetsError, _PrefixLLM


class ChatTokenizer:
    """A deterministic chat template with explicit role and termination tokens."""
    pad_token_id = 0
    eos_token_id = 2
    marker_id = 6

    def __init__(self):
        self.calls = []

    def encode(self, text):
        pieces = text.split(MM_MARKER)
        tokens = []
        for index, piece in enumerate(pieces):
            if index:
                tokens.append(self.marker_id)
            tokens.extend(byte + 8 for byte in piece.encode('utf-8'))
        return tokens

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, date_string, return_dict):
        self.calls.append([dict(message) for message in messages])
        tokens = [1]
        for message in messages:
            tokens.append(3 if message['role'] == 'user' else 5)
            tokens.extend(self.encode(message['content']))
            tokens.append(4)
            if message['role'] == 'assistant':
                tokens.append(self.eos_token_id)
        if add_generation_prompt:
            tokens.append(5)
        return tokens

    def batch_decode(self, rows, skip_special_tokens):
        self.decoded_ids = rows.detach().clone()
        return [' ' + ' '.join(f'token_{token}' for token in row.tolist() if not skip_special_tokens or token > 7) + ' '
                for row in rows]


class PrefixLLMSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size = 272, hidden_size = 32, intermediate_size = 64,
                             num_hidden_layers = 1, num_attention_heads = 4, num_key_value_heads = 2,
                             max_position_embeddings = 256, attention_dropout = 0.0,
                             bos_token_id = 1, eos_token_id = 2, pad_token_id = 0)
        config._attn_implementation = 'eager'
        # Exercise the real prefix methods and decoder without downloading pretrained weights.
        self.llm = _PrefixLLM.__new__(_PrefixLLM)
        nn.Module.__init__(self.llm)
        self.llm.model = LlamaForCausalLM(config)
        self.llm.tokenizer = ChatTokenizer()
        self.llm.marker_id = self.llm.tokenizer.marker_id
        self.llm.compute_dtype = torch.float32
        self.llm.logits_to_keep_supported = True
        self.llm.eval()

    def test_training_keeps_complete_answers_and_skips_over_budget_rows(self):
        questions = ['Which animal?', 'Which color?']
        answers = ['cat', 'blue']
        prompts = [[1, 3] + self.llm.tokenizer.encode('\n' + question) + [4, 5] for question in questions]
        targets = [self.llm.tokenizer.encode(answer) + [4, 2] for answer in answers]
        lengths = [len(prompt) + len(target) for prompt, target in zip(prompts, targets)]

        rows, labels, kept, positions = self.llm.tokenize_training_chats(
            questions, answers, max_total_tokens = [lengths[0], lengths[1] - 1])

        self.assertEqual(kept, [0])
        self.assertEqual(positions, [2])
        self.assertEqual(rows[0].tolist(), prompts[0] + targets[0])
        self.assertEqual(labels[0].tolist(), [-100] * len(prompts[0]) + targets[0])
        self.assertNotIn(self.llm.marker_id, rows[0].tolist())
        with self.assertRaises(NoValidTargetsError):
            self.llm.tokenize_training_chats(questions[:1], answers[:1], max_total_tokens = lengths[0] - 1)

    def test_generation_preserves_the_original_question_and_enforces_budget(self):
        question = 'Which animal?\nA. cat\nB. dog'
        expected = [1, 3] + self.llm.tokenizer.encode('\n' + question) + [4, 5]

        rows, positions = self.llm.tokenize_generation_chats([question], max_prompt_tokens = len(expected))

        self.assertEqual(self.llm.tokenizer.calls[-1], [{'role': 'user', 'content': MM_MARKER + '\n' + question}])
        self.assertEqual(rows[0].tolist(), expected)
        self.assertEqual(positions, [2])
        with self.assertRaisesRegex(ValueError, 'Generation prompt too long'):
            self.llm.tokenize_generation_chats([question], max_prompt_tokens = len(expected) - 1)

    def test_real_decoder_generation_is_bounded_deterministic_and_returns_answers(self):
        questions = ['Which animal?\nA. cat\nB. dog', 'What color?']
        original_questions = list(questions)
        rows, positions = self.llm.tokenize_generation_chats(questions)
        prefix = torch.randn(2, 3, 32)
        prefix_mask = torch.tensor([[True, False, True], [True, True, True]])
        embeds, mask, _, _ = self.llm.build_inputs(rows, positions, prefix, prefix_mask = prefix_mask)
        original_embeds, original_mask = embeds.clone(), mask.clone()
        # Disable early EOS so the real decoder must exercise the exact generation limit.
        self.llm.model.generation_config.eos_token_id = None

        answers = self.llm.generate(embeds, mask, max_new_tokens = 3)
        first_ids = self.llm.tokenizer.decoded_ids.clone()
        repeated = self.llm.generate(embeds, mask, max_new_tokens = 3)

        self.assertIsInstance(answers, list)
        self.assertEqual(len(answers), len(questions))
        self.assertTrue(all(isinstance(answer, str) and answer == answer.strip() for answer in answers))
        self.assertEqual(tuple(first_ids.shape), (2, 3))
        self.assertEqual(repeated, answers)
        torch.testing.assert_close(self.llm.tokenizer.decoded_ids, first_ids)
        torch.testing.assert_close(embeds, original_embeds)
        torch.testing.assert_close(mask, original_mask)
        self.assertEqual(questions, original_questions)

        # Keep the image encoder out of this decoder test while exercising IPMoE's public return contract.
        ipmoe = IPMoE.__new__(IPMoE)
        nn.Module.__init__(ipmoe)
        ipmoe.llm = self.llm
        ipmoe.eval()
        with patch.object(ipmoe, 'encode_prefix', return_value = (prefix, prefix_mask)):
            generated = ipmoe.generate(questions, {}, {}, context_limit = 64, max_new_tokens = 3)
        self.assertIsInstance(generated, list)
        self.assertEqual(generated, answers)
        self.assertEqual(questions, original_questions)
        self.assertEqual(self.llm.tokenizer.calls[-2:], [[{'role': 'user', 'content': MM_MARKER + '\n' + question}]
                                                       for question in questions])

        for invalid in (0, -1):
            with self.subTest(max_new_tokens = invalid), self.assertRaisesRegex(ValueError, 'max_new_tokens must be positive'):
                self.llm.generate(embeds, mask, max_new_tokens = invalid)

    def test_prefix_insertion_masks_labels_and_left_padding(self):
        rows = [torch.tensor([1, 8, 9, 2]), torch.tensor([1, 10, 2])]
        labels = [torch.tensor([-100, -100, 9, 2]), torch.tensor([-100, 10, 2])]
        prefix = torch.arange(2 * 3 * 32, dtype = torch.float32).reshape(2, 3, 32) / 100
        prefix_mask = torch.tensor([[True, False, True], [False, True, False]])

        embeds, mask, positions, targets = self.llm.build_inputs(rows, [1, 2], prefix, labels, prefix_mask)

        embed = self.llm.model.get_input_embeddings()
        expected_first = torch.cat((embed(rows[0][:1]), prefix[0, [0, 2]], embed(rows[0][1:])))
        expected_second = torch.cat((torch.zeros(2, 32), embed(rows[1][:2]), prefix[1, 1:2], embed(rows[1][2:])))
        torch.testing.assert_close(embeds, torch.stack((expected_first, expected_second)))
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
        self.assertEqual(positions.tolist(), [[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 2, 3]])
        self.assertEqual(targets.tolist(), [[-100, -100, -100, -100, 9, 2], [-100, -100, -100, 10, -100, 2]])

    def test_optimized_causal_loss_and_gradients_match_full_logits(self):
        rows, labels, kept, positions = self.llm.tokenize_training_chats(['Name?', 'Name the color?'], ['cat', 'blue'])
        self.assertEqual(kept, [0, 1])
        prefix_values = torch.randn(2, 3, 32)
        prefix_mask = torch.tensor([[True, False, True], [True, True, True]])
        results = []

        for optimized in (True, False):
            self.llm.zero_grad(set_to_none = True)
            self.llm.logits_to_keep_supported = optimized
            prefix = prefix_values.clone().requires_grad_()
            inputs = self.llm.build_inputs(rows, positions, prefix, labels, prefix_mask)
            loss = self.llm(*inputs)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            gradients = {name: parameter.grad.detach().clone() for name, parameter in self.llm.model.named_parameters()}
            self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients.values()))
            self.assertGreater(prefix.grad[prefix_mask].abs().sum().item(), 0)
            self.assertEqual(prefix.grad[~prefix_mask].abs().sum().item(), 0)
            results.append((loss.detach(), prefix.grad.clone(), gradients))

        torch.testing.assert_close(results[0][0], results[1][0], rtol = 1e-5, atol = 1e-6)
        torch.testing.assert_close(results[0][1], results[1][1], rtol = 1e-4, atol = 1e-6)
        for name in results[0][2]:
            torch.testing.assert_close(results[0][2][name], results[1][2][name], rtol = 1e-4, atol = 1e-6)


if __name__ == '__main__':
    unittest.main()
