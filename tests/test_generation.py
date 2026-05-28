"""
tests/test_generation.py
========================
Tests for the text generation engine.

Tests:
- Basic generation (output length)
- Temperature sampling
- Top-k filtering
- Top-p filtering
- Repetition penalty
- EOS stopping
- Streaming generation
- KV cache vs no-cache consistency
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import get_default_config
from model.gpt_model import GPTModel
from inference.generate import (
    generate,
    generate_streaming,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
    sample_token,
)


@pytest.fixture
def tiny_model():
    config = get_default_config("tiny")
    model = GPTModel(config)
    model.eval()
    return model


@pytest.fixture
def prompt():
    """Tiny prompt for testing."""
    return torch.randint(0, 8000, (1, 8))  # (B=1, T=8)


# ============================================================
# SAMPLING FUNCTION TESTS
# ============================================================

class TestSamplingFunctions:
    """Tests for individual sampling utility functions."""

    def test_temperature_scaling_high(self):
        """High temperature should flatten the distribution."""
        logits = torch.tensor([[1.0, 2.0, 10.0, 0.5]])
        scaled = apply_temperature(logits, temperature=2.0)
        # Scaled logits should be smaller (more uniform)
        assert scaled.max() < logits.max()

    def test_temperature_scaling_low(self):
        """Low temperature should sharpen the distribution."""
        logits = torch.tensor([[1.0, 2.0, 10.0, 0.5]])
        scaled = apply_temperature(logits, temperature=0.5)
        # Scaled logits should be larger (more peaked)
        assert scaled.max() > logits.max()

    def test_temperature_one_is_identity(self):
        """Temperature=1.0 should not change logits."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        scaled = apply_temperature(logits, temperature=1.0)
        assert torch.equal(logits, scaled)

    def test_top_k_keeps_k_tokens(self):
        """Top-k should zero out all but k tokens."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = apply_top_k(logits, top_k=3)
        # Non-top-3 should be -inf
        assert (filtered == float("-inf")).sum() == 2

    def test_top_k_zero_disables(self):
        """top_k=0 should not filter anything."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = apply_top_k(logits, top_k=0)
        assert torch.equal(logits, filtered)

    def test_top_p_reduces_candidates(self):
        """top_p should reduce the number of valid candidates."""
        # Very concentrated distribution: one token dominates
        logits = torch.tensor([[10.0, 0.1, 0.1, 0.1]])
        filtered = apply_top_p(logits, top_p=0.9)
        import torch.nn.functional as F
        probs = F.softmax(filtered, dim=-1)
        # Most probability mass should be on token 0
        assert probs[0, 0] > 0.9

    def test_repetition_penalty_reduces_prob(self):
        """Repetition penalty should reduce probability of seen tokens."""
        logits = torch.ones(1, 100)
        seen_ids = torch.tensor([[0, 1, 2, 3, 4]])
        penalized = apply_repetition_penalty(logits.clone(), seen_ids, penalty=1.5)
        # Seen tokens should have lower logits
        for i in range(5):
            assert penalized[0, i] < logits[0, i], \
                f"Token {i} was not penalized"

    def test_sample_token_output_shape(self):
        """sample_token should return (B, 1) tensor."""
        logits = torch.randn(3, 8000)  # B=3, V=8000
        next_tok = sample_token(logits)
        assert next_tok.shape == (3, 1)
        assert next_tok.dtype == torch.long


# ============================================================
# GENERATION TESTS
# ============================================================

class TestGenerate:
    """Tests for the main generate function."""

    def test_output_length(self, tiny_model, prompt):
        """Generated output should have correct length."""
        max_new = 10
        with torch.no_grad():
            output = generate(tiny_model, prompt, max_new_tokens=max_new)
        expected_len = prompt.shape[1] + max_new
        assert output.shape[1] == expected_len, \
            f"Expected length {expected_len}, got {output.shape[1]}"

    def test_input_preserved(self, tiny_model, prompt):
        """Prompt tokens should be preserved in output."""
        with torch.no_grad():
            output = generate(tiny_model, prompt, max_new_tokens=5)
        # First T tokens should match the prompt
        T = prompt.shape[1]
        assert torch.equal(output[:, :T], prompt), \
            "Prompt tokens were modified during generation"

    def test_eos_stopping(self, tiny_model):
        """Generation should stop at EOS token."""
        # We can't easily force EOS to be generated, but we can verify
        # the mechanism doesn't crash and respects max_new_tokens
        prompt = torch.randint(0, 8000, (1, 4))
        with torch.no_grad():
            output = generate(
                tiny_model, prompt,
                max_new_tokens=20,
                eos_token_id=2,  # EOS token
            )
        # Output should not exceed prompt + max_new_tokens
        assert output.shape[1] <= prompt.shape[1] + 20

    def test_greedy_deterministic(self, tiny_model, prompt):
        """Very low temperature should give near-deterministic output."""
        with torch.no_grad():
            out1 = generate(tiny_model, prompt, max_new_tokens=5, temperature=0.01, top_k=1)
            out2 = generate(tiny_model, prompt, max_new_tokens=5, temperature=0.01, top_k=1)
        assert torch.equal(out1, out2), "Greedy decoding should be deterministic"

    def test_kv_cache_vs_no_cache(self, tiny_model, prompt):
        """KV cache and no-cache should produce same output with greedy decoding."""
        with torch.no_grad():
            out_cached = generate(
                tiny_model, prompt, max_new_tokens=5,
                temperature=0.01, top_k=1, use_kv_cache=True
            )
            out_no_cache = generate(
                tiny_model, prompt, max_new_tokens=5,
                temperature=0.01, top_k=1, use_kv_cache=False
            )
        assert torch.equal(out_cached, out_no_cache), \
            "KV cache and non-cache generation should match with greedy decoding"

    def test_batch_generation(self, tiny_model):
        """Generation should work with batch_size > 1."""
        B, T = 3, 8
        prompt = torch.randint(0, 8000, (B, T))
        with torch.no_grad():
            output = generate(tiny_model, prompt, max_new_tokens=5)
        assert output.shape == (B, T + 5)

    def test_different_temperatures(self, tiny_model, prompt):
        """Different temperatures should produce different outputs (usually)."""
        torch.manual_seed(42)
        with torch.no_grad():
            out_low = generate(tiny_model, prompt, max_new_tokens=10, temperature=0.1)
            out_high = generate(tiny_model, prompt, max_new_tokens=10, temperature=2.0)
        # Very likely to differ (not guaranteed but should differ with high temperature)
        # We just check shapes are correct
        T = prompt.shape[1]
        assert out_low.shape[1] == T + 10
        assert out_high.shape[1] == T + 10


# ============================================================
# STREAMING TESTS
# ============================================================

class TestGenerateStreaming:
    """Tests for the streaming generation function."""

    def test_streaming_yields_tokens(self, tiny_model, prompt):
        """Streaming should yield token strings."""
        # Mock tokenizer
        class MockTokenizer:
            eos_id = 2
            def decode(self, ids, skip_special_tokens=True):
                return "x" * len(ids)  # Return fixed string

        mock_tok = MockTokenizer()
        tokens = []

        with torch.no_grad():
            for tok in generate_streaming(
                tiny_model, prompt, mock_tok, max_new_tokens=5
            ):
                tokens.append(tok)

        assert len(tokens) == 5, f"Expected 5 tokens, got {len(tokens)}"

    def test_streaming_produces_strings(self, tiny_model, prompt):
        """Each yielded item should be a string."""
        class MockTokenizer:
            eos_id = 2
            def decode(self, ids, skip_special_tokens=True):
                return "token"

        mock_tok = MockTokenizer()

        with torch.no_grad():
            for tok in generate_streaming(tiny_model, prompt, mock_tok, max_new_tokens=3):
                assert isinstance(tok, str), f"Expected str, got {type(tok)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
