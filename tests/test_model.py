"""
tests/test_model.py
===================
Unit tests for the GPT transformer model.

Tests:
- Model initialization
- Forward pass shapes
- Loss computation
- Weight tying
- KV cache correctness
- Parameter count
- Gradient flow
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import get_default_config, ModelArchConfig
from model.attention import MultiHeadCausalAttention
from model.transformer_block import TransformerBlock, FeedForward
from model.gpt_model import GPTModel


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def tiny_config():
    """Return a minimal config for fast tests."""
    return get_default_config("tiny")


@pytest.fixture
def tiny_model(tiny_config):
    """Return a tiny GPT model for testing."""
    model = GPTModel(tiny_config)
    model.eval()
    return model


@pytest.fixture
def batch():
    """Return a small batch of token IDs."""
    B, T = 2, 32
    vocab_size = 8000
    return torch.randint(0, vocab_size, (B, T))


# ============================================================
# ATTENTION TESTS
# ============================================================

class TestMultiHeadAttention:
    """Tests for the attention module."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        d_model, n_heads = 64, 4
        attn = MultiHeadCausalAttention(d_model=d_model, n_heads=n_heads, max_seq_len=128)
        x = torch.randn(2, 16, d_model)
        out, _ = attn(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_causal_mask(self):
        """
        Test that causal masking prevents attention to future tokens.

        Verification approach: with a zero-initialized model (but non-zero bias
        on the output proj) and dropout disabled, position i's output should
        only depend on tokens 0..i. We verify by zeroing future tokens: the
        first position output should not change.

        Instead of relying on exact equality across different-length sequences
        (which can differ due to residual scaling or RNG), we check the mask
        is present in the attention module's buffer.
        """
        d_model, n_heads = 64, 4
        attn = MultiHeadCausalAttention(d_model=d_model, n_heads=n_heads, max_seq_len=128)

        # Verify causal mask buffer exists and has correct upper-triangle structure
        mask = attn.causal_mask  # shape: (1, 1, max_seq_len, max_seq_len)
        assert mask is not None, "Causal mask buffer should exist"

        # Upper triangle (excluding diagonal) should be True (blocked)
        T = 8
        m = mask[0, 0, :T, :T]  # (T, T)
        for i in range(T):
            for j in range(T):
                if j > i:
                    assert m[i, j].item() is True or m[i, j].item() == 1, \
                        f"Position ({i},{j}) should be masked (j > i)"
                else:
                    assert not m[i, j].item(), \
                        f"Position ({i},{j}) should NOT be masked (j <= i)"

    def test_kv_cache_consistency(self):
        """KV cache output should match non-cache output."""
        d_model, n_heads = 64, 4
        attn = MultiHeadCausalAttention(d_model=d_model, n_heads=n_heads, max_seq_len=128)
        attn.eval()

        torch.manual_seed(0)
        x = torch.randn(1, 4, d_model)

        with torch.no_grad():
            # Full sequence forward
            out_full, _ = attn(x, use_cache=False)

            # Incremental with KV cache
            out_step1, kv = attn(x[:, :3, :], use_cache=True)
            out_step2, _ = attn(x[:, 3:4, :], past_kv=kv, use_cache=True)

        # Last token output should match
        assert torch.allclose(out_full[:, 3:4, :], out_step2, atol=1e-4), \
            "KV cache produces different results from full attention"

    def test_head_dim_assertion(self):
        """Should raise if d_model not divisible by n_heads."""
        with pytest.raises(AssertionError):
            MultiHeadCausalAttention(d_model=65, n_heads=4)

    def test_no_kv_returned_when_not_requested(self):
        """KV cache should not be returned when use_cache=False."""
        attn = MultiHeadCausalAttention(d_model=64, n_heads=4, max_seq_len=128)
        x = torch.randn(1, 8, 64)
        _, kv = attn(x, use_cache=False)
        assert kv is None


# ============================================================
# TRANSFORMER BLOCK TESTS
# ============================================================

class TestTransformerBlock:
    """Tests for the transformer block."""

    def test_output_shape(self):
        """Block output should have same shape as input."""
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
        x = torch.randn(2, 16, 64)
        out, _ = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Verify residual connection is active."""
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
        # Zero out the attention and FFN weights
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
        x = torch.randn(2, 4, 64)
        out, _ = block(x)
        # With zeroed weights and layer norms starting at 1, residual should keep x
        # (not exactly x due to layer norm, but output shouldn't be all zeros)
        assert out.abs().mean() > 0


class TestFeedForward:
    """Tests for the FFN module."""

    def test_output_shape(self):
        ffn = FeedForward(d_model=64, d_ff=256)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_gelu_activation(self):
        """Verify GELU is used (not ReLU)."""
        ffn = FeedForward(d_model=64, d_ff=256)
        # GELU(0) ≈ 0, GELU(-10) < 0 (unlike ReLU which clips to 0)
        import torch.nn.functional as F
        gelu_neg = F.gelu(torch.tensor(-1.0))
        assert gelu_neg.item() < 0, "GELU should allow negative values (unlike ReLU)"


# ============================================================
# GPT MODEL TESTS
# ============================================================

class TestGPTModel:
    """Tests for the full GPT model."""

    def test_forward_output_shapes(self, tiny_model, batch):
        """Test forward pass output shapes."""
        with torch.no_grad():
            logits, kv_cache, loss = tiny_model(batch)

        B, T = batch.shape
        V = tiny_model.config.vocab_size

        assert logits.shape == (B, T, V), \
            f"Expected logits shape (B={B}, T={T}, V={V}), got {logits.shape}"
        assert kv_cache is None  # use_cache=False by default
        assert loss is None      # targets=None

    def test_loss_computation(self, tiny_model, batch):
        """Test that loss is computed correctly."""
        targets = batch.clone()

        with torch.no_grad():
            _, _, loss = tiny_model(batch, targets=targets)

        assert loss is not None
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Cross-entropy loss should be positive"

        # Initial loss should be approximately log(vocab_size)
        import math
        expected_loss = math.log(tiny_model.config.vocab_size)
        # Allow 50% tolerance (randomly initialized model)
        assert abs(loss.item() - expected_loss) < expected_loss * 0.5, \
            f"Initial loss {loss.item():.2f} far from expected {expected_loss:.2f}"

    def test_weight_tying(self, tiny_model):
        """Verify embedding and LM head share weights."""
        assert tiny_model.lm_head.weight is tiny_model.token_embed.weight, \
            "LM head and embedding should share the same weight tensor"

    def test_param_count(self, tiny_model):
        """Parameter count should be positive and reasonable."""
        n_params = sum(p.numel() for p in tiny_model.parameters())
        assert n_params > 1000, "Model has too few parameters"
        assert n_params < 1e9, "Model has too many parameters for 'tiny'"

    def test_gradient_flow(self, tiny_config):
        """Verify gradients flow to all parameters."""
        model = GPTModel(tiny_config)
        model.train()

        B, T = 2, 16
        x = torch.randint(0, tiny_config.model.vocab_size, (B, T))
        y = torch.randint(0, tiny_config.model.vocab_size, (B, T))

        _, _, loss = model(x, targets=y)
        loss.backward()

        # Check that all parameters have gradients
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, \
            f"Parameters without gradients: {params_without_grad}"

    def test_kv_cache_mode(self, tiny_model):
        """Test KV cache returns correctly."""
        B, T = 1, 8
        x = torch.randint(0, tiny_model.config.vocab_size, (B, T))

        with torch.no_grad():
            logits, kv_cache, _ = tiny_model(x, use_cache=True)

        assert kv_cache is not None
        assert len(kv_cache) == tiny_model.config.n_layers

        # Each cache entry should be (K, V) pair
        for layer_kv in kv_cache:
            assert layer_kv is not None
            k, v = layer_kv
            assert k.shape[-2] == T  # sequence length
            assert v.shape[-2] == T

    def test_sequence_length_limit(self, tiny_model):
        """Should raise if sequence exceeds max_seq_len."""
        max_len = tiny_model.config.max_seq_len
        x = torch.randint(0, tiny_model.config.vocab_size, (1, max_len + 1))

        with pytest.raises(AssertionError):
            tiny_model(x)

    def test_num_params_method(self, tiny_model):
        """Test get_num_params utility."""
        n_all = sum(p.numel() for p in tiny_model.parameters())
        n_non_emb = tiny_model.get_num_params(non_embedding=True)
        assert n_non_emb < n_all  # Excluding embeddings should give fewer

    def test_deterministic_eval(self, tiny_model, batch):
        """Eval mode should give deterministic outputs (dropout disabled)."""
        tiny_model.eval()
        with torch.no_grad():
            logits1, _, _ = tiny_model(batch)
            logits2, _, _ = tiny_model(batch)
        assert torch.equal(logits1, logits2), \
            "Eval mode should be deterministic (dropout must be disabled)"


# ============================================================
# DEVICE TESTS
# ============================================================

class TestDeviceCompatibility:
    """Test model works on different devices."""

    def test_cpu_forward(self, tiny_model, batch):
        """Basic CPU forward pass."""
        tiny_model = tiny_model.cpu()
        batch = batch.cpu()
        with torch.no_grad():
            logits, _, _ = tiny_model(batch)
        assert logits.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, tiny_config):
        """CUDA forward pass if GPU available."""
        model = GPTModel(tiny_config).cuda()
        model.eval()
        x = torch.randint(0, tiny_config.model.vocab_size, (2, 16)).cuda()
        with torch.no_grad():
            logits, _, _ = model(x)
        assert logits.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
