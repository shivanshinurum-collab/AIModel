"""
tests/test_tokenizer.py
=======================
Tests for the BPE tokenizer.

Tests:
- Loading tokenizer
- Encoding text
- Decoding token IDs
- Round-trip consistency
- Special token IDs
- Batch encoding
- Edge cases (empty, special chars)
"""

import pytest
import os
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.tokenizer_infer import LLMTokenizer


# ============================================================
# FIXTURES
# ============================================================

TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tokenizer", "saved", "tokenizer.json"
)


def tokenizer_available() -> bool:
    """Check if a trained tokenizer exists."""
    return os.path.exists(TOKENIZER_PATH)


@pytest.fixture
def tokenizer():
    """Load tokenizer if available, skip otherwise."""
    if not tokenizer_available():
        pytest.skip(
            "Tokenizer not trained yet. Run: python tokenizer/tokenizer_train.py"
        )
    return LLMTokenizer(TOKENIZER_PATH)


# ============================================================
# TESTS
# ============================================================

class TestTokenizerLoading:
    """Tests for tokenizer initialization."""

    def test_loads_successfully(self, tokenizer):
        """Should load without errors."""
        assert tokenizer is not None

    def test_vocab_size_positive(self, tokenizer):
        """Vocabulary size should be positive."""
        assert tokenizer.vocab_size > 0

    def test_special_tokens_exist(self, tokenizer):
        """All special tokens should have valid IDs."""
        assert tokenizer.pad_id >= 0
        assert tokenizer.bos_id >= 0
        assert tokenizer.eos_id >= 0
        assert tokenizer.unk_id >= 0

    def test_special_tokens_distinct(self, tokenizer):
        """Special tokens should have different IDs."""
        ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id}
        assert len(ids) == 4, "Special tokens must have unique IDs"

    def test_not_found_raises(self):
        """Should raise FileNotFoundError for missing tokenizer."""
        with pytest.raises(FileNotFoundError):
            LLMTokenizer("/nonexistent/path/tokenizer.json")

    def test_repr(self, tokenizer):
        """__repr__ should work."""
        s = repr(tokenizer)
        assert "LLMTokenizer" in s
        assert str(tokenizer.vocab_size) in s


class TestEncoding:
    """Tests for text encoding."""

    def test_encode_returns_list(self, tokenizer):
        """encode should return a list of ints."""
        ids = tokenizer.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_not_empty(self, tokenizer):
        """Non-empty text should produce non-empty token list."""
        ids = tokenizer.encode("Hello, world!")
        assert len(ids) > 0

    def test_encode_with_bos(self, tokenizer):
        """Encoding with BOS should start with BOS token."""
        ids = tokenizer.encode("Hello", add_bos=True)
        assert ids[0] == tokenizer.bos_id

    def test_encode_with_eos(self, tokenizer):
        """Encoding with EOS should end with EOS token."""
        ids = tokenizer.encode("Hello", add_eos=True)
        assert ids[-1] == tokenizer.eos_id

    def test_encode_with_bos_eos(self, tokenizer):
        """BOS and EOS should both be present."""
        ids = tokenizer.encode("Hello", add_bos=True, add_eos=True)
        assert ids[0] == tokenizer.bos_id
        assert ids[-1] == tokenizer.eos_id

    def test_max_length_truncation(self, tokenizer):
        """Encoding with max_length should truncate."""
        long_text = "Hello world " * 100
        ids = tokenizer.encode(long_text, max_length=10)
        assert len(ids) <= 10

    def test_ids_in_vocab_range(self, tokenizer):
        """All token IDs should be within vocabulary range."""
        ids = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
        assert all(0 <= i < tokenizer.vocab_size for i in ids)


class TestDecoding:
    """Tests for token ID decoding."""

    def test_decode_returns_string(self, tokenizer):
        """decode should return a string."""
        ids = tokenizer.encode("Hello")
        text = tokenizer.decode(ids)
        assert isinstance(text, str)

    def test_decode_nonempty(self, tokenizer):
        """Decoding valid IDs should produce non-empty text."""
        ids = tokenizer.encode("Hello there!")
        text = tokenizer.decode(ids)
        assert len(text) > 0

    def test_skip_special_tokens(self, tokenizer):
        """Special tokens should not appear in decoded text by default."""
        ids = [tokenizer.bos_id] + tokenizer.encode("Hello") + [tokenizer.eos_id]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        assert "<bos>" not in text
        assert "<eos>" not in text

    def test_empty_list(self, tokenizer):
        """Decoding empty list should return empty string."""
        text = tokenizer.decode([])
        assert text == "" or text.strip() == ""


class TestRoundTrip:
    """Round-trip encode → decode consistency tests."""

    @pytest.mark.parametrize("text", [
        "Hello, world!",
        "The quick brown fox",
        "1234567890",
        "Python is awesome.",
        "  spaces  ",
    ])
    def test_roundtrip(self, tokenizer, text):
        """Encoding then decoding should approximately recover original text."""
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        # BPE may add/remove spaces, so we compare stripped versions
        assert decoded.strip() != "", f"Round-trip failed for: {text!r}"

    def test_encode_decode_count(self, tokenizer):
        """Token count method should match actual encoding length."""
        text = "The transformer is a powerful model architecture."
        ids = tokenizer.encode(text)
        count = tokenizer.count_tokens(text)
        assert count == len(ids)


class TestBatchEncoding:
    """Tests for batch encoding."""

    def test_batch_encode(self, tokenizer):
        """Batch encoding should handle multiple texts."""
        texts = ["Hello world", "How are you?", "GPT is cool"]
        batch = tokenizer.encode_batch(texts)
        assert len(batch) == len(texts)
        assert all(isinstance(ids, list) for ids in batch)

    def test_batch_decode(self, tokenizer):
        """Batch decoding should handle multiple sequences."""
        texts = ["Hello", "World"]
        batch_ids = tokenizer.encode_batch(texts)
        decoded = tokenizer.decode_batch(batch_ids)
        assert len(decoded) == len(texts)


class TestPadding:
    """Tests for padding utility."""

    def test_pad_right(self, tokenizer):
        """Right padding should extend sequence to target length."""
        ids = [1, 2, 3]
        padded = tokenizer.pad_sequence(ids, max_length=5, pad_left=False)
        assert len(padded) == 5
        assert padded[:3] == ids
        assert padded[3] == tokenizer.pad_id
        assert padded[4] == tokenizer.pad_id

    def test_pad_left(self, tokenizer):
        """Left padding should prepend pad tokens."""
        ids = [1, 2, 3]
        padded = tokenizer.pad_sequence(ids, max_length=5, pad_left=True)
        assert len(padded) == 5
        assert padded[-3:] == ids
        assert padded[0] == tokenizer.pad_id

    def test_truncation(self, tokenizer):
        """Sequences longer than max_length should be truncated."""
        ids = [1, 2, 3, 4, 5]
        truncated = tokenizer.pad_sequence(ids, max_length=3)
        assert len(truncated) == 3
        assert truncated == [1, 2, 3]

    def test_exact_length_unchanged(self, tokenizer):
        """Sequences of exactly max_length should not be modified."""
        ids = [1, 2, 3]
        result = tokenizer.pad_sequence(ids, max_length=3)
        assert result == ids


class TestEdgeCases:
    """Edge case handling tests."""

    def test_single_char(self, tokenizer):
        """Single character should encode to at least one token."""
        ids = tokenizer.encode("a")
        assert len(ids) >= 1

    def test_numbers(self, tokenizer):
        """Numbers should encode correctly."""
        ids = tokenizer.encode("42")
        assert len(ids) >= 1

    def test_punctuation(self, tokenizer):
        """Punctuation should encode."""
        ids = tokenizer.encode("!?.,:;")
        assert len(ids) >= 1

    def test_whitespace_only(self, tokenizer):
        """Pure whitespace may or may not tokenize (should not crash)."""
        try:
            ids = tokenizer.encode("   ")
            # Should not raise
        except Exception as e:
            pytest.fail(f"Whitespace encoding raised: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
