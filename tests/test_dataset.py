"""
tests/test_dataset.py
=====================
Tests for the dataset pipeline.

Tests:
- TextDataset creation
- DataLoader batching
- Sequence chunking
- Token caching
- Data cleaning functions
"""

import os
import sys
import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# MOCK TOKENIZER
# ============================================================

class MockTokenizer:
    """Minimal mock tokenizer for dataset tests (no file needed)."""
    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    vocab_size = 100

    def encode_for_training(self, text, max_length=None):
        """Return character codes as mock token IDs."""
        ids = [1] + [ord(c) % 90 + 10 for c in text[:50]] + [2]
        if max_length:
            ids = ids[:max_length]
        return ids

    def count_tokens(self, text):
        return len(text.split())


# ============================================================
# TEXT DATASET TESTS
# ============================================================

class TestTextDataset:
    """Tests for the TextDataset class."""

    def _make_text_file(self, tmpdir, n_docs=20, words_per_doc=50):
        """Create a temporary text file with fake documents."""
        path = os.path.join(tmpdir, "train.txt")
        with open(path, "w") as f:
            for i in range(n_docs):
                doc = " ".join([f"word{j}" for j in range(words_per_doc)])
                f.write(doc + "\n\n")
        return path

    def test_dataset_creation(self, tmp_path):
        """TextDataset should create without error."""
        from training.dataloader import TextDataset

        text_file = self._make_text_file(str(tmp_path))
        tok = MockTokenizer()

        dataset = TextDataset(
            text_path=text_file,
            tokenizer=tok,
            max_seq_len=32,
            split="train",
            cache_dir=str(tmp_path),
        )
        assert len(dataset) > 0

    def test_getitem_shapes(self, tmp_path):
        """Each item should be (input, target) of length max_seq_len."""
        from training.dataloader import TextDataset

        seq_len = 32
        text_file = self._make_text_file(str(tmp_path))
        tok = MockTokenizer()

        dataset = TextDataset(
            text_path=text_file,
            tokenizer=tok,
            max_seq_len=seq_len,
            cache_dir=str(tmp_path),
        )

        x, y = dataset[0]
        assert x.shape == (seq_len,), f"Expected ({seq_len},), got {x.shape}"
        assert y.shape == (seq_len,), f"Expected ({seq_len},), got {y.shape}"

    def test_next_token_prediction(self, tmp_path):
        """Target should be input shifted by 1."""
        from training.dataloader import TextDataset

        text_file = self._make_text_file(str(tmp_path), n_docs=50)
        tok = MockTokenizer()

        dataset = TextDataset(
            text_path=text_file,
            tokenizer=tok,
            max_seq_len=16,
            cache_dir=str(tmp_path),
        )

        x, y = dataset[0]
        # The full chunk is x[0], x[1], ..., x[T-1], y[T-1]
        # So y should be x shifted: y[i] should follow x[i]

        # We can verify using the raw token array
        start = 0
        chunk = dataset.token_ids[start : start + 17].astype(np.int64)
        assert np.array_equal(x.numpy(), chunk[:16])
        assert np.array_equal(y.numpy(), chunk[1:17])

    def test_cache_created(self, tmp_path):
        """Tokenized cache file should be created."""
        from training.dataloader import TextDataset

        text_file = self._make_text_file(str(tmp_path))
        tok = MockTokenizer()

        TextDataset(
            text_path=text_file,
            tokenizer=tok,
            max_seq_len=16,
            cache_dir=str(tmp_path),
        )

        cache_files = list(tmp_path.glob("*_tokens.npy"))
        assert len(cache_files) == 1, "Cache .npy file should be created"

    def test_cache_loads_faster(self, tmp_path):
        """Second creation should use cached file (not re-tokenize)."""
        from training.dataloader import TextDataset

        text_file = self._make_text_file(str(tmp_path))

        # First creation: tokenizes and saves cache
        ds1 = TextDataset(text_path=text_file, tokenizer=MockTokenizer(),
                          max_seq_len=16, cache_dir=str(tmp_path))

        # Verify cache exists
        cache_files = list(tmp_path.glob("*_tokens.npy"))
        assert len(cache_files) == 1

        # Second creation should load from cache (same token array)
        ds2 = TextDataset(text_path=text_file, tokenizer=MockTokenizer(),
                          max_seq_len=16, cache_dir=str(tmp_path))

        # Both datasets should have identical token arrays
        import numpy as np
        assert np.array_equal(ds1.token_ids, ds2.token_ids), \
            "Cached and re-loaded datasets should have identical tokens"

    def test_missing_file_raises(self, tmp_path):
        """Missing text file should raise FileNotFoundError."""
        from training.dataloader import TextDataset

        with pytest.raises(FileNotFoundError):
            TextDataset(
                text_path=str(tmp_path / "nonexistent.txt"),
                tokenizer=MockTokenizer(),
                max_seq_len=16,
                cache_dir=str(tmp_path),
            )


# ============================================================
# DATALOADER TESTS
# ============================================================

class TestDataLoader:
    """Tests for the DataLoader factory."""

    def _make_text_file(self, tmpdir, n_docs=50):
        path = os.path.join(tmpdir, "train.txt")
        with open(path, "w") as f:
            for i in range(n_docs):
                doc = f"This is document {i}. " * 10
                f.write(doc + "\n\n")
        return path

    def test_batch_shapes(self, tmp_path):
        """DataLoader should produce correctly shaped batches."""
        from training.dataloader import create_dataloader

        batch_size = 4
        seq_len = 16

        text_file = self._make_text_file(str(tmp_path))
        tok = MockTokenizer()

        loader = create_dataloader(
            text_path=text_file,
            tokenizer=tok,
            max_seq_len=seq_len,
            batch_size=batch_size,
            split="train",
            num_workers=0,
            pin_memory=False,
            cache_dir=str(tmp_path),
        )

        x, y = next(iter(loader))
        assert x.shape == (batch_size, seq_len)
        assert y.shape == (batch_size, seq_len)

    def test_tensor_types(self, tmp_path):
        """Batch tensors should be LongTensor."""
        from training.dataloader import create_dataloader

        text_file = self._make_text_file(str(tmp_path))

        loader = create_dataloader(
            text_path=text_file,
            tokenizer=MockTokenizer(),
            max_seq_len=16,
            batch_size=2,
            split="train",
            num_workers=0,
            pin_memory=False,
            cache_dir=str(tmp_path),
        )

        x, y = next(iter(loader))
        assert x.dtype == torch.long
        assert y.dtype == torch.long


# ============================================================
# CLEANING TESTS
# ============================================================

class TestDataCleaning:
    """Tests for the data cleaning pipeline."""

    def test_fix_encoding_no_crash(self):
        """fix_encoding should handle all inputs without crashing."""
        from datasets.clean_dataset import fix_encoding

        texts = ["Hello", "café", "你好", "мир", ""]
        for text in texts:
            result = fix_encoding(text)
            assert isinstance(result, str)

    def test_normalize_whitespace(self):
        """Whitespace normalization should collapse multiple spaces."""
        from datasets.clean_dataset import normalize_whitespace

        text = "Hello   world\n\n\n\nGoodbye"
        result = normalize_whitespace(text)
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_remove_html(self):
        """HTML tags should be removed."""
        from datasets.clean_dataset import remove_html

        text = "<b>Hello</b> <i>world</i>"
        result = remove_html(text)
        assert "<b>" not in result
        assert "<i>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_is_valid_text_short(self):
        """Very short texts should be filtered out."""
        from datasets.clean_dataset import is_valid_text

        assert not is_valid_text("Hi")
        assert not is_valid_text("Short text.")

    def test_is_valid_text_long(self):
        """Long enough texts should pass."""
        from datasets.clean_dataset import is_valid_text

        text = "This is a longer text that should definitely pass the length filter. " * 3
        assert is_valid_text(text)

    def test_dedup_fingerprint_consistent(self):
        """Same text should always produce same fingerprint."""
        from datasets.clean_dataset import dedup_fingerprint

        text = "Hello, this is a test document for deduplication."
        fp1 = dedup_fingerprint(text)
        fp2 = dedup_fingerprint(text)
        assert fp1 == fp2

    def test_dedup_fingerprint_different(self):
        """Different texts should produce different fingerprints."""
        from datasets.clean_dataset import dedup_fingerprint

        fp1 = dedup_fingerprint("First document about cats.")
        fp2 = dedup_fingerprint("Second document about dogs.")
        assert fp1 != fp2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
