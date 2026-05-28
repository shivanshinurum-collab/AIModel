"""
training/dataloader.py
======================
Dataset and DataLoader for GPT-style language model training.

DATA FORMAT
===========
Training uses "next-token prediction" (causal language modeling):

Given a sequence of tokens: [t0, t1, t2, t3, t4]
    Input  (x): [t0, t1, t2, t3]     ← tokens to predict from
    Target (y): [t1, t2, t3, t4]     ← next tokens to predict

The model learns P(t_{i+1} | t_0, t_1, ..., t_i) for all positions.

SEQUENCE CHUNKING
=================
Long documents are chunked into fixed-length windows of size max_seq_len.
Documents shorter than max_seq_len are skipped (or padded — we skip by default).

Memory layout:
    token_ids = [t0, t1, t2, ..., t_{total_tokens-1}]  (flat array)

    chunk_0: token_ids[0 : max_seq_len+1]
    chunk_1: token_ids[max_seq_len : 2*max_seq_len+1]
    ...

    x_i = chunk_i[:-1]   → (max_seq_len,)
    y_i = chunk_i[1:]    → (max_seq_len,)

TOKEN CACHING
=============
Tokenizing large datasets is slow. We cache tokenized data as a
numpy memory-mapped file (.npy) so it only needs to be done once.

On first run: reads text, tokenizes, saves tokens.npy
On subsequent runs: loads tokens.npy directly (much faster)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger
from tokenizer.tokenizer_infer import LLMTokenizer

log = get_logger("dataloader")


# ============================================================
# TOKENIZED DATASET
# ============================================================

class TextDataset(Dataset):
    """
    Dataset for causal language modeling.

    Tokenizes a text file once and caches it as a numpy array.
    Returns (input_ids, target_ids) pairs of length max_seq_len.

    Args:
        text_path     : Path to the text file (with double-newline doc separators)
        tokenizer     : Trained LLMTokenizer instance
        max_seq_len   : Length of each training sequence
        split         : 'train' or 'val' (for logging)
        cache_dir     : Directory to save tokenized cache files
        rebuild_cache : Force re-tokenization even if cache exists

    Shapes:
        __getitem__ returns:
            input_ids: LongTensor (max_seq_len,)
            target_ids: LongTensor (max_seq_len,)

    Example:
        dataset = TextDataset("datasets/processed/train.txt", tokenizer, 256)
        x, y = dataset[0]  # x = input, y = next-token targets
    """

    def __init__(
        self,
        text_path: str,
        tokenizer: LLMTokenizer,
        max_seq_len: int,
        split: str = "train",
        cache_dir: Optional[str] = None,
        rebuild_cache: bool = False,
    ):
        self.text_path = text_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split

        # Determine cache path
        if cache_dir is None:
            cache_dir = os.path.dirname(text_path)
        basename = os.path.splitext(os.path.basename(text_path))[0]
        self.cache_path = os.path.join(cache_dir, f"{basename}_tokens.npy")

        # Load or build tokenized data
        self.token_ids = self._load_or_tokenize(rebuild_cache)

        # Number of complete chunks we can form
        # Each chunk needs max_seq_len + 1 tokens (for input + target)
        self.n_chunks = (len(self.token_ids) - 1) // max_seq_len

        log.info(
            f"Dataset ({split}): {len(self.token_ids):,} tokens | "
            f"{self.n_chunks:,} chunks | "
            f"seq_len={max_seq_len}"
        )

    def _load_or_tokenize(self, rebuild: bool = False) -> np.ndarray:
        """
        Load tokenized data from cache or tokenize from scratch.

        Returns numpy array of dtype uint16 (saves 2x memory vs int32).
        uint16 supports up to 65535 — sufficient for vocab sizes up to 64K.
        """
        if not rebuild and os.path.exists(self.cache_path):
            log.info(f"Loading tokenized cache: {self.cache_path}")
            return np.load(self.cache_path, allow_pickle=False)

        log.info(f"Tokenizing {self.text_path} ...")
        if not os.path.exists(self.text_path):
            raise FileNotFoundError(
                f"Data file not found: {self.text_path}\n"
                f"Run datasets/download_dataset.py and datasets/merge_dataset.py first"
            )

        # Tokenize all documents
        all_tokens = []

        # Read file and split into documents (double-newline separated)
        with open(self.text_path, "r", encoding="utf-8") as f:
            content = f.read()

        documents = [d.strip() for d in content.split("\n\n") if d.strip()]
        log.info(f"Found {len(documents):,} documents to tokenize")

        from tqdm import tqdm
        for doc in tqdm(documents, desc=f"Tokenizing ({self.split})"):
            # Encode with BOS and EOS tokens
            ids = self.tokenizer.encode_for_training(doc)
            all_tokens.extend(ids)

        # Convert to numpy uint16 for memory efficiency
        # uint16 range: 0-65535 — supports most vocab sizes
        token_array = np.array(all_tokens, dtype=np.uint16)

        # Save cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.save(self.cache_path, token_array)
        log.info(f"Saved token cache: {self.cache_path} ({len(token_array):,} tokens)")

        return token_array

    def __len__(self) -> int:
        """Number of training chunks."""
        return self.n_chunks

    def __getitem__(self, idx: int):
        """
        Get the idx-th chunk as (input_ids, target_ids).

        Each chunk is max_seq_len tokens:
            x[i] = token_ids[i]
            y[i] = token_ids[i+1]  (next token prediction)

        Returns:
            x: input_ids  LongTensor (max_seq_len,)
            y: target_ids LongTensor (max_seq_len,)
        """
        # Start position of this chunk
        start = idx * self.max_seq_len

        # Get max_seq_len + 1 tokens
        chunk = self.token_ids[start : start + self.max_seq_len + 1]

        # Handle edge case: last chunk might be short
        if len(chunk) < self.max_seq_len + 1:
            # Pad with zeros (pad token)
            chunk = np.pad(
                chunk,
                (0, self.max_seq_len + 1 - len(chunk)),
                constant_values=self.tokenizer.pad_id,
            )

        # Convert uint16 → int64 for PyTorch
        chunk = chunk.astype(np.int64)

        # Input: first max_seq_len tokens
        x = torch.from_numpy(chunk[:-1])  # (max_seq_len,)
        # Target: last max_seq_len tokens (shifted by 1)
        y = torch.from_numpy(chunk[1:])   # (max_seq_len,)

        return x, y


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_dataloader(
    text_path: str,
    tokenizer: LLMTokenizer,
    max_seq_len: int,
    batch_size: int,
    split: str = "train",
    num_workers: int = 2,
    pin_memory: bool = True,
    cache_dir: Optional[str] = None,
    rebuild_cache: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for language model training.

    Args:
        text_path    : Path to text file
        tokenizer    : Trained LLMTokenizer
        max_seq_len  : Sequence length for training
        batch_size   : Batch size (sequences per batch)
        split        : 'train' or 'val' (affects shuffling)
        num_workers  : DataLoader worker processes
        pin_memory   : Pin memory for GPU transfer (recommended for CUDA)
        cache_dir    : Directory for tokenized cache
        rebuild_cache: Force re-tokenization

    Returns:
        PyTorch DataLoader that yields (input_ids, target_ids) batches

    Batch shapes:
        input_ids:  (batch_size, max_seq_len)
        target_ids: (batch_size, max_seq_len)
    """
    dataset = TextDataset(
        text_path=text_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split=split,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),   # Shuffle training, not validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,               # Drop incomplete final batch
        persistent_workers=(num_workers > 0),
    )

    return loader


# ============================================================
# INFINITE DATA ITERATOR
# ============================================================

class InfiniteDataLoader:
    """
    Wraps a DataLoader to cycle infinitely.

    Training loops often need more data than one epoch.
    This wraps any DataLoader to restart automatically at the end.

    Usage:
        loader = InfiniteDataLoader(dataloader)
        for step in range(max_steps):
            x, y = next(loader)
            # train...
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._iter = iter(dataloader)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            # Restart iterator at end of dataset
            self._iter = iter(self.dataloader)
            return next(self._iter)

    def __iter__(self):
        return self
