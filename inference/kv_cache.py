"""
inference/kv_cache.py
=====================
Key-Value Cache implementation for efficient autoregressive inference.

WHY KV CACHE?
=============
During autoregressive generation, at each step:
    - We generate ONE new token
    - But the model processes ALL tokens from the beginning

Without KV cache:
    Step 1: Process [t0]                    → O(1)
    Step 2: Process [t0, t1]                → O(2)
    Step k: Process [t0, ..., tk]           → O(k)
    Total: O(n^2) complexity

With KV cache:
    Step 1: Compute K1,V1 for t0, cache them
    Step 2: Only compute K,V for t1, attend to [K1,V1,K2,V2]  → O(1)
    Step k: Only compute K,V for new token, attend to cached   → O(1) per step
    Total: O(n) complexity

The KV cache stores the Key and Value tensors from each attention layer
so they don't need to be recomputed for past tokens.

MEMORY COST
===========
KV cache memory per token:
    = n_layers * 2 * n_heads * head_dim * bytes_per_element
    = n_layers * 2 * d_model * bytes_per_element

For medium model (d_model=512, n_layers=12, fp32):
    = 12 * 2 * 512 * 4 = 49,152 bytes ≈ 48KB per token

For a 1024-token context:
    = 48KB * 1024 = 49MB

IMPLEMENTATION
==============
The KV cache is implemented directly in the attention module.
This module provides:
    - KVCache class for managing cache state
    - Helper for initializing empty cache
    - Cache statistics utilities
"""

import torch
from typing import List, Optional, Tuple


class KVCache:
    """
    Key-Value cache manager for efficient autoregressive inference.

    Stores past key-value tensors from each transformer layer.
    Updated incrementally as new tokens are generated.

    Attributes:
        n_layers   : Number of transformer layers
        cache      : List of (K, V) tuples, one per layer
        current_len: Number of tokens currently cached

    Usage:
        # Initialize empty cache
        kv_cache = KVCache(n_layers=12)

        # First forward pass (processes full prompt)
        logits, cache_state, _ = model(prompt_ids, use_cache=True)
        kv_cache.update(cache_state)

        # Subsequent passes (one token at a time)
        for step in range(max_new_tokens):
            logits, cache_state, _ = model(
                new_token,
                past_kv=kv_cache.get(),
                use_cache=True
            )
            kv_cache.update(cache_state)
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.current_len: int = 0

    def update(
        self,
        new_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        Update cache with new key-value tensors.

        The model's forward pass returns (K, V) for the current step.
        For first step: K,V contain past + current tokens.
        For subsequent steps: K,V are appended to cached K,V.

        Args:
            new_kv: List of (K, V) tuples from model forward pass
                    Each K,V has shape (B, H, T_total, Dh)
        """
        self.cache = new_kv
        if new_kv and new_kv[0] is not None:
            self.current_len = new_kv[0][0].shape[2]  # T dimension of K

    def get(self) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get the current KV cache for passing to the model.

        Returns:
            List of (K, V) tuples or None if cache is empty
        """
        return self.cache

    def reset(self):
        """Clear the cache (use when starting a new generation)."""
        self.cache = None
        self.current_len = 0

    def memory_bytes(self) -> int:
        """Estimate cache memory usage in bytes."""
        if self.cache is None:
            return 0
        total = 0
        for k, v in self.cache:
            if k is not None:
                total += k.nelement() * k.element_size()
                total += v.nelement() * v.element_size()
        return total

    def memory_mb(self) -> float:
        """Cache memory in megabytes."""
        return self.memory_bytes() / (1024 ** 2)

    def __repr__(self) -> str:
        return (
            f"KVCache(layers={self.n_layers}, "
            f"cached_tokens={self.current_len}, "
            f"memory={self.memory_mb():.1f}MB)"
        )


def estimate_kv_cache_size(
    n_layers: int,
    d_model: int,
    n_heads: int,
    max_seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 4,  # fp32=4, fp16=2, bf16=2
) -> dict:
    """
    Estimate KV cache memory requirements.

    Formula:
        memory = n_layers * 2 * batch_size * n_heads * max_seq_len * head_dim * bytes
              = n_layers * 2 * batch_size * max_seq_len * d_model * bytes

    Args:
        n_layers  : Number of transformer layers
        d_model   : Model hidden dimension
        n_heads   : Number of attention heads
        max_seq_len: Maximum sequence length
        batch_size : Batch size for generation
        dtype_bytes: Bytes per parameter (4=fp32, 2=fp16/bf16)

    Returns:
        dict with memory stats per token and total
    """
    head_dim = d_model // n_heads

    # Per token: n_layers * 2 (K+V) * n_heads * head_dim * bytes
    bytes_per_token = n_layers * 2 * n_heads * head_dim * dtype_bytes

    # Total for max sequence
    total_bytes = bytes_per_token * max_seq_len * batch_size

    return {
        "bytes_per_token": bytes_per_token,
        "kb_per_token": bytes_per_token / 1024,
        "total_mb": total_bytes / (1024 ** 2),
        "total_gb": total_bytes / (1024 ** 3),
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
    }
