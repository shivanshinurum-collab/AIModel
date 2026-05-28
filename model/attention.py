"""
model/attention.py
==================
Multi-Head Causal Self-Attention for the GPT-style transformer.

MATH OVERVIEW
=============

Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
    Q = Query matrix  [batch, heads, seq_len, head_dim]
    K = Key matrix    [batch, heads, seq_len, head_dim]
    V = Value matrix  [batch, heads, seq_len, head_dim]
    d_k = head_dim = d_model / n_heads

The sqrt(d_k) scaling prevents the dot products from growing
too large (which would cause softmax to saturate and produce
near-zero gradients).

Multi-Head Attention:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

    head_i = Attention(Q*W_Qi, K*W_Ki, V*W_Vi)

Where W_Q, W_K, W_V are learned projections and W_O is the output
projection. Using multiple heads lets the model attend to different
aspects of the input simultaneously.

Causal Masking:
    For autoregressive generation, we apply a causal mask so that
    position i can only attend to positions j <= i.

    mask[i][j] = -inf if j > i else 0

    This ensures the model can only see "past" tokens, making it
    suitable for language modeling.

KV Cache (for inference):
    During generation, at each new token we only need to compute
    Q for the new token, but K and V for all previous tokens are
    the same. We cache past K, V tensors to avoid recomputation.

    Without KV cache: O(n^2) attention per token during generation
    With KV cache:    O(n) per new token
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention module.

    Implements the attention mechanism used in GPT-style transformers:
    - Projects input to Q, K, V
    - Applies scaled dot-product attention with causal masking
    - Supports KV cache for efficient inference
    - Applies output projection

    Args:
        d_model  : Model hidden dimension (e.g., 256, 512, 1024)
        n_heads  : Number of attention heads (must divide d_model evenly)
        dropout  : Attention dropout probability
        bias     : Whether to use bias in linear projections
        max_seq_len: Maximum sequence length (for causal mask precomputation)

    Shapes (with B=batch, T=seq_len, D=d_model, H=n_heads, Dh=d_model/H):
        Input:  (B, T, D)
        Q,K,V:  (B, H, T, Dh)
        Attn:   (B, H, T, T)
        Output: (B, T, D)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # Dh = D / H
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(Dh) for scaling

        # ---- Linear projections ----
        # Combined Q, K, V projection (more efficient than 3 separate layers)
        # Input: (B, T, D) → Output: (B, T, 3*D) → split into Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # Output projection: combines all heads back into d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Attention dropout (applied to attention weights before softmax)
        self.attn_dropout = nn.Dropout(dropout)
        # Residual dropout (applied to output)
        self.resid_dropout = nn.Dropout(dropout)

        # ---- Causal mask ----
        # Register as buffer so it moves to GPU automatically with .to(device)
        # Upper triangle (excluding diagonal) = -inf → can't attend to future
        # Shape: (1, 1, max_seq_len, max_seq_len)
        mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head causal attention.

        Args:
            x         : Input tensor of shape (B, T, D)
            past_kv   : Cached (key, value) tensors for inference
                        Each has shape (B, H, T_past, Dh)
            use_cache : If True, return current K,V for caching

        Returns:
            Tuple of:
                output  : Attended output (B, T, D)
                kv_cache: Tuple of (K, V) tensors if use_cache, else None

        Computation steps:
            1. Project x → Q, K, V
            2. Reshape to multi-head format
            3. Concatenate with cached K, V (if inference)
            4. Apply causal mask
            5. Compute scaled dot-product attention
            6. Concatenate heads and project output
        """
        B, T, D = x.shape  # batch, sequence length, d_model

        # ---- Step 1: Project to Q, K, V ----
        # (B, T, D) → (B, T, 3D) → split into 3 × (B, T, D)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        # Each: (B, T, D)

        # ---- Step 2: Reshape to multi-head format ----
        # (B, T, D) → (B, T, H, Dh) → (B, H, T, Dh)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Each: (B, H, T, Dh)

        # ---- Step 3: KV Cache (for inference) ----
        if past_kv is not None:
            past_k, past_v = past_kv
            # Concatenate past and current along sequence dimension
            k = torch.cat([past_k, k], dim=2)  # (B, H, T_past+T, Dh)
            v = torch.cat([past_v, v], dim=2)  # (B, H, T_past+T, Dh)

        # Store current K, V for caching
        present_kv = (k, v) if use_cache else None

        T_kv = k.shape[2]  # Total key/value sequence length

        # ---- Step 4: Scaled dot-product attention ----
        # Attention scores: Q @ K^T / sqrt(Dh)
        # (B, H, T, Dh) @ (B, H, Dh, T_kv) → (B, H, T, T_kv)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # ---- Step 5: Causal mask ----
        # Only mask during training (when T == T_kv, i.e., no past_kv)
        if past_kv is None:
            # Apply causal mask: set future positions to -inf
            # causal_mask shape: (1, 1, max_seq_len, max_seq_len)
            mask = self.causal_mask[:, :, :T, :T_kv]  # (1, 1, T, T_kv)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # ---- Step 6: Softmax + dropout ----
        # Convert scores to probabilities
        # Shape: (B, H, T, T_kv)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # ---- Step 7: Attend to values ----
        # (B, H, T, T_kv) @ (B, H, T_kv, Dh) → (B, H, T, Dh)
        out = torch.matmul(attn_weights, v)

        # ---- Step 8: Merge heads and project ----
        # (B, H, T, Dh) → (B, T, H, Dh) → (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection: (B, T, D) → (B, T, D)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out, present_kv

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}"
        )
