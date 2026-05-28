"""
model/transformer_block.py
==========================
A single GPT-style transformer block (decoder layer).

ARCHITECTURE
============

Each transformer block has two sub-layers with residual connections:

    x → LayerNorm → MHA → + → LayerNorm → FFN → + → output
         (pre-norm)   ↑             (pre-norm)  ↑
                      x                         x

This is "Pre-LayerNorm" (also called Pre-LN), where normalization
is applied BEFORE the attention/FFN sublayer (unlike the original
"Post-LN" architecture in "Attention is All You Need").

Pre-LN advantages:
- More stable training (gradients don't vanish as easily)
- Used in GPT-2, GPT-3, PaLM, and most modern LLMs

RESIDUAL CONNECTIONS
====================
Residual connections (skip connections) add the input directly to
the output: y = sublayer(x) + x

This enables:
- Gradient flow: gradients can skip layers entirely
- Identity shortcut: layers learn residual functions
- Deeper networks without degradation

FEED-FORWARD NETWORK (FFN)
===========================
The FFN consists of two linear layers with GELU activation:

    FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2

    Input:  (B, T, D)
    W_1:    (D, D_ff)  — expand by 4x
    W_2:    (D_ff, D)  — project back
    Output: (B, T, D)

GELU Activation:
    GELU(x) = x * Φ(x)  where Φ is the CDF of standard normal

    GELU is smoother than ReLU and helps the model learn
    more complex representations. Used in BERT, GPT-2, GPT-3.

LAYER NORMALIZATION
===================
LayerNorm normalizes the activations across the feature dimension:

    LayerNorm(x) = γ * (x - μ) / (σ + ε) + β

    where μ = mean(x), σ = std(x), γ/β are learned params, ε = 1e-5

This stabilizes training by keeping activations in a reasonable range.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from model.attention import MultiHeadCausalAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    Applied independently to each position in the sequence.
    Uses two linear layers with GELU activation and dropout.

    Architecture:
        Input (D) → Linear(D, D_ff) → GELU → Dropout → Linear(D_ff, D) → Output

    Args:
        d_model: Hidden dimension (input and output size)
        d_ff   : Inner dimension (typically 4 * d_model)
        dropout: Dropout probability
        bias   : Use bias in linear layers

    Shape:
        Input:  (B, T, D)
        Hidden: (B, T, D_ff)
        Output: (B, T, D)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Expand to d_ff dimensions
            nn.Linear(d_model, d_ff, bias=bias),
            # GELU activation — smoother alternative to ReLU
            nn.GELU(),
            # Dropout for regularization
            nn.Dropout(dropout),
            # Project back to d_model
            nn.Linear(d_ff, d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FFN to input.

        Args:
            x: Input tensor (B, T, D)

        Returns:
            Output tensor (B, T, D)
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single GPT-style transformer decoder block.

    Uses Pre-LayerNorm for training stability.

    Architecture:
        x → LN → MHA → + x → LN → FFN → + x → output

    Components:
        - LayerNorm before attention (Pre-LN)
        - Multi-Head Causal Self-Attention (with KV cache support)
        - Residual connection after attention
        - LayerNorm before FFN (Pre-LN)
        - Feed-Forward Network
        - Residual connection after FFN

    Args:
        d_model    : Hidden dimension
        n_heads    : Number of attention heads
        d_ff       : FFN inner dimension
        dropout    : Dropout probability
        bias       : Use bias in linear layers
        max_seq_len: Maximum sequence length (for causal mask)

    Shapes:
        Input:  (B, T, D)
        Output: (B, T, D)  — same shape as input
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        # ---- Pre-Attention Layer Norm ----
        # Normalizes input before attention
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)

        # ---- Multi-Head Causal Self-Attention ----
        self.attention = MultiHeadCausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            max_seq_len=max_seq_len,
        )

        # ---- Pre-FFN Layer Norm ----
        # Normalizes input before feed-forward
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)

        # ---- Feed-Forward Network ----
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through one transformer block.

        Pre-LN Residual Stream:
            1. attn_out, kv = MHA(LN(x), past_kv)   ← Attention sub-layer
            2. x = x + attn_out                        ← Residual connection
            3. x = x + FFN(LN(x))                     ← FFN sub-layer + residual

        Args:
            x       : Input tensor (B, T, D)
            past_kv : Cached key-value tensors for inference (or None)
            use_cache: Return present KV for caching

        Returns:
            Tuple of:
                output   : (B, T, D) transformed tensor
                present_kv: (K, V) cache tuple if use_cache, else None
        """
        # ---- Sub-layer 1: Multi-Head Attention with pre-norm ----
        # Apply LayerNorm to x before attention (Pre-LN)
        attn_out, present_kv = self.attention(
            self.ln1(x),     # normalized input
            past_kv=past_kv,
            use_cache=use_cache,
        )
        # Add residual connection: x + attention_output
        x = x + attn_out

        # ---- Sub-layer 2: FFN with pre-norm ----
        # Apply LayerNorm to x before FFN
        x = x + self.ffn(self.ln2(x))

        return x, present_kv

    def extra_repr(self) -> str:
        return (
            f"d_model={self.attention.d_model}, "
            f"n_heads={self.attention.n_heads}, "
            f"d_ff={self.ffn.net[0].out_features}"
        )
