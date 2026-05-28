"""
model/gpt_model.py
==================
Complete GPT-style autoregressive language model.

FULL ARCHITECTURE
=================

Token Embedding Layer:
    Maps token IDs to dense vectors.
    Shape: (vocab_size, d_model)

Positional Embedding Layer:
    Learned position encodings — each position gets its own embedding.
    Shape: (max_seq_len, d_model)
    (Unlike Transformers with sinusoidal PE, GPT uses learned PE)

The full forward pass:
    token_ids (B, T)
        ↓
    token_embed (B, T, D)        ← token embedding lookup
        +
    pos_embed (T, D)             ← positional embedding (broadcast across batch)
        ↓
    x = dropout(token_embed + pos_embed)
        ↓
    TransformerBlock_1
        ↓
    TransformerBlock_2
        ...
        ↓
    TransformerBlock_N
        ↓
    LayerNorm (final)
        ↓
    LM Head: Linear(D, vocab_size)   ← NO softmax here; use cross_entropy
        ↓
    logits (B, T, vocab_size)

WEIGHT TYING
============
The LM head (output projection to vocab) shares weights with the
token embedding table. This is called "weight tying" and:
- Reduces parameter count by vocab_size * d_model
- Often improves performance (embeddings and unembeddings are symmetric)
- Used in GPT-2, BERT, T5, and most modern LLMs

PARAMETER INITIALIZATION
=========================
Following the GPT-2 paper:
- Linear layers: normal(0, 0.02)
- Embeddings: normal(0, 0.02)
- Residual projections: normal(0, 0.02 / sqrt(2 * n_layers))
  (scaled down by sqrt(2N) because residuals accumulate gradients)
- LayerNorm: weight=1, bias=0

MEMORY ANALYSIS
===============
Parameters per layer (d=512, heads=8, dff=2048):
- QKV projection: 3 × d × d = 3 × 512 × 512 = 786K
- Output projection: d × d = 262K
- FFN: d × dff + dff × d = 512×2048 + 2048×512 = 2.1M
- LayerNorms: 4 × d = 2K
Total per layer: ~3.15M params

For 12 layers (medium model):
12 × 3.15M + embeddings (512 × 32000 ≈ 16M) + pos_embed (512K) ≈ 54M
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union

from utils.config_loader import ModelConfig
from model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """
    GPT-style autoregressive language model.

    A decoder-only transformer that predicts the next token at each position.
    Trained with cross-entropy loss on token sequences.

    Args:
        config: ModelConfig object (or ModelArchConfig)

    Attributes:
        token_embed  : Token embedding table (vocab_size, d_model)
        pos_embed    : Learned positional embeddings (max_seq_len, d_model)
        drop         : Input dropout
        blocks       : List of TransformerBlock layers
        ln_f         : Final layer normalization
        lm_head      : Output linear projection to vocab_size

    Usage:
        config = load_config("configs/model_tiny.yaml")
        model = GPTModel(config)
        logits, _ = model(token_ids)           # training
        logits, kv = model(token_ids, use_cache=True)  # inference

    Training:
        # logits: (B, T, vocab_size)
        # targets: (B, T) — shifted input (next token prediction)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    Inference:
        # Generate one token at a time
        for step in range(max_new_tokens):
            logits, kv_cache = model(next_token, past_kv=kv_cache, use_cache=True)
            next_token = sample(logits[:, -1, :])
    """

    def __init__(self, config: Union[ModelConfig, "ModelArchConfig"]):
        super().__init__()

        # Support both ModelConfig and ModelArchConfig
        if hasattr(config, "model"):
            self.config = config.model  # ModelConfig → extract ModelArchConfig
        else:
            self.config = config        # ModelArchConfig directly

        cfg = self.config

        # ---- Embedding layers ----
        # Token embeddings: maps each token ID to a D-dimensional vector
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Positional embeddings: one learnable vector per position
        # The model learns where each position is relative to others
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # Input dropout applied after embedding addition
        self.drop = nn.Dropout(cfg.dropout)

        # ---- Transformer Blocks ----
        # Stack of N identical transformer decoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                bias=cfg.bias,
                max_seq_len=cfg.max_seq_len,
            )
            for _ in range(cfg.n_layers)
        ])

        # ---- Final Layer Norm ----
        # Applied after the last transformer block before the LM head
        # Essential for stable training
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=1e-5)

        # ---- Language Model Head ----
        # Projects from d_model → vocab_size to get logits for each token
        # NOTE: We do NOT apply softmax here. Use cross_entropy during training
        # and softmax/top-k/top-p during inference.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # ---- Weight Tying ----
        # Share weights between token embedding and LM head
        # Token embeddings: input space → d_model
        # LM head: d_model → output space (same input/output vocab)
        # This reduces params and improves performance
        self.lm_head.weight = self.token_embed.weight

        # ---- Initialize weights ----
        self.apply(self._init_weights)

        # Scale residual projections by 1/sqrt(2 * n_layers)
        # This prevents gradient explosion from accumulated residuals
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers)
                )

    def _init_weights(self, module: nn.Module):
        """
        Initialize model weights following GPT-2 paper.

        - Linear layers: N(0, 0.02)
        - Embedding tables: N(0, 0.02)
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List], Optional[torch.Tensor]]:
        """
        Forward pass of the GPT model.

        Args:
            input_ids: Token ID tensor of shape (B, T)
                       During inference with KV cache, T=1 (only new token)
            past_kv  : List of (K, V) tuples from previous forward passes
                       Length = n_layers. Used for efficient autoregressive generation.
            use_cache: If True, return current KV cache for next step
            targets  : Optional target token IDs (B, T) for loss computation
                       If provided, returns (logits, kv_cache, loss)

        Returns:
            Tuple of:
                logits   : (B, T, vocab_size) — unnormalized next-token predictions
                kv_cache : List of (K, V) per layer if use_cache, else None
                loss     : Cross-entropy loss if targets provided, else None

        Tensor shapes throughout:
            input_ids:      (B, T)
            token_embeds:   (B, T, D)
            pos_ids:        (T,)
            pos_embeds:     (T, D)  → broadcast to (B, T, D)
            x:              (B, T, D)
            after N blocks: (B, T, D)
            logits:         (B, T, V)  where V = vocab_size
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Check sequence length
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # ---- Compute position IDs ----
        # During training: positions are [0, 1, 2, ..., T-1]
        # During inference with KV cache: position is [T_past + current_offset]
        past_length = 0
        if past_kv is not None and past_kv[0] is not None:
            past_length = past_kv[0][0].shape[2]  # T_past from K tensor
        pos_ids = torch.arange(past_length, past_length + T, device=device)
        # Shape: (T,)

        # ---- Embeddings ----
        # Token embedding lookup: (B, T) → (B, T, D)
        tok_emb = self.token_embed(input_ids)

        # Positional embedding lookup: (T,) → (T, D)
        pos_emb = self.pos_embed(pos_ids)

        # Combine and apply dropout
        # Broadcasting: (B, T, D) + (T, D) → (B, T, D)
        x = self.drop(tok_emb + pos_emb)
        # Shape: (B, T, D)

        # ---- Transformer Blocks ----
        present_kvs = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            # Get cached KV for this layer (if any)
            layer_past = past_kv[i] if past_kv is not None else None

            # Forward through block
            x, present_kv = block(x, past_kv=layer_past, use_cache=use_cache)

            if use_cache:
                present_kvs.append(present_kv)

        # ---- Final Layer Norm ----
        x = self.ln_f(x)
        # Shape: (B, T, D)

        # ---- Language Model Head ----
        # Project to vocabulary size to get logits
        logits = self.lm_head(x)
        # Shape: (B, T, vocab_size)

        # ---- Optional Loss Computation ----
        loss = None
        if targets is not None:
            # Flatten for cross_entropy: (B*T, vocab_size) vs (B*T,)
            # Cross-entropy expects (N, C) and (N,)
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # Ignore padding positions
            )

        return logits, present_kvs, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Count trainable parameters.

        Args:
            non_embedding: If True, exclude embedding parameters
                          (useful since they're tied with LM head)

        Returns:
            Total parameter count
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract token + positional embedding params
            n_params -= self.token_embed.weight.numel()
            n_params -= self.pos_embed.weight.numel()
        return n_params

    def configure_optimizers(
        self,
        learning_rate: float,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with weight decay applied only to non-bias params.

        Following GPT-2/3 setup:
        - Weight decay only on 2D tensors (linear weights)
        - No weight decay on biases, LayerNorm params, embeddings
        - Separate param groups for decayed/non-decayed

        Args:
            learning_rate: Peak learning rate
            weight_decay : L2 regularization coefficient
            betas        : Adam beta coefficients
            device_type  : 'cuda', 'mps', or 'cpu'

        Returns:
            Configured AdamW optimizer
        """
        # Separate parameters into decayed and non-decayed groups
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                # 2D+ tensors: linear weights, embedding weights → apply decay
                decay_params.append(param)
            else:
                # 1D tensors: biases, LayerNorm weights/biases → no decay
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW on CUDA for ~2x speedup
        use_fused = (device_type == "cuda") and (
            "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused if use_fused else None,
        )

        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        print(
            f"Optimizer: AdamW | "
            f"Decay params: {n_decay:,} | "
            f"No-decay params: {n_no_decay:,} | "
            f"Fused: {use_fused}"
        )
        return optimizer

    def estimate_mfu(self, batch_size: int, dt: float) -> float:
        """
        Estimate Model FLOPs Utilization (MFU) in units of A100 BF16 peak FLOPS.

        MFU = actual_flops / peak_flops

        For a transformer, the forward pass FLOPS are approximately:
            6 * N * T * D^2  (for N params, T tokens, D model dim)

        Args:
            batch_size: Batch size
            dt        : Time elapsed for one forward+backward pass (seconds)

        Returns:
            MFU as a fraction (0 to 1, where 1 = 100% utilization)
        """
        cfg = self.config
        N = self.get_num_params(non_embedding=True)
        # Approx: 6 * params * seq_len * batch_size FLOPs per step
        flops_per_step = 6 * N * cfg.max_seq_len * batch_size
        # A100 BF16 peak: 312 TFLOPS
        a100_peak_flops = 312e12
        mfu = flops_per_step / (a100_peak_flops * dt)
        return mfu

    def __repr__(self) -> str:
        cfg = self.config
        n_params = sum(p.numel() for p in self.parameters())
        if n_params > 1e9:
            size_str = f"{n_params/1e9:.2f}B"
        elif n_params > 1e6:
            size_str = f"{n_params/1e6:.1f}M"
        else:
            size_str = f"{n_params/1e3:.1f}K"
        return (
            f"GPTModel(\n"
            f"  params={size_str}\n"
            f"  vocab={cfg.vocab_size}, seq_len={cfg.max_seq_len}\n"
            f"  layers={cfg.n_layers}, heads={cfg.n_heads}, d_model={cfg.d_model}\n"
            f"  d_ff={cfg.d_ff}, dropout={cfg.dropout}\n"
            f")"
        )
