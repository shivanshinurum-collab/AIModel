"""
inference/generate.py
=====================
Text generation engine for the GPT-style LLM.

AUTOREGRESSIVE GENERATION
==========================
GPT models generate text one token at a time:

    for t in range(max_new_tokens):
        logits = model(input_tokens)    # (B, T, V)
        next_logits = logits[:, -1, :]  # Take last position: (B, V)
        next_token = sample(next_logits) # Sample one token: (B,)
        input_tokens = concat(input_tokens, next_token)  # Extend

This is O(T^2) without KV cache (each step reprocesses all tokens).
With KV cache, it becomes O(T) since past K,V are cached.

SAMPLING STRATEGIES
===================

Greedy (temperature → 0):
    Always pick the highest-probability token.
    Deterministic but often repetitive.

Temperature Scaling:
    logits = logits / temperature
    Higher temperature → more random (creative)
    Lower temperature → more focused (coherent)
    temperature=1.0 → no change

Top-K Sampling:
    Keep only the K highest-probability tokens, sample from those.
    Prevents very low-probability (weird) tokens from being chosen.
    typical k=40-200

Top-P (Nucleus) Sampling:
    Keep the smallest set of tokens whose cumulative probability >= p.
    Adapts to the model's confidence — uses fewer tokens when confident.
    typical p=0.9-0.95

Repetition Penalty:
    Divide logits of already-generated tokens by penalty (>1.0).
    Prevents the model from repeating the same phrases.
    Typical penalty: 1.1-1.3

Combined strategy (recommended):
    temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Generator, Union
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits.

    temperature < 1.0: sharper distribution (more confident/focused)
    temperature > 1.0: flatter distribution (more random/creative)
    temperature = 1.0: no change

    Args:
        logits     : (B, V) unnormalized logits
        temperature: Scaling factor (must be > 0)

    Returns:
        Scaled logits (B, V)
    """
    if temperature == 1.0:
        return logits
    return logits / max(temperature, 1e-8)


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage repeating tokens.

    For each token in input_ids:
        if logit > 0: logit = logit / penalty
        if logit < 0: logit = logit * penalty

    This reduces the probability of already-seen tokens.

    Args:
        logits   : (B, V) logits
        input_ids: (B, T) all tokens generated so far
        penalty  : Penalty factor > 1.0 (1.0 = no penalty)

    Returns:
        Modified logits (B, V)
    """
    if penalty == 1.0:
        return logits

    for i in range(input_ids.shape[0]):  # For each batch item
        # Get unique token IDs seen so far
        unique_ids = input_ids[i].unique()
        # Get the logit values at those positions
        score = logits[i, unique_ids]
        # Apply penalty: divide positive, multiply negative
        score = torch.where(score > 0, score / penalty, score * penalty)
        logits[i, unique_ids] = score

    return logits


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Apply top-k filtering: zero out all tokens except the top k.

    After filtering, only the k highest-logit tokens remain.
    Sampling from this set prevents choosing very unlikely tokens.

    Args:
        logits: (B, V) logits
        top_k : Number of top tokens to keep

    Returns:
        Filtered logits (B, V) with non-top-k tokens set to -inf
    """
    if top_k <= 0:
        return logits

    # Get the k-th largest logit value (per batch item)
    top_k = min(top_k, logits.shape[-1])
    # kth_vals: (B, 1) — the minimum logit value we keep
    kth_vals, _ = torch.topk(logits, top_k, dim=-1)
    threshold = kth_vals[:, -1, None]  # (B, 1)

    # Set all logits below threshold to -inf (they won't be sampled)
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling: keep tokens covering cumulative probability >= p.

    Algorithm:
        1. Sort tokens by probability (descending)
        2. Compute cumulative probabilities
        3. Remove tokens where cumulative prob > p
        4. Normalize remaining probabilities

    This adapts the effective vocabulary size to the model's confidence.
    When the model is confident, only a few tokens are kept.
    When uncertain, more tokens are kept.

    Args:
        logits: (B, V) logits
        top_p : Cumulative probability threshold (0 < top_p <= 1.0)

    Returns:
        Filtered logits (B, V)
    """
    if top_p >= 1.0:
        return logits

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sort probabilities descending (for cumulative sum)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where cumulative prob > top_p
    # (shift cumsum right by 1 to include the token that pushes over top_p)
    remove_mask = cumsum_probs - sorted_probs > top_p

    # Set removed tokens to 0
    sorted_probs[remove_mask] = 0.0

    # Scatter back to original order
    probs = torch.zeros_like(logits).scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

    # Convert back to log space (to return logits-like values)
    # Add small epsilon to avoid log(0)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    # Return as log probs (or just return the filtered probs directly)
    # We return filtered logits-equivalent for use with multinomial
    return torch.log(probs + 1e-10)


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Sample the next token from logits using the configured strategy.

    Applies: temperature → top_k → top_p → multinomial sampling

    Args:
        logits     : (B, V) unnormalized logits for next token
        temperature: Randomness control (default=1.0)
        top_k      : Keep only top k tokens (0=disabled)
        top_p      : Nucleus sampling threshold (1.0=disabled)

    Returns:
        next_token: (B, 1) sampled token IDs
    """
    # Apply temperature scaling
    logits = apply_temperature(logits, temperature)

    # Apply top-k filtering
    if top_k > 0:
        logits = apply_top_k(logits, top_k)

    # Apply top-p filtering (changes logits to log-probs)
    if top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    # Sample from the distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

    return next_token


# ============================================================
# MAIN GENERATION FUNCTION
# ============================================================

@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """
    Autoregressive text generation with sampling.

    Generates max_new_tokens tokens one at a time, appending each
    to the context and feeding it back to the model.

    Args:
        model             : GPTModel instance (in eval mode)
        input_ids         : Starting token IDs (B, T)
        max_new_tokens    : Maximum tokens to generate
        temperature       : Sampling temperature (lower = more focused)
        top_k             : Top-k filtering (0 = disabled)
        top_p             : Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = none)
        eos_token_id      : Stop generation when this token is produced
        pad_token_id      : Padding token ID (for ignoring in penalty)
        use_kv_cache      : Use KV cache for faster generation

    Returns:
        Generated token IDs including the input: (B, T + max_new_tokens)
    """
    device = input_ids.device
    model.eval()

    # Current sequence being generated
    generated = input_ids.clone()

    # KV cache: list of (K, V) per layer
    past_kv = None

    for _ in range(max_new_tokens):
        # ---- Prepare input for this step ----
        if use_kv_cache and past_kv is not None:
            # KV cache mode: only feed the latest token
            model_input = generated[:, -1:]  # (B, 1)
        else:
            # No cache: feed the full sequence
            # Truncate to max_seq_len if needed
            max_seq = model.config.max_seq_len
            model_input = generated[:, -max_seq:]

        # ---- Forward pass ----
        logits, past_kv, _ = model(
            model_input,
            past_kv=past_kv if use_kv_cache else None,
            use_cache=use_kv_cache,
        )

        # Get logits for the last position (next token prediction)
        next_logits = logits[:, -1, :]  # (B, V)

        # ---- Apply repetition penalty ----
        if repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(
                next_logits, generated, repetition_penalty
            )

        # ---- Sample next token ----
        next_token = sample_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )  # (B, 1)

        # ---- Append to generated sequence ----
        generated = torch.cat([generated, next_token], dim=1)

        # ---- Check for EOS ----
        if eos_token_id is not None:
            # If all batch items have generated EOS, stop
            if (next_token == eos_token_id).all():
                break

    return generated


# ============================================================
# STREAMING GENERATION
# ============================================================

@torch.no_grad()
def generate_streaming(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    eos_token_id: Optional[int] = None,
    use_kv_cache: bool = True,
) -> Generator[str, None, None]:
    """
    Generator that yields decoded tokens one at a time as they're generated.

    Used for streaming output to terminal or API responses.

    Usage:
        for token_text in generate_streaming(model, input_ids, tokenizer):
            print(token_text, end="", flush=True)

    Args:
        model           : GPTModel
        input_ids       : (B, T) input token IDs (B=1 for streaming)
        tokenizer       : LLMTokenizer for decoding
        max_new_tokens  : Maximum tokens to generate
        temperature     : Sampling temperature
        top_k           : Top-k filtering
        top_p           : Nucleus sampling
        repetition_penalty: Repetition penalty
        eos_token_id    : Stop on this token
        use_kv_cache    : Use KV cache

    Yields:
        Decoded token strings, one at a time
    """
    device = input_ids.device
    model.eval()

    generated = input_ids.clone()
    past_kv = None

    for _ in range(max_new_tokens):
        # Prepare input
        if use_kv_cache and past_kv is not None:
            model_input = generated[:, -1:]
        else:
            max_seq = model.config.max_seq_len
            model_input = generated[:, -max_seq:]

        # Forward pass
        logits, past_kv, _ = model(
            model_input,
            past_kv=past_kv if use_kv_cache else None,
            use_cache=use_kv_cache,
        )

        next_logits = logits[:, -1, :]

        # Repetition penalty
        if repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(next_logits, generated, repetition_penalty)

        # Sample
        next_token = sample_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)

        # Check EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # Append
        generated = torch.cat([generated, next_token], dim=1)

        # Decode and yield the new token
        token_text = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=True)
        yield token_text
