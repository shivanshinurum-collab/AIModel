"""
model/__init__.py
=================
Model package — exports GPT model and components.
"""

from model.attention import MultiHeadCausalAttention
from model.transformer_block import TransformerBlock, FeedForward
from model.gpt_model import GPTModel

__all__ = [
    "MultiHeadCausalAttention",
    "TransformerBlock",
    "FeedForward",
    "GPTModel",
]
