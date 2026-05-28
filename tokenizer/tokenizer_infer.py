"""
tokenizer/tokenizer_infer.py
============================
Tokenizer inference wrapper for the GPT LLM.

Wraps the trained BPE tokenizer with a clean API for:
- Encoding text to token IDs
- Decoding token IDs back to text
- Batch encoding
- Special token handling
- Sequence length management

Usage:
    from tokenizer.tokenizer_infer import LLMTokenizer

    tok = LLMTokenizer("tokenizer/saved/tokenizer.json")

    ids = tok.encode("Hello, world!")
    text = tok.decode(ids)

    # Batch encoding
    batch = tok.encode_batch(["Hello!", "How are you?"])
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# SPECIAL TOKEN CONSTANTS
# ============================================================
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class LLMTokenizer:
    """
    Tokenizer wrapper for the trained BPE tokenizer.

    Provides a clean interface for all tokenization operations
    needed during training and inference.

    Example:
        tok = LLMTokenizer("tokenizer/saved/tokenizer.json")

        # Encode (returns list of int IDs)
        ids = tok.encode("Hello there!")

        # Decode (returns string)
        text = tok.decode(ids)

        # Round-trip test
        assert tok.decode(tok.encode("Hello")) == "Hello"

    Attributes:
        vocab_size  : Total vocabulary size
        pad_id      : ID of the <pad> token
        bos_id      : ID of the <bos> token
        eos_id      : ID of the <eos> token
        unk_id      : ID of the <unk> token
    """

    def __init__(self, tokenizer_path: str):
        """
        Load a trained tokenizer from disk.

        Args:
            tokenizer_path: Path to tokenizer.json file

        Raises:
            FileNotFoundError: If tokenizer file doesn't exist
            ImportError      : If tokenizers library is not installed
        """
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}\n"
                f"Run: python tokenizer/tokenizer_train.py"
            )

        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "tokenizers library not installed. Run: pip install tokenizers"
            )

        self._tokenizer = Tokenizer.from_file(tokenizer_path)

        # Cache special token IDs for fast access
        self.pad_id: int = self._get_id(PAD_TOKEN)
        self.bos_id: int = self._get_id(BOS_TOKEN)
        self.eos_id: int = self._get_id(EOS_TOKEN)
        self.unk_id: int = self._get_id(UNK_TOKEN)
        self.vocab_size: int = self._tokenizer.get_vocab_size()

        # Disable automatic BOS/EOS addition by default
        # (we handle this manually in training/inference for more control)
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def _get_id(self, token: str) -> int:
        """Get token ID, raise error if not found."""
        idx = self._tokenizer.token_to_id(token)
        if idx is None:
            raise ValueError(
                f"Special token '{token}' not in vocabulary. "
                f"Retrain tokenizer with special tokens."
            )
        return idx

    # ============================================================
    # ENCODING
    # ============================================================

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text to a list of token IDs.

        Args:
            text      : Input text string
            add_bos   : Prepend <bos> token
            add_eos   : Append <eos> token
            max_length: Truncate to this length (including special tokens)

        Returns:
            List of integer token IDs

        Example:
            tok.encode("Hello!")  → [234, 567, 89]
            tok.encode("Hi", add_bos=True, add_eos=True) → [1, 45, 678, 2]
        """
        # The tokenizer's post_processor adds BOS/EOS by default
        # We encode without them and add manually for flexibility
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Encode a batch of texts to lists of token IDs.

        Args:
            texts     : List of input strings
            add_bos   : Prepend <bos> to each sequence
            add_eos   : Append <eos> to each sequence
            max_length: Truncate each sequence to this length

        Returns:
            List of lists of integer token IDs
        """
        encodings = self._tokenizer.encode_batch(
            texts, add_special_tokens=False
        )
        results = []
        for enc in encodings:
            ids = enc.ids
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]
            if max_length is not None:
                ids = ids[:max_length]
            results.append(ids)
        return results

    def encode_for_training(
        self,
        text: str,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text for training (adds BOS and EOS).

        This is the standard format for causal language modeling:
        <bos> token1 token2 ... tokenN <eos>

        Args:
            text      : Input text
            max_length: Optional max token count

        Returns:
            List of token IDs with BOS/EOS
        """
        return self.encode(text, add_bos=True, add_eos=True, max_length=max_length)

    # ============================================================
    # DECODING
    # ============================================================

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token IDs to text.

        Args:
            ids                : List of integer token IDs
            skip_special_tokens: Remove special tokens from output (default: True)

        Returns:
            Decoded text string

        Example:
            tok.decode([234, 567, 89]) → "Hello!"
        """
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            batch_ids          : List of token ID lists
            skip_special_tokens: Remove special tokens from output

        Returns:
            List of decoded strings
        """
        return self._tokenizer.decode_batch(
            batch_ids, skip_special_tokens=skip_special_tokens
        )

    # ============================================================
    # VOCABULARY ACCESS
    # ============================================================

    def token_to_id(self, token: str) -> Optional[int]:
        """Get the ID for a token string."""
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        """Get the token string for an ID."""
        return self._tokenizer.id_to_token(idx)

    def get_vocab(self) -> dict:
        """Return the full vocabulary as {token: id} dict."""
        return self._tokenizer.get_vocab()

    # ============================================================
    # UTILITIES
    # ============================================================

    def pad_sequence(
        self,
        ids: List[int],
        max_length: int,
        pad_left: bool = False,
    ) -> List[int]:
        """
        Pad a token sequence to max_length using <pad> tokens.

        Args:
            ids       : Token ID sequence
            max_length: Target length
            pad_left  : Pad on left side instead of right

        Returns:
            Padded sequence of length max_length
        """
        if len(ids) >= max_length:
            return ids[:max_length]
        padding = [self.pad_id] * (max_length - len(ids))
        if pad_left:
            return padding + ids
        return ids + padding

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text without storing them."""
        return len(self.encode(text))

    def __repr__(self) -> str:
        return (
            f"LLMTokenizer(vocab_size={self.vocab_size}, "
            f"pad={self.pad_id}, bos={self.bos_id}, "
            f"eos={self.eos_id}, unk={self.unk_id})"
        )


# ============================================================
# CLI / TEST INTERFACE
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the trained LLM tokenizer"
    )
    parser.add_argument(
        "--tokenizer", type=str,
        default=os.path.join(os.path.dirname(__file__), "saved", "tokenizer.json"),
        help="Path to tokenizer.json"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Text to tokenize (interactive mode if not specified)"
    )
    parser.add_argument(
        "--roundtrip", action="store_true",
        help="Run round-trip encode/decode test"
    )

    args = parser.parse_args()

    tok = LLMTokenizer(args.tokenizer)
    print(f"\n{tok}")
    print(f"Special tokens: PAD={tok.pad_id}, BOS={tok.bos_id}, EOS={tok.eos_id}, UNK={tok.unk_id}")

    def test_text(text: str):
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        print(f"\nInput:   {text!r}")
        print(f"Tokens:  {[tok.id_to_token(i) for i in ids[:20]]}")
        print(f"IDs:     {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"Length:  {len(ids)} tokens")
        print(f"Decoded: {decoded!r}")
        if args.roundtrip:
            match = text.strip() == decoded.strip()
            print(f"Round-trip: {'✓ PASS' if match else '✗ FAIL'}")

    if args.text:
        test_text(args.text)
    else:
        # Interactive mode
        print("\nEnter text to tokenize (Ctrl+C to exit):")
        while True:
            try:
                text = input("\n> ").strip()
                if text:
                    test_text(text)
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
