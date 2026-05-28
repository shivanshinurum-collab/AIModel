"""
tokenizer/tokenizer_train.py
============================
BPE Tokenizer training from scratch.

Trains a Byte-Pair Encoding (BPE) tokenizer using the HuggingFace
`tokenizers` library. This trains the tokenizer on raw text data —
no pretrained vocabulary is used.

How BPE works:
1. Start with byte-level vocabulary (256 bytes)
2. Find the most frequent adjacent pair of tokens
3. Merge that pair into a new single token
4. Repeat until vocab_size is reached

Special tokens added:
    <pad>  — padding token (id=0)
    <bos>  — beginning of sequence (id=1)
    <eos>  — end of sequence (id=2)
    <unk>  — unknown token (id=3)

Usage:
    python tokenizer/tokenizer_train.py \\
        --data datasets/processed/train.txt \\
        --vocab_size 8000 \\
        --output tokenizer/saved

    # Test it
    python tokenizer/tokenizer_train.py --test "Hello, world!"
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

log = get_logger("tokenizer_train")

# ============================================================
# SPECIAL TOKENS
# ============================================================
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Default output directory
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")


# ============================================================
# TOKENIZER TRAINING
# ============================================================

def train_tokenizer(
    data_path: str,
    vocab_size: int = 8000,
    save_dir: str = DEFAULT_SAVE_DIR,
    min_frequency: int = 2,
):
    """
    Train a BPE tokenizer from scratch on the given text file.

    Architecture:
        - Pre-tokenizer: Byte-level BPE (splits at whitespace + punctuation)
        - Model: BPE (Byte-Pair Encoding)
        - Post-processor: Adds <bos>/<eos> tokens
        - Normalizer: NFC unicode normalization

    Args:
        data_path    : Path to training text file (one document per 2 newlines)
        vocab_size   : Target vocabulary size (recommend 8K–50K)
        save_dir     : Directory to save tokenizer.json and vocab.txt
        min_frequency: Minimum frequency for a pair to be merged

    Returns:
        Trained tokenizer object

    Files saved:
        tokenizer.json  — Full tokenizer (model + config)
        vocab.txt       — Human-readable vocabulary list
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.normalizers import NFC
        from tokenizers.processors import TemplateProcessing
    except ImportError:
        log.error("tokenizers library not found. Install: pip install tokenizers")
        sys.exit(1)

    if not os.path.exists(data_path):
        log.error(f"Training data not found: {data_path}")
        log.error("Run datasets/download_dataset.py first")
        sys.exit(1)

    log.section("BPE Tokenizer Training")
    log.info(f"Data:       {data_path}")
    log.info(f"Vocab size: {vocab_size:,}")
    log.info(f"Save dir:   {save_dir}")

    # ---- Initialize BPE model ----
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # ---- Normalizer: NFC unicode normalization ----
    tokenizer.normalizer = NFC()

    # ---- Pre-tokenizer: ByteLevel splits ----
    # Handles all Unicode via byte-level encoding, similar to GPT-2
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # ---- Decoder: reconstruct original text ----
    tokenizer.decoder = ByteLevelDecoder()

    # ---- Trainer configuration ----
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),  # Start with byte alphabet
    )

    # ---- Train ----
    log.info("Training BPE tokenizer...")
    tokenizer.train(files=[data_path], trainer=trainer)

    # ---- Post-processor: add BOS/EOS automatically ----
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0",
        pair=f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0 $B:0 {EOS_TOKEN}:0",
        special_tokens=[
            (BOS_TOKEN, bos_id),
            (EOS_TOKEN, eos_id),
        ],
    )

    # ---- Save ----
    os.makedirs(save_dir, exist_ok=True)

    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    log.info(f"Saved tokenizer → {tokenizer_path}")

    # Save human-readable vocabulary
    vocab = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
    vocab_path = os.path.join(save_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, idx in vocab_sorted:
            f.write(f"{idx}\t{repr(token)}\n")
    log.info(f"Saved vocabulary → {vocab_path}")

    # Save config metadata
    import json
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {
            "pad": {"token": PAD_TOKEN, "id": tokenizer.token_to_id(PAD_TOKEN)},
            "bos": {"token": BOS_TOKEN, "id": bos_id},
            "eos": {"token": EOS_TOKEN, "id": eos_id},
            "unk": {"token": UNK_TOKEN, "id": tokenizer.token_to_id(UNK_TOKEN)},
        },
        "model": "BPE",
        "pre_tokenizer": "ByteLevel",
        "data_path": os.path.abspath(data_path),
        "min_frequency": min_frequency,
    }
    config_path = os.path.join(save_dir, "tokenizer_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Saved config    → {config_path}")

    # Quick test
    test_text = "Hello, how are you doing today?"
    encoded = tokenizer.encode(test_text)
    log.info(f"\nQuick test:")
    log.info(f"  Input:  {test_text!r}")
    log.info(f"  Tokens: {encoded.tokens}")
    log.info(f"  IDs:    {encoded.ids}")
    log.info(f"  Decoded: {tokenizer.decode(encoded.ids)!r}")

    actual_vocab_size = tokenizer.get_vocab_size()
    log.info(f"\nFinal vocab size: {actual_vocab_size:,}")
    log.info(f"Special token IDs: PAD={tokenizer.token_to_id(PAD_TOKEN)}, "
             f"BOS={bos_id}, EOS={eos_id}, UNK={tokenizer.token_to_id(UNK_TOKEN)}")

    return tokenizer


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer from scratch"
    )
    parser.add_argument(
        "--data", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "datasets", "processed", "train.txt"
        ),
        help="Path to training text file"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=8000,
        help="Vocabulary size (default: 8000)"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_SAVE_DIR,
        help=f"Output directory (default: {DEFAULT_SAVE_DIR})"
    )
    parser.add_argument(
        "--min_frequency", type=int, default=2,
        help="Minimum pair frequency for merging (default: 2)"
    )

    args = parser.parse_args()

    train_tokenizer(
        data_path=args.data,
        vocab_size=args.vocab_size,
        save_dir=args.output,
        min_frequency=args.min_frequency,
    )

    log.info("\nTokenizer training complete!")
    log.info("Next steps:")
    log.info("  1. Train your model: python training/train.py")
    log.info("  2. Test tokenizer: python tokenizer/tokenizer_infer.py --text 'Hello!'")


if __name__ == "__main__":
    main()
