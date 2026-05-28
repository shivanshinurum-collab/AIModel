"""
datasets/merge_dataset.py
=========================
Dataset merge and train/val split script.

Takes multiple cleaned dataset files and combines them into:
- datasets/processed/train.txt  (90% of data by default)
- datasets/processed/val.txt    (10% of data by default)

Also:
- Shuffles documents before splitting (reproducible with seed)
- Computes basic statistics (total tokens, chars, docs)
- Saves a metadata.json file with dataset info

Usage:
    python datasets/merge_dataset.py
    python datasets/merge_dataset.py --input_dir datasets/cleaned --val_ratio 0.05
    python datasets/merge_dataset.py --seed 42 --val_ratio 0.1
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

log = get_logger("merger")


# ============================================================
# DEFAULT PATHS
# ============================================================
CLEANED_DIR = os.path.join(os.path.dirname(__file__), "cleaned")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


# ============================================================
# DOCUMENT ITERATOR
# ============================================================

def iter_documents(filepath: str):
    """
    Yield documents from a text file.
    Documents are separated by double newlines.
    """
    buffer = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "" and buffer:
                doc = "\n".join(buffer).strip()
                if doc:
                    yield doc
                buffer = []
            else:
                buffer.append(line.rstrip())
    if buffer:
        doc = "\n".join(buffer).strip()
        if doc:
            yield doc


# ============================================================
# MERGE LOGIC
# ============================================================

def load_all_documents(input_dir: str) -> List[str]:
    """
    Load all documents from .txt files in a directory.

    Args:
        input_dir: Directory containing cleaned .txt files

    Returns:
        List of document strings
    """
    txt_files = list(Path(input_dir).glob("*.txt"))
    if not txt_files:
        log.error(f"No .txt files found in {input_dir}")
        sys.exit(1)

    log.info(f"Found {len(txt_files)} input files in {input_dir}")

    all_docs = []
    for filepath in txt_files:
        log.info(f"  Loading: {filepath.name}")
        docs = list(iter_documents(str(filepath)))
        log.info(f"  → {len(docs):,} documents")
        all_docs.extend(docs)

    log.info(f"Total documents loaded: {len(all_docs):,}")
    return all_docs


def split_documents(
    docs: List[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Shuffle and split documents into train/val sets.

    Args:
        docs     : List of document strings
        val_ratio: Fraction of data for validation (default: 10%)
        seed     : Random seed for reproducibility

    Returns:
        (train_docs, val_docs) tuple
    """
    random.seed(seed)
    random.shuffle(docs)

    val_size = max(1, int(len(docs) * val_ratio))
    val_docs = docs[:val_size]
    train_docs = docs[val_size:]

    log.info(f"Split: {len(train_docs):,} train / {len(val_docs):,} val")
    return train_docs, val_docs


def write_documents(docs: List[str], filepath: str):
    """
    Write documents to a text file, separated by double newlines.

    Args:
        docs    : List of document strings
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            f.write(doc.strip())
            f.write("\n\n")
    log.info(f"  Written: {filepath}")


def compute_stats(docs: List[str]) -> dict:
    """
    Compute basic text statistics.

    Args:
        docs: List of document strings

    Returns:
        dict with num_docs, total_chars, approx_words, approx_tokens
    """
    total_chars = sum(len(d) for d in docs)
    total_words = sum(len(d.split()) for d in docs)
    # Rough approximation: ~4 chars per token (BPE)
    approx_tokens = total_chars // 4

    return {
        "num_docs": len(docs),
        "total_chars": total_chars,
        "approx_words": total_words,
        "approx_tokens": approx_tokens,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def merge_and_split(
    input_dir: str = CLEANED_DIR,
    output_dir: str = PROCESSED_DIR,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Full merge and split pipeline.

    1. Load all cleaned .txt files
    2. Shuffle documents
    3. Split into train/val
    4. Write to output files
    5. Save metadata.json

    Args:
        input_dir : Directory with cleaned dataset files
        output_dir: Directory to write train.txt and val.txt
        val_ratio : Fraction of data for validation
        seed      : Random seed
    """
    log.section("Dataset Merge & Split Pipeline")

    # Load
    all_docs = load_all_documents(input_dir)

    if len(all_docs) == 0:
        log.error("No documents found! Run download_dataset.py and clean_dataset.py first.")
        sys.exit(1)

    # Split
    train_docs, val_docs = split_documents(all_docs, val_ratio=val_ratio, seed=seed)

    # Write
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")

    log.info("Writing train set...")
    write_documents(train_docs, train_path)

    log.info("Writing validation set...")
    write_documents(val_docs, val_path)

    # Statistics
    train_stats = compute_stats(train_docs)
    val_stats = compute_stats(val_docs)

    metadata = {
        "train": train_stats,
        "val": val_stats,
        "total_docs": len(all_docs),
        "val_ratio": val_ratio,
        "seed": seed,
        "train_path": os.path.abspath(train_path),
        "val_path": os.path.abspath(val_path),
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    log.section("Dataset Statistics")
    log.info(f"Train: {train_stats['num_docs']:>8,} docs | "
             f"{train_stats['approx_tokens']/1e6:.1f}M tokens | "
             f"{train_stats['total_chars']/1e6:.1f}M chars")
    log.info(f"Val:   {val_stats['num_docs']:>8,} docs | "
             f"{val_stats['approx_tokens']/1e6:.1f}M tokens | "
             f"{val_stats['total_chars']/1e6:.1f}M chars")
    log.info(f"Saved metadata → {metadata_path}")

    return metadata


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge and split cleaned datasets for LLM training"
    )
    parser.add_argument(
        "--input_dir", type=str, default=CLEANED_DIR,
        help=f"Directory with cleaned .txt files (default: {CLEANED_DIR})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=PROCESSED_DIR,
        help=f"Output directory (default: {PROCESSED_DIR})"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Fraction of data for validation (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible shuffling (default: 42)"
    )

    args = parser.parse_args()
    merge_and_split(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
