"""
datasets/download_dataset.py
============================
Dataset downloader for GPT LLM training.

Downloads open-source text datasets from HuggingFace:
- TinyStories (small, fast, great for tiny/small models)
- Wikipedia (English, 10K articles subset)
- OpenWebText (web crawl text)
- BookCorpus (books text)

All datasets are saved as raw .txt files in datasets/raw/

Usage:
    python datasets/download_dataset.py --datasets tinystories wikipedia
    python datasets/download_dataset.py --datasets all --max_samples 100000

Requirements:
    pip install datasets huggingface-hub
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

# Add project root to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

log = get_logger("downloader")


# ============================================================
# DATASET REGISTRY
# ============================================================

# WE NEED TO ADD HERE DATASETS - TRAININNG DATA

AVAILABLE_DATASETS = {
    "tinystories": {
        "hf_name": "roneneldan/TinyStories",
        "split": "train",
        "text_column": "text",
        "description": "Short children's stories — great for tiny/small models",
        "approx_size": "~2M samples, ~470MB",
    },
    "wikipedia": {
        "hf_name": "wikipedia",
        "hf_config": "20220301.en",
        "split": "train",
        "text_column": "text",
        "description": "English Wikipedia articles",
        "approx_size": "~6.5M articles, ~20GB",
    },
    "openwebtext": {
        "hf_name": "Skylion007/openwebtext",
        "split": "train",
        "text_column": "text",
        "description": "OpenWebText web crawl — similar to GPT-2 training data",
        "approx_size": "~8M documents, ~38GB",
    },
    "bookcorpus": {
        "hf_name": "bookcorpus",
        "split": "train",
        "text_column": "text",
        "description": "Books corpus used in BERT training",
        "approx_size": "~74M sentences",
    },
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


# ============================================================
# DOWNLOAD FUNCTIONS
# ============================================================

def download_dataset(
    name: str,
    max_samples: Optional[int] = None,
    output_dir: str = RAW_DIR,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Download a single dataset and save as raw text file.

    Args:
        name        : Dataset name from AVAILABLE_DATASETS
        max_samples : Maximum number of samples to download (None = all)
        output_dir  : Directory to save raw text files
        cache_dir   : HuggingFace cache directory

    Returns:
        Path to the saved .txt file

    Data flow:
        HuggingFace Hub → datasets library → iterate → write .txt
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(AVAILABLE_DATASETS.keys())}"
        )

    info = AVAILABLE_DATASETS[name]
    log.info(f"Downloading: {name} — {info['description']}")
    log.info(f"Estimated size: {info['approx_size']}")

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets library not found. Install: pip install datasets")
        sys.exit(1)

    # Build load_dataset kwargs
    load_kwargs = {
        "path": info["hf_name"],
        "split": info["split"],
        "trust_remote_code": True,
    }
    if "hf_config" in info:
        load_kwargs["name"] = info["hf_config"]
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    log.info(f"Loading from HuggingFace Hub: {info['hf_name']} ...")
    dataset = load_dataset(**load_kwargs)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.txt")

    text_col = info["text_column"]
    total = len(dataset) if hasattr(dataset, "__len__") else None
    count = 0

    log.info(f"Writing text to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            if max_samples and count >= max_samples:
                break

            text = sample.get(text_col, "")
            if not text or not isinstance(text, str):
                continue

            text = text.strip()
            if len(text) < 50:  # Skip very short texts
                continue

            # Write with document separator
            f.write(text)
            f.write("\n\n")  # Double newline between documents
            count += 1

            # Progress logging
            if count % 10000 == 0:
                pct = f"{100*count/total:.1f}%" if total else f"{count}"
                log.info(f"  Written {count:,} samples ({pct})")

    log.info(f"Done! Saved {count:,} samples → {output_path}")
    return output_path


def download_all(
    datasets: List[str],
    max_samples: Optional[int] = None,
    output_dir: str = RAW_DIR,
):
    """
    Download multiple datasets.

    Args:
        datasets   : List of dataset names
        max_samples: Maximum samples per dataset
        output_dir : Output directory
    """
    downloaded = []
    for name in datasets:
        try:
            path = download_dataset(name, max_samples=max_samples, output_dir=output_dir)
            downloaded.append(path)
        except Exception as e:
            log.error(f"Failed to download '{name}': {e}")

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        "downloaded": downloaded,
        "max_samples": max_samples,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"\nDownloaded {len(downloaded)}/{len(datasets)} datasets")
    log.info(f"Manifest: {manifest_path}")
    return downloaded


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for GPT LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
{'='*60}
""" + "\n".join(
            f"  {k:15s} — {v['description']}\n"
            f"              Size: {v['approx_size']}"
            for k, v in AVAILABLE_DATASETS.items()
        )
    )

    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(AVAILABLE_DATASETS.keys()) + ["all"],
        default=["tinystories"],
        help="Datasets to download (default: tinystories)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum samples per dataset (None = all)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=RAW_DIR,
        help=f"Output directory (default: {RAW_DIR})"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        print("=" * 60)
        for name, info in AVAILABLE_DATASETS.items():
            print(f"  {name:15s} {info['description']}")
            print(f"  {'':15s} Size: {info['approx_size']}")
            print()
        return

    datasets = list(AVAILABLE_DATASETS.keys()) if "all" in args.datasets else args.datasets
    download_all(datasets, max_samples=args.max_samples, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
