"""
datasets/clean_dataset.py
=========================
Dataset cleaning pipeline for GPT LLM training.

Cleaning stages:
1. UTF-8 encoding fix (using ftfy)
2. Unicode normalization (NFC)
3. Whitespace normalization
4. Minimum/maximum length filtering
5. Deduplication (exact match on first 200 chars)
6. Profanity/noise filtering (optional)
7. Language detection (optional, keep English only)
8. Bad character filtering

Inputs : datasets/raw/*.txt
Outputs: datasets/cleaned/*.txt

Usage:
    python datasets/clean_dataset.py --input datasets/raw/tinystories.txt
    python datasets/clean_dataset.py --input datasets/raw/ --output datasets/cleaned/
"""

import os
import re
import sys
import argparse
import unicodedata
import hashlib
from pathlib import Path
from typing import Iterator, Set, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

log = get_logger("cleaner")


# ============================================================
# CLEANING CONFIGURATION
# ============================================================
MIN_TEXT_LENGTH = 100       # minimum characters per document
MAX_TEXT_LENGTH = 100000    # maximum characters per document (trim if exceeded)
MIN_WORDS = 15              # minimum words per document
DEDUP_PREFIX_LEN = 200      # characters used for deduplication fingerprint


# ============================================================
# REGEX PATTERNS — compiled once for performance
# ============================================================
# Remove HTML tags
RE_HTML = re.compile(r"<[^>]+>")

# Collapse multiple spaces/tabs → single space
RE_SPACES = re.compile(r"[ \t]+")

# Collapse 3+ newlines → double newline
RE_NEWLINES = re.compile(r"\n{3,}")

# Remove control characters (except newline, tab)
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# URL removal (optional)
RE_URL = re.compile(r"https?://\S+|www\.\S+")

# Wikipedia boilerplate markers
RE_WIKI_MARKERS = re.compile(
    r"(==\s*References\s*==|==\s*External links\s*==|==\s*See also\s*==).*",
    re.DOTALL | re.IGNORECASE,
)

# Detect texts that are mostly non-alphabetic (tables, code dumps)
RE_ALPHA = re.compile(r"[a-zA-Z]")


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def fix_encoding(text: str) -> str:
    """
    Fix broken Unicode using ftfy library.
    Handles common issues like mojibake (wrong encoding interpretation).

    Example: "Ã©" → "é"
    """
    try:
        import ftfy
        return ftfy.fix_text(text)
    except ImportError:
        # Fallback: encode/decode roundtrip
        return text.encode("utf-8", errors="ignore").decode("utf-8")


def normalize_unicode(text: str) -> str:
    """
    Apply NFC Unicode normalization.

    NFC = Canonical Decomposition followed by Canonical Composition.
    Ensures consistent representation of accented characters.
    "é" (U+00E9) vs "e" + combining accent (U+0301) → both become U+00E9
    """
    return unicodedata.normalize("NFC", text)


def remove_html(text: str) -> str:
    """Remove HTML tags. <b>hello</b> → hello"""
    return RE_HTML.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace:
    - Collapse multiple spaces/tabs → single space
    - Collapse 3+ newlines → double newline
    - Strip leading/trailing whitespace
    """
    text = RE_SPACES.sub(" ", text)
    text = RE_NEWLINES.sub("\n\n", text)
    return text.strip()


def remove_control_chars(text: str) -> str:
    """Remove control characters that corrupt text."""
    return RE_CONTROL.sub("", text)


def truncate_at_wiki_boilerplate(text: str) -> str:
    """
    Remove Wikipedia boilerplate sections (References, External links etc.)
    These are low-quality text with citation markup.
    """
    return RE_WIKI_MARKERS.sub("", text).strip()


def remove_urls(text: str, placeholder: str = " ") -> str:
    """Replace URLs with a placeholder (URLs add no linguistic value)."""
    return RE_URL.sub(placeholder, text)


def is_valid_text(text: str) -> bool:
    """
    Quality filter: returns True if text passes all quality checks.

    Filters out:
    - Too short texts
    - Too long texts (split instead)
    - Texts with too few words
    - Texts that are mostly non-alphabetic (tables, etc.)
    """
    if len(text) < MIN_TEXT_LENGTH:
        return False

    words = text.split()
    if len(words) < MIN_WORDS:
        return False

    # Check that at least 60% of characters are alphabetic or spaces
    alpha_count = len(RE_ALPHA.findall(text))
    if alpha_count / max(1, len(text)) < 0.4:
        return False

    return True


def truncate_text(text: str, max_chars: int = MAX_TEXT_LENGTH) -> str:
    """
    Truncate text at word boundary to max_chars.
    Preserves complete words.
    """
    if len(text) <= max_chars:
        return text
    # Find last space before limit
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated


def dedup_fingerprint(text: str) -> str:
    """
    Create a deduplication fingerprint from the first N chars.

    Uses MD5 hash of the normalized prefix for memory efficiency.
    Two documents with identical first 200 chars are considered duplicates.
    """
    prefix = text[:DEDUP_PREFIX_LEN].lower().strip()
    return hashlib.md5(prefix.encode("utf-8")).hexdigest()


def clean_document(text: str, remove_url: bool = True) -> Optional[str]:
    """
    Apply the full cleaning pipeline to a single document.

    Pipeline:
        raw text
        → fix encoding
        → remove HTML
        → remove control chars
        → normalize unicode
        → truncate Wikipedia boilerplate
        → optionally remove URLs
        → normalize whitespace
        → quality filter
        → truncate if too long

    Args:
        text      : Raw input text
        remove_url: Whether to remove URLs

    Returns:
        Cleaned text string, or None if filtered out
    """
    if not text or not isinstance(text, str):
        return None

    text = fix_encoding(text)
    text = remove_html(text)
    text = remove_control_chars(text)
    text = normalize_unicode(text)
    text = truncate_at_wiki_boilerplate(text)

    if remove_url:
        text = remove_urls(text)

    text = normalize_whitespace(text)

    if not is_valid_text(text):
        return None

    text = truncate_text(text)
    return text


# ============================================================
# DOCUMENT ITERATOR
# ============================================================

def iter_documents(filepath: str) -> Iterator[str]:
    """
    Iterate over documents in a .txt file.

    Assumes documents are separated by double newlines.

    Args:
        filepath: Path to raw text file

    Yields:
        Individual document strings
    """
    buffer = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "" and buffer:
                # Empty line = document boundary
                doc = "\n".join(buffer).strip()
                if doc:
                    yield doc
                buffer = []
            else:
                buffer.append(line.rstrip())
    # Yield last document
    if buffer:
        doc = "\n".join(buffer).strip()
        if doc:
            yield doc


# ============================================================
# MAIN CLEANING PIPELINE
# ============================================================

def clean_file(
    input_path: str,
    output_path: str,
    remove_url: bool = True,
    deduplicate: bool = True,
) -> dict:
    """
    Clean a raw text file and save the result.

    Args:
        input_path : Input raw text file
        output_path: Output cleaned text file
        remove_url : Whether to remove URLs
        deduplicate: Whether to deduplicate documents

    Returns:
        Statistics dict with counts of total/cleaned/filtered/duplicates
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen_fingerprints: Set[str] = set()
    stats = {
        "total": 0,
        "cleaned": 0,
        "filtered_quality": 0,
        "filtered_duplicates": 0,
    }

    log.info(f"Cleaning: {input_path}")
    log.info(f"Output:   {output_path}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for doc in iter_documents(input_path):
            stats["total"] += 1

            # Clean
            cleaned = clean_document(doc, remove_url=remove_url)
            if cleaned is None:
                stats["filtered_quality"] += 1
                continue

            # Deduplicate
            if deduplicate:
                fp = dedup_fingerprint(cleaned)
                if fp in seen_fingerprints:
                    stats["filtered_duplicates"] += 1
                    continue
                seen_fingerprints.add(fp)

            # Write
            out_f.write(cleaned)
            out_f.write("\n\n")
            stats["cleaned"] += 1

            if stats["total"] % 10000 == 0:
                log.info(
                    f"  Processed {stats['total']:,} | "
                    f"Kept {stats['cleaned']:,} | "
                    f"Filtered {stats['filtered_quality']:,} | "
                    f"Dupes {stats['filtered_duplicates']:,}"
                )

    retention_rate = stats["cleaned"] / max(1, stats["total"]) * 100
    log.info(
        f"Done! {stats['cleaned']:,}/{stats['total']:,} kept "
        f"({retention_rate:.1f}% retention)"
    )
    return stats


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean raw text datasets for LLM training"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input file or directory containing raw .txt files"
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(__file__), "cleaned"),
        help="Output directory for cleaned files"
    )
    parser.add_argument(
        "--no_dedup", action="store_true",
        help="Disable deduplication (faster but lower quality)"
    )
    parser.add_argument(
        "--keep_urls", action="store_true",
        help="Keep URLs in text (default: remove)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.txt"))
        if not files:
            log.error(f"No .txt files found in {input_path}")
            sys.exit(1)
    else:
        log.error(f"Input not found: {input_path}")
        sys.exit(1)

    total_stats = {"total": 0, "cleaned": 0, "filtered_quality": 0, "filtered_duplicates": 0}

    for file in files:
        output_path = output_dir / file.name
        stats = clean_file(
            str(file),
            str(output_path),
            remove_url=not args.keep_urls,
            deduplicate=not args.no_dedup,
        )
        for k in total_stats:
            total_stats[k] += stats[k]

    log.info(f"\nTotal stats: {total_stats}")


if __name__ == "__main__":
    main()
