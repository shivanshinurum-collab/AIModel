"""
training/train.py
=================
Training entry point for the GPT-style LLM.

Usage:
    # Train tiny model from scratch
    python training/train.py --config configs/model_tiny.yaml

    # Resume from checkpoint
    python training/train.py --config configs/model_tiny.yaml \\
        --resume checkpoints/step_0001000.pt

    # Override specific hyperparameters
    python training/train.py --config configs/model_small.yaml \\
        --max_steps 5000 --batch_size 16 --lr 2e-4

    # Quick sanity check (100 steps)
    python training/train.py --config configs/model_tiny.yaml --max_steps 100

Quick Start:
    # 1. Download data
    python datasets/download_dataset.py --datasets tinystories --max_samples 50000

    # 2. Clean data
    python datasets/clean_dataset.py --input datasets/raw/tinystories.txt

    # 3. Merge to train/val
    python datasets/merge_dataset.py

    # 4. Train tokenizer
    python tokenizer/tokenizer_train.py --vocab_size 8000

    # 5. Train model
    python training/train.py --config configs/model_tiny.yaml
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config, get_default_config
from utils.logger import get_logger
from training.trainer import Trainer

log = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GPT-style LLM from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config
    parser.add_argument(
        "--config", type=str, default="configs/model_tiny.yaml",
        help="Path to YAML config file (default: configs/model_tiny.yaml)"
    )

    # Resume
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )

    # Override hyperparameters (override config values)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", "--learning_rate", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)

    # Data/model overrides
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)

    # Misc
    parser.add_argument(
        "--no_compile", action="store_true",
        help="Disable torch.compile even if config enables it"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Set random seeds ----
    import torch
    import random
    import numpy as np

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Load config ----
    if os.path.exists(args.config):
        log.info(f"Loading config: {args.config}")
        config = load_config(args.config)
    else:
        log.warning(f"Config not found: {args.config}")
        log.warning("Using default tiny config")
        config = get_default_config("tiny")

    # ---- Apply CLI overrides ----
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
        log.info(f"Override: max_steps = {args.max_steps}")

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        log.info(f"Override: batch_size = {args.batch_size}")

    if args.lr is not None:
        config.training.learning_rate = args.lr
        log.info(f"Override: learning_rate = {args.lr}")

    if args.grad_accum is not None:
        config.training.gradient_accumulation_steps = args.grad_accum
        log.info(f"Override: gradient_accumulation_steps = {args.grad_accum}")

    if args.eval_interval is not None:
        config.training.eval_interval = args.eval_interval

    if args.save_interval is not None:
        config.training.save_interval = args.save_interval

    if args.train_data is not None:
        config.data.dataset_path = args.train_data

    if args.val_data is not None:
        config.data.val_path = args.val_data

    if args.tokenizer is not None:
        config.data.tokenizer_path = args.tokenizer

    if args.no_compile:
        config.training.compile_model = False

    # ---- Visualize LR schedule ----
    from utils.visualization import Visualizer
    import os
    os.makedirs(config.logging.log_dir, exist_ok=True)
    viz = Visualizer(log_dir=config.logging.log_dir)
    viz.plot_lr_schedule(
        max_steps=config.training.max_steps,
        warmup_steps=config.training.warmup_steps,
        max_lr=config.training.learning_rate,
        min_lr=config.training.min_lr,
    )

    # ---- Train ----
    trainer = Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
