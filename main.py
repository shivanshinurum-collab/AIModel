"""
main.py
=======
Quick-start entry point for the GPT LLM project.

Provides a unified interface to:
- Download datasets
- Train tokenizer
- Train model
- Chat interactively
- Start API server
- Run tests

Usage:
    python main.py setup          # Download data + train tokenizer
    python main.py train          # Train tiny model
    python main.py chat           # Chat with trained model
    python main.py server         # Start API server
    python main.py test           # Run all tests
    python main.py generate --prompt "Once upon a time"
"""

import os
import sys
import argparse
from pathlib import Path

# ============================================================
# COLORFUL BANNER
# ============================================================

BANNER = """
\033[96m\033[1m
  ██████╗ ██████╗ ████████╗    ██╗     ██╗     ███╗   ███╗
 ██╔════╝ ██╔══██╗╚══██╔══╝    ██║     ██║     ████╗ ████║
 ██║  ███╗██████╔╝   ██║       ██║     ██║     ██╔████╔██║
 ██║   ██║██╔═══╝    ██║       ██║     ██║     ██║╚██╔╝██║
 ╚██████╔╝██║        ██║       ███████╗███████╗██║ ╚═╝ ██║
  ╚═════╝ ╚═╝        ╚═╝       ╚══════╝╚══════╝╚═╝     ╚═╝
\033[0m
\033[93m  GPT-Style Language Model — Built From Scratch\033[0m
\033[2m  PyTorch | BPE Tokenizer | Transformer | FastAPI\033[0m
"""


def print_banner():
    print(BANNER)


# ============================================================
# COMMANDS
# ============================================================

def cmd_setup(args):
    """Download datasets and train tokenizer."""
    print("\n\033[92m[1/3] Downloading TinyStories dataset...\033[0m")
    os.system(
        f"python datasets/download_dataset.py "
        f"--datasets tinystories --max_samples {args.max_samples}"
    )

    print("\n\033[92m[2/3] Cleaning and merging dataset...\033[0m")
    os.system("python datasets/clean_dataset.py --input datasets/raw/tinystories.txt")
    os.system("python datasets/merge_dataset.py")

    print("\n\033[92m[3/3] Training BPE tokenizer...\033[0m")
    os.system(f"python tokenizer/tokenizer_train.py --vocab_size {args.vocab_size}")

    print("\n\033[92m✓ Setup complete!\033[0m")
    print("Next: python main.py train")


def cmd_train(args):
    """Train the GPT model."""
    cmd = f"python training/train.py --config {args.config}"
    if args.max_steps:
        cmd += f" --max_steps {args.max_steps}"
    if args.resume:
        cmd += f" --resume {args.resume}"
    print(f"\n\033[92mStarting training...\033[0m")
    print(f"Config: {args.config}")
    os.system(cmd)


def cmd_chat(args):
    """Start interactive chat."""
    checkpoint = args.checkpoint or "checkpoints/best.pt"
    if not os.path.exists(checkpoint):
        print(f"\033[91mError: Checkpoint not found: {checkpoint}\033[0m")
        print("Train a model first: python main.py train")
        sys.exit(1)

    cmd = f"python cli/chat.py --checkpoint {checkpoint}"
    if args.temperature:
        cmd += f" --temperature {args.temperature}"
    os.system(cmd)


def cmd_server(args):
    """Start FastAPI inference server."""
    checkpoint = args.checkpoint or "checkpoints/best.pt"
    if not os.path.exists(checkpoint):
        print(f"\033[91mError: Checkpoint not found: {checkpoint}\033[0m")
        sys.exit(1)

    cmd = (
        f"python api/server.py "
        f"--checkpoint {checkpoint} "
        f"--port {args.port}"
    )
    print(f"\n\033[92mStarting API server on port {args.port}...\033[0m")
    print(f"API docs: http://localhost:{args.port}/docs")
    os.system(cmd)


def cmd_generate(args):
    """Quick text generation from CLI."""
    import torch
    from utils.config_loader import load_config, get_default_config
    from model.gpt_model import GPTModel
    from tokenizer.tokenizer_infer import LLMTokenizer
    from inference.generate import generate

    checkpoint = args.checkpoint or "checkpoints/best.pt"
    if not os.path.exists(checkpoint):
        print(f"\033[91mCheckpoint not found: {checkpoint}\033[0m")
        sys.exit(1)

    device = torch.device("cpu")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    ckpt_config = checkpoint_data.get("config", {})

    config = get_default_config("tiny")
    if ckpt_config:
        for key in ["d_model", "n_layers", "n_heads", "d_ff", "vocab_size", "max_seq_len"]:
            if key in ckpt_config:
                setattr(config.model, key, ckpt_config[key])

    model = GPTModel(config)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    tokenizer = LLMTokenizer("tokenizer/saved/tokenizer.json")

    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    print(f"\033[94mPrompt:\033[0m {args.prompt}")
    print(f"\033[92mGenerating...\033[0m")

    with torch.no_grad():
        output_ids = generate(
            model, prompt_tensor,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    generated = tokenizer.decode(
        output_ids[0, len(prompt_ids):].tolist(),
        skip_special_tokens=True
    )
    print(f"\033[92mOutput:\033[0m {generated}")


def cmd_test(args):
    """Run the test suite."""
    cmd = "python -m pytest tests/ -v"
    if args.test_file:
        cmd = f"python -m pytest {args.test_file} -v"
    os.system(cmd)


def cmd_info(args):
    """Show project info and status."""
    print("\n\033[96mProject Status:\033[0m")

    checks = [
        ("Raw dataset", "datasets/raw/tinystories.txt"),
        ("Cleaned dataset", "datasets/cleaned/tinystories.txt"),
        ("Train set", "datasets/processed/train.txt"),
        ("Val set", "datasets/processed/val.txt"),
        ("Tokenizer", "tokenizer/saved/tokenizer.json"),
        ("Best checkpoint", "checkpoints/best.pt"),
    ]

    for label, path in checks:
        exists = os.path.exists(path)
        icon = "✓" if exists else "✗"
        color = "\033[92m" if exists else "\033[91m"
        size = ""
        if exists:
            size_bytes = os.path.getsize(path)
            if size_bytes > 1e9:
                size = f"({size_bytes/1e9:.1f} GB)"
            elif size_bytes > 1e6:
                size = f"({size_bytes/1e6:.0f} MB)"
            elif size_bytes > 1e3:
                size = f"({size_bytes/1e3:.0f} KB)"
        print(f"  {color}{icon}\033[0m  {label:20s} {size}")

    print()
    print("\033[96mQuick commands:\033[0m")
    print("  python main.py setup          # Download data + train tokenizer")
    print("  python main.py train          # Train tiny model (~7M params)")
    print("  python main.py chat           # Chat with trained model")
    print("  python main.py server         # Start FastAPI server")
    print("  python main.py test           # Run test suite")


# ============================================================
# ARGUMENT PARSER
# ============================================================

def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="GPT LLM — trained from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # setup
    p_setup = subparsers.add_parser("setup", help="Download data + train tokenizer")
    p_setup.add_argument("--max_samples", type=int, default=100000)
    p_setup.add_argument("--vocab_size", type=int, default=8000)

    # train
    p_train = subparsers.add_parser("train", help="Train the GPT model")
    p_train.add_argument("--config", type=str, default="configs/model_tiny.yaml")
    p_train.add_argument("--max_steps", type=int, default=None)
    p_train.add_argument("--resume", type=str, default=None)

    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--checkpoint", type=str, default=None)
    p_chat.add_argument("--temperature", type=float, default=0.8)

    # server
    p_server = subparsers.add_parser("server", help="Start API server")
    p_server.add_argument("--checkpoint", type=str, default=None)
    p_server.add_argument("--port", type=int, default=8000)

    # generate
    p_gen = subparsers.add_parser("generate", help="Quick text generation")
    p_gen.add_argument("--prompt", type=str, required=True)
    p_gen.add_argument("--checkpoint", type=str, default=None)
    p_gen.add_argument("--max_tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=0.8)

    # test
    p_test = subparsers.add_parser("test", help="Run test suite")
    p_test.add_argument("--test_file", type=str, default=None)

    # info
    subparsers.add_parser("info", help="Show project status")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "server":
        cmd_server(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        print("\n")
        cmd_info(args)


if __name__ == "__main__":
    main()