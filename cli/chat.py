"""
cli/chat.py
===========
Interactive terminal chat interface for the GPT LLM.

FEATURES
========
- Rich colored terminal UI
- Streaming token output (one token at a time)
- Multi-turn conversation history
- Context management (auto-truncation)
- Commands: /help, /clear, /stats, /quit
- Model loading from checkpoint

Usage:
    python cli/chat.py --checkpoint checkpoints/best.pt
    python cli/chat.py --checkpoint checkpoints/best.pt --temperature 0.9
    python cli/chat.py --checkpoint checkpoints/best.pt --no_stream

Example session:
    You: Tell me a story about a cat
    Assistant: Once upon a time, there was a curious cat named Whiskers...

    You: /stats
    [Context: 47 tokens | Turns: 2]

    You: /clear
    [Conversation cleared]
"""

import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt_model import GPTModel
from tokenizer.tokenizer_infer import LLMTokenizer
from inference.generate import generate, generate_streaming
from chat.chat_format import ConversationManager
from utils.config_loader import load_config, get_default_config
from utils.logger import get_logger

log = get_logger("cli_chat")


# ============================================================
# TERMINAL COLORS
# ============================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"


def colored(text: str, color: str) -> str:
    """Wrap text in ANSI color codes."""
    return f"{color}{text}{Colors.RESET}"


def print_banner():
    """Print the welcome banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
  ╔══════════════════════════════════════╗
  ║          GPT LLM Chat Interface      ║
  ║       Trained from scratch!          ║
  ╚══════════════════════════════════════╝
{Colors.RESET}
{Colors.DIM}Commands: /help  /clear  /stats  /quit{Colors.RESET}
"""
    print(banner)


def print_help():
    """Print available commands."""
    help_text = f"""
{Colors.YELLOW}Available Commands:{Colors.RESET}
  {Colors.CYAN}/help{Colors.RESET}    — Show this help message
  {Colors.CYAN}/clear{Colors.RESET}   — Clear conversation history
  {Colors.CYAN}/stats{Colors.RESET}   — Show context statistics
  {Colors.CYAN}/temp N{Colors.RESET}  — Set temperature (e.g. /temp 0.7)
  {Colors.CYAN}/quit{Colors.RESET}    — Exit the chat
  {Colors.CYAN}/exit{Colors.RESET}    — Exit the chat

{Colors.YELLOW}Generation Settings:{Colors.RESET}
  Temperature: Controls randomness (0=focused, 2=creative)
  Top-k: Limits token candidates
  Top-p: Nucleus sampling threshold
"""
    print(help_text)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_for_chat(
    checkpoint_path: str,
    config_path: str = None,
) -> tuple:
    """
    Load model and tokenizer for interactive chat.

    Args:
        checkpoint_path: Path to .pt checkpoint
        config_path    : Optional YAML config path

    Returns:
        Tuple of (model, tokenizer, device)
    """
    print(colored("Loading model...", Colors.DIM))

    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(colored(f"Device: {device}", Colors.DIM))

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(colored(f"Error: Checkpoint not found: {checkpoint_path}", Colors.RED))
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", {})

    # Build config
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = get_default_config("tiny")
        if ckpt_config:
            cfg = config.model
            for key in ["d_model", "n_layers", "n_heads", "d_ff", "vocab_size", "max_seq_len"]:
                if key in ckpt_config:
                    setattr(cfg, key, ckpt_config[key])

    # Load model
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = os.path.join(
        os.path.dirname(checkpoint_path), "..", "tokenizer", "saved", "tokenizer.json"
    )
    # Try standard location
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "tokenizer/saved/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(colored("Warning: Tokenizer not found. Using default path.", Colors.YELLOW))
        tokenizer_path = "tokenizer/saved/tokenizer.json"

    tokenizer = LLMTokenizer(tokenizer_path)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    step = checkpoint.get("step", 0)
    print(colored(
        f"Model loaded: {n_params:.1f}M params | "
        f"Trained for {step:,} steps",
        Colors.GREEN
    ))

    return model, tokenizer, device


# ============================================================
# CHAT LOOP
# ============================================================

def chat_loop(
    model: GPTModel,
    tokenizer: LLMTokenizer,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    max_tokens: int = 300,
    stream: bool = True,
    system_prompt: str = None,
):
    """
    Main interactive chat loop.

    Processes user input, generates responses, handles commands.

    Args:
        model             : Loaded GPTModel
        tokenizer         : LLMTokenizer
        device            : Compute device
        temperature       : Sampling temperature
        top_k             : Top-k filtering
        top_p             : Nucleus sampling
        repetition_penalty: Repetition penalty
        max_tokens        : Max tokens per response
        stream            : Stream tokens as generated
        system_prompt     : Optional system prompt
    """
    print_banner()

    # Initialize conversation manager
    conv = ConversationManager(
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        max_seq_len=model.config.max_seq_len,
    )

    print(colored(f"Generation settings:", Colors.DIM))
    print(colored(
        f"  temperature={temperature} | top_k={top_k} | top_p={top_p} | "
        f"max_tokens={max_tokens}",
        Colors.DIM
    ))
    print()

    while True:
        try:
            # ---- Get user input ----
            user_input = input(f"{Colors.BLUE}{Colors.BOLD}You: {Colors.RESET}").strip()

            if not user_input:
                continue

            # ---- Handle commands ----
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]

                if cmd in ("/quit", "/exit", "/q"):
                    print(colored("\nGoodbye!", Colors.CYAN))
                    break

                elif cmd == "/help":
                    print_help()
                    continue

                elif cmd == "/clear":
                    conv.clear()
                    print(colored("[Conversation cleared]", Colors.DIM))
                    continue

                elif cmd == "/stats":
                    ctx_len = conv.get_context_length()
                    turns = len(conv.history)
                    print(colored(
                        f"[Context: {ctx_len} tokens | Turns: {turns} | "
                        f"Max: {model.config.max_seq_len} tokens]",
                        Colors.DIM
                    ))
                    continue

                elif cmd == "/temp":
                    try:
                        temperature = float(user_input.split()[1])
                        print(colored(f"[Temperature set to {temperature}]", Colors.DIM))
                    except (IndexError, ValueError):
                        print(colored("Usage: /temp 0.7", Colors.YELLOW))
                    continue

                else:
                    print(colored(f"Unknown command: {user_input}. Type /help for help.", Colors.YELLOW))
                    continue

            # ---- Add user message ----
            conv.add_user(user_input)

            # ---- Build prompt ----
            prompt = conv.build_prompt()
            prompt_ids = tokenizer.encode(prompt, add_bos=True)
            prompt_tensor = torch.tensor([prompt_ids], device=device)

            # ---- Generate response ----
            print(f"{Colors.GREEN}{Colors.BOLD}Assistant: {Colors.RESET}", end="", flush=True)

            response_tokens = []

            if stream:
                # Streaming mode: print tokens as they're generated
                with torch.no_grad():
                    for token_text in generate_streaming(
                        model=model,
                        input_ids=prompt_tensor,
                        tokenizer=tokenizer,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=tokenizer.eos_id,
                        use_kv_cache=True,
                    ):
                        print(token_text, end="", flush=True)
                        response_tokens.append(token_text)

                response = "".join(response_tokens)
            else:
                # Non-streaming mode: generate all at once
                with torch.no_grad():
                    output_ids = generate(
                        model=model,
                        input_ids=prompt_tensor,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=tokenizer.eos_id,
                        use_kv_cache=True,
                    )

                # Decode full output and extract response
                full_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
                response = conv.extract_response(full_text)
                print(response, end="")

            print()  # New line after response
            print()  # Blank line between turns

            # ---- Save response to history ----
            # Extract clean response (stop at User:)
            clean_response = response.split("\nUser:")[0].strip()
            conv.add_assistant(clean_response)

        except KeyboardInterrupt:
            print(colored("\n\nInterrupted. Type /quit to exit.", Colors.YELLOW))
        except EOFError:
            print(colored("\nGoodbye!", Colors.CYAN))
            break


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with your trained GPT LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint (default: checkpoints/best.pt)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config (optional)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k filtering (default: 50, 0=disabled)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus sampling threshold (default: 0.95)"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.1,
        help="Repetition penalty (default: 1.1)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=300,
        help="Maximum tokens per response (default: 300)"
    )
    parser.add_argument(
        "--no_stream", action="store_true",
        help="Disable streaming output"
    )
    parser.add_argument(
        "--system", type=str, default=None,
        help="System prompt for the assistant"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_model_for_chat(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )

    # Start chat
    chat_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        stream=not args.no_stream,
        system_prompt=args.system,
    )


if __name__ == "__main__":
    main()
