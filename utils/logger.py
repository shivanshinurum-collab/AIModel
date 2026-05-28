"""
utils/logger.py
===============
Logging system for the GPT LLM project.

Features:
- Rich-formatted console output with colors and icons
- File-based logging (plain text)
- Log level support: DEBUG, INFO, WARNING, ERROR
- Context-aware named loggers (per-module)
- Training-specific helpers (step, loss, metrics)

Usage:
    from utils.logger import get_logger
    log = get_logger("trainer")
    log.info("Training started")
    log.step(100, loss=2.34, lr=3e-4)
"""

import os
import sys
import logging
import datetime
from typing import Optional


# ============================================================
# RICH CONSOLE SETUP
# ============================================================
try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich import print as rprint
    RICH_AVAILABLE = True
    _console = Console(stderr=True)
except ImportError:
    RICH_AVAILABLE = False
    _console = None


# ============================================================
# GLOBAL LOG REGISTRY — avoid creating duplicate handlers
# ============================================================
_loggers: dict[str, "LLMLogger"] = {}


class LLMLogger:
    """
    Custom logger that wraps Python's logging module with:
    - Rich-formatted console output (if rich is installed)
    - Plain text file logging
    - Training-specific convenience methods
    """

    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        self.name = name
        self.log_dir = log_dir
        self._logger = self._setup(name, log_dir, level)

    def _setup(self, name: str, log_dir: str, level: int) -> logging.Logger:
        """Configure handlers for the underlying Python logger."""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Avoid adding duplicate handlers on re-initialization
        if logger.handlers:
            return logger

        fmt = "%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        # ----- Console handler -----
        if RICH_AVAILABLE:
            console_handler = RichHandler(
                console=_console,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
            console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(fmt, datefmt))
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # ----- File handler -----
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(fmt, datefmt))
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        return logger

    # ---- Standard log levels ----
    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    # ---- Training-specific helpers ----
    def step(
        self,
        step: int,
        loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        perplexity: Optional[float] = None,
    ):
        """Log a training step with key metrics."""
        parts = [f"step={step:>6d}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if perplexity is not None:
            parts.append(f"ppl={perplexity:.2f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        if tokens_per_sec is not None:
            parts.append(f"tok/s={tokens_per_sec:.0f}")
        self._logger.info("  ".join(parts))

    def section(self, title: str):
        """Print a section divider."""
        sep = "=" * 60
        self._logger.info(f"\n{sep}\n  {title}\n{sep}")

    def config_summary(self, config_str: str):
        """Log a configuration summary block."""
        self._logger.info(f"\n--- Config ---\n{config_str}\n--------------")

    def checkpoint_saved(self, path: str, step: int):
        self._logger.info(f"[CHECKPOINT] Saved step={step:>6d} → {path}")

    def checkpoint_loaded(self, path: str, step: int):
        self._logger.info(f"[CHECKPOINT] Loaded step={step:>6d} ← {path}")


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> LLMLogger:
    """
    Get or create a named logger.

    Args:
        name    : Logger name (e.g. 'trainer', 'tokenizer')
        log_dir : Directory for log files (default: 'logs')
        level   : Logging level (default: logging.INFO)

    Returns:
        LLMLogger instance
    """
    global _loggers
    if name not in _loggers:
        _loggers[name] = LLMLogger(name, log_dir=log_dir, level=level)
    return _loggers[name]
