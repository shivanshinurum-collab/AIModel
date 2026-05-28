"""
utils/metrics.py
================
Training metrics tracking for the GPT LLM.

Tracks:
- Training loss (smoothed exponential moving average)
- Validation loss
- Perplexity (exp(loss))
- Tokens per second throughput
- Learning rate schedule

Usage:
    tracker = MetricsTracker()
    tracker.update(step=100, loss=2.4, lr=3e-4)
    print(tracker.summary())
"""

import math
import time
from collections import deque
from typing import Optional, Dict, List


class MovingAverage:
    """
    Exponential moving average for smoothing training metrics.

    Math:
        EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}

    Args:
        alpha: Smoothing factor (0 < alpha <= 1). Higher = less smoothing.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._value: Optional[float] = None
        self._count = 0

    def update(self, value: float) -> float:
        """Update EMA with new value and return smoothed result."""
        self._count += 1
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self):
        self._value = None
        self._count = 0


class MetricsTracker:
    """
    Central metrics tracker for LLM training.

    Tracks training loss, validation loss, perplexity,
    throughput, and learning rate over time.
    Stores full history for plotting.

    Example:
        tracker = MetricsTracker()
        for step in range(max_steps):
            loss = train_one_step(...)
            tracker.update(step, loss=loss, lr=scheduler.get_lr())
        tracker.plot("logs/training_curves.png")
    """

    def __init__(self, window_size: int = 100):
        # EMA for smooth logging
        self.loss_ema = MovingAverage(alpha=0.1)

        # Full history for plotting
        self.steps: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_steps: List[int] = []
        self.lrs: List[float] = []
        self.perplexities: List[float] = []

        # Throughput tracking
        self._tokens_window: deque = deque(maxlen=window_size)
        self._time_window: deque = deque(maxlen=window_size)
        self._last_time = time.time()

        # Best validation metrics
        self.best_val_loss: float = float("inf")
        self.best_step: int = 0

    def update(
        self,
        step: int,
        loss: float,
        lr: float = 0.0,
        tokens_processed: int = 0,
    ):
        """
        Update training metrics for a step.

        Args:
            step           : Current training step number
            loss           : Raw training loss value
            lr             : Current learning rate
            tokens_processed: Tokens processed in this step (for throughput)
        """
        now = time.time()
        elapsed = now - self._last_time
        self._last_time = now

        # Store history
        self.steps.append(step)
        self.train_losses.append(loss)
        self.lrs.append(lr)

        # Update EMA
        self.loss_ema.update(loss)

        # Throughput
        if tokens_processed > 0 and elapsed > 0:
            self._tokens_window.append(tokens_processed)
            self._time_window.append(elapsed)

    def update_val(self, step: int, val_loss: float):
        """
        Record a validation loss measurement.

        Also updates best_val_loss and best_step tracking.
        """
        self.val_steps.append(step)
        self.val_losses.append(val_loss)

        # Perplexity = exp(loss)
        ppl = math.exp(min(val_loss, 30.0))  # clip to avoid overflow
        self.perplexities.append(ppl)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_step = step

    def tokens_per_second(self) -> Optional[float]:
        """Compute recent throughput in tokens/second."""
        if not self._tokens_window:
            return None
        total_tokens = sum(self._tokens_window)
        total_time = sum(self._time_window)
        if total_time == 0:
            return None
        return total_tokens / total_time

    @property
    def smoothed_loss(self) -> Optional[float]:
        """EMA-smoothed training loss."""
        return self.loss_ema.value

    @property
    def latest_val_perplexity(self) -> Optional[float]:
        """Most recent validation perplexity."""
        return self.perplexities[-1] if self.perplexities else None

    def summary(self) -> Dict:
        """Return a dictionary of the most recent metrics."""
        return {
            "step": self.steps[-1] if self.steps else 0,
            "train_loss": self.train_losses[-1] if self.train_losses else None,
            "smoothed_loss": self.smoothed_loss,
            "val_loss": self.val_losses[-1] if self.val_losses else None,
            "perplexity": self.latest_val_perplexity,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "lr": self.lrs[-1] if self.lrs else None,
            "tokens_per_sec": self.tokens_per_second(),
        }

    def __repr__(self) -> str:
        s = self.summary()
        parts = []
        if s["train_loss"] is not None:
            parts.append(f"loss={s['train_loss']:.4f}")
        if s["val_loss"] is not None:
            parts.append(f"val={s['val_loss']:.4f}")
        if s["perplexity"] is not None:
            parts.append(f"ppl={s['perplexity']:.1f}")
        if s["lr"] is not None:
            parts.append(f"lr={s['lr']:.2e}")
        return f"MetricsTracker({', '.join(parts)})"


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss)

    A perplexity of 1 means perfect prediction.
    A perplexity of V (vocab size) means random prediction.
    Lower is better.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity score
    """
    return math.exp(min(loss, 50.0))  # cap at exp(50) to prevent overflow
