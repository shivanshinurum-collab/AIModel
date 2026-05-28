"""
utils/visualization.py
=======================
Training metrics visualization for the GPT LLM project.

Generates:
- Training/validation loss curves
- Perplexity curves
- Learning rate schedule
- Combined dashboard plot

Also supports TensorBoard logging.

Usage:
    from utils.visualization import Visualizer
    viz = Visualizer(log_dir="logs")
    viz.plot_training_curves(tracker, save_path="logs/curves.png")
"""

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.metrics import MetricsTracker


# ============================================================
# MATPLOTLIB SETUP — defer import to avoid load-time cost
# ============================================================
def _get_plt():
    """Lazily import matplotlib to avoid loading it unless needed."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for server environments
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ============================================================
# TENSORBOARD WRITER
# ============================================================
class TensorBoardWriter:
    """
    Thin wrapper around TensorBoard SummaryWriter.
    Gracefully degrades if tensorboard is not installed.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir)
            print(f"[TensorBoard] Logging to: {log_dir}")
            print(f"[TensorBoard] View with: tensorboard --logdir={log_dir}")
        except ImportError:
            print("[TensorBoard] Not available. Install with: pip install tensorboard")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar metric."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: dict, step: int):
        """Log multiple scalars at once."""
        for tag, value in metrics.items():
            if value is not None:
                self.log_scalar(tag, value, step)

    def close(self):
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()


# ============================================================
# MATPLOTLIB VISUALIZATION
# ============================================================
class Visualizer:
    """
    Generates training curve plots using matplotlib.

    Typical usage:
        viz = Visualizer(log_dir="logs")
        viz.plot_training_curves(tracker)    # saves to log_dir/training_curves.png
        viz.plot_lr_schedule(steps, lrs)     # saves to log_dir/lr_schedule.png
    """

    # Color scheme
    COLORS = {
        "train_loss": "#4E9AF1",      # bright blue
        "val_loss": "#FF6B6B",        # coral red
        "perplexity": "#FFD93D",      # amber
        "lr": "#6BCB77",              # green
        "grid": "#2A2A2A",
        "background": "#1A1A2E",
        "text": "#E0E0E0",
    }

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def plot_training_curves(
        self,
        tracker: "MetricsTracker",
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """
        Generate a 2x2 dashboard with:
        - Training loss curve (with EMA overlay)
        - Validation loss curve
        - Perplexity
        - Learning rate schedule

        Args:
            tracker  : MetricsTracker with training history
            save_path: Where to save the plot (default: logs/training_curves.png)
            show     : Whether to display interactively (requires GUI)
        """
        plt = _get_plt()
        if plt is None:
            print("[Visualization] matplotlib not available, skipping plot")
            return

        if not tracker.steps:
            print("[Visualization] No data to plot yet")
            return

        save_path = save_path or os.path.join(self.log_dir, "training_curves.png")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(self.COLORS["background"])

        for ax in axes.flat:
            ax.set_facecolor("#16213E")
            ax.tick_params(colors=self.COLORS["text"])
            ax.xaxis.label.set_color(self.COLORS["text"])
            ax.yaxis.label.set_color(self.COLORS["text"])
            ax.title.set_color(self.COLORS["text"])
            for spine in ax.spines.values():
                spine.set_edgecolor("#333366")
            ax.grid(True, color=self.COLORS["grid"], linewidth=0.5, alpha=0.7)

        # ---- Plot 1: Training Loss ----
        ax = axes[0, 0]
        ax.plot(
            tracker.steps, tracker.train_losses,
            color=self.COLORS["train_loss"], alpha=0.3, linewidth=0.8, label="Raw loss"
        )
        # Compute smoothed loss from raw values for display
        if len(tracker.train_losses) > 1:
            import numpy as np
            ema_alpha = 0.1
            smoothed = [tracker.train_losses[0]]
            for v in tracker.train_losses[1:]:
                smoothed.append(ema_alpha * v + (1 - ema_alpha) * smoothed[-1])
            ax.plot(
                tracker.steps, smoothed,
                color=self.COLORS["train_loss"], linewidth=2.0, label="EMA loss"
            )
        ax.set_title("Training Loss", fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend(facecolor="#1A1A2E", labelcolor=self.COLORS["text"])

        # ---- Plot 2: Validation Loss ----
        ax = axes[0, 1]
        if tracker.val_steps:
            ax.plot(
                tracker.val_steps, tracker.val_losses,
                color=self.COLORS["val_loss"], linewidth=2.0, marker="o",
                markersize=4, label="Val loss"
            )
            # Mark best
            best_idx = tracker.val_losses.index(min(tracker.val_losses))
            ax.scatter(
                [tracker.val_steps[best_idx]], [tracker.val_losses[best_idx]],
                color="#FFD93D", s=100, zorder=5, label=f"Best: {tracker.best_val_loss:.4f}"
            )
            ax.legend(facecolor="#1A1A2E", labelcolor=self.COLORS["text"])
        ax.set_title("Validation Loss", fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-Entropy Loss")

        # ---- Plot 3: Perplexity ----
        ax = axes[1, 0]
        if tracker.perplexities:
            ax.plot(
                tracker.val_steps, tracker.perplexities,
                color=self.COLORS["perplexity"], linewidth=2.0, marker="s", markersize=4
            )
        ax.set_title("Validation Perplexity", fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity (lower is better)")

        # ---- Plot 4: Learning Rate ----
        ax = axes[1, 1]
        if tracker.lrs:
            ax.plot(
                tracker.steps, tracker.lrs,
                color=self.COLORS["lr"], linewidth=1.5
            )
        ax.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        fig.suptitle(
            "LLM Training Dashboard",
            fontsize=16, fontweight="bold",
            color=self.COLORS["text"], y=1.01
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Visualization] Saved training curves → {save_path}")

        if show:
            plt.show()
        plt.close(fig)

    def plot_lr_schedule(
        self,
        max_steps: int,
        warmup_steps: int,
        max_lr: float,
        min_lr: float,
        save_path: Optional[str] = None,
    ):
        """
        Preview the cosine LR schedule before training.

        Shows warmup phase + cosine decay phase.
        """
        plt = _get_plt()
        if plt is None:
            return

        import math
        steps = list(range(max_steps))
        lrs = []
        for step in steps:
            if step < warmup_steps:
                lr = max_lr * step / max(1, warmup_steps)
            else:
                progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
            lrs.append(lr)

        save_path = save_path or os.path.join(self.log_dir, "lr_schedule.png")
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(self.COLORS["background"])
        ax.set_facecolor("#16213E")
        ax.plot(steps, lrs, color=self.COLORS["lr"], linewidth=2.0)
        ax.axvline(warmup_steps, color="#FF6B6B", linestyle="--", alpha=0.7, label="Warmup end")
        ax.set_title("Cosine LR Schedule with Warmup", color=self.COLORS["text"], fontweight="bold")
        ax.set_xlabel("Step", color=self.COLORS["text"])
        ax.set_ylabel("Learning Rate", color=self.COLORS["text"])
        ax.tick_params(colors=self.COLORS["text"])
        ax.legend(facecolor="#1A1A2E", labelcolor=self.COLORS["text"])
        ax.grid(True, color=self.COLORS["grid"], linewidth=0.5, alpha=0.7)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Visualization] Saved LR schedule → {save_path}")
        plt.close(fig)
