"""
training/trainer.py
===================
Full training loop for GPT-style language model.

TRAINING FLOW
=============

1. Load data → create DataLoader
2. Initialize model → move to device
3. Initialize optimizer (AdamW) + LR scheduler (cosine with warmup)
4. Training loop:
   a. Sample batch (x, y)
   b. Forward pass → logits, loss
   c. Scale loss by 1/gradient_accumulation_steps
   d. Backward pass (compute gradients)
   e. Every N steps: clip gradients, optimizer step, zero grads
   f. Every eval_interval: run validation, log metrics
   g. Every save_interval: save checkpoint

GRADIENT ACCUMULATION
=====================
To simulate larger batch sizes without more memory:
    effective_batch = batch_size * gradient_accumulation_steps

We accumulate gradients over multiple micro-batches before taking an
optimizer step. This allows training with larger effective batch sizes
on limited hardware.

MIXED PRECISION TRAINING
=========================
Uses torch.amp (Automatic Mixed Precision) to run forward passes in
fp16/bf16, while keeping optimizer state in fp32.

Benefits:
- ~2x memory reduction
- ~2x speedup on GPUs with tensor cores (V100, A100, RTX 30xx+)
- No loss in model quality (with careful loss scaling)

LEARNING RATE SCHEDULE
======================
Cosine decay with linear warmup:

    if step < warmup_steps:
        lr = max_lr * step / warmup_steps     ← linear warmup

    else:
        progress = (step - warmup) / (max_steps - warmup)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))

The warmup prevents early large gradient updates from destabilizing training.
The cosine decay smoothly reduces LR as training converges.

CHECKPOINTING
=============
Saves:
    - Model state dict
    - Optimizer state dict
    - Scheduler state
    - Training step / loss history
    - Config

Supports resuming from any checkpoint.
"""

import os
import sys
import math
import time
import glob
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from contextlib import nullcontext
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt_model import GPTModel
from training.dataloader import create_dataloader, InfiniteDataLoader
from tokenizer.tokenizer_infer import LLMTokenizer
from utils.config_loader import ModelConfig
from utils.logger import get_logger
from utils.metrics import MetricsTracker
from utils.visualization import Visualizer, TensorBoardWriter

log = get_logger("trainer")


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """
    Compute learning rate at a given training step.

    Uses cosine decay with linear warmup.

    Args:
        step        : Current training step (0-indexed)
        max_lr      : Peak learning rate
        min_lr      : Minimum learning rate (at end of training)
        warmup_steps: Steps for linear warmup phase
        max_steps   : Total training steps

    Returns:
        Learning rate for this step

    Example:
        lr = get_lr(1000, max_lr=3e-4, min_lr=3e-5, warmup=500, max_steps=10000)
    """
    # Linear warmup phase
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)

    # After training completes, return min_lr
    if step >= max_steps:
        return min_lr

    # Cosine decay phase
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    # Cosine curve: 1 at progress=0, 0 at progress=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.

    Saves:
        checkpoint_dir/step_{N}.pt  — per-interval checkpoints
        checkpoint_dir/best.pt       — best validation loss checkpoint

    Keeps only the last N checkpoints to save disk space.
    """

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
        config: ModelConfig,
        is_best: bool = False,
        val_loss: Optional[float] = None,
        metrics_history: Optional[dict] = None,
    ) -> str:
        """
        Save a training checkpoint.

        Checkpoint contains:
            - model_state_dict: Model weights
            - optimizer_state_dict: Adam moment buffers (for resume)
            - step: Training step
            - loss: Last training loss
            - val_loss: Validation loss (if available)
            - config: Model + training config (for reconstruction)

        Args:
            model    : GPTModel instance
            optimizer: AdamW optimizer
            step     : Current training step
            loss     : Current training loss
            config   : Full ModelConfig
            is_best  : If True, also save as 'best.pt'
            val_loss : Validation loss (for best model tracking)

        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "val_loss": val_loss,
            "config": {
                # Save enough info to reconstruct the model
                "d_model": config.model.d_model,
                "n_layers": config.model.n_layers,
                "n_heads": config.model.n_heads,
                "d_ff": config.model.d_ff,
                "vocab_size": config.model.vocab_size,
                "max_seq_len": config.model.max_seq_len,
                "dropout": config.model.dropout,
                "bias": config.model.bias,
                "model_name": config.model.name,
            },
            "metrics_history": metrics_history,
        }

        # Step checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, f"step_{step:07d}.pt")
        torch.save(checkpoint, ckpt_path)
        log.checkpoint_saved(ckpt_path, step)

        # Best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            log.info(f"[CHECKPOINT] New best model! val_loss={val_loss:.4f}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return ckpt_path

    def load(
        self,
        model: GPTModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
    ) -> int:
        """
        Load a checkpoint into model (and optionally optimizer).

        Args:
            model           : GPTModel to load weights into
            optimizer       : Optional optimizer to restore state
            checkpoint_path : Specific checkpoint to load (None = latest)
            load_best       : Load 'best.pt' instead of latest

        Returns:
            Step number from checkpoint (for resuming training)
        """
        if load_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best.pt")
        elif checkpoint_path is None:
            # Find the latest checkpoint
            checkpoint_path = self._get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            log.warning("No checkpoint found. Starting from scratch.")
            return 0

        log.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        step = checkpoint.get("step", 0)
        val_loss = checkpoint.get("val_loss", None)
        log.checkpoint_loaded(checkpoint_path, step)

        if val_loss is not None:
            log.info(f"  Checkpoint val_loss: {val_loss:.4f}")

        return step

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Find the checkpoint with the highest step number."""
        pattern = os.path.join(self.checkpoint_dir, "step_*.pt")
        checkpoints = sorted(glob.glob(pattern))
        return checkpoints[-1] if checkpoints else None

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        pattern = os.path.join(self.checkpoint_dir, "step_*.pt")
        checkpoints = sorted(glob.glob(pattern))
        # Remove oldest if we exceed keep_last_n
        while len(checkpoints) > self.keep_last_n:
            old = checkpoints.pop(0)
            os.remove(old)
            log.debug(f"Removed old checkpoint: {old}")


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    """
    Full training loop for GPT-style language model.

    Handles:
    - DataLoader setup
    - Device selection (CUDA / MPS / CPU)
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Gradient clipping
    - Validation loop
    - Checkpointing
    - Metrics logging
    - TensorBoard logging

    Usage:
        config = load_config("configs/model_tiny.yaml")
        trainer = Trainer(config)
        trainer.train()

    Or to resume from checkpoint:
        trainer = Trainer(config)
        trainer.train(resume_from="checkpoints/step_0001000.pt")
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.cfg_model = config.model
        self.cfg_train = config.training
        self.cfg_data = config.data
        self.cfg_log = config.logging
        self.cfg_ckpt = config.checkpoint

        # ---- Device setup ----
        self.device = self._get_device()
        self.device_type = "cuda" if "cuda" in str(self.device) else \
                           "mps" if "mps" in str(self.device) else "cpu"
        log.info(f"Device: {self.device}")

        # ---- Mixed precision setup ----
        # bf16 preferred on A100/RTX 30xx; fp16 on older GPUs; none on CPU
        self.dtype = torch.bfloat16 if (
            self.device_type == "cuda" and
            torch.cuda.is_bf16_supported()
        ) else torch.float16 if self.device_type == "cuda" else torch.float32

        self.autocast_ctx = (
            torch.amp.autocast(device_type=self.device_type, dtype=self.dtype)
            if self.device_type in ("cuda", "mps") and self.cfg_train.mixed_precision
            else nullcontext()
        )

        # GradScaler for fp16 (not needed for bf16 which doesn't underflow)
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.device_type == "cuda" and self.dtype == torch.float16
            else None
        )

        log.info(f"Mixed precision: {self.cfg_train.mixed_precision} | dtype: {self.dtype}")

    def _get_device(self) -> torch.device:
        """
        Automatically detect best available device.

        Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
            log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Apple Silicon MPS backend enabled")
        else:
            device = torch.device("cpu")
            log.info("Running on CPU (training will be slow)")
        return device

    def setup(self, resume_from: Optional[str] = None) -> int:
        """
        Initialize all components for training.

        Sets up:
        - Tokenizer
        - DataLoaders (train + val)
        - Model
        - Optimizer
        - Checkpoint manager
        - Metrics tracker
        - Visualization

        Args:
            resume_from: Path to checkpoint to resume from

        Returns:
            Starting step (0 if training from scratch, else checkpoint step)
        """
        log.section("Training Setup")
        log.config_summary(self.config.summary())

        # ---- Tokenizer ----
        log.info("Loading tokenizer...")
        self.tokenizer = LLMTokenizer(self.cfg_data.tokenizer_path)
        log.info(f"Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")

        # ---- Dataloaders ----
        log.info("Creating dataloaders...")
        train_loader = create_dataloader(
            text_path=self.cfg_data.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.cfg_model.max_seq_len,
            batch_size=self.cfg_train.batch_size,
            split="train",
            num_workers=self.cfg_data.num_workers,
            pin_memory=(self.device_type == "cuda"),
        )
        val_loader = create_dataloader(
            text_path=self.cfg_data.val_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.cfg_model.max_seq_len,
            batch_size=self.cfg_train.batch_size,
            split="val",
            num_workers=self.cfg_data.num_workers,
            pin_memory=(self.device_type == "cuda"),
        )
        self.train_loader = InfiniteDataLoader(train_loader)
        self.val_loader = val_loader

        # ---- Model ----
        log.info("Initializing model...")
        self.model = GPTModel(self.config)
        self.model = self.model.to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"Model parameters: {n_params:,}")
        log.info(str(self.model))

        # Optional: torch.compile for ~20% speedup (PyTorch 2.0+)
        if self.cfg_train.compile_model:
            log.info("Compiling model with torch.compile (may take a few minutes)...")
            self.model = torch.compile(self.model)

        # ---- Optimizer ----
        self.optimizer = self.model.configure_optimizers(
            learning_rate=self.cfg_train.learning_rate,
            weight_decay=self.cfg_train.weight_decay,
            device_type=self.device_type,
        )

        # ---- Checkpoint Manager ----
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=self.cfg_ckpt.checkpoint_dir,
            keep_last_n=self.cfg_ckpt.keep_last_n,
        )

        # ---- Metrics ----
        self.metrics = MetricsTracker()

        # ---- Visualization ----
        self.viz = Visualizer(log_dir=self.cfg_log.log_dir)
        self.tb_writer = (
            TensorBoardWriter(log_dir=self.cfg_log.log_dir)
            if self.cfg_log.tensorboard else None
        )

        # ---- Resume from checkpoint ----
        start_step = 0
        if resume_from:
            start_step = self.ckpt_manager.load(
                self.model, self.optimizer, checkpoint_path=resume_from
            )

        return start_step

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> float:
        """
        Run validation loop and return average validation loss.

        Uses no_grad and eval mode for efficiency.

        Args:
            max_batches: Maximum validation batches to evaluate
                         (full validation can be slow for large datasets)

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for i, (x, y) in enumerate(self.val_loader):
            if i >= max_batches:
                break

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with self.autocast_ctx:
                _, _, loss = self.model(x, targets=y)

            total_loss += loss.item()
            n_batches += 1

        self.model.train()
        return total_loss / max(1, n_batches)

    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop.

        Gradient Accumulation:
            The effective batch size is batch_size * gradient_accumulation_steps.
            We accumulate gradients over N micro-batches before each optimizer step.
            This simulates training with a larger batch using the same GPU memory.

        Args:
            resume_from: Checkpoint path to resume training from
        """
        start_step = self.setup(resume_from=resume_from)

        cfg_train = self.cfg_train
        accum_steps = cfg_train.gradient_accumulation_steps

        log.section("Training Started")
        log.info(
            f"Max steps: {cfg_train.max_steps:,} | "
            f"Effective batch: {cfg_train.batch_size * accum_steps}"
        )

        t0 = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        for step in range(start_step, cfg_train.max_steps):

            # ---- Learning Rate Scheduling ----
            # Update LR at every step
            lr = get_lr(
                step=step,
                max_lr=cfg_train.learning_rate,
                min_lr=cfg_train.min_lr,
                warmup_steps=cfg_train.warmup_steps,
                max_steps=cfg_train.max_steps,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # ---- Gradient Accumulation Loop ----
            # Accumulate gradients over accum_steps micro-batches
            total_loss = 0.0

            for micro_step in range(accum_steps):
                x, y = next(self.train_loader)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # Forward pass (with mixed precision)
                with self.autocast_ctx:
                    _, _, loss = self.model(x, targets=y)
                    # Scale loss for gradient accumulation
                    # Without scaling, gradients would be N times too large
                    loss = loss / accum_steps

                total_loss += loss.item()

                # Backward pass
                if self.scaler is not None:
                    # FP16: scale loss to prevent underflow
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            # ---- Optimizer Step ----
            # Unscale gradients, clip, step
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping: prevents gradient explosion
            # Clips the global gradient norm to grad_clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                cfg_train.grad_clip,
            )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory

            # ---- Throughput Calculation ----
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_processed = (
                cfg_train.batch_size * accum_steps * self.cfg_model.max_seq_len
            )

            # ---- Metrics Update ----
            self.metrics.update(
                step=step,
                loss=total_loss,
                lr=lr,
                tokens_processed=tokens_processed,
            )

            # ---- Logging ----
            if step % self.cfg_log.log_interval == 0:
                tok_per_sec = self.metrics.tokens_per_second()
                log.step(
                    step=step,
                    loss=total_loss,
                    lr=lr,
                    tokens_per_sec=tok_per_sec,
                )

                if self.tb_writer:
                    self.tb_writer.log_scalars({
                        "train/loss": total_loss,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "throughput/tokens_per_sec": tok_per_sec or 0,
                    }, step=step)

            # ---- Validation ----
            if step > 0 and step % cfg_train.eval_interval == 0:
                val_loss = self.evaluate()
                is_best = val_loss < self.metrics.best_val_loss

                from utils.metrics import compute_perplexity
                ppl = compute_perplexity(val_loss)
                log.step(step=step, val_loss=val_loss, perplexity=ppl)

                self.metrics.update_val(step=step, val_loss=val_loss)

                if self.tb_writer:
                    self.tb_writer.log_scalars({
                        "val/loss": val_loss,
                        "val/perplexity": ppl,
                    }, step=step)

                # Update metrics chart
                self.viz.plot_training_curves(
                    self.metrics,
                    save_path=os.path.join(self.cfg_log.log_dir, "training_curves.png"),
                )

            # ---- Checkpointing ----
            if step > 0 and step % cfg_train.save_interval == 0:
                val_loss = self.metrics.val_losses[-1] if self.metrics.val_losses else None
                is_best = (val_loss is not None and val_loss <= self.metrics.best_val_loss)

                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=step,
                    loss=total_loss,
                    config=self.config,
                    is_best=is_best,
                    val_loss=val_loss,
                    metrics_history={
                        "steps": self.metrics.steps,
                        "train_losses": self.metrics.train_losses,
                        "val_losses": self.metrics.val_losses,
                        "val_steps": self.metrics.val_steps,
                    },
                )

        # ---- Training Complete ----
        log.section("Training Complete")
        log.info(f"Best val loss: {self.metrics.best_val_loss:.4f} at step {self.metrics.best_step}")
        log.info(f"Final training curves saved to: {self.cfg_log.log_dir}/training_curves.png")

        # Final plot
        self.viz.plot_training_curves(
            self.metrics,
            save_path=os.path.join(self.cfg_log.log_dir, "training_curves_final.png"),
        )

        # Save final checkpoint
        self.ckpt_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            step=cfg_train.max_steps,
            loss=total_loss,
            config=self.config,
            is_best=False,
        )

        if self.tb_writer:
            self.tb_writer.close()
