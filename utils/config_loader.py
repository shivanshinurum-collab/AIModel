"""
utils/config_loader.py
======================
YAML-based configuration system for the GPT LLM project.

Provides a strongly-typed ModelConfig dataclass that loads from YAML files.
Supports all four model size presets: tiny, small, medium, large.

Usage:
    config = load_config("configs/model_tiny.yaml")
    print(config.model.d_model)  # 256

Design notes:
- Uses dataclasses for type-safety and IDE autocompletion
- YAML format for human readability and easy editing
- Nested config groups: model, training, data, logging, checkpoint
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# SUB-CONFIG DATACLASSES
# ============================================================

@dataclass
class ModelArchConfig:
    """Transformer architecture hyperparameters."""
    name: str = "tiny"
    vocab_size: int = 8000          # BPE vocabulary size
    max_seq_len: int = 256          # Maximum context window
    d_model: int = 256              # Embedding / hidden dimension
    n_layers: int = 4               # Number of transformer blocks
    n_heads: int = 4                # Number of attention heads
    d_ff: int = 1024                # Feed-forward inner dimension
    dropout: float = 0.1           # Dropout probability
    bias: bool = True              # Use bias in linear layers


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    eval_interval: int = 500
    save_interval: int = 1000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5            # cosine LR decay floor
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    mixed_precision: bool = True
    compile_model: bool = False     # torch.compile — requires Python 3.11+


@dataclass
class DataConfig:
    """Dataset and tokenizer paths."""
    dataset_path: str = "datasets/processed/train.txt"
    val_path: str = "datasets/processed/val.txt"
    tokenizer_path: str = "tokenizer/saved/tokenizer.json"
    num_workers: int = 2


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "logs"
    tensorboard: bool = True
    log_interval: int = 10


@dataclass
class CheckpointConfig:
    """Checkpoint saving/loading configuration."""
    checkpoint_dir: str = "checkpoints"
    keep_last_n: int = 3            # Number of recent checkpoints to keep


@dataclass
class ModelConfig:
    """
    Top-level configuration container.
    Holds all nested sub-configs.

    Attributes:
        model   : Transformer architecture settings
        training: Training loop hyperparameters
        data    : Dataset and tokenizer paths
        logging : Logging settings
        checkpoint: Checkpoint management settings
        config_path: Path to the source YAML file (for reference)
    """
    model: ModelArchConfig = field(default_factory=ModelArchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    config_path: Optional[str] = None

    def __post_init__(self):
        """Validate config consistency after initialization."""
        assert self.model.d_model % self.model.n_heads == 0, (
            f"d_model ({self.model.d_model}) must be divisible by "
            f"n_heads ({self.model.n_heads})"
        )
        assert self.model.n_heads > 0, "n_heads must be > 0"
        assert self.model.d_ff == self.model.d_model * 4 or self.model.d_ff > 0, (
            "d_ff must be positive (typically 4 * d_model)"
        )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.model.d_model // self.model.n_heads

    @property
    def param_count_estimate(self) -> str:
        """Rough estimate of model parameter count."""
        # Embedding table + positional embeddings
        embed = self.model.vocab_size * self.model.d_model
        pos = self.model.max_seq_len * self.model.d_model

        # Per-layer: 4 attention matrices + 2 FF matrices + layer norms
        # Attention: Q, K, V, O projections = 4 * d_model^2
        attn = 4 * (self.model.d_model ** 2)
        # FFN: two linear layers
        ff = 2 * self.model.d_model * self.model.d_ff
        # Layer norms: 2 per block, each with 2*d_model params
        ln = 4 * self.model.d_model
        per_layer = attn + ff + ln

        total = embed + pos + (per_layer * self.model.n_layers)
        # LM head (tied with embedding — not doubled)

        if total < 1e6:
            return f"{total/1e3:.1f}K"
        elif total < 1e9:
            return f"{total/1e6:.1f}M"
        else:
            return f"{total/1e9:.2f}B"

    def summary(self) -> str:
        """Human-readable config summary."""
        lines = [
            f"Model: {self.model.name}  (~{self.param_count_estimate} params)",
            f"  Architecture: {self.model.n_layers}L × {self.model.n_heads}H × {self.model.d_model}d",
            f"  Vocab: {self.model.vocab_size}  |  Max Seq: {self.model.max_seq_len}",
            f"  FFN dim: {self.model.d_ff}  |  Dropout: {self.model.dropout}",
            f"Training:",
            f"  Batch: {self.training.batch_size} × {self.training.gradient_accumulation_steps} accum",
            f"  LR: {self.training.learning_rate} → {self.training.min_lr}  |  Warmup: {self.training.warmup_steps}",
            f"  Max steps: {self.training.max_steps}  |  Mixed precision: {self.training.mixed_precision}",
        ]
        return "\n".join(lines)


# ============================================================
# CONFIG LOADER
# ============================================================

def load_config(config_path: str) -> ModelConfig:
    """
    Load a ModelConfig from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ModelConfig: Fully populated configuration object.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required keys are missing.

    Example:
        config = load_config("configs/model_tiny.yaml")
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Parse nested groups
    model_cfg = ModelArchConfig(**raw.get("model", {}))
    train_cfg = TrainingConfig(**raw.get("training", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    log_cfg = LoggingConfig(**raw.get("logging", {}))
    ckpt_cfg = CheckpointConfig(**raw.get("checkpoint", {}))

    config = ModelConfig(
        model=model_cfg,
        training=train_cfg,
        data=data_cfg,
        logging=log_cfg,
        checkpoint=ckpt_cfg,
        config_path=os.path.abspath(config_path),
    )
    return config


def get_default_config(size: str = "tiny") -> ModelConfig:
    """
    Get a default config by size name without reading a YAML file.

    Args:
        size: One of 'tiny', 'small', 'medium', 'large'

    Returns:
        ModelConfig with preset values.
    """
    presets = {
        "tiny": dict(
            name="tiny", vocab_size=8000, max_seq_len=256,
            d_model=256, n_layers=4, n_heads=4, d_ff=1024, dropout=0.1,
        ),
        "small": dict(
            name="small", vocab_size=16000, max_seq_len=512,
            d_model=384, n_layers=6, n_heads=6, d_ff=1536, dropout=0.1,
        ),
        "medium": dict(
            name="medium", vocab_size=32000, max_seq_len=1024,
            d_model=512, n_layers=12, n_heads=8, d_ff=2048, dropout=0.1,
        ),
        "large": dict(
            name="large", vocab_size=50000, max_seq_len=2048,
            d_model=1024, n_layers=24, n_heads=16, d_ff=4096, dropout=0.1,
        ),
    }

    if size not in presets:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(presets.keys())}")

    model_cfg = ModelArchConfig(**presets[size])
    return ModelConfig(model=model_cfg)
