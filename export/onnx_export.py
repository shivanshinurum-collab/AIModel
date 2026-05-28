"""
export/onnx_export.py
=====================
ONNX export for the GPT LLM model.

WHY ONNX?
=========
ONNX (Open Neural Network Exchange) is a standard format for ML models
that allows deployment across different runtimes:
- ONNX Runtime (fast CPU/GPU inference)
- TensorRT (NVIDIA GPU optimization)
- OpenVINO (Intel)
- CoreML (Apple)

LIMITATIONS
===========
ONNX export of transformer models has some limitations:
- KV cache is complex to export (we export without cache)
- Dynamic shapes require extra care
- Some PyTorch ops may not be supported

We export the model in "static mode" (without KV cache) which is
suitable for batch inference on fixed-length sequences.

Usage:
    python export/onnx_export.py \\
        --checkpoint checkpoints/best.pt \\
        --output export/model.onnx
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.config_loader import load_config, get_default_config
from utils.logger import get_logger

log = get_logger("onnx_export")


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "export/model.onnx",
    config_path: str = None,
    opset_version: int = 17,
    seq_len: int = 128,
) -> str:
    """
    Export GPT model to ONNX format.

    Args:
        checkpoint_path: Path to .pt checkpoint
        output_path    : Output .onnx file path
        config_path    : YAML config path (optional)
        opset_version  : ONNX opset version (17 recommended)
        seq_len        : Fixed sequence length for export

    Returns:
        Path to exported ONNX file
    """
    try:
        import onnx
    except ImportError:
        log.error("onnx not installed. Run: pip install onnx onnxruntime")
        sys.exit(1)

    from model.gpt_model import GPTModel

    log.info(f"Loading checkpoint: {checkpoint_path}")
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

    # Load model (CPU, eval mode, no dropout)
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy input for tracing
    vocab_size = config.model.vocab_size
    dummy_input = torch.randint(0, vocab_size, (1, seq_len))  # (B=1, T=seq_len)

    # Output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    log.info(f"Exporting to ONNX: {output_path}")
    log.info(f"Input shape: {dummy_input.shape}")
    log.info(f"Opset version: {opset_version}")

    # Wrapper that returns only logits (ONNX can't handle tuple with Nones well)
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            logits, _, _ = self.model(input_ids, use_cache=False)
            return logits

    onnx_model = ONNXWrapper(model)

    with torch.no_grad():
        torch.onnx.export(
            onnx_model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits":    {0: "batch_size", 1: "sequence_length"},
            },
            do_constant_folding=True,
            export_params=True,
        )

    # Verify ONNX model
    onnx_model_loaded = onnx.load(output_path)
    onnx.checker.check_model(onnx_model_loaded)
    log.info("ONNX model verified successfully!")

    # Test with ONNX Runtime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        outputs = sess.run(None, {"input_ids": dummy_input.numpy()})
        log.info(f"ONNX Runtime test: output shape = {outputs[0].shape}")
        log.info("ONNX Runtime inference: OK")
    except ImportError:
        log.warning("onnxruntime not installed. Run: pip install onnxruntime")

    model_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    log.info(f"Export complete: {output_path} ({model_size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export GPT model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="export/model.onnx")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        opset_version=args.opset,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
