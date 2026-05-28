# 🤖 GPT-Style LLM — Built From Scratch

A **complete, production-grade GPT-style Large Language Model** built entirely from scratch using PyTorch. No pretrained weights. Every component — tokenizer, transformer architecture, training pipeline, inference engine, and API server — is custom-built from first principles.

---

## ✨ Features

| Component | Implementation |
|-----------|---------------|
| **Tokenizer** | BPE trained from scratch (HuggingFace tokenizers) |
| **Architecture** | GPT-2 style decoder-only transformer |
| **Attention** | Multi-head causal self-attention + KV cache |
| **Training** | AdamW + cosine LR + warmup + gradient accumulation |
| **Mixed Precision** | FP16/BF16 via `torch.amp` |
| **Inference** | Temperature, Top-k, Top-p, repetition penalty |
| **Chat** | Multi-turn conversation with context management |
| **API** | FastAPI with SSE streaming |
| **Export** | ONNX + INT8/BF16 quantization |
| **Visualization** | TensorBoard + Matplotlib training dashboards |

---

## 📁 Project Structure

```
AIModel/
├── configs/              # Model size YAML configs (tiny/small/medium/large)
├── datasets/             # Data pipeline (download, clean, merge)
├── tokenizer/            # BPE tokenizer train + inference
├── model/                # Transformer architecture
│   ├── attention.py      # Multi-head causal attention
│   ├── transformer_block.py  # Single transformer block
│   └── gpt_model.py      # Full GPT model
├── training/             # Training loop + data loading
│   ├── dataloader.py     # Dataset + batching
│   ├── trainer.py        # Full training loop
│   └── train.py          # Entry point
├── inference/            # Text generation
│   ├── generate.py       # Sampling strategies
│   └── kv_cache.py       # KV cache for fast inference
├── chat/                 # Conversation management
├── api/                  # FastAPI server
├── cli/                  # Terminal chat interface
├── export/               # ONNX + quantization
├── utils/                # Logger, metrics, visualization, config
├── tests/                # Unit test suite
├── checkpoints/          # Saved model checkpoints
├── logs/                 # Training logs + plots
├── requirements.txt
├── setup.py
└── main.py               # Unified entry point
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .  # Install as local package
```

### 2. Full Setup (One Command)

```bash
# Downloads TinyStories, cleans it, trains tokenizer
python main.py setup --max_samples 100000 --vocab_size 8000
```

### 3. Train Model

```bash
# Train tiny model (~7M params) — works on CPU or any GPU
python main.py train --config configs/model_tiny.yaml

# Or with custom steps
python training/train.py --config configs/model_tiny.yaml --max_steps 5000
```

### 4. Chat

```bash
# Interactive terminal chat
python main.py chat --checkpoint checkpoints/best.pt

# With custom temperature
python cli/chat.py --checkpoint checkpoints/best.pt --temperature 0.9
```

### 5. API Server

```bash
python main.py server --checkpoint checkpoints/best.pt --port 8000

# Test it
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_tokens": 100}'
```

---

## 📊 Model Size Presets

| Model | Layers | Heads | d_model | FFN | Params | GPU Req |
|-------|--------|-------|---------|-----|--------|---------|
| **tiny** | 4 | 4 | 256 | 1024 | ~7M | CPU/Any |
| **small** | 6 | 6 | 384 | 1536 | ~25M | 4GB VRAM |
| **medium** | 12 | 8 | 512 | 2048 | ~85M | 8GB VRAM |
| **large** | 24 | 16 | 1024 | 4096 | ~350M | 24GB VRAM |

---

## 🏗️ Architecture Deep Dive

### Transformer Architecture

```
Input IDs (B, T)
      ↓
Token Embedding (B, T, D)  +  Positional Embedding (T, D)
      ↓ dropout
[TransformerBlock × N]
│  Pre-LayerNorm
│  Multi-Head Causal Attention  ← Q, K, V projections + scaled dot-product
│  + Residual
│  Pre-LayerNorm
│  Feed-Forward (GELU activation, 4x expansion)
│  + Residual
      ↓
Final LayerNorm
      ↓
LM Head: Linear(D → vocab_size)  [weights tied with token embedding]
      ↓
Logits (B, T, V)
```

### Attention Mechanism

```python
# Scaled dot-product attention
Attention(Q, K, V) = softmax(QK^T / sqrt(head_dim)) × V

# Causal mask prevents attending to future positions
# Position i can only attend to positions 0..i
mask[i][j] = -inf if j > i else 0
```

### KV Cache for Inference

Without KV cache: O(T²) per token
With KV cache: O(T) per token

```python
# First step: process full prompt, cache K,V
logits, kv_cache = model(prompt, use_cache=True)

# Subsequent steps: only process new token
for step in range(max_tokens):
    logits, kv_cache = model(new_token, past_kv=kv_cache, use_cache=True)
    next_token = sample(logits)
```

---

## 📈 Training Guide

### Data Pipeline

```bash
# Step 1: Download datasets
python datasets/download_dataset.py --datasets tinystories --max_samples 500000

# Step 2: Clean (UTF-8, dedup, length filter)
python datasets/clean_dataset.py --input datasets/raw/tinystories.txt

# Step 3: Merge + train/val split
python datasets/merge_dataset.py --val_ratio 0.1
```

### Tokenizer Training

```bash
# Train BPE tokenizer (8K vocab for tiny model)
python tokenizer/tokenizer_train.py \
    --data datasets/processed/train.txt \
    --vocab_size 8000 \
    --output tokenizer/saved

# Test the tokenizer
python tokenizer/tokenizer_infer.py --text "Hello, world!"
```

### Model Training

```bash
# Full training with config
python training/train.py --config configs/model_tiny.yaml

# Resume from checkpoint
python training/train.py --config configs/model_tiny.yaml \
    --resume checkpoints/step_0005000.pt

# Override hyperparameters
python training/train.py --config configs/model_small.yaml \
    --lr 2e-4 --batch_size 8 --max_steps 50000

# Monitor with TensorBoard
tensorboard --logdir logs/
```

### Training Hyperparameters

| Setting | Tiny | Small | Medium |
|---------|------|-------|--------|
| Batch size (effective) | 128 | 128 | 128 |
| Learning rate | 3e-4 | 2.5e-4 | 1.5e-4 |
| Warmup steps | 500 | 1000 | 2000 |
| Max steps | 10K | 50K | 100K |
| Gradient clip | 1.0 | 1.0 | 1.0 |
| Weight decay | 0.1 | 0.1 | 0.1 |

---

## 🔮 Inference Guide

### Generation Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | 0.8 | Lower = more focused, Higher = more creative |
| `top_k` | 50 | Keep only top-k tokens (0 = disabled) |
| `top_p` | 0.95 | Nucleus sampling (1.0 = disabled) |
| `repetition_penalty` | 1.1 | Penalize repeated tokens |
| `max_tokens` | 200 | Maximum tokens to generate |

```bash
# Generate with custom parameters
python inference/generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "Once upon a time" \
    --temperature 0.9 --top_k 40 --max_tokens 200
```

---

## 🌐 API Reference

Start server:
```bash
python api/server.py --checkpoint checkpoints/best.pt --port 8000
```

Interactive docs: http://localhost:8000/docs

### Endpoints

#### `GET /health`
```json
{
    "status": "ok",
    "model_loaded": true,
    "model_name": "tiny",
    "vocab_size": 8000,
    "device": "cpu"
}
```

#### `POST /generate`
```json
// Request
{
    "prompt": "The transformer architecture",
    "max_tokens": 150,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95
}

// Response
{
    "generated_text": "...",
    "prompt_tokens": 5,
    "generated_tokens": 150,
    "total_tokens": 155
}
```

#### `POST /chat`
```json
// Request
{
    "messages": [
        {"role": "user", "content": "What is attention?"}
    ],
    "system_prompt": "You are a helpful AI assistant.",
    "max_tokens": 200
}

// Response
{
    "response": "Attention is a mechanism that...",
    "prompt_tokens": 42,
    "generated_tokens": 87
}
```

#### `POST /chat/stream` — SSE Streaming
```
data: {"token": "Attention", "done": false}
data: {"token": " is", "done": false}
data: {"token": " a", "done": false}
...
data: {"token": "", "done": true}
```

#### `POST /tokenize`
```json
// Request
{"text": "Hello world!"}

// Response
{"tokens": [234, 567, 89], "token_strings": ["Hello", " world", "!"], "count": 3}
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_model.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `model/attention.py` | Causal mask, KV cache, output shapes |
| `model/gpt_model.py` | Forward pass, weight tying, gradients |
| `tokenizer/` | Round-trip, special tokens, batch ops |
| `inference/generate.py` | Temperature, top-k, top-p, streaming |
| `datasets/` | Cleaning, chunking, caching |

---

## 📦 Export

### ONNX Export
```bash
python export/onnx_export.py \
    --checkpoint checkpoints/best.pt \
    --output export/model.onnx \
    --seq_len 128
```

### INT8 Quantization (4x smaller, ~2x faster on CPU)
```bash
python export/quantize.py \
    --checkpoint checkpoints/best.pt \
    --method dynamic_int8 \
    --output export/model_int8.pt \
    --benchmark
```

### BF16 Conversion (2x smaller, minimal accuracy loss)
```bash
python export/quantize.py \
    --checkpoint checkpoints/best.pt \
    --method bf16 \
    --output export/model_bf16.pt
```

---

## ⚡ GPU Optimization Tips

### CUDA (NVIDIA)
- Mixed precision (BF16 on A100, FP16 on V100/RTX)
- `torch.compile` for ~20% speedup (enable in config)
- Pin memory (`pin_memory: true` in config)
- Fused AdamW (automatic on CUDA)

### Apple Silicon (MPS)
```bash
# MPS is auto-detected. Verify:
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Memory Optimization
- Gradient accumulation to simulate larger batches
- `set_to_none=True` in `optimizer.zero_grad()` (saves memory)
- KV cache for inference (O(T) vs O(T²) memory)

---

## 🔬 Model Details

### Weight Initialization (GPT-2 style)
- Linear layers: N(0, 0.02)
- Embeddings: N(0, 0.02)
- Residual projections: N(0, 0.02 / √(2N)) where N = n_layers
- LayerNorm: weight=1, bias=0

### Why Weight Tying?
The LM head (vocabulary projection) shares weights with the token embedding.
This reduces parameters by `vocab_size × d_model` and typically improves
perplexity because the input/output spaces are symmetric.

### Pre-LayerNorm vs Post-LayerNorm
We use **Pre-LayerNorm** (apply norm before sublayer):
```
x → LN → Sublayer → + x
```
vs **Post-LayerNorm** (original "Attention is All You Need"):
```
x → Sublayer → + x → LN
```
Pre-LN is more stable for deep networks and is used in GPT-2/3/4.

---

## 📉 Expected Training Progress

| Dataset | Model | Steps | Expected Loss | Expected PPL |
|---------|-------|-------|---------------|--------------|
| TinyStories | Tiny | 5K | ~3.5 | ~33 |
| TinyStories | Tiny | 10K | ~3.0 | ~20 |
| TinyStories | Small | 50K | ~2.5 | ~12 |
| Wikipedia | Medium | 100K | ~2.8 | ~16 |

Lower perplexity = better model (1.0 = perfect, vocab_size = random).

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) — GPT-2
- [Training language models to follow instructions](https://arxiv.org/abs/2203.02155) — InstructGPT
- [TinyStories: How Small Can Language Models Be?](https://arxiv.org/abs/2305.07759)
- [The Pile: An 800GB Dataset](https://arxiv.org/abs/2101.00027)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built from scratch — no pretrained models, no borrowed weights, just math and PyTorch.*
