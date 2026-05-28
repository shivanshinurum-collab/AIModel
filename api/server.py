"""
api/server.py
=============
FastAPI inference server for the GPT LLM.

ENDPOINTS
=========
    GET  /health          → Server health check
    POST /tokenize        → Tokenize text, return token count and IDs
    POST /generate        → Generate text from prompt
    POST /chat            → Chat with conversation history
    POST /chat/stream     → Streaming chat (Server-Sent Events)

STREAMING
=========
Server-Sent Events (SSE) allow the server to push data to the client
as tokens are generated. This enables real-time token streaming.

SSE Format:
    data: {"token": "Hello", "done": false}
    data: {"token": " world", "done": false}
    data: {"token": "", "done": true}

Usage:
    # Start server
    python api/server.py --checkpoint checkpoints/best.pt \\
                          --config configs/model_tiny.yaml

    # Or via uvicorn
    uvicorn api.server:app --host 0.0.0.0 --port 8000

    # Test
    curl -X POST http://localhost:8000/generate \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "Once upon a time", "max_tokens": 100}'
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Optional, List, AsyncGenerator

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from model.gpt_model import GPTModel
from tokenizer.tokenizer_infer import LLMTokenizer
from inference.generate import generate, generate_streaming
from chat.chat_format import ConversationManager
from utils.config_loader import load_config, get_default_config
from utils.logger import get_logger

log = get_logger("api_server")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="GPT LLM API",
    description="Inference API for custom-trained GPT-style language model",
    version="0.1.0",
    docs_url="/docs",   # Swagger UI at /docs
    redoc_url="/redoc", # ReDoc UI at /redoc
)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

class ModelState:
    """Singleton holding loaded model and tokenizer."""
    model: Optional[GPTModel] = None
    tokenizer: Optional[LLMTokenizer] = None
    device: Optional[torch.device] = None
    model_name: str = "unknown"
    vocab_size: int = 0

    @classmethod
    def is_loaded(cls) -> bool:
        return cls.model is not None and cls.tokenizer is not None

    @classmethod
    def assert_loaded(cls):
        if not cls.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Start server with --checkpoint flag."
            )


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class GenerateRequest(BaseModel):
    """Request body for /generate endpoint."""
    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(200, ge=1, le=2000, description="Maximum tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, le=1000, description="Top-k filtering (0=disabled)")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stream: bool = Field(False, description="Enable streaming response")


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    max_tokens: int = Field(200, ge=1, le=2000)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=1000)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)


class TokenizeRequest(BaseModel):
    """Request body for /tokenize endpoint."""
    text: str = Field(..., description="Text to tokenize")
    add_special_tokens: bool = Field(False, description="Add BOS/EOS tokens")


class GenerateResponse(BaseModel):
    """Response from /generate endpoint."""
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """Response from /chat endpoint."""
    response: str
    prompt_tokens: int
    generated_tokens: int


class TokenizeResponse(BaseModel):
    """Response from /tokenize endpoint."""
    tokens: List[int]
    token_strings: List[str]
    count: int


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str
    model_loaded: bool
    model_name: str
    vocab_size: int
    device: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns server status and model information.
    Use this to verify the server is running before sending requests.
    """
    return HealthResponse(
        status="ok",
        model_loaded=ModelState.is_loaded(),
        model_name=ModelState.model_name,
        vocab_size=ModelState.vocab_size,
        device=str(ModelState.device) if ModelState.device else "none",
    )


@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """
    Tokenize text and return token IDs and strings.

    Useful for debugging tokenization and counting tokens.
    """
    ModelState.assert_loaded()
    tok = ModelState.tokenizer

    if request.add_special_tokens:
        ids = tok.encode(request.text, add_bos=True, add_eos=True)
    else:
        ids = tok.encode(request.text)

    token_strings = [tok.id_to_token(i) or f"<id:{i}>" for i in ids]

    return TokenizeResponse(
        tokens=ids,
        token_strings=token_strings,
        count=len(ids),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text from a prompt.

    If stream=True, returns a streaming SSE response instead.
    """
    ModelState.assert_loaded()

    if request.stream:
        return await _stream_generate(request.prompt, request)

    tok = ModelState.tokenizer
    model = ModelState.model
    device = ModelState.device

    # Tokenize prompt
    prompt_ids = tok.encode(request.prompt, add_bos=True)
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    # Generate
    with torch.no_grad():
        output_ids = generate(
            model=model,
            input_ids=prompt_tensor,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            eos_token_id=tok.eos_id,
            use_kv_cache=True,
        )

    # Decode
    generated_ids = output_ids[0, prompt_len:].tolist()
    generated_text = tok.decode(generated_ids, skip_special_tokens=True)

    return GenerateResponse(
        generated_text=generated_text,
        prompt_tokens=prompt_len,
        generated_tokens=len(generated_ids),
        total_tokens=prompt_len + len(generated_ids),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the model using conversation history.

    Maintains conversation format and manages context window.
    """
    ModelState.assert_loaded()

    tok = ModelState.tokenizer
    model = ModelState.model
    device = ModelState.device

    # Build conversation prompt
    conv = ConversationManager(
        system_prompt=request.system_prompt,
        tokenizer=tok,
        max_seq_len=model.config.max_seq_len,
    )

    # Add message history
    for msg in request.messages:
        if msg.role == "user":
            conv.add_user(msg.content)
        elif msg.role == "assistant":
            conv.add_assistant(msg.content)

    # Build prompt
    prompt = conv.build_prompt()

    # Tokenize
    prompt_ids = tok.encode(prompt, add_bos=True)
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    # Generate
    with torch.no_grad():
        output_ids = generate(
            model=model,
            input_ids=prompt_tensor,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            eos_token_id=tok.eos_id,
            use_kv_cache=True,
        )

    # Decode and extract response
    full_text = tok.decode(output_ids[0].tolist(), skip_special_tokens=True)
    response = conv.extract_response(full_text)

    return ChatResponse(
        response=response,
        prompt_tokens=prompt_len,
        generated_tokens=len(output_ids[0]) - prompt_len,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Tokens are streamed as they're generated, enabling real-time display.

    SSE event format:
        data: {"token": "Hello", "done": false}
        data: {"token": " world", "done": false}
        data: {"token": "", "done": true}
    """
    ModelState.assert_loaded()

    tok = ModelState.tokenizer
    model = ModelState.model
    device = ModelState.device

    # Build conversation prompt
    conv = ConversationManager(
        system_prompt=request.system_prompt,
        tokenizer=tok,
        max_seq_len=model.config.max_seq_len,
    )

    for msg in request.messages:
        if msg.role == "user":
            conv.add_user(msg.content)
        elif msg.role == "assistant":
            conv.add_assistant(msg.content)

    prompt = conv.build_prompt()
    prompt_ids = tok.encode(prompt, add_bos=True)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    async def event_stream() -> AsyncGenerator[str, None]:
        """Async generator that yields SSE-formatted events."""
        try:
            # Run generation in executor to avoid blocking async loop
            loop = asyncio.get_event_loop()

            # For streaming, we use the synchronous generator in a thread
            def sync_generate():
                tokens = []
                for token_text in generate_streaming(
                    model=model,
                    input_ids=prompt_tensor,
                    tokenizer=tok,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    eos_token_id=tok.eos_id,
                    use_kv_cache=True,
                ):
                    tokens.append(token_text)
                return tokens

            # Run in thread pool (non-blocking)
            tokens = await loop.run_in_executor(None, sync_generate)

            # Stream tokens
            for token_text in tokens:
                event = json.dumps({"token": token_text, "done": False})
                yield f"data: {event}\n\n"
                await asyncio.sleep(0)  # Yield control to event loop

            # Done event
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

        except Exception as e:
            error_event = json.dumps({"error": str(e), "done": True})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        config_path    : Path to YAML config (optional — uses checkpoint config)
        device         : Device to load on ('cuda', 'mps', 'cpu', or None for auto)
    """
    log.info(f"Loading model from: {checkpoint_path}")

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    ModelState.device = torch.device(device)
    log.info(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", {})

    # Load config
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        # Reconstruct from checkpoint config
        config = get_default_config("tiny")
        if ckpt_config:
            cfg = config.model
            cfg.d_model = ckpt_config.get("d_model", cfg.d_model)
            cfg.n_layers = ckpt_config.get("n_layers", cfg.n_layers)
            cfg.n_heads = ckpt_config.get("n_heads", cfg.n_heads)
            cfg.d_ff = ckpt_config.get("d_ff", cfg.d_ff)
            cfg.vocab_size = ckpt_config.get("vocab_size", cfg.vocab_size)
            cfg.max_seq_len = ckpt_config.get("max_seq_len", cfg.max_seq_len)

    # Initialize model
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(ModelState.device)
    model.eval()

    ModelState.model = model
    ModelState.model_name = ckpt_config.get("model_name", config.model.name)

    # Load tokenizer
    tokenizer_path = "tokenizer/saved/tokenizer.json"
    ModelState.tokenizer = LLMTokenizer(tokenizer_path)
    ModelState.vocab_size = ModelState.tokenizer.vocab_size

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model loaded: {n_params:,} parameters")
    log.info(f"Tokenizer: vocab_size={ModelState.vocab_size}")

    return model


# ============================================================
# SERVER ENTRY POINT
# ============================================================

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="GPT LLM Inference API Server")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config (optional, uses checkpoint config if not provided)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="Device to run on (auto-detected if not specified)")
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    # Load model at startup
    load_model(args.checkpoint, config_path=args.config, device=args.device)

    log.info(f"Starting API server at http://{args.host}:{args.port}")
    log.info(f"API docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
