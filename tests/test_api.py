"""
tests/test_api.py
=================
Tests for the FastAPI inference server.

Uses FastAPI TestClient to test endpoints without starting
a real server. Tests use a mock model to avoid needing
trained weights.
"""

import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ============================================================
# MOCK SETUP
# ============================================================

class MockTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    vocab_size = 100

    def encode(self, text, add_bos=False, add_eos=False, max_length=None):
        ids = [ord(c) % 90 + 10 for c in text[:10]]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "mock response text"

    def count_tokens(self, text):
        return max(1, len(text) // 5)

    def id_to_token(self, idx):
        return f"tok{idx}"

    def encode_batch(self, texts, **kwargs):
        return [self.encode(t) for t in texts]


class MockModel:
    """Mock GPT model for API testing."""

    class config:
        max_seq_len = 256
        vocab_size = 100
        n_layers = 2

    def __call__(self, input_ids, past_kv=None, use_cache=False, targets=None):
        B, T = input_ids.shape
        logits = torch.randn(B, T, 100)
        kv = [(torch.zeros(B, 2, T, 32), torch.zeros(B, 2, T, 32))
              for _ in range(2)] if use_cache else None
        loss = None
        if targets is not None:
            loss = torch.tensor(3.5)
        return logits, kv, loss

    def eval(self):
        return self

    def parameters(self):
        return [torch.zeros(10)]


# ============================================================
# TEST CLIENT FIXTURE
# ============================================================

@pytest.fixture
def client():
    """Create test client with mocked model."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")

    from api import server as api_server

    # Inject mock model state
    api_server.ModelState.model = MockModel()
    api_server.ModelState.tokenizer = MockTokenizer()
    api_server.ModelState.device = torch.device("cpu")
    api_server.ModelState.model_name = "tiny_test"
    api_server.ModelState.vocab_size = 100

    with TestClient(api_server.app) as c:
        yield c

    # Cleanup
    api_server.ModelState.model = None
    api_server.ModelState.tokenizer = None


# ============================================================
# TESTS
# ============================================================

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_ok(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        """Should report model as loaded."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_fields(self, client):
        """Should contain all expected fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "vocab_size" in data
        assert "device" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestTokenizeEndpoint:
    """Tests for /tokenize endpoint."""

    def test_tokenize_basic(self, client):
        """Should tokenize text and return IDs."""
        response = client.post("/tokenize", json={"text": "Hello world"})
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "count" in data
        assert data["count"] > 0

    def test_tokenize_returns_list(self, client):
        """Tokens should be a list."""
        response = client.post("/tokenize", json={"text": "Test"})
        data = response.json()
        assert isinstance(data["tokens"], list)

    def test_tokenize_empty_text(self, client):
        """Empty text should return empty or minimal tokens."""
        response = client.post("/tokenize", json={"text": ""})
        assert response.status_code == 200


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestGenerateEndpoint:
    """Tests for /generate endpoint."""

    def test_generate_basic(self, client):
        """Should generate text from prompt."""
        response = client.post("/generate", json={
            "prompt": "Hello",
            "max_tokens": 10,
            "temperature": 0.8,
        })
        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data

    def test_generate_token_counts(self, client):
        """Should return token counts."""
        response = client.post("/generate", json={
            "prompt": "Hello",
            "max_tokens": 5,
        })
        data = response.json()
        assert "prompt_tokens" in data
        assert "generated_tokens" in data
        assert "total_tokens" in data
        assert data["total_tokens"] == data["prompt_tokens"] + data["generated_tokens"]

    def test_generate_invalid_temperature(self, client):
        """Invalid temperature should return 422."""
        response = client.post("/generate", json={
            "prompt": "Hello",
            "temperature": -1.0,  # Invalid
        })
        assert response.status_code == 422


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestChatEndpoint:
    """Tests for /chat endpoint."""

    def test_chat_basic(self, client):
        """Should handle single-turn chat."""
        response = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 20,
        })
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_chat_multi_turn(self, client):
        """Should handle multi-turn conversation."""
        response = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "max_tokens": 20,
        })
        assert response.status_code == 200

    def test_chat_with_system_prompt(self, client):
        """Should accept system prompt."""
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "system_prompt": "You are a helpful assistant.",
            "max_tokens": 20,
        })
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
