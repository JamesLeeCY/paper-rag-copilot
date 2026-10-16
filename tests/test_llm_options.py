"""Per-request Ollama options (num_thread, num_ctx): no server. Run: python -m pytest tests"""
import io
import json

import pytest

import config
from src import llm as L


@pytest.fixture
def client():
    c = L.LLMClient.__new__(L.LLMClient)
    c.backend, c.model, c._ok, c._thinks, c._client = "ollama", "phi4:latest", True, False, None
    return c


def sent_options(monkeypatch, client):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen.update(json.loads(req.data)["options"])
        return io.BytesIO(json.dumps({"message": {"content": "{}"}}).encode())

    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    client.complete("sys", "user", max_tokens=300)
    return seen


def test_defaults_send_neither(monkeypatch, client):
    monkeypatch.setattr(config, "OLLAMA_NUM_THREAD", 0)
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 0)
    opts = sent_options(monkeypatch, client)
    assert "num_thread" not in opts and "num_ctx" not in opts


def test_settings_are_sent_per_request(monkeypatch, client):
    monkeypatch.setattr(config, "OLLAMA_NUM_THREAD", 2)
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 8192)
    opts = sent_options(monkeypatch, client)
    assert opts["num_thread"] == 2 and opts["num_ctx"] == 8192


def test_ollama_version(monkeypatch):
    monkeypatch.setattr(L.urllib.request, "urlopen",
                        lambda url, timeout=None: io.BytesIO(b'{"version": "0.40.5"}'))
    assert L.ollama_version() == "0.40.5"

    def down(url, timeout=None):
        raise L.urllib.error.URLError("refused")
    monkeypatch.setattr(L.urllib.request, "urlopen", down)
    assert L.ollama_version() == ""
