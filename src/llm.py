"""Pluggable LLM wrapper for generation + verification.

Backends
--------
* ``claude`` — Anthropic API (needs ANTHROPIC_API_KEY).
* ``ollama`` — local Ollama server (free, offline). Uses the native /api/chat
  endpoint over stdlib urllib, so no extra Python dependency is required.

Backend is chosen by ``config.LLM_BACKEND`` (auto: Claude if a key is present,
else Ollama). If the selected backend is unreachable, ``available`` is False and
callers fall back to their deterministic mock so the pipeline still runs.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import config


class LLMClient:
    def __init__(self, backend: str | None = None, model: str | None = None):
        self.backend = backend or config.LLM_BACKEND
        self._client = None            # anthropic client (claude backend)
        self._ok = False

        if self.backend == "claude":
            self.model = model or config.CLAUDE_MODEL
            if config.ANTHROPIC_API_KEY:
                try:
                    import anthropic

                    self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                    self._ok = True
                except Exception as e:  # pragma: no cover - defensive
                    print(f"[llm] Anthropic init failed ({e}); running in mock mode.")
        elif self.backend == "ollama":
            self.model = model or config.OLLAMA_MODEL
            self._ok = self._ollama_reachable()
            if not self._ok:
                print(
                    f"[llm] Ollama not reachable at {config.OLLAMA_HOST}; running in "
                    f"mock mode. Start it with `ollama serve` and `ollama pull "
                    f"{self.model}`."
                )
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {self.backend}")

    # -- capability ----------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._ok

    def describe(self) -> str:
        return f"{self.backend}:{self.model}" + ("" if self._ok else " (mock)")

    # -- ollama helpers ------------------------------------------------------
    def _ollama_reachable(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{config.OLLAMA_HOST}/api/tags", timeout=3
            ) as resp:
                json.loads(resp.read().decode("utf-8"))
            return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _ollama_complete(self, system, user, max_tokens, temperature, json_mode):
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            f"{config.OLLAMA_HOST}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Local generation on CPU can be slow; give it room.
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]

    # -- public --------------------------------------------------------------
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        if not self.available:
            raise RuntimeError(
                f"LLM backend '{self.backend}' unavailable; client is in mock mode."
            )
        max_tokens = max_tokens or config.LLM_MAX_TOKENS
        temperature = config.LLM_TEMPERATURE if temperature is None else temperature

        if self.backend == "claude":
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if b.type == "text")

        return self._ollama_complete(system, user, max_tokens, temperature, json_mode)
