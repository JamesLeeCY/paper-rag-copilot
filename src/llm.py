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
import re
import urllib.error
import urllib.request

import config

# Reasoning models (e.g. DeepSeek-R1) emit their chain of thought in a
# <think>…</think> block before the answer; it must not reach the parsers.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def request_timeout(prompt_chars: int, max_tokens: int) -> float:
    """Seconds to wait for a local generation on CPU.

    Covers both phases: reading the prompt and writing up to ``max_tokens``.
    The prompt is estimated at ~2.5 characters per token (mixed Chinese and
    English) and weighted at half a generated token, since prompt evaluation
    is faster per token. Sizing by output alone (the earlier rule) let a judge
    reading a long passage on 3 threads time out at 300 s.
    """
    prompt_tokens = prompt_chars / 2.5
    work = max_tokens + 0.5 * prompt_tokens
    return max(300.0, 60.0 + work * config.OLLAMA_SECONDS_PER_TOKEN)


class LLMClient:
    def __init__(self, backend: str | None = None, model: str | None = None):
        self.backend = backend or config.LLM_BACKEND
        self._client = None            # anthropic client (claude backend)
        self._ok = False
        self._thinks = False           # ollama model reports "thinking" capability

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
            if self._ok:
                self._thinks = "thinking" in self._ollama_capabilities()
            if not self._ok:
                print(
                    f"[llm] Ollama model '{self.model}' not available at "
                    f"{config.OLLAMA_HOST} (server down or model not pulled); "
                    f"running in mock mode. Start it with `ollama serve` and "
                    f"`ollama pull {self.model}`."
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
        """Server answers AND this model is pulled (a missing model would
        otherwise only fail on the first generation call)."""
        try:
            with urllib.request.urlopen(
                f"{config.OLLAMA_HOST}/api/tags", timeout=3
            ) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names

    def _ollama_capabilities(self) -> list[str]:
        """Model capabilities from /api/show (metadata only; loads no model).
        Older servers omit the field, which reads as no thinking."""
        req = urllib.request.Request(
            f"{config.OLLAMA_HOST}/api/show",
            data=json.dumps({"model": self.model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8")).get("capabilities", [])
        except (urllib.error.URLError, OSError, ValueError):
            return []

    def _ollama_complete(self, system, user, max_tokens, temperature, json_mode):
        if self._thinks:
            # Thinking tokens count toward num_predict (see OLLAMA_THINK_BUDGET).
            max_tokens += config.OLLAMA_THINK_BUDGET
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
        if config.OLLAMA_NUM_THREAD > 0:
            payload["options"]["num_thread"] = config.OLLAMA_NUM_THREAD
        if json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            f"{config.OLLAMA_HOST}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = request_timeout(len(system) + len(user), max_tokens)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["message"].get("content", "")
        if not content.strip() and data["message"].get("thinking"):
            print(
                f"[llm] {self.model}: empty answer after thinking "
                f"(done_reason={data.get('done_reason')}, "
                f"eval_count={data.get('eval_count')}); raise OLLAMA_THINK_BUDGET."
            )
        return content

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

        return strip_thinking(
            self._ollama_complete(system, user, max_tokens, temperature, json_mode)
        )
