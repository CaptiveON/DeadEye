"""OpenAI-compatible HTTP backend (vLLM, llama.cpp server, Ollama, LM Studio, OpenRouter, ...).

Lets peers evaluate open-weight models they already serve, without a GPU in the evaluation process.
``generate`` uses ``/chat/completions``. ``score_choices`` uses *first-token* log-probabilities
(``logprobs`` + ``top_logprobs`` on a one-token completion), which is exact only when every choice
starts with a distinct token: use ``action_format: letter`` in the policy config for that. Exact
full-sequence scoring needs the local ``hf`` backend.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from deadeye.models.base import GenerationResult, LanguageModel, Message, ModelInfo, ScoreResult


class OpenAICompatModel(LanguageModel):
    capabilities = frozenset({"generate", "score"})

    def __init__(self, model_id: str, base_url: str = "http://localhost:8000/v1", api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", extra_body: dict[str, Any] | None = None, timeout: float = 120.0,
                 max_retries: int = 3, top_logprobs: int = 20, seed: int | None = None, **_: object) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.extra_body = dict(extra_body or {})
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.top_logprobs = int(top_logprobs)
        self._seed = seed
        self.info = ModelInfo(id=model_id, backend="openai", precision="served", extra={"base_url": self.base_url})

    def _describe(self) -> str:
        return f"{self.model_id} @ {self.base_url}"

    def seed(self, s: int) -> None:
        self._seed = int(s)

    # ------------------------------------------------------------------ transport
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                retryable = e.code in (408, 409, 429, 500, 502, 503, 504)
                if not retryable or attempt == self.max_retries:
                    detail = e.read().decode(errors="replace")[:500]
                    raise RuntimeError(f"{url} -> HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"{url} unreachable: {e}") from e
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _msgs(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    # ------------------------------------------------------------------ primitives
    def generate(self, messages, max_new_tokens=64, temperature=0.0, stop=None) -> GenerationResult:
        payload: dict[str, Any] = {"model": self.model_id, "messages": self._msgs(messages),
                                   "max_tokens": int(max_new_tokens), "temperature": float(temperature)}
        if stop:
            payload["stop"] = list(stop)
        if self._seed is not None:
            payload["seed"] = self._seed
        payload.update(self.extra_body)
        t0 = time.perf_counter()
        resp = self._post("chat/completions", payload)
        latency = time.perf_counter() - t0
        choice = resp["choices"][0]
        msg = choice.get("message", {})
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning and "<think>" not in text:
            text = f"<think>{reasoning}</think>{text}"
        usage = resp.get("usage", {}) or {}
        return GenerationResult(text=text, prompt_tokens=int(usage.get("prompt_tokens", 0)),
                                completion_tokens=int(usage.get("completion_tokens", 0)), latency_s=latency, raw=resp)

    def score_choices(self, messages, choices, length_norm=False) -> ScoreResult:
        payload: dict[str, Any] = {"model": self.model_id, "messages": self._msgs(messages), "max_tokens": 1,
                                   "temperature": 0.0, "logprobs": True, "top_logprobs": self.top_logprobs}
        payload.update(self.extra_body)
        t0 = time.perf_counter()
        resp = self._post("chat/completions", payload)
        latency = time.perf_counter() - t0
        content = ((resp["choices"][0].get("logprobs") or {}).get("content") or [])
        top: dict[str, float] = {}
        if content:
            for entry in content[0].get("top_logprobs", []):
                tok = entry["token"].strip()
                if tok:
                    top[tok] = max(top.get(tok, -1e9), float(entry["logprob"]))
            own = content[0]
            if own.get("token", "").strip():
                top[own["token"].strip()] = max(top.get(own["token"].strip(), -1e9), float(own["logprob"]))
        floor = (min(top.values()) - 5.0) if top else -100.0
        scores = []
        for c in choices:
            cands = [lp for tok, lp in top.items() if c.startswith(tok) or tok.startswith(c)]
            scores.append(max(cands) if cands else floor)
        usage = resp.get("usage", {}) or {}
        return ScoreResult(logprobs=scores, prompt_tokens=int(usage.get("prompt_tokens", 0)), latency_s=latency,
                           choice_tokens=[1] * len(choices))
