"""Client for "System One" decision-model APIs: TypeSafe's Jev and the Jev-compatible open-weight models
(Kev's local server, Cloudflare's Clef on Workers AI, and any server that speaks the same protocol).

These models do not generate text. A request carries a ``state`` (the situation as text) and typed
``questions``; a ``choice`` question lists the allowed answers as ``criteria`` and the answer comes back as
the top option, a probability for every option, and a confidence. DeadEye maps its decision problems onto
that protocol: the rendered prompt is the state, the legal actions are the criteria, and the returned
probabilities are used exactly as the log-likelihoods of :class:`PromptScorePolicy`, so hosted or local
decision models are evaluated on the same tasks, seeds and metrics as every other conversion method.
The request and response shape follows the System One API as documented by TypeSafe and implemented by
Kev's reference server (``POST /v1/systemone``).
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any

from deadeye.models.base import LanguageModel, Message, ModelInfo, ScoreResult

DEFAULT_INSTRUCTIONS = "Choose the single best action for the current state, following the task rules given in the state."


class DecisionAPIModel(LanguageModel):
    capabilities = frozenset({"score"})

    def __init__(self, model_id: str = "jev", base_url: str = "http://127.0.0.1:8009/v1/systemone", api_key: str | None = None,
                 api_key_env: str = "TYPESAFE_API_KEY", timeout: float = 60.0, max_retries: int = 3,
                 instructions: str = DEFAULT_INSTRUCTIONS, question_id: str = "action", extra_body: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None, **_: object) -> None:
        self.model_id = model_id
        self.url = os.path.expandvars(base_url)
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.instructions = instructions
        self.question_id = question_id
        self.extra_body = dict(extra_body or {})
        self.headers = dict(headers or {})
        self.info = ModelInfo(id=model_id, backend="decision", precision="served", instruct=None,
                              extra={"base_url": base_url, "protocol": "systemone"})

    def _describe(self) -> str:
        return f"{self.model_id} @ {self.url}"

    # ------------------------------------------------------------------ transport
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise RuntimeError(f"{self.url} -> HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"{self.url} unreachable: {e}") from e
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ primitives
    @staticmethod
    def state_text(messages: list[Message]) -> str:
        """The state is the task description followed by the observation, without any answer instruction."""
        parts = [m.content.strip() for m in messages if m.role in ("system", "user")]
        text = "\n\n".join(parts)
        # the score-mode prompt ends with a one-line answer instruction meant for text models; drop it
        tail = "\nAnswer with only the"
        if tail in text:
            text = text[:text.rfind(tail)]
        return text.strip()

    def score_choices(self, messages, choices, length_norm=False) -> ScoreResult:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "state": self.state_text(messages),
            "questions": {self.question_id: {"type": "choice", "instructions": self.instructions,
                                             "criteria": {c: c for c in choices}}},
        }
        payload.update(self.extra_body)
        t0 = time.perf_counter()
        resp = self._post(payload)
        latency = time.perf_counter() - t0
        answer = (resp.get("answers") or {}).get(self.question_id) or {}
        probs = answer.get("probabilities") or {}
        floor = 1e-9
        logprobs = [math.log(max(float(probs.get(c, 0.0)), floor)) for c in choices]
        usage = resp.get("usage") or {}
        tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        if "latency_ms" in resp:
            latency = min(latency, float(resp["latency_ms"]) / 1000.0) if latency > 0 else float(resp["latency_ms"]) / 1000.0
        return ScoreResult(logprobs=logprobs, prompt_tokens=tokens, latency_s=latency, choice_tokens=[1] * len(choices),
                           extra={"confidence": answer.get("confidence"), "choice": answer.get("choice"), "server_latency_ms": resp.get("latency_ms")})
