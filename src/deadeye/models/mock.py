"""Deterministic mock language model for tests, CI and dry runs.

It reads the ``Legal actions:`` line that every DeadEye prompt contains and answers with one of them,
optionally with a configurable rate of malformed outputs (to exercise the illegal-action fallback).
``embed`` uses the hashing trick over word tokens plus any numbers in the prompt, so a linear probe
trained on top of it can learn something non-trivial. ``score_choices`` returns seeded noise.
"""
from __future__ import annotations

import hashlib
import re
import time

import numpy as np

from deadeye.models.base import (EmbedResult, GenerationResult, LanguageModel, Message, ModelInfo, ScoreResult)

LEGAL_RE = re.compile(r"Legal actions:\s*(.+)")
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class MockModel(LanguageModel):
    capabilities = frozenset({"generate", "score", "embed"})

    def __init__(self, model_id: str = "mock", strategy: str = "random", format_failure_rate: float = 0.0,
                 seed: int = 0, embed_dim: int = 256, params: int | None = 1_000, **_: object) -> None:
        self.info = ModelInfo(id=model_id, backend="mock", params=params, family="mock", instruct=True, precision="none")
        self.strategy = strategy
        self.format_failure_rate = float(format_failure_rate)
        self.rng = np.random.default_rng(seed)
        self.embed_dim = int(embed_dim)

    def _describe(self) -> str:
        return f"mock strategy={self.strategy}"

    @staticmethod
    def _legal(messages: list[Message]) -> list[str]:
        text = "\n".join(m.content for m in messages)
        m = LEGAL_RE.findall(text)
        if not m:
            return []
        return [a.strip() for a in m[-1].split(",") if a.strip()]

    def generate(self, messages, max_new_tokens=64, temperature=0.0, stop=None) -> GenerationResult:
        t0 = time.perf_counter()
        legal = self._legal(messages)
        if self.rng.random() < self.format_failure_rate or not legal:
            text = "I am not sure what to do here."
        else:
            if self.strategy == "first":
                choice = legal[0]
            elif self.strategy == "last":
                choice = legal[-1]
            else:
                choice = legal[int(self.rng.integers(len(legal)))]
            text = f"Action: {choice}"
        n_prompt = sum(len(m.content.split()) for m in messages)
        return GenerationResult(text=text, prompt_tokens=n_prompt, completion_tokens=len(text.split()),
                                latency_s=time.perf_counter() - t0)

    def score_choices(self, messages, choices, length_norm=False) -> ScoreResult:
        t0 = time.perf_counter()
        scores = [float(x) for x in self.rng.normal(-2.0, 1.0, size=len(choices))]
        n_prompt = sum(len(m.content.split()) for m in messages)
        return ScoreResult(logprobs=scores, prompt_tokens=n_prompt, latency_s=time.perf_counter() - t0,
                           choice_tokens=[max(1, len(c.split())) for c in choices])

    def embed(self, messages, layer="mid") -> EmbedResult:
        t0 = time.perf_counter()
        text = "\n".join(m.content for m in messages if m.role != "system")
        vec = np.zeros(self.embed_dim, dtype=np.float32)
        for tok in re.findall(r"[A-Za-z_]+|\d+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.embed_dim] += 1.0 if (h >> 8) % 2 else -1.0
        nums = [float(x) for x in NUM_RE.findall(text)][:32]
        for i, v in enumerate(nums):
            vec[(self.embed_dim - 32) + i] += np.tanh(v / 100.0)
        n_prompt = sum(len(m.content.split()) for m in messages)
        return EmbedResult(vector=vec, prompt_tokens=n_prompt, latency_s=time.perf_counter() - t0, layer=0)
