"""Language-model backend protocol.

A backend exposes up to three primitives, each of which supports a family of conversion methods:

* ``generate``      - free-form text (prompt-generate policies)
* ``score_choices`` - log-likelihood of candidate continuations (prompt-score policies)
* ``embed``         - a hidden-state vector for the prompt (probe policies)

Backends declare what they support in ``capabilities``. Every call reports latency and token counts
so that the report can build cost-vs-quality frontiers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ModelInfo:
    id: str
    backend: str
    params: int | None = None  # total parameter count when known
    family: str | None = None
    instruct: bool | None = None
    precision: str | None = None  # e.g. "bf16", "int4"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "backend": self.backend, "params": self.params, "family": self.family,
                "instruct": self.instruct, "precision": self.precision, **{f"extra_{k}": v for k, v in self.extra.items()}}


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    raw: Any = None


@dataclass
class ScoreResult:
    logprobs: list[float]  # one summed log-probability per choice
    prompt_tokens: int
    latency_s: float
    choice_tokens: list[int] = field(default_factory=list)


@dataclass
class EmbedResult:
    vector: np.ndarray
    prompt_tokens: int
    latency_s: float
    layer: int = -1


def render_messages_plain(messages: list[Message]) -> str:
    """Fallback rendering for models without a chat template (base models, mocks)."""
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(m.content.strip())
        elif m.role == "user":
            parts.append("Input:\n" + m.content.strip())
        else:
            parts.append("Output:\n" + m.content.strip())
    return "\n\n".join(parts) + "\n\nOutput:\n"


class LanguageModel(ABC):
    info: ModelInfo
    capabilities: frozenset[str] = frozenset()

    def generate(self, messages: list[Message], max_new_tokens: int = 64, temperature: float = 0.0,
                 stop: list[str] | None = None) -> GenerationResult:
        raise NotImplementedError(f"{self.info.id} does not support generate")

    def score_choices(self, messages: list[Message], choices: list[str], length_norm: bool = False) -> ScoreResult:
        raise NotImplementedError(f"{self.info.id} does not support score_choices")

    def embed(self, messages: list[Message], layer: int | str = "mid") -> EmbedResult:
        raise NotImplementedError(f"{self.info.id} does not support embed")

    def close(self) -> None:
        pass

    @abstractmethod
    def _describe(self) -> str:  # pragma: no cover
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._describe()}>"


def resolve_layer(layer: int | str, n_layers: int) -> int:
    """Map "mid"/"last"/int to a non-negative index into the hidden-state stack (0 = embeddings)."""
    if isinstance(layer, str):
        if layer == "last":
            return n_layers
        if layer == "mid":
            return n_layers // 2
        if layer.lstrip("-").isdigit():
            layer = int(layer)
        else:
            raise ValueError(f"bad layer spec {layer!r}")
    if layer < 0:
        layer = n_layers + 1 + layer
    if not 0 <= layer <= n_layers:
        raise ValueError(f"layer {layer} out of range for {n_layers} layers")
    return int(layer)
