"""Policy protocol: a conversion method turns a language model (or nothing) into an actor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from deadeye.envs.base import Environment, Observation


@dataclass
class HistoryStep:
    obs_text: str
    action: str
    reward: float


@dataclass
class Decision:
    action: str | None  # parsed action; None when the output could not be mapped to a legal action
    raw_output: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Policy(ABC):
    name: str = "policy"
    #: True when ``prepare`` changes the underlying model weights (LoRA); the runner then loads a fresh model per cell.
    mutates_model: bool = False
    #: True when the policy needs ``prepare`` (an offline training phase) before acting.
    needs_prepare: bool = False

    def __init__(self) -> None:
        self.env: Environment | None = None

    def bind(self, env: Environment) -> None:
        """Attach the environment instance the policy will act in (same instance the runner steps)."""
        self.env = env

    def prepare(self, env_factory: Callable[[], Environment], train_seeds: list[int]) -> dict[str, Any]:
        """Optional offline phase (probe training, behaviour cloning) on seeds disjoint from evaluation."""
        return {}

    def reset(self) -> None:
        pass

    @abstractmethod
    def act(self, obs: Observation, history: list[HistoryStep]) -> Decision:  # pragma: no cover
        ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}
