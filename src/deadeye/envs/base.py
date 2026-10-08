"""Environment protocol shared by every DeadEye task.

Design rules that every environment follows:

1. ``reset(seed)`` fully determines the episode instance (hidden parameters, pre-drawn noise).
   Two policies evaluated on the same seed face the same instance, which enables paired
   comparisons (common random numbers) and lowers variance.
2. The observation carries a *text* rendering for language models, an optional numeric
   *feature* vector for probes and classical baselines, and the list of *legal actions*.
   Every environment is Markov in its text rendering: the text alone is enough to act optimally.
3. ``oracle_action()`` returns an optimal (or canonical near-optimal) action for the current
   state. Oracles are used (a) as the upper reference point for score normalisation, (b) as the
   label source for probes and behaviour cloning, and (c) to compute regret where it is defined.
4. ``describe()`` returns the task rules in plain language. Policies put it in the system prompt.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    text: str
    legal_actions: list[str]
    features: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    obs: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """Base class for all tasks. Subclasses set ``name``, ``action_labels`` and ``max_steps``."""

    name: str = "base"
    #: Every action label that can appear in ``legal_actions`` over the life of the environment.
    action_labels: list[str] = []
    #: Hard cap on decisions per episode (the runner enforces it as a safety net).
    max_steps: int = 100
    #: Name of the env-specific headline metric reported alongside the return, and its direction.
    primary_metric: str = "return"
    higher_is_better: bool = True

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._done = True

    # ------------------------------------------------------------------ protocol
    @abstractmethod
    def reset(self, seed: int) -> Observation:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def step(self, action: str) -> StepResult:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def oracle_action(self) -> str:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def describe(self) -> str:  # pragma: no cover - abstract
        ...

    # ------------------------------------------------------------------ optional hooks
    def episode_metrics(self) -> dict[str, float]:
        """Env-specific metrics for the finished episode (success, regret, ...)."""
        return {}

    @property
    def done(self) -> bool:
        return self._done

    @property
    def t(self) -> int:
        return self._t

    # ------------------------------------------------------------------ helpers
    def _check_action(self, action: str, legal: list[str]) -> None:
        if action not in legal:
            raise ValueError(f"illegal action {action!r}; legal: {legal}")

    @staticmethod
    def fmt(x: float, nd: int = 2) -> str:
        return f"{x:.{nd}f}"
