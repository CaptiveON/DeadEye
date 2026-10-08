"""Contextual bandit with hidden linear-logistic reward functions.

Each round shows a numeric context ``x`` (``dim`` features, abstract names). Each action ``a`` has a
hidden weight vector ``w_a``; the expected reward is ``sigmoid(w_a . x / sqrt(dim))`` and realised
rewards are Bernoulli. The agent only sees past (context, action, reward) triples, so it must learn
the mapping in-context. The oracle knows ``w`` and picks the arm with the highest expected reward.
"""
from __future__ import annotations

import numpy as np

from deadeye.envs.base import Environment, Observation, StepResult


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class ContextualBanditEnv(Environment):
    name = "contextual_bandit"
    primary_metric = "regret"
    higher_is_better = False

    def __init__(self, n_actions: int = 3, dim: int = 4, horizon: int = 30, history_window: int = 10,
                 weight_scale: float = 2.0, **params) -> None:
        super().__init__(n_actions=n_actions, dim=dim, horizon=horizon, history_window=history_window,
                         weight_scale=weight_scale, **params)
        self.n_actions = int(n_actions)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.history_window = int(history_window)
        self.weight_scale = float(weight_scale)
        self.action_labels = [f"option_{chr(ord('A') + i)}" for i in range(self.n_actions)]
        self.max_steps = self.horizon

    def reset(self, seed: int) -> Observation:
        self._rng = np.random.default_rng(seed)
        self.W = self._rng.normal(0, self.weight_scale, size=(self.n_actions, self.dim))
        self.contexts = np.round(self._rng.normal(0, 1, size=(self.horizon, self.dim)), 2)
        self.probs = _sigmoid(self.contexts @ self.W.T / np.sqrt(self.dim))  # (T, A)
        self.reward_table = (self._rng.random((self.horizon, self.n_actions)) < self.probs).astype(float)
        self.history: list[tuple[np.ndarray, int, float]] = []
        self._t = 0
        self._done = False
        self._regret = 0.0
        return self._obs()

    def step(self, action: str) -> StepResult:
        self._check_action(action, self.action_labels)
        a = self.action_labels.index(action)
        r = float(self.reward_table[self._t, a])
        best = float(self.probs[self._t].max())
        self._regret += best - float(self.probs[self._t, a])
        self.history.append((self.contexts[self._t], a, r))
        optimal = a == int(np.argmax(self.probs[self._t]))
        self._t += 1
        self._done = self._t >= self.horizon
        return StepResult(self._obs(), r, self._done, {"optimal": optimal})

    def oracle_action(self) -> str:
        return self.action_labels[int(np.argmax(self.probs[self._t]))]

    def episode_metrics(self) -> dict[str, float]:
        n = max(1, len(self.history))
        opt = sum(1 for (x, a, r) in self.history if a == int(np.argmax(_sigmoid(x @ self.W.T / np.sqrt(self.dim)))))
        return {"regret": self._regret, "optimal_action_frac": opt / n}

    def describe(self) -> str:
        names = ", ".join(self.action_labels)
        return (
            f"You are making {self.horizon} sequential decisions. Each round you see a context made of {self.dim} "
            f"numeric features (x1..x{self.dim}) and must choose one of {self.n_actions} options ({names}). "
            "Each option's success probability depends on the context through a fixed but unknown rule. "
            "You receive a reward of 1 (success) or 0 (failure). Use the history of past contexts, choices and "
            "rewards to infer which option works best for which contexts, and maximise total reward."
        )

    def _ctx_str(self, x: np.ndarray) -> str:
        return ", ".join(f"x{i + 1}={v:+.2f}" for i, v in enumerate(x))

    def _obs(self) -> Observation:
        t = min(self._t, self.horizon - 1)
        lines = [f"Round {self._t + 1} of {self.horizon}."]
        if self.history:
            lines.append(f"Past rounds (most recent last, showing up to {self.history_window}):")
            for x, a, r in self.history[-self.history_window:]:
                lines.append(f"  context [{self._ctx_str(x)}] -> chose {self.action_labels[a]} -> reward {int(r)}")
        else:
            lines.append("No past rounds yet.")
        lines.append(f"Current context: [{self._ctx_str(self.contexts[t])}]")
        return Observation(text="\n".join(lines), legal_actions=list(self.action_labels),
                           features=self.contexts[t].astype(np.float32), info={"t": self._t})
