"""Stochastic multi-armed bandit: the canonical exploration/exploitation task.

Instance structure follows Krishnamurthy et al. (2024): one best arm with mean ``0.5 + gap/2``
and ``n_arms - 1`` arms with mean ``0.5 - gap/2`` (``means_mode="gap"``), or arm means drawn
uniformly (``means_mode="uniform"``). Rewards are Bernoulli. The whole reward table is pre-drawn
at reset so every policy sees the same potential outcomes for a given seed.
"""
from __future__ import annotations

import numpy as np

from deadeye.envs.base import Environment, Observation, StepResult


class BanditEnv(Environment):
    name = "bandit"
    primary_metric = "regret"
    higher_is_better = False

    def __init__(
        self,
        n_arms: int = 5,
        horizon: int = 50,
        gap: float = 0.2,
        means_mode: str = "gap",
        history_format: str = "summary",
        **params,
    ) -> None:
        super().__init__(n_arms=n_arms, horizon=horizon, gap=gap, means_mode=means_mode,
                         history_format=history_format, **params)
        self.n_arms = int(n_arms)
        self.horizon = int(horizon)
        self.gap = float(gap)
        self.means_mode = means_mode
        self.history_format = history_format
        self.action_labels = [f"arm_{i + 1}" for i in range(self.n_arms)]
        self.max_steps = self.horizon

    # ------------------------------------------------------------------
    def reset(self, seed: int) -> Observation:
        self._rng = np.random.default_rng(seed)
        if self.means_mode == "gap":
            means = np.full(self.n_arms, 0.5 - self.gap / 2)
            best = int(self._rng.integers(self.n_arms))
            means[best] = 0.5 + self.gap / 2
        elif self.means_mode == "uniform":
            means = self._rng.uniform(0.1, 0.9, size=self.n_arms)
        else:
            raise ValueError(f"unknown means_mode {self.means_mode!r}")
        self.means = means
        self.best_arm = int(np.argmax(means))
        # Potential outcomes for every (round, arm): common random numbers across policies.
        self.reward_table = (self._rng.random((self.horizon, self.n_arms)) < means[None, :]).astype(float)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)
        self.pulls: list[int] = []
        self.rewards: list[float] = []
        self._t = 0
        self._done = False
        self._regret = 0.0
        return self._obs()

    def step(self, action: str) -> StepResult:
        legal = self.action_labels
        self._check_action(action, legal)
        arm = legal.index(action)
        r = float(self.reward_table[self._t, arm])
        self.counts[arm] += 1
        self.sums[arm] += r
        self.pulls.append(arm)
        self.rewards.append(r)
        self._regret += float(self.means[self.best_arm] - self.means[arm])
        self._t += 1
        self._done = self._t >= self.horizon
        return StepResult(self._obs(), r, self._done, {"arm": arm, "optimal": arm == self.best_arm})

    def oracle_action(self) -> str:
        return self.action_labels[self.best_arm]

    def episode_metrics(self) -> dict[str, float]:
        n = max(1, len(self.pulls))
        return {
            "regret": self._regret,
            "suboptimal_pull_frac": float(sum(1 for a in self.pulls if a != self.best_arm) / n),
            "best_arm_frac_last10": float(np.mean([a == self.best_arm for a in self.pulls[-10:]])) if self.pulls else 0.0,
        }

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return (
            f"You are playing a {self.n_arms}-armed bandit for {self.horizon} rounds. Each round you choose one arm "
            "and receive a reward of 1 or 0. Each arm has a fixed but unknown success probability. "
            "Your goal is to maximise the total reward over all rounds, which requires balancing exploration "
            "(trying arms to learn about them) and exploitation (choosing the arm that looks best)."
        )

    def _obs(self) -> Observation:
        lines = [f"Round {self._t + 1} of {self.horizon}."]
        if self.history_format == "summary":
            lines.append("Summary of past pulls:")
            for i, label in enumerate(self.action_labels):
                if self.counts[i] == 0:
                    lines.append(f"  {label}: never pulled")
                else:
                    lines.append(f"  {label}: pulled {self.counts[i]} times, average reward {self.sums[i] / self.counts[i]:.2f}")
        elif self.history_format == "raw":
            if self.pulls:
                hist = ", ".join(f"{self.action_labels[a]}->{int(r)}" for a, r in zip(self.pulls, self.rewards))
                lines.append(f"History (arm->reward): {hist}")
            else:
                lines.append("History: none yet.")
        else:
            raise ValueError(f"unknown history_format {self.history_format!r}")
        feats = np.concatenate([self.counts / max(1, self._t), np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.5)])
        return Observation(text="\n".join(lines), legal_actions=list(self.action_labels), features=feats.astype(np.float32),
                           info={"t": self._t})


class UCB1Policy:
    """Classical algorithmic baseline for the bandit (Auer et al., 2002). Not a language model."""

    name = "ucb1"

    def __init__(self, env: BanditEnv) -> None:
        self.env = env

    def reset(self) -> None:
        pass

    def act(self, obs: Observation, history) -> str:
        env = self.env
        for i in range(env.n_arms):
            if env.counts[i] == 0:
                return env.action_labels[i]
        t = env._t
        ucb = env.sums / env.counts + np.sqrt(2 * np.log(t + 1) / env.counts)
        return env.action_labels[int(np.argmax(ucb))]
