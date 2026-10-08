"""Non-LLM reference policies: random (lower anchor), oracle (upper anchor), classical algorithms."""
from __future__ import annotations

import numpy as np

from deadeye.envs.bandit import BanditEnv, UCB1Policy
from deadeye.envs.base import Observation
from deadeye.policies.base import Decision, HistoryStep, Policy


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Observation, history: list[HistoryStep]) -> Decision:
        return Decision(action=str(self.rng.choice(obs.legal_actions)), raw_output="random")


class OraclePolicy(Policy):
    name = "oracle"

    def act(self, obs: Observation, history: list[HistoryStep]) -> Decision:
        assert self.env is not None
        return Decision(action=self.env.oracle_action(), raw_output="oracle")


class UCB1(Policy):
    """Bandit-only classical baseline (Auer et al. 2002)."""

    name = "ucb1"

    def bind(self, env) -> None:
        if not isinstance(env, BanditEnv):
            raise TypeError("ucb1 only applies to the bandit environment")
        super().bind(env)
        self.algo = UCB1Policy(env)

    def act(self, obs: Observation, history: list[HistoryStep]) -> Decision:
        return Decision(action=self.algo.act(obs, history), raw_output="ucb1")
