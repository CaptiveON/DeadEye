from __future__ import annotations

from deadeye.envs.bandit import BanditEnv
from deadeye.envs.base import Environment
from deadeye.envs.blackjack import BlackjackEnv
from deadeye.envs.contextual_bandit import ContextualBanditEnv
from deadeye.envs.gridworld import GridworldEnv
from deadeye.envs.loan import LoanEnv
from deadeye.envs.tictactoe import TicTacToeEnv

ENV_REGISTRY: dict[str, type[Environment]] = {
    BanditEnv.name: BanditEnv,
    ContextualBanditEnv.name: ContextualBanditEnv,
    LoanEnv.name: LoanEnv,
    GridworldEnv.name: GridworldEnv,
    TicTacToeEnv.name: TicTacToeEnv,
    BlackjackEnv.name: BlackjackEnv,
}


def make_env(name: str, **params) -> Environment:
    try:
        cls = ENV_REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"unknown environment {name!r}; known: {sorted(ENV_REGISTRY)}") from e
    return cls(**params)
