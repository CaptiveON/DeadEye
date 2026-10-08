"""Tic-tac-toe against a fixed opponent: adversarial decision-making with perfect information.

Cells are numbered 1-9 left-to-right, top-to-bottom. The oracle is minimax with a preference for
faster wins / slower losses, with deterministic tie-breaking. Against a random opponent the oracle
wins most games; against the minimax opponent the best achievable outcome is a draw.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from deadeye.envs.base import Environment, Observation, StepResult

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(board: tuple[str, ...]) -> str | None:
    for a, b, c in LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    if "." not in board:
        return "draw"
    return None


@lru_cache(maxsize=None)
def minimax(board: tuple[str, ...], player: str, me: str) -> tuple[float, int | None]:
    """Return (value for `me`, best move) with depth-aware scoring (faster wins preferred)."""
    w = winner(board)
    if w is not None:
        empties = board.count(".")
        if w == "draw":
            return 0.0, None
        return (1.0 + empties) if w == me else -(1.0 + empties), None
    other = "O" if player == "X" else "X"
    best_val, best_move = None, None
    for i, cell in enumerate(board):
        if cell != ".":
            continue
        nb = board[:i] + (player,) + board[i + 1:]
        val, _ = minimax(nb, other, me)
        if best_val is None or (player == me and val > best_val) or (player != me and val < best_val):
            best_val, best_move = val, i
    return best_val, best_move


class TicTacToeEnv(Environment):
    name = "tictactoe"
    primary_metric = "win"
    higher_is_better = True

    def __init__(self, opponent: str = "random", agent_mark: str = "alternate", **params) -> None:
        super().__init__(opponent=opponent, agent_mark=agent_mark, **params)
        self.opponent = opponent
        self.agent_mark_mode = agent_mark
        self.action_labels = [str(i) for i in range(1, 10)]
        self.max_steps = 5
        self.me, self.opp = ("O", "X") if agent_mark == "O" else ("X", "O")
        self.board = tuple(["."] * 9)
        self._outcome = 0.0

    def reset(self, seed: int) -> Observation:
        self._rng = np.random.default_rng(seed + 10_007)
        if self.agent_mark_mode == "alternate":
            self.me = "X" if seed % 2 == 0 else "O"
        else:
            self.me = self.agent_mark_mode
        self.opp = "O" if self.me == "X" else "X"
        self.board = tuple(["."] * 9)
        self._t = 0
        self._done = False
        self._outcome = 0.0
        if self.me == "O":
            self._opponent_move()
        return self._obs()

    def _legal(self) -> list[str]:
        return [str(i + 1) for i, c in enumerate(self.board) if c == "."]

    def _place(self, i: int, mark: str) -> None:
        self.board = self.board[:i] + (mark,) + self.board[i + 1:]

    def _opponent_move(self) -> None:
        if winner(self.board) is not None:
            return
        if self.opponent == "random":
            empties = [i for i, c in enumerate(self.board) if c == "."]
            self._place(int(self._rng.choice(empties)), self.opp)
        elif self.opponent == "minimax":
            _, mv = minimax(self.board, self.opp, self.opp)
            self._place(int(mv), self.opp)
        else:
            raise ValueError(f"unknown opponent {self.opponent!r}")

    def step(self, action: str) -> StepResult:
        self._check_action(action, self._legal())
        self._place(int(action) - 1, self.me)
        self._t += 1
        w = winner(self.board)
        if w is None:
            self._opponent_move()
            w = winner(self.board)
        reward = 0.0
        if w is not None:
            self._done = True
            if w == self.me:
                reward = 1.0
            elif w == "draw":
                reward = 0.0
            else:
                reward = -1.0
            self._outcome = reward
        return StepResult(self._obs(), reward, self._done, {"winner": w})

    def oracle_action(self) -> str:
        _, mv = minimax(self.board, self.me, self.me)
        return str(int(mv) + 1)

    def episode_metrics(self) -> dict[str, float]:
        return {"win": float(self._outcome > 0), "draw": float(self._outcome == 0), "loss": float(self._outcome < 0)}

    def describe(self) -> str:
        return (
            f"You are playing tic-tac-toe as '{self.me}' against an opponent playing '{self.opp}'. The board cells are "
            "numbered 1 to 9 from left to right, top to bottom. Three of your marks in a row, column or diagonal wins. "
            "Choose the number of an empty cell. Win if you can, otherwise force a draw; never lose if avoidable."
        )

    def _obs(self) -> Observation:
        cells = [c if c != "." else str(i + 1) for i, c in enumerate(self.board)]
        rows = [" ".join(cells[r * 3:(r + 1) * 3]) for r in range(3)]
        text = f"You are '{self.me}'. Board (numbers mark empty cells):\n" + "\n".join(rows)
        feats = np.array([1.0 if c == self.me else (-1.0 if c == self.opp else 0.0) for c in self.board], dtype=np.float32)
        return Observation(text=text, legal_actions=self._legal(), features=feats, info={"t": self._t})
