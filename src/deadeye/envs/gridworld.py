"""Gridworld navigation: deterministic multi-step planning with full observability.

The agent sees an ASCII map plus coordinates and must reach the goal in as few steps as possible.
Walls are placed randomly; instances are re-drawn until a path exists. The oracle is the first step
of a breadth-first-search shortest path (ties broken in a fixed order so labels are deterministic).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from deadeye.envs.base import Environment, Observation, StepResult

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class GridworldEnv(Environment):
    name = "gridworld"
    primary_metric = "success"
    higher_is_better = True

    def __init__(self, size: int = 5, n_walls: int | None = None, max_steps: int | None = None,
                 step_penalty: float = 0.02, min_distance: int | None = None, **params) -> None:
        super().__init__(size=size, n_walls=n_walls, max_steps=max_steps, step_penalty=step_penalty,
                         min_distance=min_distance, **params)
        self.size = int(size)
        self.min_distance = int(min_distance) if min_distance is not None else self.size
        self.n_walls = int(n_walls) if n_walls is not None else self.size
        self.max_steps = int(max_steps) if max_steps is not None else 4 * self.size
        self.step_penalty = float(step_penalty)
        self.action_labels = list(MOVES)

    # ------------------------------------------------------------------ instance
    def _bfs(self, start: tuple[int, int]) -> dict[tuple[int, int], int]:
        """Distance-to-goal map from every reachable cell (BFS from the goal)."""
        dist = {self.goal: 0}
        q = deque([self.goal])
        while q:
            cur = q.popleft()
            for d in MOVES.values():
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if self._free(nxt) and nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist

    def _free(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.size and 0 <= c < self.size and not self.walls[r, c]

    def reset(self, seed: int) -> Observation:
        self._rng = np.random.default_rng(seed)
        while True:
            self.walls = np.zeros((self.size, self.size), dtype=bool)
            cells = [(r, c) for r in range(self.size) for c in range(self.size)]
            idx = self._rng.permutation(len(cells))
            self.pos = cells[idx[0]]
            self.goal = cells[idx[1]]
            for k in idx[2:2 + self.n_walls]:
                self.walls[cells[k]] = True
            self.dist = self._bfs(self.pos)
            if self.pos in self.dist and self.dist[self.pos] >= self.min_distance:
                break
        self.optimal_steps = self.dist[self.pos]
        self._t = 0
        self._done = False
        self._success = False
        return self._obs()

    def _legal(self) -> list[str]:
        return [m for m, d in MOVES.items() if self._free((self.pos[0] + d[0], self.pos[1] + d[1]))]

    def step(self, action: str) -> StepResult:
        legal = self._legal()
        self._check_action(action, legal)
        d = MOVES[action]
        self.pos = (self.pos[0] + d[0], self.pos[1] + d[1])
        self._t += 1
        reward = -self.step_penalty
        if self.pos == self.goal:
            reward += 1.0
            self._success = True
            self._done = True
        elif self._t >= self.max_steps:
            self._done = True
        return StepResult(self._obs(), reward, self._done, {"pos": self.pos})

    def oracle_action(self) -> str:
        best, best_d = None, None
        for m in self.action_labels:  # fixed order => deterministic tie-break
            d = MOVES[m]
            nxt = (self.pos[0] + d[0], self.pos[1] + d[1])
            if nxt in self.dist and (best_d is None or self.dist[nxt] < best_d):
                best, best_d = m, self.dist[nxt]
        assert best is not None
        return best

    def episode_metrics(self) -> dict[str, float]:
        return {"success": float(self._success), "steps": float(self._t), "optimal_steps": float(self.optimal_steps),
                "excess_steps": float(self._t - self.optimal_steps) if self._success else float(self.max_steps - self.optimal_steps)}

    def describe(self) -> str:
        return (
            f"You are navigating a {self.size}x{self.size} grid. 'A' marks your position, 'G' the goal, '#' walls and "
            f"'.' free cells. Rows are numbered top to bottom and columns left to right, both starting at 0. "
            "Each move goes one cell up, down, left or right; you cannot move into walls or off the grid. "
            f"Reach the goal in as few moves as possible (at most {self.max_steps} moves)."
        )

    def _obs(self) -> Observation:
        rows = []
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) == self.pos:
                    row += "A"
                elif (r, c) == self.goal:
                    row += "G"
                elif self.walls[r, c]:
                    row += "#"
                else:
                    row += "."
            rows.append(row)
        text = (f"Move {self._t + 1} (max {self.max_steps}).\nMap:\n" + "\n".join(rows) +
                f"\nYou are at row {self.pos[0]}, column {self.pos[1]}. The goal is at row {self.goal[0]}, column {self.goal[1]}.")
        feats = np.concatenate([[self.pos[0] / self.size, self.pos[1] / self.size, self.goal[0] / self.size, self.goal[1] / self.size],
                                self.walls.astype(np.float32).ravel()])
        return Observation(text=text, legal_actions=self._legal(), features=feats.astype(np.float32), info={"t": self._t})
