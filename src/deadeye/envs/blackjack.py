"""Blackjack (hit/stand) against the house: decisions under risk with a known optimal policy.

Rules follow the Gymnasium convention: infinite deck (cards 1-10, with 10 four times as likely),
dealer hits below 17, a natural (two-card 21) pays 1.5. Only *hit* and *stand* are available so the
oracle is the standard basic-strategy hit/stand table. Player and dealer cards come from two separate
pre-drawn streams per seed, so a policy's hits never change the cards the dealer will draw and every
policy faces the same dealer hand (tighter pairing across policies).
"""
from __future__ import annotations

import numpy as np

from deadeye.envs.base import Environment, Observation, StepResult

DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def hand_value(hand: list[int]) -> tuple[int, bool]:
    total = sum(hand)
    usable = 1 in hand and total + 10 <= 21
    return (total + 10 if usable else total), usable


class BlackjackEnv(Environment):
    name = "blackjack"
    primary_metric = "return"
    higher_is_better = True

    def __init__(self, natural_bonus: bool = True, hands_per_episode: int = 1, **params) -> None:
        super().__init__(natural_bonus=natural_bonus, hands_per_episode=hands_per_episode, **params)
        self.natural_bonus = bool(natural_bonus)
        self.hands = int(hands_per_episode)
        self.action_labels = ["hit", "stand"]
        self.max_steps = 12 * self.hands

    def _draw_player(self) -> int:
        card = DECK[int(self._pstream[self._pk])]
        self._pk += 1
        return card

    def _draw_dealer(self) -> int:
        card = DECK[int(self._dstream[self._dk])]
        self._dk += 1
        return card

    def reset(self, seed: int) -> Observation:
        self._rng = np.random.default_rng(seed + 20_011)
        self._pstream = self._rng.integers(len(DECK), size=16 * self.hands)
        self._dstream = self._rng.integers(len(DECK), size=16 * self.hands)
        self._pk = self._dk = 0
        self.hand_idx = 0
        self._outcomes: list[float] = []
        self._busts = 0
        self._deal()
        self._t = 0
        self._done = False
        self._outcome = 0.0
        self._bust = False
        return self._obs()

    def _deal(self) -> None:
        self.player = [self._draw_player(), self._draw_player()]
        self.dealer = [self._draw_dealer(), self._draw_dealer()]

    def step(self, action: str) -> StepResult:
        self._check_action(action, self.action_labels)
        self._t += 1
        reward = 0.0
        hand_over = False
        if action == "hit":
            self.player.append(self._draw_player())
            total, _ = hand_value(self.player)
            if total > 21:
                hand_over, reward = True, -1.0
                self._busts += 1
        else:
            hand_over = True
            ptotal, _ = hand_value(self.player)
            while hand_value(self.dealer)[0] < 17:
                self.dealer.append(self._draw_dealer())
            dtotal, _ = hand_value(self.dealer)
            if dtotal > 21 or ptotal > dtotal:
                reward = 1.0
                if self.natural_bonus and len(self.player) == 2 and ptotal == 21:
                    reward = 1.5
            elif ptotal == dtotal:
                reward = 0.0
            else:
                reward = -1.0
        if hand_over:
            self._outcomes.append(reward)
            self._outcome += reward
            if self.hand_idx + 1 < self.hands:
                self.hand_idx += 1
                self._deal()
            else:
                self._done = True
        return StepResult(self._obs(), reward, self._done, {"hand": self.hand_idx, "hand_over": hand_over})

    def oracle_action(self) -> str:
        total, usable = hand_value(self.player)
        up = self.dealer[0]
        if usable:  # soft totals
            if total >= 19:
                return "stand"
            if total == 18:
                return "stand" if up in (2, 3, 4, 5, 6, 7, 8) else "hit"
            return "hit"
        if total >= 17:
            return "stand"
        if 13 <= total <= 16:
            return "stand" if 2 <= up <= 6 else "hit"
        if total == 12:
            return "stand" if 4 <= up <= 6 else "hit"
        return "hit"

    def episode_metrics(self) -> dict[str, float]:
        n = max(1, len(self._outcomes))
        return {"win": float(sum(1 for o in self._outcomes if o > 0) / n), "loss": float(sum(1 for o in self._outcomes if o < 0) / n),
                "bust": float(self._busts / n), "hands": float(len(self._outcomes))}

    def describe(self) -> str:
        return (
            "You are playing blackjack against the dealer. Cards 2-10 are worth their number, face cards are worth 10 "
            "and an ace is worth 11 if that does not bust you, otherwise 1. You may 'hit' (take another card) or "
            "'stand' (end your turn). If you exceed 21 you bust and lose. After you stand, the dealer draws until "
            "reaching 17 or more. You win if your total is higher than the dealer's or the dealer busts; equal totals "
            "tie. A two-card 21 pays extra. Maximise your expected winnings."
            + (f" You play {self.hands} hands in a row; each hand is settled separately." if self.hands > 1 else "")
        )

    def _obs(self) -> Observation:
        total, usable = hand_value(self.player)
        prefix = f"Hand {self.hand_idx + 1} of {self.hands} (running total {self._outcome:+.1f}).\n" if self.hands > 1 else ""
        text = (prefix + f"Your cards: {', '.join(str(c) for c in self.player)} (total {total}{', with an ace counted as 11' if usable else ''}).\n"
                f"Dealer shows: {self.dealer[0]}.")
        feats = np.array([total / 21, float(usable), self.dealer[0] / 10], dtype=np.float32)
        return Observation(text=text, legal_actions=list(self.action_labels), features=feats, info={"t": self._t})
