"""Prompt construction shared by every prompt-based conversion method.

The prompt has a fixed skeleton so that the *only* differences between conditions are the ones the
experiment manipulates (action format, history window, few-shot demonstrations, chain-of-thought).
Every prompt contains a ``Legal actions:`` line, which the mock backend also relies on.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any

from deadeye.envs.base import Environment, Observation
from deadeye.models.base import Message
from deadeye.policies.base import HistoryStep

LETTERS = string.ascii_uppercase


@dataclass
class PromptConfig:
    action_format: str = "name"  # "name" | "letter"
    history_window: int = 0  # number of past (observation, action, reward) steps to show
    few_shot: int = 0  # number of oracle demonstrations (from training seeds) to prepend
    cot: bool = False  # ask for brief reasoning before the final action line
    mode: str = "generate"  # "generate" | "score" (changes only the answer instruction)
    extra_system: str = ""  # free text appended to the system prompt (for ablations)
    demos: list[dict[str, Any]] = field(default_factory=list)  # filled by the policy from training seeds

    def to_dict(self) -> dict[str, Any]:
        return {"action_format": self.action_format, "history_window": self.history_window, "few_shot": self.few_shot,
                "cot": self.cot, "mode": self.mode, "extra_system": self.extra_system}


def label_map(legal: list[str], action_format: str) -> dict[str, str]:
    """Map the label the model should emit -> the environment action."""
    if action_format == "name":
        return {a: a for a in legal}
    if action_format == "letter":
        return {LETTERS[i]: a for i, a in enumerate(legal)}
    raise ValueError(f"unknown action_format {action_format!r}")


def legal_block(legal: list[str], action_format: str) -> str:
    lm = label_map(legal, action_format)
    if action_format == "name":
        return "Legal actions: " + ", ".join(legal)
    lines = ["Legal actions: " + ", ".join(lm)]
    lines += [f"  {k} = {v}" for k, v in lm.items()]
    return "\n".join(lines)


def answer_instruction(cfg: PromptConfig) -> str:
    what = "the letter" if cfg.action_format == "letter" else "the action name"
    if cfg.mode == "score":
        return f"Answer with only {what} of your chosen action."
    if cfg.cot:
        return (f"Think step by step in at most three short sentences, then end your reply with a final line of the "
                f"form `Action: <{what.split()[-1]}>`.")
    return f"Reply with exactly one line of the form `Action: <{what.split()[-1]}>` and nothing else."


def system_prompt(env: Environment, cfg: PromptConfig) -> str:
    text = env.describe()
    if cfg.extra_system:
        text += "\n\n" + cfg.extra_system
    return text


def user_message(obs: Observation, history: list[HistoryStep], cfg: PromptConfig) -> str:
    parts = []
    if cfg.history_window > 0 and history:
        parts.append("Recent steps:")
        for h in history[-cfg.history_window:]:
            first = h.obs_text.strip().splitlines()[0]
            parts.append(f"  {first} -> {h.action} -> reward {h.reward:+.2f}")
        parts.append("")
    parts.append(obs.text.strip())
    parts.append(legal_block(obs.legal_actions, cfg.action_format))
    parts.append(answer_instruction(cfg))
    return "\n".join(parts)


def build_messages(env: Environment, obs: Observation, history: list[HistoryStep], cfg: PromptConfig) -> tuple[list[Message], dict[str, str]]:
    msgs = [Message("system", system_prompt(env, cfg))]
    for demo in cfg.demos[:cfg.few_shot]:
        demo_obs = Observation(text=demo["obs_text"], legal_actions=demo["legal_actions"])
        msgs.append(Message("user", user_message(demo_obs, [], cfg)))
        lm = label_map(demo["legal_actions"], cfg.action_format)
        label = next(k for k, v in lm.items() if v == demo["action"])
        if cfg.mode == "score":
            msgs.append(Message("assistant", label))
        elif cfg.cot:
            msgs.append(Message("assistant", f"{demo.get('rationale', 'Choosing the best available option.')}\nAction: {label}"))
        else:
            msgs.append(Message("assistant", f"Action: {label}"))
    msgs.append(Message("user", user_message(obs, history, cfg)))
    return msgs, label_map(obs.legal_actions, cfg.action_format)


def collect_demos(env_factory, seeds: list[int], n: int) -> list[dict[str, Any]]:
    """Oracle demonstrations: one (observation, action) pair per training seed, at a random step."""
    import numpy as np
    demos = []
    rng = np.random.default_rng(12345)
    for seed in seeds:
        if len(demos) >= n:
            break
        env = env_factory()
        obs = env.reset(seed)
        # roll forward a random number of oracle steps so demos cover mid-episode states too
        k = int(rng.integers(0, max(1, min(5, env.max_steps))))
        for _ in range(k):
            if env.done:
                break
            obs = env.step(env.oracle_action()).obs
        if env.done:
            obs = env.reset(seed)
        demos.append({"obs_text": obs.text, "legal_actions": list(obs.legal_actions), "action": env.oracle_action()})
    return demos
