"""LoRA behaviour cloning: fine-tune the model on oracle (observation -> action) pairs, then act.

Inference after training uses either exact action scoring (``infer="score"``, default) or free-form
generation (``infer="generate"``). Requires the ``hf`` backend.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from deadeye.envs.base import Environment, Observation
from deadeye.models.base import LanguageModel, Message
from deadeye.policies.base import Decision, HistoryStep, Policy
from deadeye.policies.prompting import PromptGeneratePolicy, PromptScorePolicy
from deadeye.policies.prompts import PromptConfig, build_messages, label_map


class LoraSFTPolicy(Policy):
    name = "lora_sft"
    mutates_model = True
    needs_prepare = True

    def __init__(self, model: LanguageModel, prompt: PromptConfig, n_train_episodes: int = 40, oracle_mix: float = 0.7,
                 infer: str = "score", max_train_states: int = 4000, seed: int = 0, lora: dict[str, Any] | None = None,
                 max_new_tokens: int = 16) -> None:
        super().__init__()
        if "train" not in model.capabilities:
            raise TypeError("lora_sft requires a trainable backend (hf)")
        self.model = model
        self.prompt = prompt
        self.n_train_episodes = int(n_train_episodes)
        self.oracle_mix = float(oracle_mix)
        self.infer = infer
        self.max_train_states = int(max_train_states)
        self.seed = int(seed)
        self.lora = dict(lora or {})
        self.max_new_tokens = int(max_new_tokens)
        train_prompt = PromptConfig(**{**prompt.to_dict(), "mode": "score" if infer == "score" else "generate", "cot": False})
        self.train_prompt = train_prompt
        if infer == "score":
            self.actor: Policy = PromptScorePolicy(model, train_prompt)
        elif infer == "generate":
            self.actor = PromptGeneratePolicy(model, train_prompt, max_new_tokens=max_new_tokens)
        else:
            raise ValueError("infer must be 'score' or 'generate'")

    def bind(self, env: Environment) -> None:
        super().bind(env)
        self.actor.bind(env)

    def prepare(self, env_factory: Callable[[], Environment], train_seeds: list[int]) -> dict[str, Any]:
        from deadeye.models.training import train_lora_sft

        rng = np.random.default_rng(self.seed)
        examples: list[tuple[list[Message], str]] = []
        for seed in train_seeds[:self.n_train_episodes]:
            env = env_factory()
            obs = env.reset(seed)
            history: list[HistoryStep] = []
            while not env.done and len(examples) < self.max_train_states:
                oracle = env.oracle_action()
                msgs, lm = build_messages(env, obs, history, self.train_prompt)
                label = next(k for k, v in lm.items() if v == oracle)
                target = label if self.infer == "score" else f"Action: {label}"
                examples.append((msgs, target))
                action = oracle if rng.random() < self.oracle_mix else str(rng.choice(obs.legal_actions))
                res = env.step(action)
                history.append(HistoryStep(obs.text, action, res.reward))
                obs = res.obs
        stats = train_lora_sft(self.model, examples, seed=self.seed, **self.lora)
        return stats

    def act(self, obs: Observation, history: list[HistoryStep]) -> Decision:
        return self.actor.act(obs, history)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model.info.id, "infer": self.infer, "n_train_episodes": self.n_train_episodes,
                "oracle_mix": self.oracle_mix, "lora": self.lora, **self.train_prompt.to_dict()}
