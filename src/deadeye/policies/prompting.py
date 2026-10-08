"""Prompt-based conversion methods: free-form generation and log-likelihood action scoring."""
from __future__ import annotations

import time
from typing import Any, Callable

from deadeye.envs.base import Environment, Observation
from deadeye.models.base import LanguageModel
from deadeye.policies.base import Decision, HistoryStep, Policy
from deadeye.policies.parsing import parse_action
from deadeye.policies.prompts import PromptConfig, build_messages, collect_demos


class _PromptPolicy(Policy):
    def __init__(self, model: LanguageModel, prompt: PromptConfig) -> None:
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.needs_prepare = prompt.few_shot > 0

    def prepare(self, env_factory: Callable[[], Environment], train_seeds: list[int]) -> dict[str, Any]:
        if self.prompt.few_shot > 0:
            self.prompt.demos = collect_demos(env_factory, train_seeds, self.prompt.few_shot)
        return {"n_demos": len(self.prompt.demos)}

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model.info.id, **self.prompt.to_dict()}


class PromptGeneratePolicy(_PromptPolicy):
    name = "prompt_generate"

    def __init__(self, model: LanguageModel, prompt: PromptConfig, max_new_tokens: int = 32, temperature: float = 0.0,
                 stop: list[str] | None = None) -> None:
        prompt.mode = "generate"
        super().__init__(model, prompt)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.stop = stop

    def act(self, obs: Observation, history: list[HistoryStep]) -> Decision:
        assert self.env is not None
        msgs, lm = build_messages(self.env, obs, history, self.prompt)
        gen = self.model.generate(msgs, max_new_tokens=self.max_new_tokens, temperature=self.temperature, stop=self.stop)
        label, think = parse_action(gen.text, list(lm))
        action = lm.get(label) if label is not None else None
        extra: dict[str, Any] = {"label": label}
        if think:
            extra["think_chars"] = len(think)
        return Decision(action=action, raw_output=gen.text, latency_s=gen.latency_s, prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens, extra=extra)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "max_new_tokens": self.max_new_tokens, "temperature": self.temperature}


class PromptFreePolicy(PromptGeneratePolicy):
    """The 'before conversion' condition: the model answers the situation in its own words, with no output
    format imposed, and an action is extracted from the free text if one can be. Measures what an unconverted
    assistant costs (latency, tokens) and how often its reply contains a usable decision at all."""

    name = "prompt_free"

    def __init__(self, model: LanguageModel, prompt: PromptConfig, max_new_tokens: int = 128, temperature: float = 0.0,
                 stop: list[str] | None = None) -> None:
        super().__init__(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature, stop=stop)
        prompt.mode = "free"
        self.prompt = prompt


class PromptScorePolicy(_PromptPolicy):
    name = "prompt_score"

    def __init__(self, model: LanguageModel, prompt: PromptConfig, length_norm: bool = False) -> None:
        prompt.mode = "score"
        super().__init__(model, prompt)
        self.length_norm = bool(length_norm)

    def act(self, obs: Observation, history: list[HistoryStep]) -> Decision:
        assert self.env is not None
        msgs, lm = build_messages(self.env, obs, history, self.prompt)
        labels = list(lm)
        res = self.model.score_choices(msgs, labels, length_norm=self.length_norm)
        best = max(range(len(labels)), key=lambda i: res.logprobs[i])
        scores = {labels[i]: round(res.logprobs[i], 4) for i in range(len(labels))}
        return Decision(action=lm[labels[best]], raw_output=f"scores={scores}", latency_s=res.latency_s,
                        prompt_tokens=res.prompt_tokens, completion_tokens=0,
                        extra={"scores": scores, "choice_tokens": res.choice_tokens, **res.extra})

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "length_norm": self.length_norm}
