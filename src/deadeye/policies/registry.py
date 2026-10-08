"""Construct a policy from a method spec: ``{"name": ..., "params": {...}}``."""
from __future__ import annotations

from typing import Any

from deadeye.models.base import LanguageModel
from deadeye.policies.base import Policy
from deadeye.policies.baselines import UCB1, OraclePolicy, RandomPolicy
from deadeye.policies.lora_sft import LoraSFTPolicy
from deadeye.policies.probe import FeatureProbePolicy, ProbePolicy
from deadeye.policies.prompting import PromptGeneratePolicy, PromptScorePolicy
from deadeye.policies.prompts import PromptConfig

#: name -> (class, needs_model)
POLICY_REGISTRY: dict[str, tuple[type[Policy], bool]] = {
    "random": (RandomPolicy, False),
    "oracle": (OraclePolicy, False),
    "ucb1": (UCB1, False),
    "feature_probe": (FeatureProbePolicy, False),
    "prompt_generate": (PromptGeneratePolicy, True),
    "prompt_score": (PromptScorePolicy, True),
    "probe": (ProbePolicy, True),
    "lora_sft": (LoraSFTPolicy, True),
}

_PROMPT_KEYS = {"action_format", "history_window", "few_shot", "cot", "extra_system"}


def needs_model(name: str) -> bool:
    return POLICY_REGISTRY[name][1]


def make_policy(name: str, params: dict[str, Any] | None = None, model: LanguageModel | None = None) -> Policy:
    try:
        cls, wants_model = POLICY_REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"unknown method {name!r}; known: {sorted(POLICY_REGISTRY)}") from e
    params = dict(params or {})
    if wants_model:
        if model is None:
            raise ValueError(f"method {name} needs a model")
        prompt = PromptConfig(**{k: params.pop(k) for k in list(params) if k in _PROMPT_KEYS})
        return cls(model, prompt, **params)  # type: ignore[call-arg]
    return cls(**params)  # type: ignore[call-arg]
