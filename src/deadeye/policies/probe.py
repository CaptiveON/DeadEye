"""Probe conversion: a linear classifier on frozen hidden states (or on raw task features as a control).

``prepare`` rolls out a behaviour policy on the training seeds (oracle actions with probability
``oracle_mix``, random otherwise, so that off-path states are covered) and records the oracle action
as the label for every visited state. A multinomial logistic regression is fitted on the model's
hidden-state vector for the prompt (``probe``) or on the environment's numeric feature vector
(``feature_probe``; no language model involved). At decision time illegal actions are masked out.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from deadeye.envs.base import Environment, Observation
from deadeye.models.base import LanguageModel
from deadeye.policies.base import Decision, HistoryStep, Policy
from deadeye.policies.prompts import PromptConfig, build_messages


class _ProbeBase(Policy):
    needs_prepare = True

    def __init__(self, n_train_episodes: int = 40, oracle_mix: float = 0.7, C: float = 1.0, max_train_states: int = 50000,
                 seed: int = 0) -> None:
        super().__init__()
        self.n_train_episodes = int(n_train_episodes)
        self.oracle_mix = float(oracle_mix)
        self.C = float(C)
        self.max_train_states = int(max_train_states)
        self.seed = int(seed)
        self.clf = None
        self.scaler = None
        self.classes: list[str] = []
        self.feat_time = 0.0

    def features(self, env: Environment, obs: Observation, history: list[HistoryStep]) -> tuple[np.ndarray, int, float]:
        raise NotImplementedError

    def prepare(self, env_factory: Callable[[], Environment], train_seeds: list[int]) -> dict[str, Any]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(self.seed)
        X, y = [], []
        t0 = time.perf_counter()
        n_used = 0
        for seed in train_seeds[:self.n_train_episodes]:
            if len(X) >= self.max_train_states:
                break
            n_used += 1
            env = env_factory()
            obs = env.reset(seed)
            history: list[HistoryStep] = []
            while not env.done and len(X) < self.max_train_states:
                oracle = env.oracle_action()
                vec, _, _ = self.features(env, obs, history)
                X.append(vec)
                y.append(oracle)
                action = oracle if rng.random() < self.oracle_mix else str(rng.choice(obs.legal_actions))
                res = env.step(action)
                history.append(HistoryStep(obs.text, action, res.reward))
                obs = res.obs
        Xa = np.stack(X).astype(np.float64)
        self.classes = sorted(set(y))
        self.scaler = StandardScaler().fit(Xa)
        Xs = self.scaler.transform(Xa)
        if len(self.classes) == 1:
            self.clf = None  # degenerate: always the single class
        else:
            self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(Xs, y)
        train_acc = float(np.mean(self.clf.predict(Xs) == np.array(y))) if self.clf is not None else 1.0
        return {"n_train_states": len(y), "n_train_episodes_used": n_used, "truncated": len(y) >= self.max_train_states,
                "n_classes": len(self.classes), "train_acc": train_acc, "feature_dim": int(Xa.shape[1]),
                "prepare_time_s": time.perf_counter() - t0}

    def act(self, obs: Observation, history: list[HistoryStep]) -> Decision:
        assert self.env is not None
        vec, ptoks, lat = self.features(self.env, obs, history)
        t0 = time.perf_counter()
        legal = obs.legal_actions
        if self.clf is None:
            cand = [a for a in legal if a in self.classes] or legal
            action = cand[0]
            probs = {}
        else:
            p = self.clf.predict_proba(self.scaler.transform(vec[None, :].astype(np.float64)))[0]
            probs = {c: float(p[i]) for i, c in enumerate(self.clf.classes_)}
            scored = [(probs.get(a, -1.0), a) for a in legal]
            action = max(scored)[1]
        return Decision(action=action, raw_output=f"probs={ {k: round(v, 3) for k, v in probs.items()} }",
                        latency_s=lat + time.perf_counter() - t0, prompt_tokens=ptoks, completion_tokens=0,
                        extra={"probs": probs})


class ProbePolicy(_ProbeBase):
    name = "probe"

    def __init__(self, model: LanguageModel, prompt: PromptConfig, layer: int | str = "mid", **kw: Any) -> None:
        super().__init__(**kw)
        self.model = model
        self.prompt = prompt
        self.layer = layer

    def features(self, env, obs, history):
        msgs, _ = build_messages(env, obs, history, self.prompt)
        e = self.model.embed(msgs, layer=self.layer)
        return e.vector, e.prompt_tokens, e.latency_s

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model.info.id, "layer": str(self.layer), "n_train_episodes": self.n_train_episodes,
                "oracle_mix": self.oracle_mix, "C": self.C, **self.prompt.to_dict()}


class FeatureProbePolicy(_ProbeBase):
    """Control condition: same probe, but on the environment's raw numeric features (no LM)."""

    name = "feature_probe"

    def features(self, env, obs, history):
        if obs.features is None:
            raise ValueError(f"{env.name} provides no numeric features")
        return np.asarray(obs.features, dtype=np.float32), 0, 0.0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "n_train_episodes": self.n_train_episodes, "oracle_mix": self.oracle_mix, "C": self.C}
