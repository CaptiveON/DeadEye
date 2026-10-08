"""Run the factorial design in a config: (environment x model x method) cells, each over many seeds.

Outputs, per cell directory ``<output_dir>/<env>/<model>/<method>/``:
  summary.json   - aggregates + full provenance (specs, model info, versions, timings)
  episodes.jsonl - one record per episode (return, metrics, illegal-action counts, cost)
  steps.jsonl    - one record per decision (raw model output, parsed action, oracle action, reward)
Cells whose summary.json exists are skipped unless ``force`` is set, so runs are resumable.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from deadeye import __version__
from deadeye.config import EnvSpec, MethodSpec, ModelSpec, RunConfig
from deadeye.envs import Environment, make_env
from deadeye.models.base import LanguageModel
from deadeye.models.registry import load_catalog, load_model
from deadeye.policies.base import Decision, HistoryStep, Policy
from deadeye.policies.registry import make_policy, needs_model

log = logging.getLogger(__name__)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


@dataclass
class Cell:
    env: EnvSpec
    method: MethodSpec
    model: ModelSpec | None  # None for baselines

    @property
    def path_parts(self) -> tuple[str, str, str]:
        return self.env.key, (self.model.key if self.model else "_baseline"), self.method.key


def run_episode(env: Environment, policy: Policy, seed: int, illegal_action: str = "random_fallback",
                log_prompts: bool = False, rng: np.random.Generator | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = rng or np.random.default_rng(seed + 7)
    obs = env.reset(seed)
    policy.reset()
    history: list[HistoryStep] = []
    steps: list[dict[str, Any]] = []
    ret, n_illegal, n_agree, latency, ptoks, ctoks = 0.0, 0, 0, 0.0, 0, 0
    t = 0
    while not env.done and t < env.max_steps:
        oracle = env.oracle_action()
        d: Decision = policy.act(obs, history)
        legal = d.action is not None and d.action in obs.legal_actions
        action = d.action if legal else None
        if action is None:
            n_illegal += 1
            if illegal_action == "random_fallback":
                action = str(rng.choice(obs.legal_actions))
            elif illegal_action == "first_legal":
                action = obs.legal_actions[0]
            elif illegal_action == "terminate":
                steps.append({"seed": seed, "t": t, "legal_actions": obs.legal_actions, "oracle_action": oracle,
                              "parsed_action": d.action, "action": None, "legal": False, "reward": 0.0,
                              "latency_s": d.latency_s, "prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens,
                              "raw_output": d.raw_output[:2000], "extra": d.extra, **({"obs_text": obs.text} if log_prompts else {})})
                latency += d.latency_s
                ptoks += d.prompt_tokens
                ctoks += d.completion_tokens
                t += 1
                break
            else:
                raise ValueError(f"unknown illegal_action policy {illegal_action!r}")
        res = env.step(action)
        n_agree += int(legal and action == oracle)  # a fallback action is not the policy's choice
        ret += res.reward
        latency += d.latency_s
        ptoks += d.prompt_tokens
        ctoks += d.completion_tokens
        steps.append({"seed": seed, "t": t, "legal_actions": obs.legal_actions, "oracle_action": oracle, "parsed_action": d.action,
                      "action": action, "legal": legal, "reward": res.reward, "latency_s": d.latency_s,
                      "prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens,
                      "raw_output": d.raw_output[:2000], "extra": d.extra, **({"obs_text": obs.text} if log_prompts else {})})
        history.append(HistoryStep(obs.text, action, res.reward))
        obs = res.obs
        t += 1
    metrics = {k: float(v) for k, v in env.episode_metrics().items()}
    ep = {"seed": seed, "return": ret, "steps": t, "n_illegal": n_illegal, "illegal_rate": n_illegal / max(1, t),
          "oracle_agreement": n_agree / max(1, t), "latency_s": latency, "prompt_tokens": ptoks, "completion_tokens": ctoks,
          **metrics}
    return ep, steps


class Runner:
    def __init__(self, cfg: RunConfig, force: bool = False, only_env: str | None = None, only_model: str | None = None,
                 only_method: str | None = None, dry_run: bool = False, console: Any = None) -> None:
        self.cfg = cfg
        self.force = force
        self.only_env, self.only_model, self.only_method = only_env, only_model, only_method
        self.dry_run = dry_run
        self.console = console
        self.out = Path(cfg.output_dir)
        self.catalog = load_catalog(cfg.catalog)

    # ------------------------------------------------------------------ plan
    def cells(self) -> list[Cell]:
        cells: list[Cell] = []
        for env in self.cfg.envs:
            if self.only_env and self.only_env not in (env.name, env.key):
                continue
            for b in self.cfg.baselines:
                if b == "ucb1" and env.name != "bandit":
                    continue
                if self.only_method and self.only_method != b:
                    continue
                cells.append(Cell(env, MethodSpec(name=b), None))
        for model in self.cfg.models:
            if self.only_model and self.only_model not in (model.id, model.key):
                continue
            for method in self.cfg.methods:
                if self.only_method and self.only_method not in (method.name, method.key):
                    continue
                for env in self.cfg.envs:
                    if self.only_env and self.only_env not in (env.name, env.key):
                        continue
                    if needs_model(method.name):
                        cells.append(Cell(env, method, model))
                    elif model is self.cfg.models[0]:  # model-free methods (feature_probe) run once
                        cells.append(Cell(env, method, None))
        return cells

    def cell_dir(self, cell: Cell) -> Path:
        return self.out.joinpath(*cell.path_parts)

    def _say(self, msg: str) -> None:
        if self.console is not None:
            self.console.print(msg)
        else:
            log.info(msg)

    # ------------------------------------------------------------------ execution
    def run(self) -> list[Path]:
        self.out.mkdir(parents=True, exist_ok=True)
        manifest = {"name": self.cfg.name, "started": datetime.now(timezone.utc).isoformat(), "deadeye": __version__,
                    "python": platform.python_version(), "platform": platform.platform(), "config": self.cfg.raw,
                    "cells": [list(c.path_parts) for c in self.cells()]}
        try:
            import torch
            manifest["torch"] = torch.__version__
            manifest["cuda"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                manifest["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        try:
            import transformers
            manifest["transformers"] = transformers.__version__
        except Exception:
            pass
        dump_json(manifest, self.out / "run_manifest.json")
        done: list[Path] = []
        cells = self.cells()
        self._say(f"[bold]{self.cfg.name}[/bold]: {len(cells)} cells -> {self.out}")
        if self.dry_run:
            for c in cells:
                self._say("  " + "/".join(c.path_parts))
            return done
        # Baselines and model-free methods first.
        for cell in [c for c in cells if c.model is None]:
            p = self._run_cell(cell, model=None)
            if p:
                done.append(p)
        # Group by model so each model is loaded once (unless a method mutates it).
        for model_spec in self.cfg.models:
            # Cells that share one copy of the model run first; mutating cells (LoRA) come last and each gets a fresh
            # copy only after the shared one is released, so at most one copy of the weights is in memory.
            my_cells = sorted((c for c in cells if c.model is model_spec), key=lambda c: _method_mutates(c.method.name))
            pending = [c for c in my_cells if self.force or not (self.cell_dir(c) / "summary.json").exists()]
            if not pending:
                for c in my_cells:
                    self._say(f"  skip (done) {'/'.join(c.path_parts)}")
                continue
            shared: LanguageModel | None = None
            for cell in my_cells:
                if not self.force and (self.cell_dir(cell) / "summary.json").exists():
                    self._say(f"  skip (done) {'/'.join(cell.path_parts)}")
                    continue
                cls_mutates = _method_mutates(cell.method.name)
                if cls_mutates:
                    if shared is not None:
                        shared.close()
                        shared = None
                    fresh = load_model(model_spec.spec, self.catalog)
                    p = self._run_cell(cell, model=fresh)
                    fresh.close()
                else:
                    if shared is None:
                        shared = load_model(model_spec.spec, self.catalog)
                    p = self._run_cell(cell, model=shared)
                if p:
                    done.append(p)
            if shared is not None:
                shared.close()
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        dump_json(manifest, self.out / "run_manifest.json")
        return done

    def _run_cell(self, cell: Cell, model: LanguageModel | None) -> Path | None:
        cdir = self.cell_dir(cell)
        if not self.force and (cdir / "summary.json").exists():
            self._say(f"  skip (done) {'/'.join(cell.path_parts)}")
            return None
        env_factory: Callable[[], Environment] = lambda: make_env(cell.env.name, **cell.env.params)
        env = env_factory()
        if model is not None:
            missing = required_capabilities(cell.method.name) - set(model.capabilities)
            if missing:
                self._say(f"  skip (model lacks {sorted(missing)}) {'/'.join(cell.path_parts)}")
                return None
        try:
            policy = make_policy(cell.method.name, cell.method.params, model=model)
        except TypeError as e:
            self._say(f"  skip (incompatible: {e}) {'/'.join(cell.path_parts)}")
            return None
        policy.bind(env)
        t0 = time.perf_counter()
        prepare_stats = policy.prepare(env_factory, self.cfg.train_seeds) if policy.needs_prepare else {}
        prep_time = time.perf_counter() - t0
        seeds = self.cfg.episodes_for(cell.env)
        episodes: list[dict[str, Any]] = []
        cdir.mkdir(parents=True, exist_ok=True)
        steps_path = cdir / "steps.jsonl"
        ep_path = cdir / "episodes.jsonl"
        t1 = time.perf_counter()
        with open(steps_path, "w") as sf, open(ep_path, "w") as ef:
            for seed in seeds:
                if model is not None and hasattr(model, "seed"):
                    model.seed(seed)
                ep, steps = run_episode(env, policy, seed, self.cfg.illegal_action, self.cfg.log_prompts)
                episodes.append(ep)
                ef.write(json.dumps(ep, default=_json_default) + "\n")
                if self.cfg.log_steps:
                    for s in steps:
                        sf.write(json.dumps(s, default=_json_default) + "\n")
        eval_time = time.perf_counter() - t1
        summary = summarize(cell, episodes, model, policy, prepare_stats, prep_time, eval_time, self.cfg)
        dump_json(summary, cdir / "summary.json")
        self._say(f"  {'/'.join(cell.path_parts)}: return {summary['return_mean']:.3f} +- {summary['return_std']:.3f}, "
                  f"illegal {summary['illegal_rate']:.1%}, agree {summary['oracle_agreement']:.1%}, "
                  f"{summary['n_decisions']} decisions in {eval_time:.1f}s")
        return cdir


def required_capabilities(name: str) -> set[str]:
    return {"prompt_generate": {"generate"}, "prompt_score": {"score"}, "probe": {"embed"}, "lora_sft": {"train"}}.get(name, set())


def _method_mutates(name: str) -> bool:
    from deadeye.policies.registry import POLICY_REGISTRY
    return bool(getattr(POLICY_REGISTRY[name][0], "mutates_model", False))


def summarize(cell: Cell, episodes: list[dict[str, Any]], model: LanguageModel | None, policy: Policy,
              prepare_stats: dict[str, Any], prep_time: float, eval_time: float, cfg: RunConfig) -> dict[str, Any]:
    rets = np.array([e["return"] for e in episodes], dtype=float)
    n_steps = int(sum(e["steps"] for e in episodes))
    n_illegal = int(sum(e["n_illegal"] for e in episodes))
    metric_keys = [k for k in episodes[0] if k not in ("seed",)] if episodes else []
    means = {f"{k}_mean": float(np.mean([e[k] for e in episodes])) for k in metric_keys}
    stds = {f"{k}_std": float(np.std([e[k] for e in episodes], ddof=1)) if len(episodes) > 1 else 0.0 for k in metric_keys}
    return {
        "env": cell.env.name, "env_key": cell.env.key, "env_params": cell.env.params,
        "model_key": cell.model.key if cell.model else "_baseline",
        "model": model.info.to_dict() if model is not None else {"id": "_baseline", "backend": "none", "params": None},
        "model_spec": cell.model.spec if cell.model else None,
        "method": cell.method.name, "method_key": cell.method.key, "method_params": cell.method.params,
        "policy": policy.describe(),
        "n_episodes": len(episodes), "seeds": [e["seed"] for e in episodes], "n_decisions": n_steps,
        "return_mean": float(rets.mean()) if len(rets) else float("nan"),
        "return_std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "illegal_rate": n_illegal / max(1, n_steps),
        "oracle_agreement": float(np.mean([e["oracle_agreement"] for e in episodes])) if episodes else float("nan"),
        "latency_per_decision_s": float(sum(e["latency_s"] for e in episodes) / max(1, n_steps)),
        "prompt_tokens_per_decision": float(sum(e["prompt_tokens"] for e in episodes) / max(1, n_steps)),
        "completion_tokens_per_decision": float(sum(e["completion_tokens"] for e in episodes) / max(1, n_steps)),
        **means, **stds,
        "prepare_stats": prepare_stats, "prepare_time_s": prep_time, "eval_time_s": eval_time,
        "illegal_action_policy": cfg.illegal_action, "run_name": cfg.name, "deadeye": __version__,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
