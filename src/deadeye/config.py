"""Experiment configuration (YAML -> dataclasses)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _seed_list(spec: Any, default_start: int, default_n: int) -> list[int]:
    if spec is None:
        return list(range(default_start, default_start + default_n))
    if isinstance(spec, list):
        return [int(s) for s in spec]
    if isinstance(spec, dict):
        start = int(spec.get("start", default_start))
        n = int(spec.get("n", default_n))
        return list(range(start, start + n))
    raise ValueError(f"bad seed spec {spec!r}")


def short_hash(obj: Any) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:6]


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", s).strip("_")


@dataclass
class EnvSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    n_episodes: int | None = None
    label: str | None = None

    @property
    def key(self) -> str:
        return slug(self.label) if self.label else self.name


@dataclass
class MethodSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    @property
    def key(self) -> str:
        return slug(self.label) if self.label else self.name


@dataclass
class ModelSpec:
    spec: dict[str, Any]
    label: str | None = None

    @property
    def id(self) -> str:
        return str(self.spec["id"])

    @property
    def key(self) -> str:
        return slug(self.label) if self.label else slug(self.id)


@dataclass
class RunConfig:
    name: str
    output_dir: str
    seeds: list[int]
    train_seeds: list[int]
    envs: list[EnvSpec]
    models: list[ModelSpec]
    methods: list[MethodSpec]
    baselines: list[str] = field(default_factory=lambda: ["random", "oracle"])
    illegal_action: str = "random_fallback"
    log_steps: bool = True
    log_prompts: bool = False
    catalog: str | None = "configs/models.yaml"
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path | None = None) -> "RunConfig":
        envs = [EnvSpec(name=e["name"], params=dict(e.get("params") or {}), n_episodes=e.get("n_episodes"), label=e.get("label"))
                for e in d.get("envs", [])]
        models = [ModelSpec(spec={k: v for k, v in m.items() if k != "label"}, label=m.get("label")) for m in d.get("models", [])]
        methods = [MethodSpec(name=m["name"], params=dict(m.get("params") or {}), label=m.get("label")) for m in d.get("methods", [])]
        for kind, items in (("env", envs), ("model", models), ("method", methods)):
            seen: dict[str, int] = {}
            for it in items:
                seen[it.key] = seen.get(it.key, 0) + 1
            dups = sorted(k for k, n in seen.items() if n > 1)
            if dups:
                raise ValueError(f"duplicate {kind} keys {dups}: give each variant a distinct `label`")
        catalog = d.get("catalog", "configs/models.yaml")
        if catalog and base_dir is not None and not Path(catalog).is_absolute():
            for cand in (base_dir / catalog, base_dir.parent / catalog, Path(catalog)):
                if cand.exists():
                    catalog = str(cand)
                    break
        cfg = cls(name=d.get("name", "run"), output_dir=d.get("output_dir", f"results/{d.get('name', 'run')}"),
                  seeds=_seed_list(d.get("seeds"), 0, 20), train_seeds=_seed_list(d.get("train_seeds"), 100_000, 50),
                  envs=envs, models=models, methods=methods, baselines=list(d.get("baselines", ["random", "oracle"])),
                  illegal_action=d.get("illegal_action", "random_fallback"), log_steps=bool(d.get("log_steps", True)),
                  log_prompts=bool(d.get("log_prompts", False)), catalog=catalog, notes=d.get("notes", ""), raw=d)
        train = set(cfg.train_seeds)
        for env in envs:  # probe / LoRA / few-shot data come from train_seeds and must never be evaluated on
            leak = train.intersection(cfg.episodes_for(env))
            if leak:
                raise ValueError(f"evaluation seeds of env {env.key!r} overlap train_seeds (e.g. {min(leak)})")
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d, base_dir=path.resolve().parent)

    def episodes_for(self, env: EnvSpec) -> list[int]:
        if env.n_episodes is None:
            return list(self.seeds)
        start = self.seeds[0] if self.seeds else 0
        return list(range(start, start + int(env.n_episodes)))
