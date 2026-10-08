"""Build a backend from a model spec (a dict from the YAML config).

Spec fields: ``backend`` (hf | openai | mock), ``id`` (repo id, served model name, or mock name), plus
backend-specific options. Optional metadata (``params``, ``family``, ``instruct``) overrides what the
backend can infer; the catalogue in ``configs/models.yaml`` supplies it when ``catalog`` is given.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from deadeye.models.base import LanguageModel

_META_KEYS = ("params", "family", "instruct", "precision")


def load_catalog(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, dict[str, Any]] = {}
    for fam in data.get("families", []):
        for m in fam.get("models", []):
            meta = {"family": fam["name"], "params": m.get("params"), "instruct": m.get("instruct"),
                    "license": fam.get("license"), "non_embedding_params": m.get("non_embedding_params")}
            out[m["id"]] = meta
    return out


def load_model(spec: dict[str, Any], catalog: dict[str, dict[str, Any]] | None = None) -> LanguageModel:
    spec = dict(spec)
    backend = spec.pop("backend", "hf")
    model_id = spec.pop("id")
    meta = {k: spec.pop(k) for k in _META_KEYS if k in spec}
    cat = (catalog or {}).get(model_id, {})
    if backend == "mock":
        from deadeye.models.mock import MockModel
        model: LanguageModel = MockModel(model_id=model_id, **spec)
    elif backend == "hf":
        from deadeye.models.hf_backend import HFModel
        if model_id == "tiny-random":
            from deadeye.models.tiny import ensure_tiny_model
            model_id = str(ensure_tiny_model(spec.pop("cache_dir", None)))
            meta.setdefault("family", "tiny-random")
        # Base checkpoints get the plain rendering (see hf_backend). Many base tokenizers (e.g. Qwen2.5) still ship a
        # chat template, so decide from the declared `instruct` flag, not from template presence.
        if meta.get("instruct", cat.get("instruct")) is False:
            spec.setdefault("use_chat_template", False)
        model = HFModel(model_id, **spec)
    elif backend == "openai":
        from deadeye.models.openai_backend import OpenAICompatModel
        model = OpenAICompatModel(model_id, **spec)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    for k in ("params", "family", "instruct", "precision"):
        v = meta.get(k, cat.get(k))
        if v is not None:
            setattr(model.info, k, v)
    if cat.get("license"):
        model.info.extra["license"] = cat["license"]
    return model
