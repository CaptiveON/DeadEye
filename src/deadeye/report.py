"""Aggregate run outputs into normalised scores, confidence intervals, tables, figures and an HTML report.

Normalisation: for each environment (env_key) the ``random`` and ``oracle`` baseline cells define
0 and 1; every episode return is mapped onto that scale, then per-cell mean / IQM / bootstrap CI are
computed over episodes. Figures use a fixed categorical palette keyed by *method* so colours never
change between runs or when a filter removes a series.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from deadeye.metrics import bootstrap_ci, holm_correction, iqm, paired_permutation_test

# Fixed method -> colour assignment (never cycled). Light-surface values of the validated palette.
METHOD_ORDER = ["prompt_generate", "prompt_score", "probe", "lora_sft", "feature_probe", "ucb1"]
METHOD_COLORS = {"prompt_generate": "#2a78d6", "prompt_score": "#eb6834", "probe": "#1baf7a", "lora_sft": "#eda100",
                 "feature_probe": "#e87ba4", "ucb1": "#4a3aa7", "random": "#8a8984", "oracle": "#0b0b0b"}
FAMILY_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#d9d8d3"


# ---------------------------------------------------------------------- loading
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_results(dirs: Iterable[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cells, episodes) dataframes from one or more result directories."""
    cells, eps = [], []
    for d in dirs:
        for summ in sorted(Path(d).rglob("summary.json")):
            s = json.loads(summ.read_text())
            model = s.get("model") or {}
            row = {
                "run": s.get("run_name"), "env": s["env"], "env_key": s["env_key"], "model_key": s["model_key"],
                "model_id": model.get("id", "_baseline"), "backend": model.get("backend"), "params": model.get("params"),
                "family": model.get("family"), "instruct": model.get("instruct"), "precision": model.get("precision"),
                "method": s["method"], "method_key": s["method_key"], "n_episodes": s["n_episodes"], "n_decisions": s["n_decisions"],
                "return_mean": s["return_mean"], "return_std": s["return_std"], "illegal_rate": s["illegal_rate"],
                "oracle_agreement": s["oracle_agreement"], "latency_per_decision_s": s["latency_per_decision_s"],
                "prompt_tokens_per_decision": s["prompt_tokens_per_decision"],
                "completion_tokens_per_decision": s["completion_tokens_per_decision"],
                "prepare_time_s": s.get("prepare_time_s", 0.0), "eval_time_s": s.get("eval_time_s", 0.0),
                "method_params": json.dumps(s.get("method_params", {}), sort_keys=True),
                "env_params": json.dumps(s.get("env_params", {}), sort_keys=True), "dir": str(summ.parent),
            }
            for k, v in s.items():
                if k.endswith("_mean") and k not in row:
                    row[k] = v
            cells.append(row)
            for e in _read_jsonl(summ.parent / "episodes.jsonl"):
                eps.append({"env_key": s["env_key"], "model_key": s["model_key"], "method_key": s["method_key"], "method": s["method"],
                            "run": s.get("run_name"), **e})
    cells_df = pd.DataFrame(cells)
    eps_df = pd.DataFrame(eps)
    if cells_df.empty:
        raise FileNotFoundError(f"no summary.json files found under {list(dirs)}")
    return cells_df, eps_df


def _cell_id(df: pd.DataFrame) -> pd.Series:
    return df["env_key"] + "|" + df["model_key"] + "|" + df["method_key"]


def normalize_results(cells: pd.DataFrame, episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add normalised scores (random=0, oracle=1) to episodes and per-cell aggregates to cells."""
    anchors = {}
    for env_key, grp in cells.groupby("env_key"):
        r = grp[(grp["model_key"] == "_baseline") & (grp["method"] == "random")]
        o = grp[(grp["model_key"] == "_baseline") & (grp["method"] == "oracle")]
        if len(r) and len(o):
            anchors[env_key] = (float(r["return_mean"].iloc[0]), float(o["return_mean"].iloc[0]))
    episodes = episodes.copy()
    episodes["norm"] = np.nan
    for env_key, (rm, om) in anchors.items():
        mask = episodes["env_key"] == env_key
        span = om - rm
        episodes.loc[mask, "norm"] = (episodes.loc[mask, "return"] - rm) / span if abs(span) > 1e-12 else np.nan
    cells = cells.copy()
    cells["cell_id"] = _cell_id(cells)
    episodes["cell_id"] = _cell_id(episodes)
    agg = []
    for cid, grp in episodes.groupby("cell_id"):
        x = grp["norm"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(x)
        agg.append({"cell_id": cid, "norm_mean": float(np.nanmean(x)) if len(x) else np.nan, "norm_iqm": iqm(x[~np.isnan(x)]) if len(x) else np.nan,
                    "norm_ci_lo": lo, "norm_ci_hi": hi, "norm_std": float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0})
    cells = cells.merge(pd.DataFrame(agg), on="cell_id", how="left")
    cells["anchor_random"] = cells["env_key"].map({k: v[0] for k, v in anchors.items()})
    cells["anchor_oracle"] = cells["env_key"].map({k: v[1] for k, v in anchors.items()})
    return cells, episodes


# ---------------------------------------------------------------------- tables
def short_model(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _fmt_params(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "?"
    p = float(p)
    if p >= 1e9:
        return f"{p / 1e9:.2g}B"
    if p >= 1e6:
        return f"{p / 1e6:.0f}M"
    return f"{p / 1e3:.0f}K"


def method_tables(cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One wide table per method: rows = models (by size), columns = environments, values = norm score with CI."""
    tables: dict[str, pd.DataFrame] = {}
    model_cells = cells[cells["model_key"] != "_baseline"]
    for mkey, grp in model_cells.groupby("method_key", sort=False):
        grp = grp.copy()
        grp["model"] = grp["model_id"].map(short_model) + " (" + grp["params"].map(_fmt_params) + ")"
        grp["val"] = grp.apply(lambda r: f"{r['norm_mean']:.2f} [{r['norm_ci_lo']:.2f}, {r['norm_ci_hi']:.2f}]", axis=1)
        order = grp.sort_values("params", na_position="last").drop_duplicates("model")["model"].tolist()
        wide = grp.pivot_table(index="model", columns="env_key", values="val", aggfunc="first").reindex(order)
        wide.index.name = f"{mkey}: model"
        tables[mkey] = wide
    return tables


def secondary_tables(cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    model_cells = cells[cells["model_key"] != "_baseline"].copy()
    model_cells["model"] = model_cells["model_id"].map(short_model)
    out = {}
    for name, col, fmt in [("illegal_rate", "illegal_rate", "{:.1%}"), ("oracle_agreement", "oracle_agreement", "{:.1%}"),
                           ("latency_s_per_decision", "latency_per_decision_s", "{:.3f}"),
                           ("completion_tokens_per_decision", "completion_tokens_per_decision", "{:.1f}")]:
        t = model_cells.pivot_table(index=["method_key", "model"], columns="env_key", values=col, aggfunc="first")
        out[name] = t.map(lambda v: fmt.format(v) if pd.notna(v) else "")
    anchors = cells[cells["model_key"] == "_baseline"].pivot_table(index="method", columns="env_key", values="return_mean", aggfunc="first")
    out["baseline_returns"] = anchors.map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    return out


def tables_to_markdown(tables: dict[str, pd.DataFrame]) -> str:
    parts = []
    for name, t in tables.items():
        parts.append(f"### {name}\n")
        parts.append(t.to_markdown() if hasattr(t, "to_markdown") else t.to_string())
        parts.append("")
    return "\n".join(parts)


def tables_to_latex(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    out = {}
    for name, t in tables.items():
        cap = name.replace("_", " ")
        out[name] = t.to_latex(escape=True, caption=f"DeadEye results: {cap}.", label=f"tab:{name}", na_rep="--")
    return out


# ---------------------------------------------------------------------- pairwise comparisons
def pairwise_by_model(episodes: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Within each (env, model), compare every method pair with a paired permutation test over seeds."""
    rows = []
    model_eps = episodes[episodes["model_key"] != "_baseline"]
    for (env_key, model_key), grp in model_eps.groupby(["env_key", "model_key"]):
        methods = sorted(grp["method_key"].unique())
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                ea = grp[grp["method_key"] == a].set_index("seed")["norm"]
                eb = grp[grp["method_key"] == b].set_index("seed")["norm"]
                common = ea.index.intersection(eb.index)
                if len(common) < 3:
                    continue
                p = paired_permutation_test(ea.loc[common].to_numpy(), eb.loc[common].to_numpy())
                rows.append({"env_key": env_key, "model_key": model_key, "a": a, "b": b, "n": len(common),
                             "diff_mean": float((ea.loc[common] - eb.loc[common]).mean()), "p": p})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm_correction(df["p"].tolist())
    return df


# ---------------------------------------------------------------------- figures
def _style(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")


def _families(cells: pd.DataFrame) -> dict[str, str]:
    fams = sorted({f for f in cells["family"].dropna().unique()})
    return {f: FAMILY_MARKERS[i % len(FAMILY_MARKERS)] for i, f in enumerate(fams)}


def fig_scale(cells: pd.DataFrame, out: Path) -> Path | None:
    """Normalised score vs parameter count, one panel per environment, one line per method (x family)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mc = cells[(cells["model_key"] != "_baseline") & cells["params"].notna()]
    if mc.empty:
        return None
    envs = sorted(mc["env_key"].unique())
    n = len(envs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    markers = _families(mc)
    handles: dict[str, Any] = {}
    for i, (ax, env_key) in enumerate(zip(axes.ravel(), envs)):
        sub = mc[mc["env_key"] == env_key]
        ax.axhline(1.0, color=METHOD_COLORS["oracle"], linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(0.0, color=METHOD_COLORS["random"], linestyle="--", linewidth=1.0, alpha=0.6)
        ucb = cells[(cells["env_key"] == env_key) & (cells["method"] == "ucb1")]
        if len(ucb):
            ax.axhline(float(ucb["norm_mean"].iloc[0]), color=METHOD_COLORS["ucb1"], linestyle=":", linewidth=1.2, alpha=0.8)
        for method in METHOD_ORDER:
            for mk, grp in sub[sub["method"] == method].groupby("method_key"):
                for fam, g in grp.groupby(grp["family"].fillna("unknown")):
                    g = g.sort_values("params")
                    color = METHOD_COLORS.get(method, TEXT_SECONDARY)
                    h = ax.errorbar(g["params"], g["norm_mean"], yerr=[g["norm_mean"] - g["norm_ci_lo"], g["norm_ci_hi"] - g["norm_mean"]],
                                    color=color, marker=markers.get(fam, "o"), markersize=5, linewidth=1.6, capsize=2, alpha=0.95,
                                    label=f"{mk} ({fam})")
                    handles.setdefault(f"{mk} ({fam})", h)
        ax.set_xscale("log")
        _style(ax, xlabel="parameters", ylabel="normalised score (random=0, oracle=1)" if i % ncols == 0 else "", title=env_key)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(4, max(1, len(handles))), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Decision quality vs model size", color=TEXT_PRIMARY, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def fig_illegal(cells: pd.DataFrame, out: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mc = cells[(cells["model_key"] != "_baseline") & cells["params"].notna() & (cells["method"] == "prompt_generate")]
    if mc.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    markers = _families(mc)
    envs = sorted(mc["env_key"].unique())
    env_colors = {e: list(METHOD_COLORS.values())[i % 6] for i, e in enumerate(envs)}
    for env_key in envs:
        for fam, g in mc[mc["env_key"] == env_key].groupby(mc["family"].fillna("unknown")):
            g = g.sort_values("params")
            ax.plot(g["params"], g["illegal_rate"], color=env_colors[env_key], marker=markers.get(fam, "o"), markersize=5,
                    linewidth=1.6, label=f"{env_key} ({fam})")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    _style(ax, xlabel="parameters", ylabel="illegal / unparseable action rate", title="Format failures under free-form generation")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def fig_pareto(cells: pd.DataFrame, out: Path) -> Path | None:
    """Mean normalised score across environments vs latency per decision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mc = cells[cells["model_key"] != "_baseline"]
    if mc.empty:
        return None
    agg = mc.groupby(["model_key", "model_id", "method", "method_key"], dropna=False).agg(
        score=("norm_mean", "mean"), latency=("latency_per_decision_s", "mean"), params=("params", "first")).reset_index()
    agg = agg[agg["latency"] > 0]
    if agg.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for method in METHOD_ORDER:
        g = agg[agg["method"] == method]
        if g.empty:
            continue
        ax.scatter(g["latency"], g["score"], s=36, color=METHOD_COLORS.get(method, TEXT_SECONDARY), label=method, alpha=0.9,
                   edgecolors="white", linewidths=0.8)
        for _, r in g.iterrows():
            ax.annotate(short_model(r["model_id"]), (r["latency"], r["score"]), fontsize=6, color=TEXT_SECONDARY,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    _style(ax, xlabel="latency per decision (s, log)", ylabel="mean normalised score across tasks", title="Quality vs inference cost")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def fig_heatmap(cells: pd.DataFrame, out: Path) -> Path | None:
    """Model x environment heatmap of normalised score, one panel per method (single-hue sequential ramp)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    mc = cells[cells["model_key"] != "_baseline"].copy()
    if mc.empty:
        return None
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f2f6fc", "#2a78d6", "#0b2f66"])
    methods = [m for m in mc["method_key"].unique()]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods) + 2.5, 0.42 * mc["model_key"].nunique() + 2.2), squeeze=False)
    fig.subplots_adjust(wspace=0.12, left=0.18)
    for j, (ax, mk) in enumerate(zip(axes[0], methods)):
        sub = mc[mc["method_key"] == mk]
        order = sub.sort_values("params", na_position="last").drop_duplicates("model_id")["model_id"].tolist()
        piv = sub.pivot_table(index="model_id", columns="env_key", values="norm_mean", aggfunc="first").reindex(order)
        im = ax.imshow(piv.to_numpy(dtype=float), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right", fontsize=7, color=TEXT_SECONDARY)
        if j == 0:
            ax.set_yticks(range(piv.shape[0]), [short_model(m) for m in piv.index], fontsize=7, color=TEXT_SECONDARY)
        else:
            ax.set_yticks(range(piv.shape[0]), [""] * piv.shape[0])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.55 else TEXT_PRIMARY)
        ax.set_title(mk, fontsize=9, loc="left", color=TEXT_PRIMARY)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="normalised score")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


# ---------------------------------------------------------------------- html
def _img_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{path.stem}" src="data:image/png;base64,{data}">'


def write_html(out: Path, title: str, tables: dict[str, pd.DataFrame], figures: list[Path], notes: str = "") -> Path:
    css = """
    :root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#d9d8d3; --accent:#2a78d6; }
    @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --line:#3a3a38; --accent:#3987e5; } }
    :root[data-theme="dark"] { --bg:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --line:#3a3a38; --accent:#3987e5; }
    body { background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, sans-serif; margin:0; padding:24px 16px; }
    main { max-width:1100px; margin:0 auto; }
    h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; } h3 { font-size:13px; color:var(--ink2); margin:18px 0 6px; }
    table { border-collapse:collapse; font-size:12px; width:100%; overflow-x:auto; display:block; }
    th, td { border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; white-space:nowrap; }
    th { color:var(--ink2); font-weight:600; } img { max-width:100%; height:auto; background:#fcfcfb; border-radius:6px; margin:8px 0; }
    p.note { color:var(--ink2); }
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>{title}</title><style>{css}</style></head><body><main><h1>{title}</h1>"
             "<p class='note'>Scores are normalised per environment so that the random policy scores 0 and the oracle scores 1; "
             "brackets are 95% bootstrap confidence intervals over episodes.</p>"]
    if notes:
        parts.append(f"<p class='note'>{notes}</p>")
    parts.append("<h2>Figures</h2>")
    for f in figures:
        if f is not None and Path(f).exists():
            parts.append(_img_tag(Path(f)))
    parts.append("<h2>Tables</h2>")
    for name, t in tables.items():
        parts.append(f"<h3>{name}</h3>")
        parts.append(t.to_html(border=0, na_rep=""))
    parts.append("</main></body></html>")
    out.write_text("\n".join(parts))
    return out


# ---------------------------------------------------------------------- entry point
def build_report(result_dirs: list[str | Path], out_dir: str | Path, paper_dir: str | Path | None = None,
                 title: str = "DeadEye report") -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells, episodes = load_results(result_dirs)
    cells, episodes = normalize_results(cells, episodes)
    cells.to_csv(out / "cells.csv", index=False)
    episodes.to_csv(out / "episodes.csv", index=False)
    tables = {**method_tables(cells), **secondary_tables(cells)}
    (out / "tables.md").write_text(tables_to_markdown(tables))
    tex = tables_to_latex(tables)
    (out / "tables").mkdir(exist_ok=True)
    for name, s in tex.items():
        (out / "tables" / f"{name}.tex").write_text(s)
    pw = pairwise_by_model(episodes, cells)
    pw.to_csv(out / "pairwise_methods.csv", index=False)
    figs = [fig_scale(cells, out / "fig_scale"), fig_illegal(cells, out / "fig_illegal"), fig_pareto(cells, out / "fig_pareto"),
            fig_heatmap(cells, out / "fig_heatmap")]
    figs = [f for f in figs if f is not None]
    html = write_html(out / "index.html", title, tables, figs)
    if paper_dir is not None:
        pdir = Path(paper_dir)
        (pdir / "figures").mkdir(parents=True, exist_ok=True)
        (pdir / "tables").mkdir(parents=True, exist_ok=True)
        for f in figs:
            pdf = Path(f).with_suffix(".pdf")
            if pdf.exists():
                (pdir / "figures" / pdf.name).write_bytes(pdf.read_bytes())
        for name, s in tex.items():
            (pdir / "tables" / f"{name}.tex").write_text(s)
    return {"cells": cells, "episodes": episodes, "tables": tables, "figures": figs, "html": html, "pairwise": pw}
