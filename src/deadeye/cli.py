"""Command-line interface: ``deadeye --help``."""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="DeadEye: benchmark open-weight language models as decision-makers.")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


@app.command("list-envs")
def list_envs() -> None:
    """List environments with their action spaces and default parameters."""
    from deadeye.envs import ENV_REGISTRY, make_env
    t = Table("environment", "actions", "max steps", "primary metric", "default params")
    for name in ENV_REGISTRY:
        env = make_env(name)
        t.add_row(name, ", ".join(env.action_labels), str(env.max_steps), f"{env.primary_metric} ({'higher' if env.higher_is_better else 'lower'} better)",
                  json.dumps(env.params))
    console.print(t)


@app.command("list-methods")
def list_methods() -> None:
    """List conversion methods (policies) and whether they need a model."""
    from deadeye.policies.registry import POLICY_REGISTRY
    t = Table("method", "needs model", "mutates model", "class")
    for name, (cls, needs) in POLICY_REGISTRY.items():
        t.add_row(name, "yes" if needs else "no", "yes" if getattr(cls, "mutates_model", False) else "no", cls.__name__)
    console.print(t)


@app.command("show-prompt")
def show_prompt(env: str, seed: int = 0, method: str = "prompt_generate", action_format: str = "name", few_shot: int = 0,
                cot: bool = False, history_window: int = 0, steps: int = 0, params: str = "{}") -> None:
    """Print the exact messages a model would receive for an environment state."""
    from deadeye.envs import make_env
    from deadeye.policies.prompts import PromptConfig, build_messages, collect_demos
    e = make_env(env, **json.loads(params))
    obs = e.reset(seed)
    for _ in range(steps):
        if e.done:
            break
        obs = e.step(e.oracle_action()).obs
    cfg = PromptConfig(action_format=action_format, few_shot=few_shot, cot=cot, history_window=history_window,
                       mode="score" if method == "prompt_score" else "generate")
    if few_shot:
        cfg.demos = collect_demos(lambda: make_env(env, **json.loads(params)), list(range(100_000, 100_000 + few_shot)), few_shot)
    msgs, lm = build_messages(e, obs, [], cfg)
    for m in msgs:
        console.rule(m.role)
        console.print(m.content, markup=False)
    console.rule("label -> action")
    console.print(json.dumps(lm))
    console.print(f"oracle action: {e.oracle_action()}")


@app.command()
def run(config: Path, force: bool = False, only_env: Optional[str] = None, only_model: Optional[str] = None,
        only_method: Optional[str] = None, dry_run: bool = False, verbose: bool = False) -> None:
    """Run every (environment x model x method) cell of a YAML config. Resumable: finished cells are skipped."""
    _setup_logging(verbose)
    from deadeye.config import RunConfig
    from deadeye.runner import Runner
    cfg = RunConfig.load(config)
    Runner(cfg, force=force, only_env=only_env, only_model=only_model, only_method=only_method, dry_run=dry_run, console=console).run()
    console.print(f"[green]done[/green] -> {cfg.output_dir}")


@app.command()
def report(results: list[Path], out: Path = Path("report"), paper_dir: Optional[Path] = None, title: str = "DeadEye report") -> None:
    """Aggregate result directories into tables, figures and an HTML report (optionally exporting paper assets)."""
    from deadeye.report import build_report
    res = build_report([str(r) for r in results], out, paper_dir=paper_dir, title=title)
    console.print((out / "tables.md").read_text())
    console.print(f"[green]report[/green] -> {res['html']}")


@app.command()
def smoke(out: Optional[Path] = None, keep: bool = False, verbose: bool = False) -> None:
    """End-to-end pipeline check with a mock model and a tiny random model (no downloads, ~1-2 minutes)."""
    _setup_logging(verbose)
    from deadeye.config import RunConfig
    from deadeye.report import build_report
    from deadeye.runner import Runner
    out = out or Path(tempfile.mkdtemp(prefix="deadeye-smoke-"))
    cfg = RunConfig.from_dict(smoke_config(str(out / "results")))
    Runner(cfg, force=True, console=console).run()
    res = build_report([str(out / "results")], out / "report", title="DeadEye smoke test")
    console.print(f"[green]smoke test passed[/green]: {len(res['cells'])} cells -> {res['html']}")


@app.command()
def estimate(config: Path, seconds_per_decision: float = 1.0) -> None:
    """Estimate decisions, prompt tokens and wall-clock for a config (uses oracle rollouts to count steps)."""
    from deadeye.config import RunConfig
    from deadeye.envs import make_env
    from deadeye.policies.prompts import PromptConfig, build_messages
    cfg = RunConfig.load(config)
    t = Table("environment", "episodes", "decisions/episode", "decisions", "~prompt tokens/decision")
    total_dec = 0
    for es in cfg.envs:
        seeds = cfg.episodes_for(es)
        steps, words = [], []
        for s in seeds[:10]:
            env = make_env(es.name, **es.params)
            obs = env.reset(s)
            n = 0
            while not env.done:
                msgs, _ = build_messages(env, obs, [], PromptConfig())
                words.append(sum(len(m.content.split()) for m in msgs))
                obs = env.step(env.oracle_action()).obs
                n += 1
            steps.append(n)
        dpe = sum(steps) / len(steps)
        dec = int(dpe * len(seeds))
        total_dec += dec
        t.add_row(es.key, str(len(seeds)), f"{dpe:.1f}", str(dec), f"{1.4 * sum(words) / len(words):.0f}")
    console.print(t)
    n_models = max(1, len(cfg.models))
    n_methods = max(1, len([m for m in cfg.methods]))
    grand = total_dec * n_models * n_methods
    console.print(f"models x methods = {n_models} x {n_methods}; total LM decisions ~ {grand:,}; "
                  f"at {seconds_per_decision}s/decision ~ {grand * seconds_per_decision / 3600:.1f} h (+ probe/LoRA preparation)")


@app.command()
def compare(results: Path, env: str, a: str, b: str) -> None:
    """Paired permutation test between two cells: A and B are 'model_key/method_key' strings."""
    from deadeye.metrics import cohens_d_paired, paired_permutation_test
    from deadeye.report import load_results, normalize_results
    cells, eps = load_results([results])
    cells, eps = normalize_results(cells, eps)
    sel = {}
    for name, spec in (("a", a), ("b", b)):
        mk, meth = spec.split("/", 1)
        s = eps[(eps["env_key"] == env) & (eps["model_key"] == mk) & (eps["method_key"] == meth)].set_index("seed")["norm"]
        if s.empty:
            raise typer.BadParameter(f"no episodes for {spec} in {env}")
        sel[name] = s
    common = sel["a"].index.intersection(sel["b"].index)
    xa, xb = sel["a"].loc[common].to_numpy(), sel["b"].loc[common].to_numpy()
    p = paired_permutation_test(xa, xb)
    console.print(f"{env}: n={len(common)} paired episodes; mean(A)={xa.mean():.3f} mean(B)={xb.mean():.3f} "
                  f"diff={xa.mean() - xb.mean():+.3f}; paired permutation p={p:.4f}; Cohen's d={cohens_d_paired(xa, xb):.2f}")


def smoke_config(output_dir: str) -> dict:
    return {
        "name": "smoke", "output_dir": output_dir, "seeds": {"start": 0, "n": 4}, "train_seeds": {"start": 100000, "n": 6},
        "catalog": None,
        "envs": [
            {"name": "bandit", "params": {"n_arms": 3, "horizon": 8}},
            {"name": "contextual_bandit", "params": {"horizon": 5}},
            {"name": "loan", "params": {"n_applicants": 4}},
            {"name": "gridworld", "params": {"size": 4, "max_steps": 10, "min_distance": 2}},
            {"name": "tictactoe"},
            {"name": "blackjack"},
        ],
        "models": [
            {"backend": "mock", "id": "mock-random", "strategy": "random", "format_failure_rate": 0.25, "params": 1000, "family": "mock", "label": "mock"},
            {"backend": "hf", "id": "tiny-random", "params": 205120, "family": "tiny", "label": "tiny"},
        ],
        "methods": [
            {"name": "prompt_generate", "params": {"max_new_tokens": 6}},
            {"name": "prompt_score"},
            {"name": "prompt_score", "params": {"action_format": "letter"}, "label": "prompt_score_letter"},
            {"name": "probe", "params": {"n_train_episodes": 4, "max_train_states": 60}},
            {"name": "feature_probe", "params": {"n_train_episodes": 4}},
            {"name": "lora_sft", "params": {"n_train_episodes": 2, "max_train_states": 12, "lora": {"epochs": 2, "r": 4, "batch_size": 4, "lr": 0.001}}},
        ],
        "baselines": ["random", "oracle", "ucb1"],
    }


if __name__ == "__main__":  # pragma: no cover
    app()
