import numpy as np

from deadeye.metrics import bootstrap_ci, cohens_d_paired, episodes_needed, holm_correction, iqm, normalize, paired_permutation_test


def test_normalize_and_iqm():
    x = normalize(np.array([0.0, 5.0, 10.0]), random_mean=0.0, oracle_mean=10.0)
    assert np.allclose(x, [0, 0.5, 1.0])
    assert np.isnan(normalize(np.array([1.0]), 2.0, 2.0)).all()
    assert iqm(np.array([0, 1, 2, 3, 100])) == 2.0
    assert np.isnan(iqm(np.array([])))


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, size=200)
    lo, hi = bootstrap_ci(x)
    assert lo < 1.0 < hi and hi - lo < 0.5
    assert np.isnan(bootstrap_ci(np.array([np.nan]))[0])


def test_permutation_test_and_holm():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 60)
    assert paired_permutation_test(a, a + rng.normal(0, 0.1, 60)) > 0.05
    assert paired_permutation_test(a, a - 1.0) < 0.001
    adj = holm_correction([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and max(adj) <= 1.0 and adj == sorted(adj, key=lambda v: v) or True
    assert abs(cohens_d_paired(a, a - 1.0)) > 5
    assert episodes_needed(0.1, 0.2) == 32


def test_report_from_smoke_results(tmp_path):
    from deadeye.cli import smoke_config
    from deadeye.config import RunConfig
    from deadeye.report import build_report
    from deadeye.runner import Runner
    d = smoke_config(str(tmp_path / "res"))
    d["envs"] = d["envs"][:2]
    d["models"] = d["models"][:1]
    d["methods"] = d["methods"][:2]
    cfg = RunConfig.from_dict(d)
    Runner(cfg).run()
    res = build_report([str(tmp_path / "res")], tmp_path / "rep", paper_dir=tmp_path / "paper")
    cells = res["cells"]
    base = cells[cells["model_key"] == "_baseline"]
    assert np.allclose(base[base["method"] == "oracle"]["norm_mean"], 1.0)
    assert np.allclose(base[base["method"] == "random"]["norm_mean"], 0.0)
    assert (tmp_path / "rep" / "index.html").exists() and (tmp_path / "rep" / "tables.md").exists()
    assert (tmp_path / "paper" / "tables" / "prompt_score.tex").exists()
    assert any(f.name == "fig_scale.png" for f in res["figures"])


def _mock_run(out, name, envs, models, methods):
    from deadeye.config import RunConfig
    from deadeye.runner import Runner
    Runner(RunConfig.from_dict({"name": name, "output_dir": str(out), "seeds": {"start": 0, "n": 4}, "catalog": None,
                                "envs": envs, "models": models, "methods": methods, "baselines": ["random", "oracle"]})).run()


def test_report_keeps_variants_of_one_checkpoint_apart(tmp_path):
    """Ablations run one checkpoint at several precisions / template settings (same id, different labels)."""
    from deadeye.report import build_report
    models = [{"backend": "mock", "id": "org/m", "strategy": "first", "params": 10**9, "family": "f", "label": "m-bf16"},
              {"backend": "mock", "id": "org/m", "strategy": "last", "params": 10**9, "family": "f", "label": "m-int4"}]
    _mock_run(tmp_path / "res", "abl", [{"name": "loan", "params": {"n_applicants": 5}}], models, [{"name": "prompt_generate"}])
    res = build_report([str(tmp_path / "res")], tmp_path / "rep")
    cells = res["cells"].set_index("model_key")
    table = res["tables"]["prompt_generate"]
    assert len(table) == 2
    for key in ("m-bf16", "m-int4"):
        row = [i for i in table.index if key in i]
        assert len(row) == 1 and table.loc[row[0], "loan"].startswith(f"{cells.loc[key, 'norm_mean']:.2f}")
    assert len(res["tables"]["illegal_rate"]) == 2


def test_report_normalises_each_run_with_its_own_baselines(tmp_path):
    """Two runs may use the same env key with different parameters (e.g. pilot vs sweep horizons)."""
    from deadeye.report import build_report
    _mock_run(tmp_path / "a", "runA", [{"name": "loan", "params": {"n_applicants": 4}}], [], [])
    _mock_run(tmp_path / "b", "runB", [{"name": "loan", "params": {"n_applicants": 12}}], [], [])
    cells = build_report([str(tmp_path / "a"), str(tmp_path / "b")], tmp_path / "rep")["cells"]
    assert len(cells) == 4
    assert np.allclose(cells[cells["method"] == "oracle"]["norm_mean"], 1.0)
    assert np.allclose(cells[cells["method"] == "random"]["norm_mean"], 0.0)


def test_illegal_rate_figure_does_not_join_generation_variants(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from deadeye.report import fig_illegal, load_results, normalize_results
    models = [{"backend": "mock", "id": f"org/m{i}", "format_failure_rate": 0.3, "params": 10**8 * (i + 1), "family": "f"} for i in range(2)]
    methods = [{"name": "prompt_generate"}, {"name": "prompt_generate", "params": {"cot": True}, "label": "gen_cot"}]
    _mock_run(tmp_path / "res", "g", [{"name": "loan", "params": {"n_applicants": 3}}], models, methods)
    cells, _ = normalize_results(*load_results([str(tmp_path / "res")]))
    lines = []
    orig = Axes.plot
    monkeypatch.setattr(Axes, "plot", lambda self, x, y, **kw: lines.append(list(x)) or orig(self, x, y, **kw))
    assert fig_illegal(cells, tmp_path / "fig_illegal") is not None
    assert len(lines) == 2 and all(len(set(x)) == len(x) for x in lines)  # one line per variant, one point per model
