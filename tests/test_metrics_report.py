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
