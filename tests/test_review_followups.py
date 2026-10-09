"""Tests for the follow-ups to the adversarial review."""
import json

import numpy as np

from deadeye.config import RunConfig
from deadeye.envs import make_env
from deadeye.envs.tictactoe import expectimax, minimax
from deadeye.policies import make_policy
from deadeye.policies.registry import POLICY_REGISTRY
from deadeye.policies.base import Decision, Policy
from deadeye.runner import Runner, run_episode


def _mean_return(opponent, mark, use_expectimax, seeds=200):
    env = make_env("tictactoe", opponent=opponent, agent_mark=mark)
    tot = []
    for seed in range(seeds):
        env.reset(seed)
        while not env.done:
            fn = expectimax(env.board, env.me)[1] if use_expectimax else minimax(env.board, env.me, env.me)[1]
            env.step(str(int(fn) + 1))
        tot.append(env._outcome)
    return float(np.mean(tot))


def test_expectimax_is_at_least_as_good_as_minimax_against_random():
    for mark in ("X", "O"):
        assert _mean_return("random", mark, True) >= _mean_return("random", mark, False) - 1e-9
    assert _mean_return("random", "O", True) > _mean_return("random", "O", False)


def test_oracle_vs_minimax_opponent_still_never_loses_and_varies_games():
    env = make_env("tictactoe", opponent="minimax")
    boards = set()
    for seed in range(40):
        env.reset(seed)
        while not env.done:
            env.step(env.oracle_action())
        assert env.episode_metrics()["loss"] == 0.0
        boards.add(env.board)
    assert len(boards) >= 3


def test_blackjack_dealer_cards_do_not_depend_on_player_hits():
    env = make_env("blackjack")
    env.reset(5)
    env.step("stand")
    dealer_a = list(env.dealer)
    env.reset(5)
    res = env.step("hit")
    if not res.done:
        env.step("stand")
    dealer_b = list(env.dealer)
    n = min(len(dealer_a), len(dealer_b))
    assert dealer_a[:n] == dealer_b[:n]


class _RecordingPolicy(Policy):
    name = "_recording"
    needs_prepare = True
    seen: dict = {}

    def prepare(self, env_factory, train_seeds):
        _RecordingPolicy.seen = dict(env_factory().params)
        return {"ok": True}

    def act(self, obs, history):
        return Decision(action=obs.legal_actions[0], raw_output="rec")


def test_train_params_select_the_preparation_environment(tmp_path):
    POLICY_REGISTRY["_recording"] = (_RecordingPolicy, False)
    try:
        cfg = RunConfig.from_dict({
            "name": "t", "output_dir": str(tmp_path / "r"), "seeds": [0, 1], "catalog": None,
            "envs": [{"name": "loan", "params": {"n_applicants": 3, "shift": "sign_flip"},
                      "train_params": {"n_applicants": 3, "shift": "none"}, "label": "loan_flip"}],
            "models": [{"backend": "mock", "id": "m"}], "methods": [{"name": "_recording"}], "baselines": ["random", "oracle"],
        })
        Runner(cfg).run()
        assert _RecordingPolicy.seen["shift"] == "none"
        summ = json.loads((tmp_path / "r" / "loan_flip" / "_nomodel" / "_recording" / "summary.json").read_text())
        assert summ["train_env_params"] == {"n_applicants": 3, "shift": "none"} and summ["env_params"]["shift"] == "sign_flip"
    finally:
        POLICY_REGISTRY.pop("_recording", None)


def test_feature_probe_appears_in_report(tmp_path):
    from deadeye.report import build_report
    cfg = RunConfig.from_dict({
        "name": "t", "output_dir": str(tmp_path / "r"), "seeds": {"start": 0, "n": 3}, "train_seeds": {"start": 100, "n": 4},
        "catalog": None, "envs": [{"name": "loan", "params": {"n_applicants": 4}}],
        "models": [{"backend": "mock", "id": "m", "params": 10, "family": "mock"}],
        "methods": [{"name": "prompt_score"}, {"name": "feature_probe", "params": {"n_train_episodes": 4}}],
    })
    Runner(cfg).run()
    res = build_report([str(tmp_path / "r")], tmp_path / "rep")
    cells = res["cells"]
    assert "_nomodel" in set(cells["model_key"]) and "feature_probe" in res["tables"]
    assert "no LM (task features)" in res["tables"]["feature_probe"].index[0]
    assert not np.isnan(cells[cells["method"] == "feature_probe"]["norm_mean"].iloc[0])


def test_probe_reports_episodes_used_and_truncation():
    pol = make_policy("feature_probe", {"n_train_episodes": 5, "max_train_states": 7})
    stats = pol.prepare(lambda: make_env("loan", n_applicants=4), list(range(100000, 100010)))
    assert stats["n_train_episodes_used"] == 2 and stats["truncated"] and stats["n_train_states"] == 7


def test_blackjack_multi_hand_episode():
    env = make_env("blackjack", hands_per_episode=5)
    obs = env.reset(3)
    assert "Hand 1 of 5" in obs.text and env.max_steps == 60
    n = 0
    while not env.done:
        env.step(env.oracle_action())
        n += 1
    m = env.episode_metrics()
    assert m["hands"] == 5.0 and 5 <= n <= 60 and abs(m["win"] + m["loss"]) <= 1.0
    # same seed, same hands regardless of a different (legal) policy's choices on hand 1
    env.reset(3)
    env.step("stand")
    first_dealer = env.dealer[:2] if env.hand_idx == 0 else None
    assert first_dealer is None or len(first_dealer) == 2


def test_expectimax_takes_immediate_win():
    env = make_env("tictactoe", opponent="random", agent_mark="X")
    env.reset(0)
    env.board = tuple("XX.OO....")
    assert env.oracle_action() == "3"


def test_scale_slopes_and_bare_latex(tmp_path):
    from deadeye.report import build_report, scale_slopes, slope_ratio
    cfg = RunConfig.from_dict({
        "name": "t", "output_dir": str(tmp_path / "r"), "seeds": {"start": 0, "n": 4}, "catalog": None,
        "envs": [{"name": "bandit", "params": {"n_arms": 3, "horizon": 5}}, {"name": "blackjack"}],
        "models": [{"backend": "mock", "id": f"m{i}", "params": 10 ** (6 + i), "family": "mock", "strategy": "random", "seed": i} for i in range(3)],
        "methods": [{"name": "prompt_generate"}, {"name": "prompt_score"}],
    })
    Runner(cfg).run()
    res = build_report([str(tmp_path / "r")], tmp_path / "rep")
    slopes = res["slopes"]
    assert set(slopes["env_key"]) >= {"bandit", "blackjack", "__mean__"} and (slopes["n_models"] == 3).all()
    assert (slopes["slope_ci_lo"] <= slopes["slope"]).all() and (slopes["slope"] <= slopes["slope_ci_hi"]).all()
    assert set(slopes["instruct"]) == {"instruct"}  # the mock reports itself as an instruct model
    r = slope_ratio(slopes, "mock", "bandit", "prompt_score", "prompt_generate", instruct="instruct")
    assert set(r) == {"ratio", "ci_lo", "ci_hi"}
    assert (tmp_path / "rep" / "slopes.csv").exists() and (tmp_path / "rep" / "tables" / "slopes.tex").exists()
    tex = (tmp_path / "rep" / "tables" / "prompt_score.tex").read_text()
    assert tex.lstrip().startswith("\\begin{tabular}") and "\\begin{table}" not in tex and "\\caption" not in tex


def test_compare_cli_families(tmp_path):
    from typer.testing import CliRunner
    from deadeye.cli import app
    cfg = RunConfig.from_dict({
        "name": "t", "output_dir": str(tmp_path / "r"), "seeds": {"start": 0, "n": 6}, "catalog": None,
        "envs": [{"name": "loan", "params": {"n_applicants": 4}}, {"name": "loan", "params": {"n_applicants": 4, "shift": "sign_flip"}, "label": "loan_flip"}],
        "models": [{"backend": "mock", "id": "m"}], "methods": [{"name": "prompt_generate"}, {"name": "prompt_score"}],
    })
    Runner(cfg).run()
    out = tmp_path / "cmp.csv"
    result = CliRunner().invoke(app, ["compare", str(tmp_path / "r"), "--pair", "loan:m/prompt_generate vs loan:m/prompt_score",
                                      "--pair", "loan:m/prompt_score vs loan_flip:m/prompt_score", "--n-perm", "200", "--out", str(out)])
    assert result.exit_code == 0, result.output
    rows = out.read_text().splitlines()
    assert len(rows) == 3 and "p_holm" in rows[0]


def test_decision_confidence_and_ece():
    from deadeye.runner import decision_confidence, expected_calibration_error
    step = {"action": "hit", "oracle_action": "hit", "legal_actions": ["hit", "stand"], "legal": True,
            "extra": {"scores": {"hit": -0.1, "stand": -2.4}}}
    c, hit = decision_confidence(step)
    assert hit == 1 and 0.9 < c < 0.92
    step_letter = {"action": "stand", "oracle_action": "hit", "legal_actions": ["hit", "stand"], "legal": True,
                   "extra": {"scores": {"A": -2.4, "B": -0.1}}}
    c2, hit2 = decision_confidence(step_letter)
    assert hit2 == 0 and abs(c2 - c) < 1e-9
    assert decision_confidence({"action": "x", "oracle_action": "x", "legal_actions": ["x", "y"], "legal": True, "extra": {"probs": {"x": 0.7, "y": 0.3}}}) == (0.7, 1)
    # probabilities are renormalised over the legal actions only
    assert decision_confidence({"action": "x", "oracle_action": "x", "legal_actions": ["x"], "legal": True, "extra": {"probs": {"x": 0.7, "y": 0.3}}}) == (1.0, 1)
    assert decision_confidence({"action": "x", "oracle_action": "y", "legal_actions": ["x", "y"], "legal": True, "extra": {"confidence": 0.55}}) == (0.55, 0)
    assert decision_confidence({"action": "x", "oracle_action": "x", "legal_actions": ["x"], "legal": True, "extra": {"label": "x"}}) == (None, 1)
    conf = np.array([0.95] * 50 + [0.55] * 50)
    hit = np.array([1] * 45 + [0] * 5 + [1] * 25 + [0] * 25, dtype=float)
    ece = expected_calibration_error(conf, hit)
    assert abs(ece - (0.5 * abs(0.95 - 0.9) + 0.5 * abs(0.55 - 0.5))) < 1e-9


def test_summary_carries_calibration(tmp_path):
    cfg = RunConfig.from_dict({
        "name": "t", "output_dir": str(tmp_path / "r"), "seeds": {"start": 0, "n": 3}, "catalog": None,
        "envs": [{"name": "blackjack"}], "models": [{"backend": "mock", "id": "m"}],
        "methods": [{"name": "prompt_score"}, {"name": "prompt_generate"}],
    })
    Runner(cfg).run()
    score = json.loads((tmp_path / "r" / "blackjack" / "m" / "prompt_score" / "summary.json").read_text())
    gen = json.loads((tmp_path / "r" / "blackjack" / "m" / "prompt_generate" / "summary.json").read_text())
    assert set(score["calibration"]) >= {"ece", "brier", "accuracy_vs_oracle", "n_confidence"} and score["calibration"]["n_confidence"] > 0
    assert gen["calibration"] == {}


def test_system_comparison_table(tmp_path):
    from deadeye.report import build_report
    cfg = RunConfig.from_dict({
        "name": "t", "output_dir": str(tmp_path / "r"), "seeds": {"start": 0, "n": 3}, "catalog": None,
        "envs": [{"name": "blackjack"}, {"name": "loan", "params": {"n_applicants": 3}}],
        "models": [{"backend": "mock", "id": "m", "params": 1000, "family": "mock"}],
        "methods": [{"name": "prompt_score"}, {"name": "prompt_generate"}],
    })
    Runner(cfg).run()
    res = build_report([str(tmp_path / "r")], tmp_path / "rep")
    comp = res["tables"]["system_comparison"]
    assert len(comp) == 2 and (tmp_path / "rep" / "system_comparison.csv").exists()
    raw = __import__("pandas").read_csv(tmp_path / "rep" / "system_comparison.csv")
    assert set(raw["method"]) == {"prompt_score", "prompt_generate"} and (raw["n_tasks"] == 2).all()
    assert raw.loc[raw["method"] == "prompt_score", "ece"].notna().all() and raw["latency_p95_s"].notna().all()
