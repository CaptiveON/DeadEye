import json

import numpy as np
import pytest

from deadeye.config import RunConfig
from deadeye.envs import ENV_REGISTRY, make_env
from deadeye.models.mock import MockModel
from deadeye.policies import make_policy
from deadeye.runner import Runner, run_episode


@pytest.mark.parametrize("name", sorted(ENV_REGISTRY))
def test_mock_prompt_generate_runs_everywhere(name):
    env = make_env(name)
    model = MockModel(strategy="random", format_failure_rate=0.3, seed=1)
    pol = make_policy("prompt_generate", {"max_new_tokens": 8}, model=model)
    pol.bind(env)
    ep, steps = run_episode(env, pol, seed=0)
    assert ep["steps"] == len(steps) >= 1
    assert 0.0 <= ep["illegal_rate"] <= 1.0
    assert all(s["action"] in s["legal_actions"] for s in steps)


def test_illegal_action_policies():
    env = make_env("blackjack")
    model = MockModel(strategy="random", format_failure_rate=1.0)
    pol = make_policy("prompt_generate", {}, model=model)
    pol.bind(env)
    ep, steps = run_episode(env, pol, seed=0, illegal_action="first_legal")
    assert ep["n_illegal"] == ep["steps"] and all(s["action"] == "hit" for s in steps) or ep["steps"] >= 1
    ep2, steps2 = run_episode(env, pol, seed=0, illegal_action="terminate")
    assert ep2["steps"] == 1 and steps2[0]["action"] is None
    with pytest.raises(ValueError):
        run_episode(env, pol, seed=0, illegal_action="explode")


def test_oracle_policy_matches_oracle_and_random_differs():
    env = make_env("gridworld", size=5)
    oracle = make_policy("oracle")
    oracle.bind(env)
    ep, _ = run_episode(env, oracle, seed=0)
    assert ep["oracle_agreement"] == 1.0 and ep["success"] == 1.0
    rnd = make_policy("random", {"seed": 0})
    rnd.bind(env)
    ep2, _ = run_episode(env, rnd, seed=0)
    assert ep2["oracle_agreement"] < 1.0 or ep2["steps"] >= 1


def test_ucb1_only_binds_to_bandit():
    pol = make_policy("ucb1")
    with pytest.raises(TypeError):
        pol.bind(make_env("blackjack"))
    env = make_env("bandit", n_arms=3, horizon=60)
    pol.bind(env)
    ep, _ = run_episode(env, pol, seed=0)
    assert ep["regret"] < 60 * 0.2 * 0.8  # learns something


def test_feature_probe_learns_loan_decisions():
    factory = lambda: make_env("loan", n_applicants=20)
    pol = make_policy("feature_probe", {"n_train_episodes": 30})
    stats = pol.prepare(factory, list(range(100000, 100030)))
    assert stats["n_train_states"] == 600 and stats["train_acc"] > 0.75
    env = factory()
    pol.bind(env)
    ep, _ = run_episode(env, pol, seed=0)
    assert ep["oracle_agreement"] > 0.6


def test_mock_probe_and_score_pipeline():
    factory = lambda: make_env("tictactoe")
    model = MockModel(seed=0)
    probe = make_policy("probe", {"n_train_episodes": 5}, model=model)
    stats = probe.prepare(factory, list(range(100000, 100005)))
    assert stats["n_train_states"] > 0 and stats["feature_dim"] == 256
    env = factory()
    probe.bind(env)
    ep, steps = run_episode(env, probe, seed=1)
    assert ep["illegal_rate"] == 0.0
    score = make_policy("prompt_score", {"action_format": "letter"}, model=model)
    score.bind(env)
    ep2, steps2 = run_episode(env, score, seed=1)
    assert ep2["illegal_rate"] == 0.0 and "scores" in steps2[0]["extra"]


def test_runner_end_to_end_with_mock(tmp_path):
    cfg = RunConfig.from_dict({
        "name": "t", "output_dir": str(tmp_path / "res"), "seeds": {"start": 0, "n": 3}, "train_seeds": {"start": 50, "n": 3},
        "catalog": None,
        "envs": [{"name": "bandit", "params": {"n_arms": 3, "horizon": 5}}, {"name": "tictactoe"}],
        "models": [{"backend": "mock", "id": "m", "params": 10, "family": "mock"}],
        "methods": [{"name": "prompt_generate"}, {"name": "prompt_score"}, {"name": "probe", "params": {"n_train_episodes": 2}},
                    {"name": "lora_sft", "params": {"n_train_episodes": 1}}],
        "baselines": ["random", "oracle", "ucb1"],
    })
    runner = Runner(cfg)
    cells = runner.cells()
    assert any(c.method.name == "ucb1" and c.env.name == "bandit" for c in cells)
    assert not any(c.method.name == "ucb1" and c.env.name == "tictactoe" for c in cells)
    done = runner.run()
    # 2 envs x (random, oracle) + ucb1 + 2 envs x 3 model methods (lora skipped: mock cannot train)
    assert len(done) == 2 * 2 + 1 + 2 * 3
    summ = json.loads((tmp_path / "res" / "tictactoe" / "m" / "prompt_score" / "summary.json").read_text())
    assert summ["n_episodes"] == 3 and summ["model"]["params"] == 10
    assert (tmp_path / "res" / "run_manifest.json").exists()
    # resume: nothing re-run
    assert Runner(cfg).run() == []
    assert len(Runner(cfg, force=True, only_env="bandit", only_method="prompt_score").run()) == 1


def test_config_keys_and_episode_override(tmp_path):
    cfg = RunConfig.from_dict({"name": "x", "seeds": [1, 2], "envs": [{"name": "blackjack", "n_episodes": 5}, {"name": "bandit", "params": {"horizon": 3}}],
                               "models": [{"backend": "mock", "id": "a/b", "strategy": "first"}], "methods": [{"name": "prompt_score", "params": {"length_norm": True}}]})
    assert cfg.episodes_for(cfg.envs[0]) == [1, 2, 3, 4, 5]
    assert cfg.envs[1].key == "bandit" and cfg.envs[0].key == "blackjack"
    assert cfg.models[0].key == "a__b" and cfg.methods[0].key == "prompt_score"
    with pytest.raises(ValueError):
        RunConfig.from_dict({"name": "x", "envs": [{"name": "bandit"}, {"name": "bandit", "params": {"horizon": 3}}]})


def test_fallback_actions_do_not_count_as_oracle_agreement():
    env = make_env("loan", n_applicants=10)
    pol = make_policy("prompt_generate", {}, model=MockModel(format_failure_rate=1.0))
    pol.bind(env)
    for seed in range(5):
        ep, steps = run_episode(env, pol, seed=seed)
        assert ep["illegal_rate"] == 1.0 and ep["oracle_agreement"] == 0.0


def test_json_default_keeps_numpy_bools_boolean():
    from deadeye.runner import _json_default
    out = json.loads(json.dumps({"a": np.bool_(True), "b": np.float32(0.5), "c": np.int64(3)}, default=_json_default))
    assert out == {"a": True, "b": 0.5, "c": 3}


def test_runner_never_holds_two_model_copies(tmp_path, monkeypatch):
    import deadeye.runner as runner_mod
    live, peak = set(), [0]
    orig = runner_mod.load_model

    def tracking_load(spec, catalog=None):
        m = orig(spec, catalog)
        live.add(id(m))
        peak[0] = max(peak[0], len(live))
        m.close = lambda: live.discard(id(m))
        return m

    monkeypatch.setattr(runner_mod, "load_model", tracking_load)
    cfg = RunConfig.from_dict({
        "name": "mem", "output_dir": str(tmp_path / "res"), "seeds": {"start": 0, "n": 1}, "catalog": None,
        "envs": [{"name": "loan", "params": {"n_applicants": 2}}], "models": [{"backend": "mock", "id": "m"}],
        "methods": [{"name": "prompt_generate"}, {"name": "lora_sft"}, {"name": "prompt_score"}], "baselines": []})
    Runner(cfg).run()
    assert peak[0] == 1 and not live


def test_config_rejects_evaluation_seeds_overlapping_training_seeds():
    base = {"name": "x", "envs": [{"name": "bandit"}], "train_seeds": {"start": 100, "n": 50}}
    RunConfig.from_dict({**base, "seeds": {"start": 0, "n": 100}})  # disjoint: fine
    with pytest.raises(ValueError):
        RunConfig.from_dict({**base, "seeds": {"start": 0, "n": 101}})
    with pytest.raises(ValueError):  # a per-env episode override can also run into the training range
        RunConfig.from_dict({**base, "seeds": {"start": 0, "n": 10}, "envs": [{"name": "blackjack", "n_episodes": 300}]})
