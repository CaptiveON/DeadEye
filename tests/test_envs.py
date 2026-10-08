import numpy as np
import pytest

from deadeye.envs import ENV_REGISTRY, make_env
from deadeye.envs.tictactoe import minimax, winner


@pytest.mark.parametrize("name", sorted(ENV_REGISTRY))
def test_oracle_is_legal_and_terminates(name):
    env = make_env(name)
    for seed in range(5):
        obs = env.reset(seed)
        n = 0
        while not env.done:
            a = env.oracle_action()
            assert a in obs.legal_actions
            assert set(obs.legal_actions) <= set(env.action_labels)
            obs = env.step(a).obs
            n += 1
            assert n <= env.max_steps
        m = env.episode_metrics()
        assert all(isinstance(v, float) for v in m.values())


@pytest.mark.parametrize("name", sorted(ENV_REGISTRY))
def test_reset_is_deterministic(name):
    a, b = make_env(name), make_env(name)
    oa, ob = a.reset(3), b.reset(3)
    assert oa.text == ob.text and oa.legal_actions == ob.legal_actions
    rng = np.random.default_rng(0)
    while not a.done:
        act = str(rng.choice(oa.legal_actions))
        ra, rb = a.step(act), b.step(act)
        assert ra.reward == rb.reward and ra.obs.text == rb.obs.text
        oa = ra.obs


@pytest.mark.parametrize("name", sorted(ENV_REGISTRY))
def test_oracle_beats_random_on_average(name):
    env = make_env(name)
    rng = np.random.default_rng(1)

    def rollout(policy):
        tot = []
        for seed in range(40):
            obs = env.reset(seed)
            r = 0.0
            while not env.done:
                a = env.oracle_action() if policy == "oracle" else str(rng.choice(obs.legal_actions))
                res = env.step(a)
                r += res.reward
                obs = res.obs
            tot.append(r)
        return float(np.mean(tot))

    assert rollout("oracle") > rollout("random")


def test_bandit_regret_zero_for_oracle_and_positive_for_random():
    env = make_env("bandit", n_arms=4, horizon=20)
    env.reset(0)
    while not env.done:
        env.step(env.oracle_action())
    assert env.episode_metrics()["regret"] == 0.0
    env.reset(0)
    while not env.done:
        env.step("arm_1" if env.best_arm != 0 else "arm_2")
    assert env.episode_metrics()["regret"] > 0


def test_bandit_illegal_action_raises():
    env = make_env("bandit")
    env.reset(0)
    with pytest.raises(ValueError):
        env.step("arm_99")


def test_gridworld_oracle_is_shortest_path():
    env = make_env("gridworld", size=6)
    for seed in range(10):
        env.reset(seed)
        n = 0
        while not env.done:
            env.step(env.oracle_action())
            n += 1
        m = env.episode_metrics()
        assert m["success"] == 1.0 and m["steps"] == m["optimal_steps"] and n >= env.min_distance


def test_tictactoe_minimax_never_loses_vs_minimax():
    env = make_env("tictactoe", opponent="minimax")
    for seed in range(6):
        env.reset(seed)
        while not env.done:
            env.step(env.oracle_action())
        assert env.episode_metrics()["loss"] == 0.0


def test_tictactoe_helpers():
    assert winner(tuple("XXX......")) == "X"
    assert winner(tuple("XOXXOOOXX")) == "draw"
    val, mv = minimax(tuple("XX.OO...."), "X", "X")
    assert mv == 2 and val > 0


def test_blackjack_basic_strategy_examples():
    env = make_env("blackjack")
    env.reset(0)
    env.player, env.dealer = [10, 6], [10, 5]
    assert env.oracle_action() == "hit"
    env.player, env.dealer = [10, 6], [5, 5]
    assert env.oracle_action() == "stand"
    env.player, env.dealer = [1, 7], [9, 5]
    assert env.oracle_action() == "hit"
    env.player, env.dealer = [10, 10], [1, 5]
    assert env.oracle_action() == "stand"


def test_loan_shift_variants_run():
    for shift in ("none", "covariate", "sign_flip"):
        env = make_env("loan", n_applicants=6, shift=shift)
        obs = env.reset(0)
        assert obs.features is not None and len(obs.features) == 11
        while not env.done:
            obs = env.step(env.oracle_action()).obs
        assert env.episode_metrics()["regret"] == 0.0


def test_contextual_bandit_features_and_oracle():
    env = make_env("contextual_bandit", n_actions=3, dim=4, horizon=5)
    obs = env.reset(2)
    assert obs.features.shape == (4,)
    best = env.oracle_action()
    res = env.step(best)
    assert res.info["optimal"]
