"""Exercises the real transformers code path with a tiny randomly initialised model (no downloads)."""
import numpy as np
import pytest

from deadeye.envs import make_env
from deadeye.models.base import Message
from deadeye.policies import make_policy
from deadeye.runner import run_episode

MSGS = [Message("system", "You are navigating a grid."),
        Message("user", "Map:\n..A\n.G.\nLegal actions: up, down\nReply with exactly one line of the form `Action: <name>`.")]


def test_render_uses_chat_template(tiny_model):
    text = tiny_model.render(MSGS)
    assert text.startswith("<|system|>") and text.endswith("<|assistant|>\n")
    ids = tiny_model.encode(MSGS)
    assert len(ids) > 10 and tiny_model.info.instruct is True


def test_generate_greedy_is_deterministic(tiny_model):
    a = tiny_model.generate(MSGS, max_new_tokens=6)
    b = tiny_model.generate(MSGS, max_new_tokens=6)
    assert a.text == b.text and a.completion_tokens == 6 and a.prompt_tokens == len(tiny_model.encode(MSGS))
    tiny_model.seed(0)
    c = tiny_model.generate(MSGS, max_new_tokens=6, temperature=1.0)
    tiny_model.seed(0)
    d = tiny_model.generate(MSGS, max_new_tokens=6, temperature=1.0)
    assert c.text == d.text


def test_score_choices_matches_manual_logprob(tiny_model):
    import torch
    res = tiny_model.score_choices(MSGS, ["up", "down"])
    assert len(res.logprobs) == 2 and all(lp < 0 for lp in res.logprobs)
    # manual single-sequence computation for the first choice
    p = tiny_model.encode(MSGS)
    c = tiny_model.tokenizer("up", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p + c])
    with torch.no_grad():
        logp = torch.log_softmax(tiny_model.model(input_ids=ids).logits.float(), -1)[0]
    manual = sum(logp[len(p) - 1 + i, t].item() for i, t in enumerate(c))
    assert abs(manual - res.logprobs[0]) < 1e-3
    norm = tiny_model.score_choices(MSGS, ["up", "down"], length_norm=True)
    assert abs(norm.logprobs[0] - res.logprobs[0] / len(c)) < 1e-6


def test_embed_layers(tiny_model):
    mid = tiny_model.embed(MSGS, layer="mid")
    last = tiny_model.embed(MSGS, layer="last")
    assert mid.vector.shape == (64,) and last.vector.shape == (64,)
    assert mid.layer == 1 and last.layer == 2
    assert not np.allclose(mid.vector, last.vector)
    with pytest.raises(ValueError):
        tiny_model.embed(MSGS, layer=7)


def test_lora_sft_changes_model(tiny_model_path):
    from deadeye.models.hf_backend import HFModel
    from deadeye.models.training import train_lora_sft
    m = HFModel(str(tiny_model_path))
    before = m.score_choices(MSGS, ["up", "down"]).logprobs
    stats = train_lora_sft(m, [(MSGS, "up")] * 8, epochs=3, batch_size=4, lr=1e-3)
    after = m.score_choices(MSGS, ["up", "down"]).logprobs
    assert stats["trainable_params"] > 0 and stats["loss_last"] < stats["loss_first"]
    assert after[0] > before[0]  # the trained target became more likely
    assert type(m.model).__name__ == "LlamaForCausalLM"  # merged back into a plain model


def test_hf_policies_run_in_envs(tiny_model):
    env = make_env("bandit", n_arms=3, horizon=4)
    for name, params in [("prompt_generate", {"max_new_tokens": 4}), ("prompt_score", {}), ("prompt_score", {"action_format": "letter"})]:
        pol = make_policy(name, params, model=tiny_model)
        pol.bind(env)
        ep, steps = run_episode(env, pol, seed=0)
        assert ep["steps"] == 4 and all(s["action"] in s["legal_actions"] for s in steps)
    probe = make_policy("probe", {"n_train_episodes": 2, "layer": "mid"}, model=tiny_model)
    stats = probe.prepare(lambda: make_env("bandit", n_arms=3, horizon=4), [100000, 100001])
    assert stats["feature_dim"] == 64
    probe.bind(env)
    ep, _ = run_episode(env, probe, seed=0)
    assert ep["illegal_rate"] == 0.0


def test_registry_loads_tiny_and_overrides_metadata(tiny_model_path):
    from deadeye.models.registry import load_model
    m = load_model({"backend": "hf", "id": str(tiny_model_path), "params": 123, "family": "tiny"})
    assert m.info.params == 123 and m.info.family == "tiny" and m.info.extra["counted_params"] > 0
