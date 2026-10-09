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


def test_score_choices_prefix_choices_are_terminated(tiny_model):
    """A choice whose tokens extend another's (arm_1 / arm_10 under digit-splitting tokenisers) must be able to win."""
    import torch
    tok = tiny_model.tokenizer
    c0 = tok("up", add_special_tokens=False)["input_ids"]
    c1 = tok("up down", add_special_tokens=False)["input_ids"]
    assert c1[:len(c0)] == c0 and len(c1) > len(c0)  # precondition: strict token prefix
    res = tiny_model.score_choices(MSGS, ["up", "up down"])
    p = tiny_model.encode(MSGS)
    for c, lp in zip((c0, c1), res.logprobs):
        seq = c + [tok.eos_token_id]
        with torch.no_grad():
            logp = torch.log_softmax(tiny_model.model(input_ids=torch.tensor([p + seq])).logits.float(), -1)[0]
        assert abs(sum(logp[len(p) - 1 + i, t].item() for i, t in enumerate(seq)) - lp) < 1e-3
    assert res.choice_tokens == [len(c0) + 1, len(c1) + 1]
    # prefix-free choice sets are scored exactly as before (no terminator)
    assert tiny_model.score_choices(MSGS, ["up", "down"]).choice_tokens == [len(c0), len(tok("down", add_special_tokens=False)["input_ids"])]


def test_greedy_generation_ignores_checkpoint_repetition_penalty(tiny_model_path, tmp_path):
    """Checkpoint generation_config.json defaults (Qwen2.5-Instruct: repetition_penalty 1.05) must not alter greedy decoding."""
    import json
    import shutil
    from deadeye.models.hf_backend import HFModel
    ref = HFModel(str(tiny_model_path))
    want = ref.generate(MSGS, max_new_tokens=16).text
    d = tmp_path / "tiny_rp"
    shutil.copytree(tiny_model_path, d)
    gc = {"bos_token_id": ref.tokenizer.bos_token_id, "eos_token_id": ref.tokenizer.eos_token_id,
          "pad_token_id": ref.tokenizer.pad_token_id, "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
          "repetition_penalty": 1.3}
    (d / "generation_config.json").write_text(json.dumps(gc))
    assert HFModel(str(d)).generate(MSGS, max_new_tokens=16).text == want


def test_close_frees_weights_held_in_reference_cycles(tiny_model_path):
    """device_map="auto" dispatch hooks create reference cycles; close() must still release the weights so the runner
    can load a fresh copy for LoRA without holding two copies in memory."""
    import gc
    import weakref
    from accelerate.hooks import ModelHook, add_hook_to_module
    from deadeye.models.hf_backend import HFModel
    m = HFModel(str(tiny_model_path))
    add_hook_to_module(m.model, ModelHook())
    ref = weakref.ref(m.model)
    gc.disable()
    try:
        m.close()
        assert ref() is None
    finally:
        gc.enable()


def test_registry_renders_declared_base_models_plainly(tiny_model_path):
    """Base checkpoints (catalogue or spec `instruct: false`) use the plain rendering even if their tokenizer ships a
    chat template (Qwen2.5 base tokenizers do); an explicit `use_chat_template` still wins."""
    from deadeye.models.registry import load_model
    path = str(tiny_model_path)
    base = load_model({"backend": "hf", "id": path, "instruct": False})
    assert base.render(MSGS).endswith("Output:\n") and base.info.extra["chat_template"] is False
    assert load_model({"backend": "hf", "id": path}, {path: {"instruct": False}}).render(MSGS).endswith("Output:\n")
    forced = load_model({"backend": "hf", "id": path, "instruct": False, "use_chat_template": True})
    assert forced.render(MSGS).endswith("<|assistant|>\n")
    inst = load_model({"backend": "hf", "id": path})
    assert inst.render(MSGS).endswith("<|assistant|>\n") and inst.info.extra["chat_template"] is True


def test_device_and_dtype_selection():
    import types
    from deadeye.models.hf_backend import default_dtype, select_device

    class _Cuda:
        def __init__(self, ok):
            self.ok = ok

        def is_available(self):
            return self.ok

    def fake_torch(cuda: bool, mps: bool | None):
        t = types.SimpleNamespace(cuda=_Cuda(cuda), bfloat16="bf16", float16="fp16", float32="fp32")
        t.backends = types.SimpleNamespace(mps=None if mps is None else types.SimpleNamespace(is_available=lambda: mps))
        return t

    assert select_device("auto", fake_torch(True, True)) == "cuda"
    assert select_device("auto", fake_torch(False, True)) == "mps"
    assert select_device("auto", fake_torch(False, False)) == "cpu"
    assert select_device("auto", fake_torch(False, None)) == "cpu"
    assert select_device("cpu", fake_torch(True, True)) == "cpu"
    assert default_dtype("cuda", fake_torch(True, True)) == "bf16"
    assert default_dtype("cpu", fake_torch(False, False)) == "fp32"

    class _Ones:
        def __init__(self, fail):
            self.fail = fail

        def __call__(self, *a, **k):
            if self.fail:
                raise RuntimeError("no bf16 on this device")
            return types.SimpleNamespace(__mul__=lambda s, o: s, sum=lambda: types.SimpleNamespace(item=lambda: 4.0))

    t = fake_torch(False, True)
    t.ones = _Ones(fail=True)
    assert default_dtype("mps", t) == "fp16"


def test_quantisation_refused_off_cuda(tiny_model_path):
    import torch
    from deadeye.models.hf_backend import HFModel
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(ValueError):
        HFModel(str(tiny_model_path), quantization="int4")
