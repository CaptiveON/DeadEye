"""Build a tiny randomly-initialised Llama-architecture model with its own tokenizer.

Used by the test-suite and the ``smoke`` config so that the *real* Hugging Face code path
(chat template, generation, scoring, hidden states, LoRA training) runs end-to-end without any
download. Its outputs are meaningless by construction; it is a pipeline check, not a baseline.
"""
from __future__ import annotations

import os
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
)


def _corpus() -> list[str]:
    from deadeye.envs import ENV_REGISTRY, make_env
    texts = ["Action:", "Legal actions:", "Answer with exactly one action."]
    for name in ENV_REGISTRY:
        env = make_env(name)
        texts.append(env.describe())
        texts.extend(env.action_labels)
        for seed in range(3):
            obs = env.reset(seed)
            while not env.done:
                texts.append(obs.text)
                texts.append("Legal actions: " + ", ".join(obs.legal_actions))
                obs = env.step(env.oracle_action()).obs
    return texts


def build_tiny_model(path: str | Path, vocab_size: int = 1024, hidden: int = 64, n_layers: int = 2,
                     n_heads: int = 4, seed: int = 0) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    specials = ["<unk>", "<s>", "</s>", "<pad>", "<|system|>", "<|user|>", "<|assistant|>"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(_corpus(), trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                   pad_token="<pad>", additional_special_tokens=specials[4:])
    fast.chat_template = CHAT_TEMPLATE
    cfg = LlamaConfig(vocab_size=len(fast), hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=n_layers,
                      num_attention_heads=n_heads, num_key_value_heads=max(1, n_heads // 2), max_position_embeddings=2048,
                      bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id,
                      tie_word_embeddings=False)
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return path


def ensure_tiny_model(cache_dir: str | Path | None = None) -> Path:
    base = Path(cache_dir or os.environ.get("DEADEYE_CACHE", Path.home() / ".cache" / "deadeye"))
    target = base / "tiny-random"
    if not (target / "config.json").exists():
        build_tiny_model(target)
    return target
