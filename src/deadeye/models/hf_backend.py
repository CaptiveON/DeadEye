"""Local Hugging Face backend: generation, exact log-likelihood scoring, hidden-state extraction.

Works for any causal LM loadable with ``AutoModelForCausalLM``. Chat-template rendering is used when
the tokenizer ships one (instruct models) and ``use_chat_template`` is true; base models fall back
to a plain "Input/Output" rendering. Thinking-mode switches (e.g. Qwen3 ``enable_thinking``) are
passed through ``chat_template_kwargs``.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from deadeye.models.base import (EmbedResult, GenerationResult, LanguageModel, Message, ModelInfo, ScoreResult,
                                 render_messages_plain, resolve_layer)

log = logging.getLogger(__name__)

_DTYPES = {"float32": "float32", "fp32": "float32", "bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16", "fp16": "float16"}


class HFModel(LanguageModel):
    capabilities = frozenset({"generate", "score", "embed", "train"})

    def __init__(self, model_id: str, dtype: str = "auto", quantization: str | None = None, device: str = "auto",
                 use_chat_template: bool = True, chat_template_kwargs: dict[str, Any] | None = None,
                 max_length: int = 4096, trust_remote_code: bool = False, local_files_only: bool = False,
                 revision: str | None = None, **_: object) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.use_chat_template = use_chat_template
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.max_length = int(max_length)
        cuda = torch.cuda.is_available()
        self.device = ("cuda" if cuda else "cpu") if device == "auto" else device
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        else:
            torch_dtype = getattr(torch, _DTYPES[dtype])
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "local_files_only": local_files_only}
        if revision:
            kwargs["revision"] = revision
        precision = str(torch_dtype).replace("torch.", "")
        if quantization in ("int4", "int8"):
            from transformers import BitsAndBytesConfig
            if quantization == "int4":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                                                   bnb_4bit_use_double_quant=True)
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            precision = quantization
        elif quantization:
            raise ValueError(f"unknown quantization {quantization!r} (use int4|int8)")
        if self.device.startswith("cuda"):
            kwargs["device_map"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code,
                                                       local_files_only=local_files_only, revision=revision)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
        except TypeError:  # transformers < 4.56 spelled the argument torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
        if not self.device.startswith("cuda"):
            self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.quantized = quantization is not None
        n_params = sum(p.numel() for p in self.model.parameters())
        self.info = ModelInfo(id=model_id, backend="hf", params=None if self.quantized else n_params,
                              instruct=bool(self.tokenizer.chat_template), precision=precision,
                              extra={"counted_params": n_params, "device": self.device,
                                     "n_layers": int(getattr(self.model.config, "num_hidden_layers", 0) or 0)})

    # ------------------------------------------------------------------ helpers
    def _describe(self) -> str:
        return f"{self.model_id} ({self.info.precision}, {self.device})"

    def seed(self, s: int) -> None:
        self.torch.manual_seed(int(s))

    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def render(self, messages: list[Message]) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        if self.use_chat_template and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                          **self.chat_template_kwargs)
            except Exception as e:  # templates that reject system messages (some Gemma releases)
                if msgs and msgs[0]["role"] == "system" and len(msgs) > 1:
                    merged = [{"role": msgs[1]["role"], "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}] + msgs[2:]
                    return self.tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True,
                                                              **self.chat_template_kwargs)
                raise e
        return render_messages_plain(messages)

    def encode(self, messages: list[Message]) -> list[int]:
        text = self.render(messages)
        uses_template = bool(self.use_chat_template and self.tokenizer.chat_template)
        ids = self.tokenizer(text, add_special_tokens=not uses_template)["input_ids"]
        if len(ids) > self.max_length:
            log.warning("prompt of %d tokens truncated to %d (keeping the end)", len(ids), self.max_length)
            ids = ids[-self.max_length:]
        return list(ids)

    def _batch(self, seqs: list[list[int]]):
        torch = self.torch
        pad = self.tokenizer.pad_token_id
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        dev = self.model.device if hasattr(self.model, "device") else self.device
        return ids.to(dev), mask.to(dev)

    # ------------------------------------------------------------------ primitives
    def generate(self, messages, max_new_tokens=64, temperature=0.0, stop=None) -> GenerationResult:
        torch = self.torch
        prompt_ids = self.encode(messages)
        ids, mask = self._batch([prompt_ids])
        # repetition_penalty=1.0 overrides checkpoint defaults (e.g. Qwen2.5-Instruct ships 1.05, its base model does
        # not), which would otherwise silently change "greedy" decoding for some checkpoints only.
        gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "pad_token_id": self.tokenizer.pad_token_id,
                                      "repetition_penalty": 1.0}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=1.0, top_k=0)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=mask, **gen_kwargs)
        latency = time.perf_counter() - t0
        new = out[0, len(prompt_ids):]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        if stop:
            cut = min([text.find(s) for s in stop if s in text] or [len(text)])
            text = text[:cut]
        return GenerationResult(text=text, prompt_tokens=len(prompt_ids), completion_tokens=int(new.numel()), latency_s=latency)

    def score_choices(self, messages, choices, length_norm=False) -> ScoreResult:
        torch = self.torch
        prompt_ids = self.encode(messages)
        choice_ids = [self.tokenizer(c, add_special_tokens=False)["input_ids"] for c in choices]
        if any(len(c) == 0 for c in choice_ids):
            raise ValueError("a choice tokenised to zero tokens")
        # If one choice's tokens are a strict prefix of another's (e.g. "arm_1" vs "arm_10" with digit-splitting
        # tokenisers), the summed log-probability of the longer choice can never exceed the shorter one. Score
        # "choice followed by end-of-sequence" instead (the same terminator LoRA training appends to its targets).
        if any(len(a) < len(b) and b[:len(a)] == a for a in choice_ids for b in choice_ids):
            eos = self.tokenizer.eos_token_id
            term = [eos] if eos is not None else self.tokenizer("\n", add_special_tokens=False)["input_ids"]
            choice_ids = [c + term for c in choice_ids]
        seqs = [prompt_ids + c for c in choice_ids]
        ids, mask = self._batch(seqs)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
        P = len(prompt_ids)
        scores = []
        for i, c in enumerate(choice_ids):
            tgt = torch.tensor(c, dtype=torch.long, device=logp.device)
            lp = logp[i, P - 1:P - 1 + len(c)].gather(-1, tgt[:, None]).sum().item()
            scores.append(lp / len(c) if length_norm else lp)
        return ScoreResult(logprobs=scores, prompt_tokens=P, latency_s=time.perf_counter() - t0,
                           choice_tokens=[len(c) for c in choice_ids])

    def embed(self, messages, layer="mid") -> EmbedResult:
        torch = self.torch
        prompt_ids = self.encode(messages)
        ids, mask = self._batch([prompt_ids])
        li = resolve_layer(layer, self.n_layers)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        vec = out.hidden_states[li][0, -1].float().cpu().numpy().astype(np.float32)
        return EmbedResult(vector=vec, prompt_tokens=len(prompt_ids), latency_s=time.perf_counter() - t0, layer=li)

    def close(self) -> None:
        try:
            del self.model
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass
