"""Minimal LoRA supervised fine-tuning (behaviour cloning) on top of :class:`HFModel`.

Examples are ``(messages, target_text)`` pairs; the loss is computed on the target tokens only.
Written in plain PyTorch + peft so the dependency surface stays small and every step is inspectable.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from deadeye.models.base import Message
from deadeye.models.hf_backend import HFModel

log = logging.getLogger(__name__)


def train_lora_sft(hf: HFModel, examples: list[tuple[list[Message], str]], r: int = 8, alpha: int = 16,
                   dropout: float = 0.05, lr: float = 2e-4, epochs: int = 2, batch_size: int = 8,
                   target_modules: Any = "all-linear", seed: int = 0, merge: bool = True,
                   grad_accum: int = 1, max_grad_norm: float = 1.0) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    model = hf.model
    if hf.quantized:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Tokenise once.
    pad = hf.tokenizer.pad_token_id
    eos = hf.tokenizer.eos_token_id
    data = []
    for messages, target in examples:
        p = hf.encode(messages)
        t = hf.tokenizer(target, add_special_tokens=False)["input_ids"] + ([eos] if eos is not None else [])
        data.append((p + t, [-100] * len(p) + t))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    dev = model.device if hasattr(model, "device") else hf.device
    model.train()
    losses: list[float] = []
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(seed)
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(len(data), generator=g).tolist()
        for b in range(0, len(order), batch_size):
            batch = [data[i] for i in order[b:b + batch_size]]
            L = max(len(x) for x, _ in batch)
            ids = torch.full((len(batch), L), pad, dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, (x, y) in enumerate(batch):
                ids[i, :len(x)] = torch.tensor(x)
                lab[i, :len(y)] = torch.tensor(y)
                mask[i, :len(x)] = 1
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=lab.to(dev))
            loss = out.loss / grad_accum
            loss.backward()
            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss.item()))
        log.info("lora-sft epoch %d/%d mean loss %.4f", epoch + 1, epochs, sum(losses[-max(1, len(order) // batch_size):]) / max(1, len(order) // batch_size))
    model.eval()
    if merge and not hf.quantized:
        model = model.merge_and_unload()
    hf.model = model
    return {"n_examples": len(data), "trainable_params": int(trainable), "steps": step, "loss_first": losses[0] if losses else None,
            "loss_last": sum(losses[-5:]) / max(1, len(losses[-5:])) if losses else None,
            "train_time_s": time.perf_counter() - t0, "epochs": epochs, "lora_r": r}
