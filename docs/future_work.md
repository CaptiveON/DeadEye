# Future work: what this study leaves for someone with more hardware

The study is deliberately limited to what one Apple M2 Max laptop with 32 GB of unified memory can run.
Everything below is prepared for in the harness (configs, backends, methods) but not run, and each item
says what it needs. All cost figures are rough and must be checked against current prices.

## Larger rungs of the same ladders

- **Weight-level methods above 8B.** Exact likelihood scoring, hidden-state probes and LoRA on Qwen2.5 14B,
  32B and 72B, Qwen3 14B and 32B, Gemma 3 12B and 27B, Llama 3.1 70B. Needs one 80 GB GPU (A100 or H100) in
  16-bit, or 24 to 48 GB with 4-bit loading through bitsandbytes (`quantization: int4`, CUDA only). The
  base-versus-instruct comparison on Qwen3 above 8B (base checkpoints exist up to 14B).
- **The Llama ladder** (3.2 1B and 3B, 3.1 8B and 70B): gated repositories; included in the catalogue, not run.
- **Control ladders above 3B:** Pythia 6.9B and 12B, OLMo 2 13B and 32B.
- **Prompting-only coverage of the large rungs through hosted inference** (OpenRouter, Together, Fireworks and
  similar) is cheap, on the order of fifty to one hundred and fifty US dollars for generation, free reply and
  first-token scoring across eight models, but the server controls sampling and precision, probes and LoRA do
  not apply, and the rows cannot enter the within-family scale fits. A config for it is a few lines in the
  style of `configs/examples/served_models.yaml`.

## Reasoning budgets

- Thinking and chain-of-thought budgets of 2048 tokens, and the 256-token budget on the 7B and 8B rungs across
  all seven tasks rather than three. On a laptop a 2048-token reply takes minutes per decision.

## Quantisation beyond GGUF

- bitsandbytes NF4 and LLM.int8 at every rung, including the ones where 16-bit no longer fits, so that H5 is
  tested with the same weights under the same backend rather than across runtimes. CUDA only.

## Decision models

- The hosted decision models by API: Cloudflare's Clef and Clef-flash (Workers AI), Perplexity's
  pplx-decider (Decisions API), and TypeSafe's Jev if early access is granted. Their per-token prices make a
  full run of `configs/decision_models.yaml` cost well under a dollar each, but the latency would include a
  network round trip and the hosted models' sampling and precision are not controlled.
- Kev 9B and 27B, Clef 27B and the Perplexity decider run locally, so that their latency is measured on the
  same machine as the conversions.
- Adapters for Laya's package and Perplexity's Decisions API (their request schemas differ from the System One
  protocol the `decision` backend implements).

## Methods

- Reinforcement-learning conversions (DPO on oracle-versus-model action pairs; GRPO with environment reward).
  The `hf` backend exposes everything needed; the training loop is the work.
- An MLX or llama.cpp backend with exact log-likelihood scoring and hidden-state access on Apple silicon,
  which would make the laptop two to three times faster than PyTorch's Metal backend.

## Sample sizes

- The five ablation blocks at 100 episodes instead of 50 (the laptop design can detect differences of about
  0.14 in normalised score there; 100 episodes bring it to about 0.10).

## Cloud routes, for reference

| Option | What you get | Approximate cost |
|---|---|---|
| Kaggle notebooks | two 16 GB T4 GPUs, 30 hours per week | free |
| Google Colab Pro | A100 or L4 time within a monthly quota | about $10 per month |
| Rented RTX 4090 (24 GB) | everything to 14B in 16-bit, 32B in 4-bit | about $0.30 to $0.70 per hour |
| Rented A100 or H100 (80 GB) | the full original GPU sweep (about 150 GPU-hours) | about $1.5 to $3 per hour |
