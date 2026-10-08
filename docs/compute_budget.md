# Compute budget

`deadeye estimate <config> --seconds-per-decision S` counts decisions from oracle rollouts and
multiplies by models x methods. Use it after every config edit. Rules of thumb measured on the
tiny model and typical hardware (prompts are 300-600 tokens, answers 1-16 tokens):

| Setting | prompt_score (one forward, batched choices) | prompt_generate 16 tokens | embed (probe) |
|---|---|---|---|
| 135M-360M, CPU (4 cores) | 0.15-0.4 s | 0.3-0.8 s | 0.1-0.3 s |
| 1.5B, CPU | 1-2 s | 2-4 s | 0.8-1.5 s |
| 1.5B, 24 GB GPU bf16 | 0.03 s | 0.08 s | 0.02 s |
| 7B, 24 GB GPU bf16 | 0.08 s | 0.25 s | 0.06 s |
| 14B, 24 GB GPU bf16 | 0.15 s | 0.5 s | 0.12 s |
| 32B / 72B, 80 GB GPU int4 | 0.4 / 0.9 s | 1.5 / 3 s | 0.3 / 0.7 s |

Chain-of-thought (`max_new_tokens: 256`) is roughly 10x `prompt_generate`; thinking mode with
`max_new_tokens: 2048` can be 50x. Budget those cells explicitly.

LoRA: 100 training episodes yield 1000-5000 examples per environment; two epochs at batch 8 take
2-10 minutes on a GPU for 7B (QLoRA for int4 models), under a minute for sub-1B models.

Sweep (`configs/sweep_gpu.yaml`, 33 model entries, 6 methods): about 150 GPU-hours on one 80 GB
GPU. The control ladders (`configs/sweep_controls.yaml`) add about 60. The five ablation blocks in
`configs/ablations/` are 150-200 cells each; together they are of the same order as the sweep, and
each can be run and reported on its own. Check any edit with `deadeye estimate <config>`. The CPU pilot (`configs/pilot_cpu.yaml`) is 6-12 hours on a
laptop, or 25 minutes for the 135M model alone.

Memory: bf16 needs ~2 bytes/parameter plus KV cache (prompts are short); int4 ~0.6 bytes/parameter.
