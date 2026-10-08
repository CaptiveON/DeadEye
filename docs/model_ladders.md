# Open-weight model ladders for DeadEye

*Compiled 2026-10-08. Companion to `docs/literature_review.md` and `paper/refs.bib`.*

## How to read these tables

**Repository ids** are marked in one of two ways:

- **(verified)**: the exact Hugging Face id was seen in a search result URL or listing, or the exact model name was seen in an official page or README (GitHub release notes, official model tables).
- **(unverified)**: the id follows the family's official naming pattern but was not seen during this check. Confirm it on huggingface.co before use.

Direct fetches to huggingface.co were blocked from this environment, so official model cards could not be opened. Mirrors, official GitHub READMEs, vendor docs and search snippets were used instead.

**Parameter counts** are as published.

- MoE models are written as `total / active`.
- "Non-embedding" is given only where a published value was found. `n/p` means not published or not found. `(unverified)` marks a figure recalled from a model card but not re-confirmed.
- For small models the embedding share is large (e.g., ~170M of Gemma 3 270M; ~0.16B of Qwen3-0.6B's 0.6B; ~0.13B of Qwen2.5-0.5B's 0.49B), so **use non-embedding parameters as the H1 x-axis** where available, following Kaplan et al.

**Thinking mode** records whether a model can emit an explicit reasoning trace before answering, and how that is controlled.

**Licence notes.** "Apache-2.0" and "MIT" are permissive. The Llama community licences and the Gemma Terms of Use are custom licences with use restrictions and gated downloads. The Qwen Research License (Qwen2.5-3B) restricts commercial use.

---

## Qwen2.5 (Alibaba Qwen) [qwen2024qwen25]

Released 19 Sep 2024. Base and Instruct at every size, same recipe (18T pre-training tokens; SFT + DPO + GRPO post-training).

| Variant | HF repo (base) | HF repo (instruct) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 0.5B | `Qwen/Qwen2.5-0.5B` (verified) | `Qwen/Qwen2.5-0.5B-Instruct` (verified) | 0.49B | 0.36B | 32,768 (gen 8,192) | Apache-2.0 | 09/2024 | no | 24 layers |
| 1.5B | `Qwen/Qwen2.5-1.5B` (unverified) | `Qwen/Qwen2.5-1.5B-Instruct` (verified) | 1.54B (unverified) | 1.31B (unverified) | 32,768 (unverified) | Apache-2.0 | 09/2024 | no | — |
| 3B | `Qwen/Qwen2.5-3B` (verified) | `Qwen/Qwen2.5-3B-Instruct` (verified) | 3.09B | 2.77B | 32,768 (gen 8,192) | **Qwen Research License** | 09/2024 | no | 36 layers; licence restricts commercial use |
| 7B | `Qwen/Qwen2.5-7B` (unverified) | `Qwen/Qwen2.5-7B-Instruct` (verified) | 7.61B | 6.53B | 131,072 with YaRN (32,768 default config; gen 8,192) | Apache-2.0 | 09/2024 | no | 28 layers |
| 14B | `Qwen/Qwen2.5-14B` (verified: named as R1-Distill base) | `Qwen/Qwen2.5-14B-Instruct` (unverified) | 14.7B (unverified) | 13.1B (unverified) | 131,072 with YaRN (unverified) | Apache-2.0 | 09/2024 | no | base of DeepSeek-R1-Distill-Qwen-14B |
| 32B | `Qwen/Qwen2.5-32B` (verified: named as R1-Distill base) | `Qwen/Qwen2.5-32B-Instruct` (unverified) | 32.5B (unverified) | 31.0B (unverified) | 131,072 with YaRN (unverified) | Apache-2.0 | 09/2024 | no | base of DeepSeek-R1-Distill-Qwen-32B |
| 72B | `Qwen/Qwen2.5-72B` (verified) | `Qwen/Qwen2.5-72B-Instruct` (verified) | 72.7B | 70.0B | 131,072 with YaRN (GGUF card: 32,768) | **Qwen License** (custom; scale clause) | 09/2024 | no | 80 layers |

## Qwen3 [yang2025qwen3; qwen2025qwen3github]

Released 29 Apr 2025, Apache-2.0. Hybrid thinking is **on by default**. It is disabled with `enable_thinking=False` in the chat template or with `/no_think` in a message (`/think` re-enables it), and there is a user-set thinking budget.

**Base checkpoints were not released for 32B or 235B-A22B.**

| Variant | HF repo (base) | HF repo (post-trained, hybrid) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 0.6B | `Qwen/Qwen3-0.6B-Base` (unverified) | `Qwen/Qwen3-0.6B` (unverified) | 0.6B | 0.44B | 32,768 | Apache-2.0 | 04/2025 | yes, toggle in same weights | 28 layers |
| 1.7B | `Qwen/Qwen3-1.7B-Base` (unverified) | `Qwen/Qwen3-1.7B` (unverified) | 1.7B | 1.4B (unverified) | 32,768 | Apache-2.0 | 04/2025 | yes, toggle | — |
| 4B | `Qwen/Qwen3-4B-Base` (unverified) | `Qwen/Qwen3-4B` (unverified) | 4.0B | 3.6B | 32,768 native (131,072 with YaRN per model card; unverified) | Apache-2.0 | 04/2025 | yes, toggle | 36 layers |
| 8B | `Qwen/Qwen3-8B-Base` (verified) | `Qwen/Qwen3-8B` (verified) | 8.2B | 6.95B | 32,768 native; 131,072 with YaRN | Apache-2.0 | 04/2025 | yes, toggle | 36 layers |
| 14B | `Qwen/Qwen3-14B-Base` (verified) | `Qwen/Qwen3-14B` (unverified) | 14.8B | 13.2B | 32,768 native; 131,072 with YaRN | Apache-2.0 | 04/2025 | yes, toggle | 40 layers; **largest dense base** |
| 32B | **none released** | `Qwen/Qwen3-32B` (verified) | 32.8B | 31.2B | 32,768 native; 131,072 with YaRN | Apache-2.0 | 04/2025 | yes, toggle | 64 layers |
| 30B-A3B (MoE) | `Qwen/Qwen3-30B-A3B-Base` (unverified) | `Qwen/Qwen3-30B-A3B` (unverified) | 30.5B / 3.3B | 29.9B | 32,768 native; 131,072 with YaRN | Apache-2.0 | 04/2025 | yes, toggle | 48 layers, 128 experts (8 active) |
| 235B-A22B (MoE) | **none released** | `Qwen/Qwen3-235B-A22B` (unverified) | 235B / 22B | 234B (unverified) | 32,768 native; 131,072 with YaRN | Apache-2.0 | 04/2025 | yes, toggle | — |

**Qwen3-2507 refresh** (thinking and non-thinking split into separate checkpoints; 256K context, extendable to 1M from 8 Aug 2025):

| Variant | HF repo (base) | HF repo (instruct / thinking) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 4B-2507 | — | `Qwen/Qwen3-4B-Instruct-2507` (verified); `Qwen/Qwen3-4B-Thinking-2507` (verified) | 4.0B | 3.6B | 262,144 | Apache-2.0 | 08/2025 | Instruct: never; Thinking: always | same-size thinking vs non-thinking pair |
| 30B-A3B-2507 | — | `Qwen/Qwen3-30B-A3B-Instruct-2507` (verified); `Qwen/Qwen3-30B-A3B-Thinking-2507` (verified) | 30.5B / 3.3B | 29.9B | 262,144 | Apache-2.0 | 07/2025 | Instruct: never; Thinking: always | — |
| 235B-A22B-2507 | — | `Qwen/Qwen3-235B-A22B-Instruct-2507` (verified); `Qwen/Qwen3-235B-A22B-Thinking-2507` (verified) | 235B / 22B | n/p | 262,144 | Apache-2.0 | 07/2025 | Instruct: never; Thinking: always | — |

## Newer Qwen ladders [qwen2026github]

### Qwen3-Next

| Variant | HF repo (base) | HF repo (instruct / thinking) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 80B-A3B | not released (community reports) | `Qwen/Qwen3-Next-80B-A3B-Instruct` (verified); `Qwen/Qwen3-Next-80B-A3B-Thinking` (unverified) | 80B / 3B | n/p | 262,144 (unverified) | Apache-2.0 | 09/2025 | Instruct: never; Thinking: always | Gated DeltaNet + gated attention (3:1); 512 experts (10 routed + 1 shared) |

### Qwen3.5

All sizes are vision-language (vision encoder). Context is 262,144 native, extendable to ~1.01M. The licence is reported as Apache-2.0 by third-party listings and the GitHub repository; it was not confirmed on the weight cards.

| Variant | HF repo (base) | HF repo (post-trained) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 0.8B | `Qwen/Qwen3.5-0.8B-Base` (unverified) | `Qwen/Qwen3.5-0.8B` (verified) | 0.8B (HF lists 0.9B) | n/p | 262,144 | Apache-2.0 (unverified) | 03/2026 | yes (`reasoning_effort`, `preserve_thinking`) | — |
| 2B | `Qwen/Qwen3.5-2B-Base` (unverified) | `Qwen/Qwen3.5-2B` (verified) | 2B | n/p | 262,144 | Apache-2.0 (unverified) | 03/2026 | yes | — |
| 4B | `Qwen/Qwen3.5-4B-Base` (unverified) | `Qwen/Qwen3.5-4B` (verified) | 4B | n/p | 262,144 | Apache-2.0 (unverified) | 03/2026 | yes | — |
| 9B | `Qwen/Qwen3.5-9B-Base` (verified) | `Qwen/Qwen3.5-9B` (verified) | 9B | n/p | 262,144 | Apache-2.0 (unverified) | 03/2026 | yes | — |
| 27B | (unverified) | `Qwen/Qwen3.5-27B` (verified) | 27B dense | n/p | 262,144 | Apache-2.0 (unverified) | 02/2026 | yes | — |
| 35B-A3B | (unverified) | `Qwen/Qwen3.5-35B-A3B` (verified) | 35B / 3B | n/p | 262,144 | Apache-2.0 (unverified) | 02/2026 | yes | MoE |
| 122B-A10B | (unverified) | `Qwen/Qwen3.5-122B-A10B` (unverified) | 122B / 10B | n/p | 262,144 | Apache-2.0 (unverified) | 02/2026 | yes | MoE |
| 397B-A17B | (unverified) | `Qwen/Qwen3.5-397B-A17B` (verified) | 397B / 17B | n/p | 262,144 | Apache-2.0 (unverified) | 02/2026 | yes | MoE; first Qwen3.5 release (16 Feb 2026) |

### Qwen3.6 and Qwen3.8

| Variant | HF repo (base) | HF repo (post-trained) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-27B | n/p | `Qwen/Qwen3.6-27B` (verified) | 27B dense (HF lists 28B) | n/p | 262,144 | Apache-2.0 | 04/2026 | yes (unverified details) | FP8 variant available |
| Qwen3.6-35B-A3B | n/p | `Qwen/Qwen3.6-35B-A3B` (verified) | 35B / 3B | n/p | 262,144 (unverified) | Apache-2.0 | 04/2026 | yes (unverified details) | 256 experts (8 routed + 1 shared) |
| Qwen3.8-27B | n/p | `Qwen/Qwen3.8-27B` (unverified id; name verified in official README) | 27B | n/p | 262,144 in serving examples | weights licence per HF card (unverified) | 08/2026 | yes (`reasoning_effort`) | — |
| Qwen3.8-2.4T-A95B | n/p | `Qwen/Qwen3.8-2.4T-A95B` (unverified id; name verified in official README) | 2.4T / 95B | n/p | 262,144 in serving examples | (unverified) | 08/2026 | yes | out of scope for this study |

## SmolLM2 and SmolLM3 (Hugging Face) [allal2025smollm2; hf2025smollm3]

| Variant | HF repo (base) | HF repo (instruct) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` (unverified) | `HuggingFaceTB/SmolLM2-135M-Instruct` (verified) | 135M | n/p | 8,192 (unverified) | Apache-2.0 | 11/2024 (unverified; paper 02/2025) | no | — |
| SmolLM2-360M | `HuggingFaceTB/SmolLM2-360M` (unverified) | `HuggingFaceTB/SmolLM2-360M-Instruct` (verified) | 360M | n/p | 8,192 (unverified) | Apache-2.0 | 11/2024 (unverified) | no | — |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B` (verified) | `HuggingFaceTB/SmolLM2-1.7B-Instruct` (unverified) | 1.7B | n/p | 8,192 (unverified) | Apache-2.0 | 11/2024 (unverified) | no | ~11T training tokens |
| SmolLM3-3B | `HuggingFaceTB/SmolLM3-3B-Base` (verified) | `HuggingFaceTB/SmolLM3-3B` (verified) | 3B | n/p | 64K trained; 128K with YaRN | Apache-2.0 (unverified) | 07/2025 | yes: dual-mode think / no_think in one model | 11.2T tokens; six languages; NoPE + YaRN |

## Gemma 3, Gemma 3n and Gemma 4 (Google) [gemma2025gemma3; google2025gemma3n; google2026gemma4]

### Gemma 3

Gemma 3 parameter splits come from the technical report's Table 1 (vision encoder + embedding + non-embedding). The 270M figures come from third-party reports. Licence is the Gemma Terms of Use (gated).

| Variant | HF repo (base, "pt") | HF repo (instruct, "it") | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 270M | `google/gemma-3-270m` (unverified) | `google/gemma-3-270m-it` (unverified) | ~270M | ~100M (~170M are embeddings; third-party) | 32K (third-party) | Gemma Terms of Use | 08/2025 | no | text only; 256k vocab |
| 1B | `google/gemma-3-1b-pt` (verified) | `google/gemma-3-1b-it` (verified) | ~1.0B (302M emb + 698M) | 698M | 32K | Gemma Terms of Use | 03/2025 | no | text only |
| 4B | `google/gemma-3-4b-pt` (verified) | `google/gemma-3-4b-it` (verified) | ~4.3B (417M vision + 675M emb + 3,209M) | 3,209M | 128K | Gemma Terms of Use | 03/2025 | no | image + text |
| 12B | `google/gemma-3-12b-pt` (unverified) | `google/gemma-3-12b-it` (unverified) | ~12.2B (417M + 1,012M + 10,759M) | 10,759M | 128K | Gemma Terms of Use | 03/2025 | no | image + text |
| 27B | `google/gemma-3-27b-pt` (unverified) | `google/gemma-3-27b-it` (verified) | ~27.4B (417M + 1,416M + 25,600M) | 25,600M | 128K | Gemma Terms of Use | 03/2025 | no | image + text |

### Gemma 3n

| Variant | HF repo (base) | HF repo (instruct) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| E2B | `google/gemma-3n-E2B` (verified) | `google/gemma-3n-E2B-it` (verified) | raw >5B; effective ~2B (~1.91B loaded with PLE caching) | n/p | 32K | Gemma Terms of Use | 2025 (preview 05/2025; exact GA date unverified) | no | text + image + audio in; MatFormer, per-layer embeddings |
| E4B | `google/gemma-3n-E4B` (unverified) | `google/gemma-3n-E4B-it` (verified) | raw ~8B; effective ~4B | n/p | 32K | Gemma Terms of Use | 2025 | no | as above |

### Gemma 4

| Variant | HF repo (base) | HF repo (instruct) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| E2B | `google/gemma-4-E2B` (unverified) | `google/gemma-4-E2B-it` (unverified) | effective ~2B | n/p | 128K | Apache-2.0 (news coverage) | 04/2026 | yes, configurable (`enable_thinking`) | audio on small models |
| E4B | `google/gemma-4-E4B` (unverified) | `google/gemma-4-E4B-it` (unverified; `GEMMA4_E4B_IT` checkpoint in official repo) | effective ~4B | n/p | 128K | Apache-2.0 | 04/2026 | yes, configurable | — |
| 26B-A4B | `google/gemma-4-26B-A4B` (unverified) | `google/gemma-4-26B-A4B-it` (verified) | 25.2B / 3.8B | n/p | 256K | Apache-2.0 | 04/2026 | yes, configurable | MoE |
| 31B | `google/gemma-4-31B` (verified) | `google/gemma-4-31B-it` (verified) | 31B dense | n/p | 256K | Apache-2.0 | 04/2026 | yes, configurable | — |
| 12B | (unverified) | (unverified) | 12B (reported in model card by one secondary source) | n/p | (unverified) | Apache-2.0 | 2026 (unverified) | (unverified) | existence unconfirmed |

## Llama 3.x and Llama 4 (Meta) [grattafiori2024llama3; meta2025llamamodels]

All Llama releases are gated under custom community licences. **The 3.x sizes come from different releases (3.1, 3.2, 3.3) and are not a single-recipe ladder.**

| Variant | HF repo (base) | HF repo (instruct) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B` (verified) | `meta-llama/Llama-3.2-1B-Instruct` (verified) | ~1.2B (unverified exact) | n/p | 128K | Llama 3.2 Community License | 09/2024 | no | text only |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B` (unverified) | `meta-llama/Llama-3.2-3B-Instruct` (verified) | ~3.2B (unverified exact) | n/p | 128K | Llama 3.2 Community License | 09/2024 | no | — |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` (verified: named as R1-Distill base) | `meta-llama/Llama-3.1-8B-Instruct` (unverified) | 8B | n/p | 128K | Llama 3.1 Community License | 07/2024 | no | — |
| Llama 3.1 70B | `meta-llama/Llama-3.1-70B` (unverified) | `meta-llama/Llama-3.1-70B-Instruct` (unverified) | 70B | n/p | 128K | Llama 3.1 Community License | 07/2024 | no | — |
| Llama 3.1 405B | `meta-llama/Llama-3.1-405B` (unverified) | `meta-llama/Llama-3.1-405B-Instruct` (unverified) | 405B | n/p | 128K | Llama 3.1 Community License | 07/2024 | no | — |
| Llama 3.3 70B | — (instruct only) | `meta-llama/Llama-3.3-70B-Instruct` (verified: named as R1-Distill base) | 70B | n/p | 128K | Llama 3.3 Community License | 12/2024 | no | — |
| Llama 4 Scout | `meta-llama/Llama-4-Scout-17B-16E` (verified) | `meta-llama/Llama-4-Scout-17B-16E-Instruct` (verified) | ~109B / 17B (16 experts) | n/p | 10M (256K in training) | Llama 4 Community License | 04/2025 | no | image + text in; MoE |
| Llama 4 Maverick | `meta-llama/Llama-4-Maverick-17B-128E` (unverified) | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` (verified) | ~400B / 17B (128 experts) | n/p | 1M | Llama 4 Community License | 04/2025 | no | 48 layers; MoE |

## Pythia (EleutherAI): base only [biderman2023pythia]

16 models: 8 sizes, each also trained on deduplicated data, all trained on the Pile in the same order with 154 checkpoints each. Apache-2.0. Context (sequence length) 2048. Release 2023 (paper April 2023).

Non-embedding counts for 70M-2.8B are from the paper's Table 1 as reproduced in secondary sources. The 6.9B and 12B values were not re-confirmed.

| Variant | HF repo (base) | HF repo (instruct) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 70M | `EleutherAI/pythia-70m`, `-70m-deduped` (verified) | n/a | 70M | 18,915,328 | 2048 | Apache-2.0 | 2023 | no | 6 layers, d=512 |
| 160M | `EleutherAI/pythia-160m`, `-160m-deduped` (verified) | n/a | 160M | 85,056,000 | 2048 | Apache-2.0 | 2023 | no | 12 layers, d=768 |
| 410M | `EleutherAI/pythia-410m`, `-410m-deduped` (verified) | n/a | 410M | 302,311,424 | 2048 | Apache-2.0 | 2023 | no | 24 layers, d=1024 |
| 1B | `EleutherAI/pythia-1b`, `-1b-deduped` (verified) | n/a | 1B | 805,736,448 | 2048 | Apache-2.0 | 2023 | no | 16 layers, d=2048 |
| 1.4B | `EleutherAI/pythia-1.4b`, `-1.4b-deduped` (verified) | n/a | 1.4B | 1,208,602,624 | 2048 | Apache-2.0 | 2023 | no | 24 layers, d=2048 |
| 2.8B | `EleutherAI/pythia-2.8b`, `-2.8b-deduped` (verified) | n/a | 2.8B | 2,517,652,480 | 2048 | Apache-2.0 | 2023 | no | 32 layers, d=2560 |
| 6.9B | `EleutherAI/pythia-6.9b`, `-6.9b-deduped` (verified) | n/a | 6.9B | 6,444,163,072 (unverified) | 2048 | Apache-2.0 | 2023 | no | 32 layers, d=4096 |
| 12B | `EleutherAI/pythia-12b`, `-12b-deduped` (verified) | n/a | 12B | 11,327,027,200 (unverified) | 2048 | Apache-2.0 | 2023 | no | 36 layers, d=5120 |

Extra rungs: `EleutherAI/pythia-14m` and `EleutherAI/pythia-31m` (both verified, no deduplicated variants).

## OLMo 2 and Olmo 3 (Ai2): fully open data, code and checkpoints [olmo2025olmo2; olmo2025olmo3]

| Variant | HF repo (base) | HF repo (instruct / think) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| OLMo 2 1B | `allenai/OLMo-2-0425-1B` (verified) | `allenai/OLMo-2-0425-1B-Instruct` (verified) | ~1B | n/p | 4,096 (unverified) | Apache-2.0 (repository licence; weight licence unverified) | 04/2025 | no | 4T stage-1 tokens |
| OLMo 2 7B | `allenai/OLMo-2-1124-7B` (verified) | `allenai/OLMo-2-1124-7B-Instruct` (verified) | 7B | n/p | 4,096 (unverified) | Apache-2.0 | 11/2024 | no | 4T stage-1 tokens |
| OLMo 2 13B | `allenai/OLMo-2-1124-13B` (verified) | `allenai/OLMo-2-1124-13B-Instruct` (verified) | 13B | n/p | 4,096 (unverified) | Apache-2.0 | 11/2024 | no | 5T stage-1 tokens |
| OLMo 2 32B | `allenai/OLMo-2-0325-32B` (verified) | `allenai/OLMo-2-0325-32B-Instruct` (verified) | 32B | n/p | 4,096 (unverified) | Apache-2.0 | 03/2025 | no | — |
| Olmo 3 7B | `allenai/Olmo-3-1025-7B` (verified) | `allenai/Olmo-3-7B-Instruct` (verified); `allenai/Olmo-3-7B-Think` (verified) | 7B | n/p | 65,536 | Apache-2.0 | 11/2025 | Think variant (separate checkpoint) | ~5.93T tokens; RL-Zero variants also released (ids unverified) |
| Olmo 3 32B | `allenai/Olmo-3-1125-32B` (verified) | `allenai/Olmo-3-32B-Think` (unverified; `allenai/Olmo-3-32B-Think-DPO` verified) | 32B | n/p | 65,536 | Apache-2.0 | 11/2025 | Think variant | ~5.50T tokens |
| Olmo 3.1 32B | (as above) | `allenai/Olmo-3.1-32B-Think` (verified); `allenai/Olmo-3.1-32B-Instruct` (unverified) | 32B | n/p | 65,536 | Apache-2.0 | 12/2025 | Think variant | extended RL schedule |

## gpt-oss (OpenAI) [openai2025gptoss]

| Variant | HF repo (base) | HF repo (post-trained) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | n/a (no base released) | `openai/gpt-oss-20b` (verified) | 20.9B / 3.6B (README: 21B) | n/p | 128K (unverified) | Apache-2.0 | 08/2025 | yes: reasoning effort low / medium / high | 24 layers; MoE weights in MXFP4 (~4.25 bits); harmony chat format required |
| gpt-oss-120b | n/a | `openai/gpt-oss-120b` (verified) | 116.8B / 5.1B (README: 117B) | n/p | 128K (unverified) | Apache-2.0 | 08/2025 | yes: low / medium / high | 36 layers; MXFP4 |

## Phi-4 family (Microsoft) [abdin2024phi4; abdin2025phi4reasoning]

No base checkpoints were found for this family.

| Variant | HF repo (base) | HF repo (instruct / reasoning) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Phi-4-mini | n/a | `microsoft/Phi-4-mini-instruct` (verified) | 3.8B | n/p | 128K | MIT | 02/2025 (unverified) | no | 200K vocab; GQA; shared embedding |
| Phi-4-mini-reasoning | n/a | `microsoft/Phi-4-mini-reasoning` (unverified) | 3.8B | n/p | 128K | MIT | 04/2025 (unverified) | always reasons | maths-focused |
| Phi-4 | n/a | `microsoft/phi-4` (unverified) | 14.66B | n/p | 16K | MIT | 12/2024 report; weights 01/2025 | no | synthetic-data recipe |
| Phi-4-reasoning | n/a | `microsoft/Phi-4-reasoning` (unverified) | 14B | n/p | 32K | MIT | 04/2025 | always reasons | SFT on o3-mini traces |
| Phi-4-reasoning-plus | n/a | `microsoft/Phi-4-reasoning-plus` (unverified) | 14B | n/p | 32K (one listing says 128K) | MIT | 04/2025 | always reasons | + outcome-based RL |

## Mistral small models [mistral2025small3; mistral2025small31; mistral2025mistral3]

All Apache-2.0. Ministral 3 sizes pair a dense language model with a ~0.4B vision encoder.

| Variant | HF repo (base) | HF repo (instruct / reasoning) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Ministral 3 3B | `mistralai/Ministral-3-3B-Base-2512` (unverified; seen in mirrors) | `mistralai/Ministral-3-3B-Instruct-2512` (verified); `mistralai/Ministral-3-3B-Reasoning-2512` (unverified) | 3.4B LM + 0.4B vision | n/p | 256K | Apache-2.0 | 12/2025 | Reasoning variant (separate checkpoint) | — |
| Ministral 3 8B | `mistralai/Ministral-3-8B-Base-2512` (verified) | `mistralai/Ministral-3-8B-Instruct-2512` (unverified); `mistralai/Ministral-3-8B-Reasoning-2512` (verified) | 8.4B LM + 0.4B vision | n/p | 256K | Apache-2.0 | 12/2025 | Reasoning variant | — |
| Ministral 3 14B | `mistralai/Ministral-3-14B-Base-2512` (unverified) | `mistralai/Ministral-3-14B-Instruct-2512` (verified); `mistralai/Ministral-3-14B-Reasoning-2512` (verified) | 13.5B LM + 0.4B vision (HF card: 13.9B) | n/p | 256K | Apache-2.0 | 12/2025 | Reasoning variant | — |
| Mistral Small 3 (24B) | `mistralai/Mistral-Small-24B-Base-2501` (verified) | `mistralai/Mistral-Small-24B-Instruct-2501` (verified) | 24B | n/p | 32K | Apache-2.0 | 01/2025 | no | Tekken tokenizer (131k vocab) |
| Mistral Small 3.1 (24B) | `mistralai/Mistral-Small-3.1-24B-Base-2503` (verified) | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (verified) | 24B | n/p | 128K | Apache-2.0 | 03/2025 | no | adds vision |
| Mistral Small 3.2 (24B) | — | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (verified) | 24B | n/p | 128K (one host caps at 32K) | Apache-2.0 | 06/2025 | no | minor update of 3.1 |

## DeepSeek-R1 distilled models [deepseekai2025r1]

Released January 2025. All distills always reason; DeepSeek recommends starting output with `<think>\n`, temperature 0.6 and no system prompt. Weights are MIT-licensed, combined with the licence of each base model. Context length is not stated in the distill table; DeepSeek's evaluations used up to 32,768 generated tokens.

| Variant | HF repo (base it was distilled into) | HF repo (distilled reasoning model) | Total params | Non-embedding | Context | License | Release | Thinking mode | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-1.5B | `Qwen/Qwen2.5-Math-1.5B` (verified name) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (verified) | 1.5B | n/p | not stated (unverified) | MIT + Apache-2.0 (base) | 01/2025 | always | maths-specialised base |
| Qwen-7B | `Qwen/Qwen2.5-Math-7B` (verified name) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (verified) | 7B | n/p | not stated | MIT + Apache-2.0 | 01/2025 | always | maths-specialised base |
| Llama-8B | `meta-llama/Llama-3.1-8B` (verified name) | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (verified) | 8B | n/p | not stated | MIT + Llama 3.1 licence | 01/2025 | always | — |
| Qwen-14B | `Qwen/Qwen2.5-14B` (verified name) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (verified) | 14B | n/p | not stated | MIT + Apache-2.0 | 01/2025 | always | general base |
| Qwen-32B | `Qwen/Qwen2.5-32B` (verified name) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (verified) | 32B | n/p | not stated | MIT + Apache-2.0 | 01/2025 | always | general base |
| Llama-70B | `meta-llama/Llama-3.3-70B-Instruct` (verified name) | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (verified) | 70B | n/p | not stated | MIT + Llama 3.3 licence | 01/2025 | always | instruct base |

The distill ladder mixes **two families and three base types** (maths-specialised, general base, instruct). It is useful as matched pairs (distill vs its own base) for H7, not as an H1 ladder.

---

## Recommended ladders for a controlled scale study

**Primary ladder: Qwen2.5 (0.5B-72B).** It is the cleanest within-family comparison available.

- Seven dense rungs spanning more than two orders of magnitude, from one recipe and one release.
- Base and instruct checkpoints at every size, so H2 can be tested on every rung.
- Published non-embedding counts.
- It is the family most used in recent LLM-agent RL work (RAGEN, MS-GRPO, the R1 distills), which eases comparison.
- Caveats: exclude or flag the 3B (Qwen Research License) and 72B (Qwen License) rungs if permissive licensing matters, and note that the 0.5B-3B rungs have shorter (32K) context.

**H7 ladder: Qwen3 dense (0.6B-32B).** Thinking can be switched on, off or budgeted *inside the same weights*, which gives a within-checkpoint causal test of H7 across six sizes under Apache-2.0.

- Its base checkpoints stop at 14B, so H2 on Qwen3 is limited to 0.6B-14B.
- The Qwen3-4B Instruct-2507 / Thinking-2507 pair adds a between-checkpoint contrast at fixed size.

**Fully open replication ladder: OLMo 2 (1B / 7B / 13B / 32B), plus Olmo 3 (7B, 32B).** These provide base and instruct with open data (allowing contamination checks) and Apache-2.0, and Olmo 3 adds Think vs Instruct from the same base. They have fewer rungs, and the 1B was released later than the others.

**Pure-scale control: Pythia (70M-12B).** Identical data and order across eight sizes with intermediate checkpoints make it the gold-standard scale control for H1, probes and LoRA (and for separating parameters from training tokens). It is base-only and weak in absolute terms, so it cannot address H2 or H7.

**Sub-1B regime for H5 and the low end of H1.** SmolLM2 (135M / 360M / 1.7B, Apache-2.0, heavily over-trained) together with Gemma 3 270M / 1B are suitable here. Gemma's Terms of Use and very large embedding share argue for using non-embedding parameters on the x-axis.

**Small-scale H2 + H7 triplets: Ministral 3 (3B / 8B / 14B).** Each rung has base, instruct and reasoning variants under Apache-2.0, but there are only three rungs and each includes a vision encoder.

**Not recommended as primary H1 ladders.**

- **Llama 3.x:** different releases and recipes per size, gated licence, and 3.3 is instruct-only.
- **Phi-4:** no base checkpoints, non-matched sizes.
- **gpt-oss and Gemma 4 / Gemma 3n:** MoE or "effective-parameter" rungs that have no single parameter count.
- **DeepSeek-R1 distills:** mixed bases.
- **Qwen3.5 / 3.6 / 3.8:** promising, but newer, multimodal throughout, and less studied; use as a late replication.

**For all MoE models,** report active and total parameters and keep them out of the dense log-linear fit, or fit them separately.
