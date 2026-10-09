# Pre-registration: DeadEye v1 confirmatory study

Status: **draft** (to be frozen with the `prereg-v1` git tag after the Phase 1 pilot; any later change
is recorded in the "Deviations" section with a date and reason).

## 1. Title

Do small open-weight language models make good decision-makers? A controlled study of scale,
conversion method and task structure.

## 2. Research questions

RQ1. How does decision quality scale with parameter count within a model family, and does the slope
depend on the task's decision structure?
RQ2. Does the conversion method (prompting, likelihood scoring, hidden-state probe, LoRA behaviour
cloning) change the dependence on scale?
RQ3. Which other factors (instruction tuning, quantisation, reasoning budget, observation format,
prompt encoding, adaptation data budget) move decision quality, and by how much?
RQ4. What is the quality / inference-cost frontier?
RQ5 (exploratory, added 8 October 2026, not a pre-registered test). How do small open-weight models converted
here compare with the purpose-built decision models that run on the same laptop (Kev 0.8B and 4B, Laya, SemIf)
in decision quality, calibration, latency and cost on the same tasks and seeds? The hosted decision models
(Clef, Perplexity's decider, Jev) are future work.

## 3. Hypotheses, predictions and falsification criteria

Each prediction names the analysis that tests it. "Score" means normalised return (random = 0,
oracle = 1). CI means a 95% bootstrap confidence interval. Slopes are from `score = a + b log10(params)`
fitted within one family on the sweep results.

| # | Prediction | Supported if | Falsified if |
|---|---|---|---|
| H1a | Under `prompt_generate`, `b > 0` in every ladder (SmolLM2 135M-1.7B, Qwen2.5 0.5B-7B, Qwen3 0.6B-8B, Gemma 3 270M-4B) | CI of `b` excludes 0 and is positive for every ladder with >= 3 rungs | any ladder's CI includes 0 or is negative |
| H1b | Slope is larger for gridworld and tictactoe than for loan | bootstrap CI of (b_planning - b_loan) excludes 0 | CI includes 0 |
| H2a | Instruct > base under `prompt_generate` at every size | paired difference CI > 0 at >= 80% of sizes | fewer than 50% |
| H2b | The instruct - base gap under `probe` and `lora_sft` is < 0.05 | CI of the mean gap lies within [-0.05, 0.05] | CI excludes that interval |
| H3a | The smallest rung of each ladder with `lora_sft` >= the largest rung with `prompt_generate` (ratios 12-15: SmolLM2 135M vs 1.7B, Qwen2.5 0.5B vs 7B, Qwen3 0.6B vs 8B, Gemma 3 270M vs 4B), in distribution | paired difference CI >= 0 on at least 5 of 7 tasks, in at least 3 of 4 ladders | 3 or fewer tasks in 2 or more ladders |
| H3b | On `loan_sign_flip` and `loan_covariate_shift`, the small adapted model's loss relative to in-distribution is larger than the large model's | CI of the interaction excludes 0 | CI includes 0 |
| H4a | Illegal-action rate under `prompt_score` is 0 | by construction | n/a |
| H4b | `b_score` < 0.5 `b_generate` | bootstrap CI of the ratio < 0.5 | CI includes 0.5 or above |
| H5 | Q4_K_M (llama.cpp GGUF) costs < 0.05 at 7B and > 0.10 at 1.5B under generation and first-token scoring; Q8_0 < 0.05 at every size | CIs of the paired 16-bit minus quantised differences meet the thresholds | any fails |
| H6a | No model under prompting reaches UCB1's bandit score | all CIs below the UCB1 score | any CI includes or exceeds it |
| H6b | Summary history > raw history for models < 2B | paired CI > 0 for all such models | any CI includes 0 |
| H7a | A 256-token thinking / CoT budget improves gridworld + tictactoe at >= 4B (Qwen3 4B, 8B; Qwen2.5 7B) | paired CI > 0 | CI includes 0 |
| H7b | A 256-token budget does not improve models < 2B (Qwen3 1.7B; Qwen2.5 1.5B) | paired CI includes 0 or is negative | CI > 0 |
| H7c | Latency-normalised score is lower with the budget at every size | score/latency ratio lower for every size | any size higher |

## 4. Design

- Environments, parameters and seeds exactly as in `configs/mac_main.yaml`, `configs/mac_lora.yaml`,
  `configs/mac_free_reply.yaml`, `configs/mac_controls.yaml`, `configs/decision_models.yaml` and the files in
  `configs/ablations/` at the `prereg-v1` tag. Hardware: one Apple M2 Max (32 GB), PyTorch Metal backend,
  16-bit weights; LoRA on rungs up to 4B with 40 training episodes (3000 states at most).
- Evaluation seeds 0-99 in the main sweep (tic-tac-toe 0-299; blackjack 0-99 with 20 hands per episode) and
  0-49 in the ablations and the free-reply condition (tic-tac-toe 0-149; blackjack 0-49); training seeds
  100000-100099; the pilot used seeds 5000-5019 and is not part of the confirmatory data. Prompts frozen
  at the tag.
- Models: the families and sizes listed in the two configs; a model that cannot be loaded is
  dropped and listed in the appendix.
- Primary outcome: normalised return per episode. Secondary: illegal-action rate, oracle agreement,
  task metrics, latency per decision (mean, median, 95th percentile) and tokens per decision, and for every
  policy that assigns probabilities (scoring, probes, decision models) the expected calibration error and
  Brier score of the probability assigned to the chosen action against the oracle.

## 5. Sample size

For a paired comparison with smallest effect of interest 0.10, alpha = 0.05 (two-sided) and power 0.8,
the required number of paired episodes is n = ((1.96 + 0.84) * sd_diff / 0.10)^2, where sd_diff is the
standard deviation of the per-episode paired differences in normalised score. One hundred episodes
therefore suffice when sd_diff <= 0.36. From the pilot (seeds 5000-5019), sd_diff per task (to be filled
in at the tag): bandit __, contextual bandit __, loan __, gridworld __, tictactoe __, blackjack __.
Tic-tac-toe uses 300 episodes and blackjack 100 episodes of 20 hands because single-game outcomes are
-1/0/+1 and their normalised SD is near 1. The ablation blocks and the free-reply condition use 50 episodes,
which detect a difference of about 0.14 at sd_diff = 0.36 (0.20 at 0.5); that smallest detectable effect is
reported next to every ablation result. For any task whose measured sd_diff implies a detectable effect above
the stated one, we report it alongside the estimate and do not interpret a non-significant result as evidence
of no effect.

## 6. Analysis plan

Per cell: mean, IQM, 95% bootstrap CI (2000 resamples). Paired comparisons: permutation test on
sign-flipped paired differences (5000 permutations). Holm correction is applied within each
hypothesis's family of tests, which is the set of `--pair` arguments of one `deadeye compare` call:
H2a is one family per model family, H3a one family per task set, H5 one family per method, H6b and
H7a-b one family each. Scale slopes (H1, H4b): ordinary least squares of cell-mean normalised score on
log10(parameters) within a family, method and task, with 95% CIs from 1000 bootstrap resamples of
episodes within cells (`slopes.csv` from `deadeye report`; slope ratios from `slope_ratio`). All computed
by `deadeye report` and `deadeye compare`; the exact code at the `prereg-v1` tag is the analysis.

## 7. Exclusion and stopping rules

No episode is excluded for its outcome. A cell is excluded only on load failure or crash (recorded
in `run_manifest.json`). No interim analysis changes the design; the sweep runs to completion.

## 8. Deviations (append-only)

| Date | What changed | Why |
|---|---|---|
| 9 October 2026 | Scope fixed to one Apple M2 Max laptop: ladders to 8B (LoRA to 4B), GGUF quantisation through llama.cpp, 256-token reasoning budgets, local decision models only, ablations at 50 episodes; everything else moved to future work | the study has no GPU budget; the design was changed before any confirmatory run and before the prereg-v1 tag |
