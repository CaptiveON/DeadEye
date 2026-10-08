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

## 3. Hypotheses, predictions and falsification criteria

Each prediction names the analysis that tests it. "Score" means normalised return (random = 0,
oracle = 1). CI means a 95% bootstrap confidence interval. Slopes are from `score = a + b log10(params)`
fitted within one family on the sweep results.

| # | Prediction | Supported if | Falsified if |
|---|---|---|---|
| H1a | Under `prompt_generate`, `b > 0` in every family | CI of `b` excludes 0 and is positive for every family with >= 4 sizes | any family's CI includes 0 or is negative |
| H1b | Slope is larger for gridworld and tictactoe than for loan | bootstrap CI of (b_planning - b_loan) excludes 0 | CI includes 0 |
| H2a | Instruct > base under `prompt_generate` at every size | paired difference CI > 0 at >= 80% of sizes | fewer than 50% |
| H2b | The instruct - base gap under `probe` and `lora_sft` is < 0.05 | CI of the mean gap lies within [-0.05, 0.05] | CI excludes that interval |
| H3a | Best `lora_sft` model <= 1.7B >= `prompt_generate` model 10x larger (in-distribution) | paired difference CI >= 0 on at least 4 of 6 tasks | fewer than 3 tasks |
| H3b | On `loan_sign_flip` and `loan_covariate_shift`, the small adapted model's loss relative to in-distribution is larger than the large model's | CI of the interaction excludes 0 | CI includes 0 |
| H4a | Illegal-action rate under `prompt_score` is 0 | by construction | n/a |
| H4b | `b_score` < 0.5 `b_generate` | bootstrap CI of the ratio < 0.5 | CI includes 0.5 or above |
| H5 | int4 cost < 0.05 at >= 7B; > 0.10 below 2B | CIs of the paired bf16 - int4 difference meet both thresholds | either fails |
| H6a | No model under prompting reaches UCB1's bandit score | all CIs below the UCB1 score | any CI includes or exceeds it |
| H6b | Summary history > raw history for models < 2B | paired CI > 0 for all such models | any CI includes 0 |
| H7a | Thinking / CoT improves gridworld + tictactoe at >= 4B | paired CI > 0 | CI includes 0 |
| H7b | Thinking / CoT does not improve models < 2B | paired CI includes 0 or is negative | CI > 0 |
| H7c | Latency-normalised score is lower with thinking at every size | score/latency ratio lower for every size | any size higher |

## 4. Design

- Environments, parameters and seeds exactly as in `configs/sweep_gpu.yaml` and `configs/ablations.yaml`
  at the `prereg-v1` tag.
- Evaluation seeds 0-99 (blackjack 0-299); training seeds 100000-100099. Prompts frozen at the tag.
- Models: the families and sizes listed in the two configs; a model that cannot be loaded is
  dropped and listed in the appendix.
- Primary outcome: normalised return per episode. Secondary: illegal-action rate, oracle agreement,
  task metrics, latency and tokens per decision.

## 5. Sample size

From the pilot, the per-episode SD of normalised score (to be filled in): bandit __, contextual
bandit __, loan __, gridworld __, tictactoe __, blackjack __. With a smallest effect of interest of
0.10 and SD <= 0.40, 100 paired episodes give power >= 0.8 at alpha = 0.05 (two-sided); blackjack
(SD ~ 1.0 per hand) uses 300.

## 6. Analysis plan

Per cell: mean, IQM, 95% bootstrap CI (2000 resamples). Paired comparisons: permutation test on
sign-flipped paired differences (5000 permutations), Holm-corrected within each hypothesis's family
of tests. Scale slopes by ordinary least squares with bootstrap CIs over episodes. All computed by
`deadeye report` and `deadeye compare`; the exact code at the `prereg-v1` tag is the analysis.

## 7. Exclusion and stopping rules

No episode is excluded for its outcome. A cell is excluded only on load failure or crash (recorded
in `run_manifest.json`). No interim analysis changes the design; the sweep runs to completion.

## 8. Deviations (append-only)

| Date | What changed | Why |
|---|---|---|
| | | |
