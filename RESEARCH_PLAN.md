# DeadEye research plan: from open weights to decision-makers

**Question.** When an open-weight language model is converted into a decision-making policy, which
factors determine decision quality? In particular: does parameter count dominate, or can small
models compete once the *conversion method* and the *task structure* are taken into account?

**Deliverables.** (1) A peer-testable product: the `deadeye` benchmark harness in this repository,
installable with `pip`, runnable on a laptop CPU (pilot) or a GPU box (full sweep), with a public
leaderboard generated from JSONL logs. (2) A research paper (skeleton in `paper/`) whose every
table and figure is produced by `deadeye report` from logged runs. (3) A pre-registration
(`docs/preregistration.md`) that fixes the hypotheses and analysis before the main sweep runs.

This document is the end-to-end guide: where to start, what each phase produces, how to test, and
what "finished" means. Commands are real and run today.

---

## 1. Framing

### 1.1 What "converting a weighted model into a decision model" means here

A decision model is a policy: it maps a state or observation to an action. A language model is not
a policy; it is a distribution over text. There are several distinct ways to turn one into the
other, and they use the weights very differently. DeadEye treats the conversion method as a
first-class experimental factor with five levels:

| Method (`methods[].name`) | What is used from the model | Training data | Where it runs |
|---|---|---|---|
| `prompt_generate` | free-form generation, parsed into an action (zero-shot, few-shot, chain-of-thought variants) | none | any backend |
| `prompt_score` | exact log-likelihood of each legal action as a continuation (lm-eval-harness style) | none | `hf`; first-token approximation on `openai` |
| `probe` | hidden state of the prompt at one layer, with a logistic-regression head | oracle-labelled states from training seeds | `hf` |
| `lora_sft` | LoRA behaviour cloning on oracle (observation, action) pairs, then scoring | same | `hf` |
| `feature_probe` | **no language model**: the same logistic head on the environment's raw numeric features | same | anywhere |

`feature_probe` is the control that answers "does the language model's representation add anything
over the raw features?". `random` and `oracle` anchor the score scale; `ucb1` is the classical
algorithmic reference on the bandit.

### 1.2 Tasks

All environments are procedurally generated with a seed, have a known oracle, and are Markov in
their text rendering (the text alone suffices to act optimally). They span the decision structures
that matter in practice:

| Environment | Decision structure | Oracle | Headline metric |
|---|---|---|---|
| `bandit` | exploration vs exploitation, no state | best true arm | pseudo-regret |
| `contextual_bandit` | in-context learning of an abstract rule from (context, action, reward) history | argmax of hidden logistic model | regret |
| `loan` | one-shot decisions from semantically meaningful features (tabular, knowledge-driven); OOD switches | approve iff true expected value > 0 | profit, regret |
| `gridworld` | deterministic multi-step planning | BFS shortest path | success, excess steps |
| `tictactoe` | adversarial, perfect information | minimax | win rate |
| `blackjack` | risk under uncertainty | basic strategy (hit/stand) | expected return |

Every episode return is normalised so that the random policy scores 0 and the oracle scores 1 on
that environment; a normalised score is comparable across tasks and the cross-task mean is a single
headline number per (model, method).

### 1.3 Factors

| Factor | Levels in the configs | Hypothesis |
|---|---|---|
| Parameter count (within one family / recipe) | Qwen2.5 0.5B to 72B; Qwen3 0.6B to 32B; SmolLM2 135M to 1.7B; Gemma 3 270M to 27B; Pythia 70M to 12B (control) | H1 |
| Instruction tuning | base vs instruct checkpoint of the same size | H2 |
| Conversion method | the five methods above | H3, H4 |
| Quantisation | bf16, int8, int4 | H5 |
| Observation / history format | bandit `history_format: summary` vs `raw`; `history_window` | H6 |
| Reasoning budget | Qwen3 thinking on/off; `cot`; `max_new_tokens` 16 / 256 / 2048 | H7 |
| Prompt details | `action_format: name` vs `letter`; `few_shot` 0/3; `temperature` 0/0.7; `length_norm` | secondary |
| Data budget for adaptation | probe / LoRA with 10, 40, 100 training episodes | H3 |
| Task difficulty | arms 5 vs 10, grid 6 vs 9, opponent random vs minimax, loan shift none / covariate / sign-flip | H1, H3 |

### 1.4 Pre-registered hypotheses (details and falsification criteria in `docs/preregistration.md`)

- **H1 (scale).** Under `prompt_generate`, normalised score increases approximately log-linearly with
  parameters within a family, with a task-dependent slope: steep for planning and adversarial tasks
  (gridworld, tictactoe), shallow for one-shot contextual decisions (loan).
- **H2 (instruction tuning).** At equal size, instruct checkpoints beat base checkpoints under
  prompting; the gap shrinks to non-significance under `probe` and `lora_sft`.
- **H3 (adaptation vs scale).** A `lora_sft` or `probe` model of at most 1.7B parameters matches or
  exceeds a zero-shot `prompt_generate` model ten times larger on in-distribution instances, but not
  on the out-of-distribution loan variants, where the generalisation gap grows with size.
- **H4 (scoring vs generation).** `prompt_score` eliminates format failures and closes a substantial
  fraction (we predict more than half) of the gap between the smallest and largest model under
  `prompt_generate`.
- **H5 (quantisation).** int4 costs less than 0.05 normalised score at 7B and above and more than
  0.10 below 2B.
- **H6 (exploration).** On the bandit, no model at any scale reaches the UCB1 reference under
  prompting; summary history beats raw history for small models.
- **H7 (reasoning).** Thinking mode / chain-of-thought improves gridworld and tictactoe for models of
  4B and above, does not improve or hurts models below 2B, and always loses on latency-normalised
  efficiency.

---

## 2. Phases: where to start, what to do, what each phase produces

### Phase 0: harness and sanity (done; today)

What exists: six environments with oracles, three backends (`hf`, `openai`, `mock`), five methods,
a resumable runner with per-decision logging, a statistics module (bootstrap CIs, IQM, paired
permutation tests with Holm correction), and a report generator (tables in Markdown and LaTeX,
figures, HTML). Exit criteria, all met:

```bash
pip install -e ".[hf,dev]" tabulate
pytest -q                    # unit + integration tests, tiny random model, no downloads
deadeye smoke                # full pipeline end-to-end in ~30 s, writes an HTML report
deadeye list-envs; deadeye list-methods
deadeye show-prompt gridworld --seed 3 --cot --few-shot 2   # see exactly what a model sees
```

### Phase 1: CPU pilot (week 1-2)

Goal: validate the pipeline on real weights, fix prompt bugs, and get first effect sizes to size the
main sweep. Nine models up to 1.7B (SmolLM2, Qwen2.5, Qwen3), four methods, shortened horizons.

```bash
deadeye estimate configs/pilot_cpu.yaml --seconds-per-decision 0.5
deadeye run configs/pilot_cpu.yaml --only-model HuggingFaceTB/SmolLM2-135M-Instruct   # ~20 min
deadeye run configs/pilot_cpu.yaml                                                     # all, resumable
deadeye report results/pilot_cpu --out report/pilot_cpu
```

Rules for the pilot: prompts may be edited **only** in this phase, and only using the pilot seeds.
When the pilot is done, tag the repository (`git tag prereg-v1`) and freeze `docs/preregistration.md`.
Everything after the tag is confirmatory.

Outputs: `report/pilot_cpu/index.html`, a first scale curve, an estimate of the per-episode standard
deviation of normalised score on each task (used for the sample-size calculation), and a list of
format failures from `steps.jsonl` (search for `"legal": false`).

### Phase 2: main sweep on GPU (week 3-5)

The within-family ladders with all methods, 100 episodes per cell (300 for blackjack), 100 training
episodes for probes and LoRA. `configs/sweep_gpu.yaml` has 27 model entries and 6 methods; cells are
resumable so the sweep can be split across machines by `--only-model`.

```bash
deadeye estimate configs/sweep_gpu.yaml --seconds-per-decision 0.3
CUDA_VISIBLE_DEVICES=0 deadeye run configs/sweep_gpu.yaml --only-model Qwen/Qwen2.5-7B-Instruct
deadeye report results/sweep_gpu --out report/sweep_gpu --paper-dir paper
```

Hardware: everything up to 14B runs in bf16 on a 24 GB GPU; 27B-72B use `quantization: int4`
(bitsandbytes, `pip install -e ".[quant]"`), which needs about 20-45 GB. Mark quantised rows as such
in the paper; the quantisation ablation in Phase 3 measures what int4 costs at sizes where both fit.

### Phase 3: ablations (week 5-7)

`configs/ablations.yaml` isolates one factor per block on three sizes of the Qwen2.5 ladder (1.5B,
7B, 14B) plus Qwen3 4B/8B for thinking. Run blocks separately, e.g.

```bash
deadeye run configs/ablations.yaml --only-method gen_cot
deadeye run configs/ablations.yaml --only-env loan_sign_flip
```

### Phase 4: analysis and writing (week 7-9)

`deadeye report` produces every table and figure; `deadeye compare` runs the paired tests behind each
claim in the paper. Fill `paper/sections/*.tex` from the report's `tables/` and `figures/`. The
literature review in `docs/literature_review.md` and the bibliography in `paper/refs.bib` are
pre-verified.

### Phase 5: release (week 9-10)

Tag `v1.0`, build the Docker image, archive the results directory and the report with a DOI
(Zenodo), publish the leaderboard HTML, submit the paper (targets: NeurIPS Datasets & Benchmarks,
TMLR, COLM, or an agents / small-models workshop), post the preprint.

---

## 3. How to test: validity, statistics, and the checks that must pass

### 3.1 Harness validity (automatic)

- Oracle regret is 0 and the oracle's normalised score is 1 on every environment; the random
  policy's is 0 (`tests/test_envs.py`, `tests/test_metrics_report.py`).
- Each seed reproduces exactly the same instance and the same pre-drawn noise, so two policies on
  the same seed face identical potential outcomes (common random numbers; paired tests are valid).
- Exact scoring matches a manual log-probability computation; hidden states differ by layer; LoRA
  training lowers the loss and raises the probability of the trained label (`tests/test_hf_tiny.py`).
- Every prompt is inspectable with `deadeye show-prompt` and every decision is logged with the raw
  model output, the parsed action, the legal set and the oracle action.

### 3.2 Statistical protocol (pre-registered)

- Unit of analysis: the episode (seed). Primary outcome: normalised return. Secondary: illegal-action
  rate, oracle agreement, task metrics (regret, success, win), latency and tokens per decision.
- Point estimates: mean and interquartile mean; uncertainty: 95% bootstrap CI over episodes
  (2000 resamples). Paired comparisons (same seeds): permutation test on sign flips of the paired
  differences, 5000 permutations, Holm-corrected within each family of comparisons. Effect sizes:
  paired Cohen's d.
- Sample size: `deadeye.metrics.episodes_needed(effect, sd)`; with the pilot's SD of normalised
  score per task (expected 0.2-0.4) and a smallest effect of interest of 0.10, 32-130 episodes per
  cell are needed; the sweep uses 100 (300 for the high-variance blackjack task).
- Scale curves: fit `score = a + b * log10(params)` per (family, method, task) by least squares with
  bootstrap CIs on `b`; H1 is supported if `b > 0` with CI excluding 0 for `prompt_generate` and the
  slopes differ by task class; H4 is supported if `b` under `prompt_score` is less than half of `b`
  under `prompt_generate`.
- Exclusions: a cell is excluded only if the model fails to load or a run crashes; no exclusion on
  outcome. Every excluded cell is listed in the paper's appendix.

### 3.3 Leakage and confound controls

- Training seeds (probe, LoRA, few-shot demonstrations) start at 100000 and never overlap evaluation
  seeds (0-999).
- Prompt wording is frozen after Phase 1 (`prereg-v1` tag). Prompt-format factors are run as
  explicit ablations, not tuned per model.
- Action labels are compared under both `name` and `letter` encodings so that tokenisation and
  surface-form effects are measured, not hidden.
- The oracle is deterministic; "oracle agreement" is therefore a lower bound when several actions are
  optimal (gridworld, tictactoe), which is why return, not agreement, is the primary outcome.
- Quantised models report the catalogue parameter count, not the packed weight count, and are
  labelled by precision in every table.
- API-served models (`openai` backend) are non-deterministic across servers; results from them are
  reported in a separate table and never mixed into the scale curves.

### 3.4 What would make us abandon a hypothesis

Written in `docs/preregistration.md`. In brief: a confidence interval on the relevant slope or
difference that includes 0, or an effect in the opposite direction, in the confirmatory sweep.

---

## 4. Compute budget

Decisions per model per method (sweep defaults): bandit 5000, contextual bandit 3000, loan 2000,
gridworld about 1000, tictactoe about 400, blackjack about 600; roughly 12000 decisions. Per model:
`prompt_generate` x2 (plain, CoT), `prompt_score`, `probe` (plus 100 training episodes),
`lora_sft` (plus training) - roughly 60000 forward passes of 300-600 prompt tokens.

| Model size | Backend | Seconds per decision (approx.) | Hours per model (sweep) |
|---|---|---|---|
| 135M-0.6B | CPU laptop | 0.2-0.8 | 4-13 |
| 0.5B-1.7B | one 24 GB GPU | 0.03-0.08 | 0.5-1.5 |
| 7B-14B | one 24 GB GPU (bf16) | 0.1-0.25 | 2-4 |
| 32B-72B | one 80 GB GPU (int4) | 0.4-1.0 | 7-17 |

The full sweep is about 120 GPU-hours on a single 80 GB GPU or 2-3 days on 2-4 GPUs. The ablation
config is roughly the same again. `deadeye estimate <config>` recomputes these numbers for any
edit. See `docs/compute_budget.md`.

---

## 5. Threats to validity (and what the design does about them)

1. **Prompt sensitivity.** Mitigated by freezing prompts before the sweep and running format
   ablations; residual sensitivity is reported, not hidden.
2. **Tokenisation and surface-form competition in scoring.** Mitigated by name/letter encodings and
   length normalisation as explicit factors.
3. **Contamination.** Instances are generated on the fly from seeds; no instance text exists on the
   web. Rules (blackjack basic strategy, tic-tac-toe) are of course known to models; this is a
   feature of knowledge-driven decision-making, and the contextual bandit and sign-flipped loan
   variants are the tasks where prior knowledge cannot help.
4. **Cross-family confounds.** Scale claims are made within families only; cross-family plots are
   descriptive.
5. **Oracle quality.** Blackjack basic strategy without doubling/splitting is near-optimal, not
   optimal; this changes the anchor slightly but equally for all policies.
6. **Hardware non-determinism.** Greedy decoding and seeded sampling; the manifest records torch,
   transformers, CUDA, and GPU model for every run.

---

## 6. The end state ("done" checklist)

- [ ] `prereg-v1` tag with frozen prompts and `docs/preregistration.md`
- [ ] `results/sweep_gpu` and `results/ablations` complete (no missing cells in `run_manifest.json`)
- [ ] `deadeye report ... --paper-dir paper` regenerates every figure and table in the paper
- [ ] Paper: abstract, introduction, related work (from `docs/literature_review.md`), method, results
      (H1-H7 each answered with a number and a CI), limitations, reproducibility statement, appendix
      with all prompts and the cell list
- [ ] `v1.0` tag, Docker image, Zenodo DOI for code + results, leaderboard HTML published
- [ ] Peer-testing instructions verified by one person who did not write the code
      (`docs/peer_testing.md`)

---

## 7. Extensions after v1.0

- Preference / RL fine-tuning (DPO on oracle-vs-model action pairs, GRPO with environment reward)
  as a sixth conversion method; the `hf` backend already exposes everything needed.
- Text-game environments (TextWorld / ALFWorld / BabyAI-text) for long-horizon language tasks.
- Multi-agent and negotiation tasks; human-behaviour targets (Centaur-style) instead of oracles.
- Energy measurement per decision (NVML) to replace latency in the Pareto plots.
