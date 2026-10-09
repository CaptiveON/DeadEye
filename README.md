# DeadEye

DeadEye is a benchmark for one question: when an open-weight language model is turned into a
decision-maker, what decides how well it does? Parameter count is the obvious candidate, but the
way the model is converted (prompted, likelihood-scored, probed, or fine-tuned), whether it was
instruction-tuned, how it was quantised, how much it is allowed to think, and the structure of the
task itself all compete for the explanation. DeadEye measures them together.

It takes any open-weight model, as local Hugging Face weights or through an OpenAI-compatible
server, and evaluates it on seven procedurally generated tasks with known optimal policies, under
five conversion methods, with paired seeds, bootstrap confidence intervals and a log of every
decision. Every table and figure in the paper is produced by `deadeye report` from those logs.

- Research plan, phases and "how to test": [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)
- Pre-registration: [`docs/preregistration.md`](docs/preregistration.md)
- Peer-testing guide: [`docs/peer_testing.md`](docs/peer_testing.md)
- Literature review and verified bibliography: [`docs/literature_review.md`](docs/literature_review.md), [`paper/refs.bib`](paper/refs.bib)
- Model ladders: [`docs/model_ladders.md`](docs/model_ladders.md), [`configs/models.yaml`](configs/models.yaml)
- The two comparisons (small vs larger open-weight models within four ladders to 8B; small conversions vs the decision models Kev, Laya and SemIf): section 7 of the plan and [`configs/decision_models.yaml`](configs/decision_models.yaml)
- Everything the laptop excludes: [`docs/future_work.md`](docs/future_work.md)
- Step by step from here to submission: [`docs/before_you_publish.md`](docs/before_you_publish.md)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch                      # macOS wheel includes Metal; Linux: add --index-url https://download.pytorch.org/whl/cpu or your CUDA wheel
pip install -e ".[hf,dev]" tabulate    # add ".[quant]" for int4/int8 on CUDA only
```

## Try it in one minute (no downloads)

```bash
pytest -q          # unit + integration tests with a tiny randomly initialised model
deadeye smoke      # every environment x every method with a mock and the tiny model -> HTML report
```

## Run the study on a Mac (or any machine)

The whole study is designed for one Apple M2 Max with 32 GB (PyTorch's Metal backend; CUDA and CPU also work).

```bash
deadeye list-envs
deadeye list-methods
deadeye show-prompt bandit --seed 0 --steps 5        # see exactly what the model sees
deadeye estimate configs/mac_main.yaml --seconds-per-decision 0.4   # decisions, tokens, wall-clock
scripts/run_mac.sh pilot                             # then main, lora, free, controls, ablations, quant, decision
deadeye report results/pilot --out report/pilot --paper-dir paper
deadeye compare results/pilot --pair "loan:Qwen__Qwen2.5-0.5B-Instruct/prompt_score vs loan:Qwen__Qwen2.5-0.5B-Instruct/prompt_generate"
```

## What is in the box

| Layer | Contents |
|---|---|
| Environments (`deadeye.envs`) | `bandit`, `contextual_bandit`, `loan` (semantic tabular decisions, with OOD shifts), `gridworld`, `tictactoe`, `blackjack`, `support` (customer-support triage under a written policy); each seeded, with an oracle and numeric features |
| Backends (`deadeye.models`) | `hf` (generate, exact action log-likelihoods, hidden states, LoRA), `openai` (any OpenAI-compatible server; first-token scoring), `decision` (System One decision models: TypeSafe Jev, Kev, Cloudflare Clef and compatibles, scored on their returned probabilities), `mock` and a tiny random model for CI |
| Methods (`deadeye.policies`) | `prompt_free` (the unconverted assistant, for before/after comparisons), `prompt_generate` (zero/few-shot, CoT), `prompt_score`, `probe` (linear head on hidden states), `lora_sft` (behaviour cloning), `feature_probe` (no-LM control), plus `random`, `oracle`, `ucb1` |
| Runner (`deadeye.runner`) | factorial (env x model x method) over paired seeds, resumable, logs every decision with the raw output, parsed action, legal set and oracle action |
| Statistics (`deadeye.metrics`, `deadeye.runner`) | normalised scores (random = 0, oracle = 1), bootstrap CIs, IQM, paired permutation tests, Holm correction, power calculation, scale-slope fits, calibration (ECE, Brier) against the oracle |
| Report (`deadeye.report`) | Markdown/LaTeX tables, scale curves, format-failure curves, cost-quality frontier, heatmaps, HTML page |

## Config format

```yaml
name: pilot
output_dir: results/pilot
seeds: {start: 0, n: 20}            # evaluation episodes (one per seed)
train_seeds: {start: 100000, n: 40} # probes / LoRA / few-shot demos (disjoint from evaluation)
envs:
  - {name: bandit, params: {n_arms: 5, horizon: 30}}
  - {name: loan, params: {n_applicants: 10, shift: sign_flip}, label: loan_flip}
models:
  - {backend: hf, id: Qwen/Qwen2.5-0.5B-Instruct, dtype: float32}
  - {backend: hf, id: Qwen/Qwen3-1.7B, chat_template_kwargs: {enable_thinking: false}}
  - {backend: openai, id: qwen2.5:7b-instruct, base_url: "http://localhost:11434/v1", params: 7620000000, family: qwen2.5}
methods:
  - {name: prompt_generate, params: {max_new_tokens: 16, cot: false, few_shot: 0, action_format: name}}
  - {name: prompt_score, params: {length_norm: false}}
  - {name: probe, params: {layer: mid, n_train_episodes: 40}}
  - {name: lora_sft, params: {n_train_episodes: 40, lora: {r: 16, epochs: 2}}}
baselines: [random, oracle, ucb1]
illegal_action: random_fallback      # or first_legal | terminate
```

Outputs land in `results/<run>/<env>/<model>/<method>/{summary.json, episodes.jsonl, steps.jsonl}`.

## Docker

```bash
docker build -t deadeye .
docker run --rm -v $PWD/results:/app/results -v deadeye-cache:/cache deadeye smoke
```

## Citation

See `CITATION.cff`. Licence: Apache-2.0.
