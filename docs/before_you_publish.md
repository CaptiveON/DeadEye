# Before you publish: what is done, what is not, and what to do

This page is written for the person who will run the study and submit the paper. It says plainly what
exists today, what is still missing, and the order in which to do the remaining work.

## What is done

- The benchmark software is complete and tested: seven decision tasks with known optimal policies, four
  model backends (local Hugging Face weights, OpenAI-compatible servers, decision-model APIs, and a mock
  for tests), six conversion methods, a resumable runner that logs every decision, statistics (confidence
  intervals, paired tests, scale slopes, calibration), and a report generator that writes every table and
  figure the paper uses. `pytest` and `deadeye smoke` pass.
- The paper is drafted in full: abstract, introduction, related work, benchmark, setup, results structure,
  analysis plan, limitations, conclusion and appendix (every prompt verbatim), with a verified Harvard
  bibliography of 124 entries.
- The research plan, pre-registration, compute budget, peer-testing guide and known limitations are written.

## What is not done

- **No real model has been evaluated.** The sandbox in which this was built had no GPU and no access to
  Hugging Face, so every experiment ran only on a mock model and a tiny randomly initialised model, which
  prove the pipeline works and nothing else. The Metal backend was added for your Mac but could not be
  exercised here; the first thing the pilot will tell you is whether it loads and runs, and the fallback is
  `dtype: float16` or `device: cpu` in the config.
- **Every number in the paper's results is a placeholder.** Placeholders are the blue `\result{...}` marks
  (127 of them) and the red `\todo{...}` notes (23). The draft PDF prints them in colour so none can be missed.
- **The pre-registration is not frozen** (`docs/preregistration.md` still has blanks for the pilot's standard
  deviations), and nothing has been tagged.
- **Laya may need a small adapter** before it can be scored if its HTTP schema differs from the System One
  protocol; Kev speaks the protocol already implemented, and SemIf is reproduced by the letter-scoring row.
- **Two small source checks** remain: the TechCrunch bylines in the bibliography, and the Qwen3.5 repository
  ids in `configs/decision_models.yaml`, which must be confirmed on the Hugging Face hub.
- **Everything the laptop cannot run is listed in `docs/future_work.md`** and in the paper's future-work
  section: the 14B to 72B rungs, the hosted decision models, long thinking budgets, and more.

## What to do, in order

Each step says what you need, what to run, how long it takes, and what you should see. Everything runs on your
Mac; no GPU rental is needed, and none of these commands uses Claude or any paid service.

### Step 1. Install on the Mac (one hour)

```bash
git clone https://github.com/CaptiveON/DeadEye && cd DeadEye
python3 -m venv .venv && source .venv/bin/activate
pip install torch                      # the default macOS wheel includes the Metal (MPS) backend
pip install -e ".[hf,dev]" tabulate
pytest -q && deadeye smoke             # about five minutes; all tests pass, a smoke report is written
huggingface-cli login                  # free account; needed for the gated Gemma 3 repositories
python -c "import torch; print(torch.backends.mps.is_available())"   # must print True
```

Accept the Gemma licence on the Hugging Face page of `google/gemma-3-270m` once; the other families are open.

### Step 2. Run the pilot (3 to 6 hours)

```bash
scripts/run_mac.sh pilot
deadeye report results/pilot --out report/pilot
```

Open `report/pilot/index.html`. Check: `random` scores 0 and `oracle` 1 on every task; the illegal-action rate
under `prompt_generate` is below about 20% for the instruct models (if not, read the raw outputs in `steps.jsonl`
and fix the wording in `src/deadeye/policies/prompts.py`); the probe's `train_acc` in `summary.json` is well
above chance. Note the measured `latency_per_decision_s` per model size and re-estimate the sweep with
`deadeye estimate configs/mac_main.yaml --seconds-per-decision <value>`. Prompts may be edited only now.

### Step 3. Freeze the design (one hour)

Fill in the per-task standard deviations in section 5 of `docs/preregistration.md` from `report/pilot/episodes.csv`
(the standard deviation of the paired differences of `norm` between two methods on the same seeds), then:

```bash
git add -A && git commit -m "Freeze pre-registration" && git tag prereg-v1 && git push --tags
```

### Step 4. Run the main sweep (two to three weeks of overnight runs)

```bash
scripts/run_mac.sh main
```

The ladders run in the order Qwen2.5, Qwen3, SmolLM2, Gemma 3, so a stopped run still leaves complete ladders.
Rerun the same command after any interruption; finished cells are skipped. `logs/mac_main.log` has the progress
line of every cell, and `results/mac_main/run_manifest.json` the full cell list.

### Step 5. LoRA, free replies, controls (one to two weeks of overnight runs)

```bash
scripts/run_mac.sh lora
scripts/run_mac.sh free
scripts/run_mac.sh controls     # optional; Pythia and OLMo 2
```

### Step 6. Ablations (about a week)

```bash
scripts/run_mac.sh ablations    # prompting, reasoning, adaptation, tasks, tasks_lora
brew install llama.cpp          # for the quantisation block; download the Q8_0 and Q4_K_M GGUF files of
                                # Qwen2.5-Instruct 1.5B, 3B, 7B, start one llama-server per file on the ports in
                                # configs/ablations/quantisation.yaml, then:
scripts/run_mac.sh quant
```

### Step 7. Decision models (a day)

Install Kev (github.com/jaredpalmer/kev) and start its server for the 0.8B and 4B checkpoints on ports 8009 and
8010; install Laya's package and SemIf as their repositories describe; confirm the two Qwen3.5 base repository
ids on the hub and correct them in the config if needed. If Laya's HTTP schema differs from the System One
protocol, add the small adapter noted in `src/deadeye/models/decision_api.py`. Then:

```bash
scripts/run_mac.sh decision
```

### Step 8. Generate every table and figure (minutes)

```bash
deadeye report results/mac_main results/mac_lora results/mac_free_reply results/mac_controls \
  results/ablations results/decision_models --out report/paper --paper-dir paper
```

`report/paper/cells.csv`, `slopes.csv`, `pairwise_methods.csv` and `system_comparison.csv` hold every number the
paper quotes; `paper/tables/*.tex` and `paper/figures/*.pdf` are included by the paper automatically.

### Step 9. Fill the results (two to three days of careful work)

Work through `paper/sections/results.tex`, `analysis.tex`, `abstract.tex`, `introduction.tex` and `conclusion.tex`.
Replace each `\result{...}` with the measured value and its confidence interval from the CSV files, delete each
`\todo{...}` once handled, run one `deadeye compare` call per hypothesis family and quote the Holm-adjusted
p-values, then write "supported" or "falsified" against each pre-registered criterion. A falsified hypothesis
is a result.

```bash
grep -c '\\result{' paper/sections/*.tex    # must reach 0
grep -c '\\todo{' paper/sections/*.tex      # must reach 0
python scripts/check_tex.py                 # braces, environments, citation keys
```

### Step 10. Final checks, PDF, submission (a day)

Add the TechCrunch bylines to `paper/refs.bib`; reopen each model card cited in the decision-model table and
confirm sizes and licences; rerun `python scripts/check_tex.py`. Build the PDF with `cd paper && make` (TeX) or
`python scripts/build_draft_pdf.py` (no TeX needed). Archive `results/` and `report/` with a DOI (Zenodo), tag
`v1.0`, submit to arXiv and then the venue.

## Honest expectations

The design was reviewed adversarially during construction and every oracle was checked, but no study
survives contact with real models unchanged. Expect to find a prompt that a model family misreads, a
checkpoint that fails to load, or a task whose variance is higher than planned. All of these are handled
by the plan: prompts change only before the tag, failed cells are listed in the appendix, and the power
section reports the smallest effect the design could detect instead of claiming a null.
