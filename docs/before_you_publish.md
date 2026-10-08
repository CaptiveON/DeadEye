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
  prove the pipeline works and nothing else.
- **Every number in the paper's results is a placeholder.** Placeholders are the blue `\result{...}` marks
  (127 of them) and the red `\todo{...}` notes (23). The draft PDF prints them in colour so none can be missed.
- **The pre-registration is not frozen** (`docs/preregistration.md` still has blanks for the pilot's standard
  deviations), and nothing has been tagged.
- **Two systems need a small adapter** before they can be scored: Laya's package and Perplexity's Decisions
  API use their own request schema; Kev and Clef speak the protocol already implemented.
- **Two small source checks** remain: the TechCrunch bylines in the bibliography, and the Qwen3.5 repository
  ids in `configs/decision_models.yaml`, which must be confirmed on the Hugging Face hub.

## What to do, in order

Each step says what you need, what to run, how long it takes, and what you should see.

### Step 1. Get a machine and install (half a day)

You need a Linux machine with an NVIDIA GPU. One 24 GB card (for example an RTX 4090 or L4) handles every
model up to 14B parameters; the 27B to 72B rows need an 80 GB card (A100 or H100) or 4-bit loading. Cloud
rental is fine. Then:

```bash
git clone https://github.com/CaptiveON/DeadEye && cd DeadEye
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124    # pick the CUDA build for your driver
pip install -e ".[hf,quant,dev]" tabulate
pytest -q && deadeye smoke
huggingface-cli login            # needed for gated repositories (Llama, Gemma)
```

You should see all tests pass and a smoke-test report in a temporary folder.

### Step 2. Run the pilot (a few hours on a GPU; a day on a laptop CPU)

```bash
deadeye run configs/pilot_cpu.yaml
deadeye report results/pilot_cpu --out report/pilot_cpu
```

Open `report/pilot_cpu/index.html`. Check three things: the `random` row scores 0 and `oracle` scores 1 on
every task; the illegal-action rate under `prompt_generate` is below about 20% for the instruct models
(if it is much higher, read the raw outputs in `steps.jsonl` and fix the prompt wording in
`src/deadeye/policies/prompts.py`); and the probe's training accuracy in `summary.json` is well above chance.
Prompts may be edited only during this step.

### Step 3. Freeze the design (one hour)

Fill in the per-task standard deviations in section 5 of `docs/preregistration.md` from
`report/pilot_cpu/episodes.csv` (the standard deviation of the paired differences of `norm` between two
methods on the same seeds), then:

```bash
git add -A && git commit -m "Freeze pre-registration" && git tag prereg-v1 && git push --tags
```

Optionally register the same document on OSF (osf.io) for a timestamp outside your repository.

### Step 4. Run the main sweep (about 150 GPU-hours on one 80 GB card; 2 to 4 days on 2 to 4 cards)

```bash
deadeye estimate configs/sweep_gpu.yaml --seconds-per-decision 0.3   # check the budget first
scripts/run_sweep.sh configs/sweep_gpu.yaml 0 1 2 3                   # one process per GPU id
deadeye run configs/sweep_controls.yaml                               # Pythia and OLMo 2 (about 60 GPU-hours)
```

Runs are resumable: if a machine stops, run the same command again and finished cells are skipped. Watch
`results/sweep_gpu/run_manifest.json` for the cell list and `logs_*.txt` for errors.

### Step 5. Run the ablation blocks (about the same again; each block can run alone)

```bash
for b in quantisation prompting thinking adaptation tasks; do deadeye run configs/ablations/$b.yaml; done
```

### Step 6. Run the decision-model comparison (one to two days, mostly waiting on servers)

- Kev: install from github.com/jaredpalmer/kev and start one server per size on ports 8009, 8010, 8011
  as written in `configs/decision_models.yaml`.
- Clef: set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`; Workers AI bills per request.
- Jev: apply for early access at typesafe.ai and set `TYPESAFE_API_KEY`; if access is not granted, leave the
  row out and say so in the paper.
- Laya and Perplexity: write the adapter (copy `src/deadeye/models/decision_api.py`, change the request and
  response field names to match their documentation, register it in `src/deadeye/models/registry.py`).
- SemIf: run its server, or rely on the `score_letter` row, which is the same computation.
- Confirm the three Qwen3.5 base repository ids on the hub and correct them in the config if needed.

```bash
deadeye run configs/decision_models.yaml
```

### Step 7. Generate every table and figure (minutes)

```bash
deadeye report results/sweep_gpu results/sweep_controls results/ablations results/decision_models \
  --out report/paper --paper-dir paper
```

This writes `paper/tables/*.tex` and `paper/figures/*.pdf`; the paper includes them automatically.
`report/paper/cells.csv`, `slopes.csv`, `pairwise_methods.csv` and `system_comparison.csv` hold every number.

### Step 8. Fill the results (two to three days of careful work)

Work through `paper/sections/results.tex`, `analysis.tex`, `abstract.tex`, `introduction.tex` and
`conclusion.tex`. Replace each `\result{...}` with the measured value and its confidence interval, taken
from the CSV files, and delete each `\todo{...}` once it is handled. For the paired tests, run one
`deadeye compare` call per hypothesis family (the pre-registration lists them) and quote the Holm-adjusted
p-values. Then check the pre-registered criteria one by one and write "supported" or "falsified"; a
falsified hypothesis is a result, not a failure.

```bash
grep -c '\\result{' paper/sections/*.tex    # must reach 0
grep -c '\\todo{' paper/sections/*.tex      # must reach 0
python scripts/check_tex.py                 # braces, environments, citation keys
```

### Step 9. Final source checks (one hour)

Open the TechCrunch articles and add the journalists' names to `paper/refs.bib`; open each model card
cited in the decision-model table and confirm the sizes and licences still match; re-run
`python scripts/check_tex.py`.

### Step 10. Build the PDF and submit

With a TeX installation: `cd paper && make`. Without one: `python scripts/build_draft_pdf.py` builds the
same content through pandoc and Chromium. Archive the `results/` folder and the report with a DOI
(Zenodo), tag the repository `v1.0`, and submit (arXiv first, then the venue). The reproducibility
checklist in the appendix is written for this moment.

## Honest expectations

The design was reviewed adversarially during construction and every oracle was checked, but no study
survives contact with real models unchanged. Expect to find a prompt that a model family misreads, a
checkpoint that fails to load, or a task whose variance is higher than planned. All of these are handled
by the plan: prompts change only before the tag, failed cells are listed in the appendix, and the power
section reports the smallest effect the design could detect instead of claiming a null.
