# Handoff

Written 10 October 2026 for whoever continues this work, human or assistant. It records what exists, what
does not, the decisions that fixed the design, and the rules the project owner set for how the work is
carried out. Read this before touching anything; read `docs/before_you_publish.md` before running anything.

## 1. The project in one paragraph

DeadEye asks what decides how well an open-weight language model performs once it is turned into a
decision-maker: parameter count, the conversion method (free reply, formatted generation, likelihood
scoring, a probe on hidden states, LoRA behaviour cloning), instruction tuning, quantisation, reasoning
budget, or the structure of the task. It is a benchmark harness (`deadeye`, Python, tested) with seven seeded
tasks that have known optimal policies, a pre-registered study, and a paper. The study runs on one Apple M2
Max laptop with 32 GB; everything that needs more hardware is future work. The paper's results are not yet
measured.

## 2. State of the work

Done and verified:

- Harness: `src/deadeye/` (environments, backends `hf`/`openai`/`decision`/`mock`, six methods, runner,
  statistics, report, CLI). 115 tests (`pytest -q`) and the end-to-end smoke run (`deadeye smoke`) pass.
- Configs for the laptop study: `configs/pilot.yaml`, `mac_main.yaml`, `mac_lora.yaml`, `mac_free_reply.yaml`,
  `mac_controls.yaml`, `configs/ablations/*.yaml` (seven files), `decision_models.yaml`; `scripts/run_mac.sh`
  runs them in order and resumes. `configs/examples/served_models.yaml` is an example, not part of the study.
- Paper: `paper/main.tex` and `paper/sections/*.tex`, complete draft with 142 `\result{}` placeholders and
  22 `\todo{}` notes; `paper/refs.bib` with 124 verified entries in Harvard author-year style (natbib + agsm);
  `scripts/check_tex.py` passes. `scripts/build_draft_pdf.py` builds `DeadEye_paper_draft.pdf` and
  `DeadEye_handbook.pdf` without a TeX installation (pandoc, KaTeX, Chromium); both PDFs are committed at the
  repository root and are rebuilt after every paper change.
- Documents: `RESEARCH_PLAN.md`, `docs/preregistration.md` (not yet frozen), `docs/before_you_publish.md`
  (the ten steps), `docs/compute_budget.md`, `docs/peer_testing.md`, `docs/known_limitations.md`,
  `docs/future_work.md`, `docs/literature_review.md`, `docs/model_ladders.md`, `docs/references_audit.md`.

Not done, in the order it has to happen (details and commands in `docs/before_you_publish.md`):

1. No real model has been run. Everything so far ran on a mock and a tiny random model in a sandbox with no
   GPU and no access to Hugging Face. The Metal backend was added for the Mac but has never executed; the
   pilot is its first real test (`dtype: float16` or `device: cpu` are the fallbacks).
2. Pilot, pre-registration freeze (`git tag prereg-v1`), main sweep, LoRA, free replies, controls,
   ablations, decision models, then `deadeye report ... --paper-dir paper`.
3. Fill every `\result{}` and clear every `\todo{}` from the report's CSV files; run `deadeye compare` per
   hypothesis family; mark each pre-registered criterion supported or falsified.
4. Small checks before submission: TechCrunch bylines in `refs.bib`; the Qwen3.5 base repository ids in
   `configs/decision_models.yaml`; a Laya adapter if its HTTP schema differs from the System One protocol;
   reopen every model card cited in the decision-model table.

## 3. Decisions that fix the design (do not reopen without the owner)

| Date | Decision | Why |
|---|---|---|
| 8 Oct | Seven tasks with exact oracles, scores normalised random = 0 / oracle = 1, paired seeds, bootstrap CIs, paired permutation tests with Holm per hypothesis family, pre-registration before the confirmatory runs | the paper must survive expert review; every claim needs a known optimum and a pre-stated test |
| 8 Oct | Six conversion methods including the unconverted free reply as the "before" reference and a no-LM feature probe as a control | the question is what the conversion adds, so both ends must be measured |
| 8 Oct | Customer-support triage task under a written policy ("I received the package damaged. I need a refund today.") | the owner's canonical use case |
| 8 Oct | Comparison with decision models (Jev and its open relatives) as exploratory RQ5, never pre-registered | the hosted model cannot be controlled for training data |
| 9 Oct | Two-part comparison: small vs large open-weight within ladders; small conversions vs Kev, Clef, Laya, Perplexity's decider, SemIf | requested explicitly |
| 9 Oct | Calibration (ECE, Brier) of the assigned probability against the oracle, logged for every policy that returns probabilities | "calibrated decisions" is the decision models' claim; it is measured, not assumed |
| 9 Oct | Scope fixed to one Apple M2 Max (32 GB): four ladders 135M-8B with every method, LoRA to 4B, GGUF quantisation via llama.cpp, 256-token reasoning budgets, local decision models only (Kev 0.8B/4B, Laya, SemIf), ablations at 50 episodes; 14B-72B, hosted decision models, long budgets, Llama, larger controls are future work; no hosted inference rows at all | no GPU budget; the owner decided that only what the laptop runs is in the study |
| 9 Oct | Title: "From Open Weights to Decision Policies: A Controlled Study of Scale, Conversion Method and Task Structure in Language-Model Decision-Making"; RQ1-RQ4 pre-registered, RQ5 exploratory | precise and defensible |

## 4. Rules the project owner set for assistants working on this repository

These were stated during the work and apply to every future session.

- **Agents.** Any sub-agent spawned must run Opus 5.5 at maximum reasoning effort, and no more than three
  agents may run at the same time.
- **Commits.** No `Co-Authored-By` line naming an assistant, no session links, no assistant attribution of any
  kind in any commit message, file or author field. Commits are authored by the owner's identity
  (`CaptiveON <its.mohammadadnan@gmail.com>`, set in this repository's git config); commit signing is off.
  Commit messages say what changed and why, in plain prose. Work is committed and pushed to the working branch
  (`claude/tender-planck-1vf5vf`) as it progresses; the branch is the default branch of the repository.
- **Writing.** The paper, the documents and the commit messages must read as a careful person wrote them:
  varied sentence rhythm, plain verbs, no stock phrases, no templated scaffolding, no restating a paragraph in
  its last sentence. This is a style rule for the output only; it is not mentioned in the paper or in commits.
- **Citations.** Harvard author-year referencing (natbib `authoryear` with the `agsm` style; `\citet`/`\citep`
  only). Every reference is cross-checked against a primary source before it is cited, and only established
  sources are allowed: peer-reviewed venues, academic publishers, official technical reports and model cards of
  the organisations that released the models, and established press. No third-party blogs, forum posts or
  unverifiable preprints. `docs/references_audit.md` records how each entry was verified; keep it current.
- **Authenticity.** The paper will be public and reviewed by senior researchers. No fabricated numbers, dates,
  hardware or results: anything unmeasured stays a visible `\result{}` placeholder; makers' performance figures
  are quoted only as their claims with the source; every empirical table and figure is generated by
  `deadeye report`, never typed by hand.
- **Scope.** Only what the owner's Mac can run is in the study; everything else goes to `docs/future_work.md`
  and the paper's future-work section, with the hardware and approximate cost it needs. Do not add hosted
  inference rows.
- **Pre-registration discipline.** Prompts change only during the pilot and only against the pilot seeds
  (5000-5019). After the `prereg-v1` tag every design change is logged in the deviations table of
  `docs/preregistration.md` with a date and a reason. Evaluation seeds never overlap training seeds (the
  harness rejects such configs).
- **Quality gates before any push.** `pytest -q`, `deadeye smoke`, `python scripts/check_tex.py`, and a dry
  run of every changed config (`deadeye run <cfg> --dry-run`). Figures keep the fixed method-to-colour mapping
  in `deadeye/report.py`.
- **Where sessions run.** Cloud sessions cannot reach the owner's Mac and cannot download model weights, so
  they do code, documents and the paper. The experiments are plain terminal commands that need no assistant;
  the owner runs them on the Mac and commits `results/**/summary.json` and `episodes.jsonl` (not `steps.jsonl`,
  which is large) so a later session can generate the report and fill the paper from the logs.

## 5. Caveats a successor must know

- Facts about Jev, Kev, Clef, Laya, Perplexity's decider and SemIf date from September and October 2026,
  after the assistant's training data; they were confirmed through search renderings of the makers' pages
  and a direct read of Kev's repository, because several hosts were unreachable from the sandbox. Open the
  primary pages once before submission.
- The adversarial review of the harness fixed thirteen bugs; the design notes it left (oracle tie-breaking,
  blackjack variance, the contextual-bandit feature probe at chance by construction, Holm families) are in
  `docs/known_limitations.md` and in the paper's limitations.
- `configs/models.yaml` marks repository ids the ladder audit could not fetch as `verified: false`; confirm
  them on the hub before the sweep.
- The handbook PDF and the paper draft PDF are built from the sources; rebuild them with
  `python scripts/build_draft_pdf.py` after any change rather than editing them.

## 6. How to pick the work up

1. `git clone`, create a venv, `pip install torch` then `pip install -e ".[hf,dev]" tabulate`, run
   `pytest -q && deadeye smoke`.
2. Read `docs/before_you_publish.md` and follow its ten steps; `scripts/run_mac.sh <stage>` runs each stage.
3. For code or paper changes in a cloud session: keep the gates in section 4, rebuild the PDFs, commit with
   the owner's identity, push to the working branch.
