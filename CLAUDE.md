# Working rules for this repository

Read `HANDOFF.md` first: it holds the state of the project, the decisions that fix the study design, and the
rules below in full. In short:

- Sub-agents: Opus 5.5 at maximum reasoning effort, at most three at a time.
- Commits: authored by the owner's git identity set in this repository; no assistant attribution lines of any
  kind in messages, files or author fields; commit signing off; plain prose messages; push to the working branch.
- Writing: natural, careful prose in the paper, the documents and commit messages; no templated phrasing.
- Citations: Harvard author-year (natbib `authoryear`, `agsm`); every reference verified against a primary
  source; only peer-reviewed venues, academic publishers, official technical reports or model cards, and
  established press. Update `docs/references_audit.md` with every change to `paper/refs.bib`.
- Authenticity: never invent a number, date, hardware detail or result; unmeasured results stay `\result{}`
  placeholders; tables and figures come from `deadeye report` only.
- Scope: only what the owner's Apple M2 Max (32 GB) can run is in the study; everything else goes to
  `docs/future_work.md` and the paper's future-work section. No hosted inference rows.
- Before every push: `pytest -q`, `deadeye smoke`, `python scripts/check_tex.py`, dry-run changed configs,
  rebuild the PDFs with `python scripts/build_draft_pdf.py` after paper or document changes.
- Prompts may change only during the pilot; after the `prereg-v1` tag, design changes go into the deviations
  table of `docs/preregistration.md`.
