# Contributing

- Install with `pip install -e ".[hf,dev]" tabulate` and run `pytest -q` and `deadeye smoke` before opening a PR.
- New environments: subclass `deadeye.envs.base.Environment`, register in `deadeye/envs/registry.py`; the
  parametrised tests in `tests/test_envs.py` must pass (oracle legal and better than random, deterministic seeds).
- New conversion methods: subclass `deadeye.policies.base.Policy`, register in `deadeye/policies/registry.py`,
  declare `needs_prepare` / `mutates_model` honestly, and add the required backend capability to
  `deadeye.runner.required_capabilities`.
- New backends: implement `generate`, `score_choices`, `embed` as applicable and declare `capabilities`.
- Results submissions: add `results/<run>/**/summary.json` + `episodes.jsonl` + the config; never edit
  prompts for a submission without saying so.
- Keep the fixed method-to-colour mapping in `deadeye/report.py`; add new methods to `METHOD_ORDER` with a
  new palette slot rather than reusing one.
