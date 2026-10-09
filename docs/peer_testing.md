# Testing DeadEye as a peer

The goal of this page is that someone who did not write the code can (a) verify the harness,
(b) reproduce a reported number, (c) evaluate their own model, and (d) submit results.

## A. Verify the harness (no downloads, 5 minutes)

```bash
git clone https://github.com/CaptiveON/DeadEye && cd DeadEye
python -m venv .venv && source .venv/bin/activate
pip install torch                      # macOS wheel includes Metal; on Linux add --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[hf,dev]" tabulate
pytest -q                 # ~1-2 min: environments, parsing, runner, exact scoring, LoRA, report
deadeye smoke             # ~30 s: mock + tiny random model through every env and method
```

`deadeye smoke` prints the path of an HTML report. Open it: the `random` row scores 0 and the
`oracle` row scores 1 on every environment; the tiny random model is at chance.

## B. Reproduce a reported number

Every number in the paper comes from a `summary.json` in a results directory with the config that
produced it in `run_manifest.json`. To re-run one cell:

```bash
deadeye run configs/mac_main.yaml --only-env loan --only-model Qwen/Qwen2.5-1.5B-Instruct --only-method prompt_score --force
deadeye report results/mac_main --out report/check
deadeye compare results/mac_main --pair "loan:Qwen__Qwen2.5-1.5B-Instruct/prompt_score vs loan:Qwen__Qwen2.5-1.5B-Instruct/prompt_generate"
```

Greedy decoding and seeded instances make `prompt_score`, `probe` and `prompt_generate`
(temperature 0) bit-reproducible on the same hardware and library versions, and statistically
reproducible across hardware (compare CIs, not point values).

## C. Evaluate your own model

1. **Local weights** (any `AutoModelForCausalLM`): add to a config

   ```yaml
   models:
     - {backend: hf, id: your-org/your-model, dtype: bfloat16, params: 3000000000, family: yours, instruct: true}
   ```

   `params` and `family` are optional for unquantised models (they are counted), required for
   quantised ones and for the scale plots.
2. **A served model** (vLLM, llama.cpp server, Ollama, LM Studio, OpenRouter):

   ```yaml
   models:
     - {backend: openai, id: served-name, base_url: "http://localhost:8000/v1", params: 7000000000, family: yours}
   methods:
     - {name: prompt_generate, params: {max_new_tokens: 16}}
     - {name: prompt_score, params: {action_format: letter}, label: score_letter}
   ```

   Served models support `prompt_generate` and first-token `prompt_score`; probes and LoRA need
   local weights.
3. Run `deadeye run your.yaml` and `deadeye report results/yours`.

## D. Add an environment or a method

- Environment: subclass `deadeye.envs.base.Environment`, implement `reset`, `step`, `oracle_action`,
  `describe`, set `action_labels` and `max_steps`, register it in `deadeye/envs/registry.py`, and add
  it to `tests/test_envs.py` (the parametrised tests cover legality, determinism and oracle > random
  automatically).
- Method: subclass `deadeye.policies.base.Policy` (see `policies/probe.py` for an offline-trained
  example) and register it in `deadeye/policies/registry.py`.

## E. Submit results to the leaderboard

Open a pull request that adds `results/<your-run>/**/summary.json` and `episodes.jsonl` (not
`steps.jsonl`, which can be large) plus the config. CI regenerates the leaderboard from all
`results/` directories. State hardware, library versions (already in `run_manifest.json`) and
whether prompts were modified.

## F. What to look at when something seems wrong

- `steps.jsonl` has the raw model output for every decision; filter `"legal": false` to see format
  failures and `"parsed_action"` to see what the parser extracted.
- `deadeye show-prompt <env> --seed N --steps K` prints the exact messages.
- `prepare_stats` in `summary.json` reports probe training accuracy and LoRA loss before/after.
