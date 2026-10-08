"""DeadEye: a benchmark for converting open-weight language models into decision-making policies.

The package is organised around three protocols:

* :mod:`deadeye.envs`     - procedurally generated decision environments with known oracles.
* :mod:`deadeye.models`   - language-model backends (Hugging Face, OpenAI-compatible HTTP, mocks).
* :mod:`deadeye.policies` - "conversion methods" that turn a language model into a policy.

:mod:`deadeye.runner` runs the full factorial design described in a YAML config and writes JSONL logs;
:mod:`deadeye.report` aggregates those logs into tables and figures for the paper.
"""

__version__ = "0.1.0"
