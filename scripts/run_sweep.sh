#!/usr/bin/env bash
# Split a config across GPUs by model: one process per GPU, each taking every k-th model.
# Usage: scripts/run_sweep.sh configs/sweep_gpu.yaml 0 1 2 3
set -euo pipefail
CONFIG=${1:?config path}; shift
GPUS=("$@"); [ ${#GPUS[@]} -gt 0 ] || GPUS=(0)
mapfile -t MODELS < <(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for m in cfg.get("models", []):
    print(m.get("label") or m["id"])
PY
)
i=0
for m in "${MODELS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "GPU $gpu <- $m"
  CUDA_VISIBLE_DEVICES=$gpu deadeye run "$CONFIG" --only-model "$m" > "logs_${gpu}.txt" 2>&1 &
  i=$((i + 1))
  # wait for this GPU's previous job before assigning the next model to it
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
echo "sweep finished; run: deadeye report $(python -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
