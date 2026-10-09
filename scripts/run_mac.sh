#!/usr/bin/env bash
# Run the whole laptop study in order, keeping the Mac awake, logging each config, resuming after interruptions.
# Usage: scripts/run_mac.sh            (everything, in order)
#        scripts/run_mac.sh pilot      (one named stage)
# Stages: pilot, main, lora, free, controls, ablations, decision. Stop any time with Ctrl-C; rerun to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1     # lets rare unsupported ops fall back to the CPU instead of failing
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
run() { echo "== $(date '+%F %T') $1"; caffeinate -i deadeye run "$1" 2>&1 | tee -a "logs/$(basename "$1" .yaml).log"; }
stage=${1:-all}
case "$stage" in
  pilot|all)     run configs/pilot.yaml ;;& 
  main|all)      run configs/mac_main.yaml ;;&
  lora|all)      run configs/mac_lora.yaml ;;&
  free|all)      run configs/mac_free_reply.yaml ;;&
  controls|all)  run configs/mac_controls.yaml ;;&
  ablations|all) for b in prompting reasoning adaptation tasks tasks_lora; do run "configs/ablations/$b.yaml"; done ;;&
  decision|all)  run configs/decision_models.yaml ;;&
  quant)         run configs/ablations/quantisation.yaml ;;   # needs the llama.cpp servers running first
esac
echo "done: $stage"
