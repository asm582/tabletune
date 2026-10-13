#!/usr/bin/env bash
# Repeat runs 2 and 3 of every sweep behind a headline number (the existing
# sweeps are run 1), so results can be reported as mean and spread of 3 runs.
# Waits for any running pipeline to finish first. Sweeps are named <sweep>-r2,
# <sweep>-r3; average them with: python experiments/bench/aggregate_repeats.py <sweep>
#
# Usage (on the instance, repo root): bash experiments/scripts/run_repeats.sh [2 3]
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
while pgrep -f "run_draft.sh|run_ngram.sh|run_ctrl_items.sh|run_ctrl_fix.sh" > /dev/null; do sleep 60; done
source ~/venv/bin/activate
log() { echo "=== $* ($(date +%H:%M:%S)) ==="; }
REPS=${*:-2 3}
NGDATA=file:experiments/results/ngram/bench-apply.jsonl
EAGLE_REFIT=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000
             "VLLM_DSD_COSTS=experiments/results/tuner-v3-code/costs_k*.json" VLLM_DSD_PRIOR=experiments/results/tuner-v2/acceptance.json)
NG_CTRL=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=4 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000
         "VLLM_DSD_COSTS=experiments/results/ngram-tuner/costs_k*.json")

for r in $REPS; do
  log "repeat $r: EAGLE-3 code workload (code128t-t4)"
  env MAX_SEQS=128 TEMPERATURE=1.0 bash experiments/bench/run_configs.sh code128t-t4-r$r \
    "k1 k2 k3 dsd-tuned-v3 nospec" "1 2 4 8 16 32 64 96 128" instruct-coder

  log "repeat $r: EAGLE-3 controller value (ctrlval)"
  C="4 16 32 40 64 96"
  env MAX_SEQS=128 TEMPERATURE=1.0 bash experiments/bench/run_configs.sh ctrlval-off-r$r "dsd-docs dsd-rescaled" "$C" instruct-coder
  env "${EAGLE_REFIT[@]}" MAX_SEQS=128 TEMPERATURE=1.0 \
    bash experiments/bench/run_configs.sh ctrlval-refit-r$r "dsd-docs dsd-rescaled" "$C" instruct-coder
  env VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_LOG_STEPS=1000 MAX_SEQS=128 TEMPERATURE=1.0 \
    bash experiments/bench/run_configs.sh ctrlval-hyst-r$r "dsd-rescaled" "$C" instruct-coder

  log "repeat $r: n-gram code-apply (ngram-apply-t4, -v2)"
  export OSL=512
  env MAX_SEQS=128 TEMPERATURE=0 bash experiments/bench/run_configs.sh ngram-apply-t4-r$r \
    "nospec ngram-k1 ngram-k3 ngram-k5 ngram-tuned k3" "1 2 4 8 16 32 64 96 128" "$NGDATA"
  env MAX_SEQS=128 TEMPERATURE=0 bash experiments/bench/run_configs.sh ngram-apply-t4-v2-r$r \
    "ngram-tuned-v2" "1 2 4 8 16 32 64 96 128" "$NGDATA"

  log "repeat $r: n-gram controller"
  C="4 16 32 64 96 128"
  env "${NG_CTRL[@]}" VLLM_DSD_PRIOR=experiments/results/ngram-tuner/acceptance.json MAX_SEQS=128 TEMPERATURE=0 \
    bash experiments/bench/run_configs.sh ngram-ctrl-calib-r$r "ngram-untuned" "$C" "$NGDATA"
  env MAX_SEQS=128 TEMPERATURE=0 bash experiments/bench/run_configs.sh ngram-ctrl-off-r$r "ngram-untuned" "$C" "$NGDATA"
  env "${NG_CTRL[@]}" VLLM_DSD_PRIOR=experiments/results/ngram-tuner/acceptance.json MAX_SEQS=128 TEMPERATURE=0 \
    bash experiments/bench/run_configs.sh ngram-ctrl-tuned-r$r "ngram-tuned-v2" "$C" "$NGDATA"
  unset OSL
done
log "repeats done"
