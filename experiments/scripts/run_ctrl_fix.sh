#!/usr/bin/env bash
# Issue #2: the controller with exploration and prior decay on
# (VLLM_DSD_EXPLORE=20, VLLM_DSD_EXPLORE_MAX_COST=0.25, VLLM_DSD_PRIOR_DECAY=0.5).
#
#   A  n-gram run (b) again: untuned table + EAGLE-3's prior. Expect the tuner's
#      table [[1,20,5],[21,128,3]] (without the options: [[1,2,5],[3,128,3]]).
#   B  n-gram tuned v2 table + calibrated prior, options on: no harm, and the
#      throughput cost of exploring (compare with run (d), options off).
#   C  EAGLE-3: the untuned tables of experiments/results/integration §5 (docs table, hand
#      table with K=0) with spec-bench's prior on the code workload, options on
#      (compare with ctrlval-refit, options off).
#
# Usage (on the instance, repo root; vLLM with this package installed): bash experiments/scripts/run_ctrl_fix.sh
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
log() { echo "=== $* ($(date +%H:%M:%S)) ==="; }
FIX=(VLLM_DSD_EXPLORE=20 VLLM_DSD_EXPLORE_MAX_COST=0.25 VLLM_DSD_PRIOR_DECAY=0.5)
NG=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=4 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000
    "VLLM_DSD_COSTS=experiments/results/ngram-tuner/costs_k*.json" OSL=512 TEMPERATURE=0 MAX_SEQS=128)
DATA=file:experiments/results/ngram/bench-apply.jsonl

log "A: n-gram, untuned table, EAGLE-3 prior, options on"
env "${NG[@]}" "${FIX[@]}" VLLM_DSD_PRIOR=experiments/results/tuner-v3-code/acceptance.json \
  bash experiments/bench/run_configs.sh ngram-ctrl-eagleprior-fix "ngram-untuned" "4 16 32 64 96 128" "$DATA"

log "B: n-gram, tuned v2 table, calibrated prior, options on"
env "${NG[@]}" "${FIX[@]}" VLLM_DSD_PRIOR=experiments/results/ngram-tuner/acceptance.json \
  bash experiments/bench/run_configs.sh ngram-ctrl-tuned-fix "ngram-tuned-v2" "4 16 32 64 96 128" "$DATA"

log "C: EAGLE-3, untuned tables, spec-bench prior, options on"
env VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000 \
    "VLLM_DSD_COSTS=experiments/results/tuner-v3-code/costs_k*.json" VLLM_DSD_PRIOR=experiments/results/tuner-v2/acceptance.json \
    "${FIX[@]}" MAX_SEQS=128 TEMPERATURE=1.0 \
  bash experiments/bench/run_configs.sh ctrlval-refit-fix "dsd-docs dsd-rescaled" "4 16 32 40 64 96" instruct-coder
log "controller fix runs done"
