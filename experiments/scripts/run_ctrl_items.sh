#!/usr/bin/env bash
# Issue #2, open items A and B (vLLM with the controller plugin, MRv1).
# Note: VLLM_DSD_MIN_GAIN, VLLM_DSD_DECAY_SEEN_ONLY and VLLM_DSD_DEBUG (item A) were removed
# from the controller after this study (they did not fix the flips on the GPU); to
# reproduce item A, check out a commit before their removal.
#
#   A  diagnostic: n-gram tuned v2 table + calibrated prior + VLLM_DSD_DEBUG=1 at
#      c = 4, 16 (K=5 on most steps). Compare the controller's undecayed counters
#      with vLLM's spec-decode counters (accept_by_pos in summary.csv).
#   B1 correctness with VLLM_DSD_ANY_K=1: greedy smoke, EAGLE-3 and n-gram, vs. no spec.
#   B2 value: EAGLE-3 docs table (K in {1,3}) + controller + any K, code workload,
#      c = 16..64; compare with ctrlval-refit (table's Ks only) and the v3 table.
#   S  stability: run B again (n-gram tuned v2, calibrated prior, exploration and
#      prior decay) plus VLLM_DSD_MIN_GAIN=0.02 and VLLM_DSD_DECAY_SEEN_ONLY=1;
#      compare table changes (run B: 7) and throughput.
#
# Usage (on the instance, repo root): bash experiments/scripts/run_ctrl_items.sh [A] [B1] [B2] [S]
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
STAGES=${*:-A B1 B2 S}
has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "=== $* ($(date +%H:%M:%S)) ==="; }

if has A; then
  log "A: debug counters, n-gram tuned v2 + calibrated prior"
  env VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=4 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000 \
      "VLLM_DSD_COSTS=experiments/results/ngram-tuner/costs_k*.json" VLLM_DSD_PRIOR=experiments/results/ngram-tuner/acceptance.json \
      VLLM_DSD_DEBUG=1 OSL=512 TEMPERATURE=0 MAX_SEQS=128 \
    bash experiments/bench/run_configs.sh ngram-ctrl-debug "ngram-tuned-v2" "4 16" file:experiments/results/ngram/bench-apply.jsonl
fi

if has B1; then
  mkdir -p experiments/results/integration
  CTRL=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=2 VLLM_DSD_REFIT_STEPS=20 VLLM_DSD_LOG_STEPS=50
        VLLM_DSD_ANY_K=1 VLLM_USE_V2_MODEL_RUNNER=0)
  log "B1: EAGLE-3 + any K, greedy vs. no spec"
  env "${CTRL[@]}" "VLLM_DSD_COSTS=experiments/results/tuner-v3-code/costs_k*.json" \
      VLLM_DSD_PRIOR=experiments/results/tuner-v3-code/acceptance.json \
    python experiments/bench/smoke_test.py --mode dsd --tag _anyk > experiments/results/integration/smoke_anyk_eagle.log 2>&1
  log "B1: n-gram + any K, greedy vs. no spec"
  env "${CTRL[@]}" "VLLM_DSD_COSTS=experiments/results/ngram-tuner/costs_k*.json" \
      VLLM_DSD_PRIOR=experiments/results/ngram-tuner/acceptance.json \
    python experiments/bench/smoke_test.py --mode ngram-dsd --tag _anyk > experiments/results/integration/smoke_anyk_ngram.log 2>&1
  python experiments/bench/smoke_test.py --compare | grep -E "anyk|^dsd |^ngram-dsd "
  grep -ahoE "Dynamic SD controller[^\"]{0,200}" experiments/results/integration/smoke_anyk_*.log | grep -v "'steps'" | head -6
fi

if has B2; then
  log "B2: EAGLE-3 docs table + controller + any K"
  env VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000 \
      "VLLM_DSD_COSTS=experiments/results/tuner-v3-code/costs_k*.json" VLLM_DSD_PRIOR=experiments/results/tuner-v2/acceptance.json \
      VLLM_DSD_ANY_K=1 MAX_SEQS=128 TEMPERATURE=1.0 \
    bash experiments/bench/run_configs.sh ctrlval-anyk "dsd-docs" "16 32 40 64" instruct-coder
fi
if has S; then
  log "S: stability options on run B's setup"
  env VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=4 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000 \
      "VLLM_DSD_COSTS=experiments/results/ngram-tuner/costs_k*.json" VLLM_DSD_PRIOR=experiments/results/ngram-tuner/acceptance.json \
      VLLM_DSD_EXPLORE=20 VLLM_DSD_EXPLORE_MAX_COST=0.25 VLLM_DSD_PRIOR_DECAY=0.5 \
      VLLM_DSD_MIN_GAIN=0.02 VLLM_DSD_DECAY_SEEN_ONLY=1 OSL=512 TEMPERATURE=0 MAX_SEQS=128 \
    bash experiments/bench/run_configs.sh ngram-ctrl-stable "ngram-tuned-v2" "4 16 32 64 96 128" file:experiments/results/ngram/bench-apply.jsonl
fi
log "items done"
