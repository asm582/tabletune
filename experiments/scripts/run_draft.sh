#!/usr/bin/env bash
# Issue #4: the tuner and the controller with a separate draft model
# (Qwen3-0.6B drafting for Qwen3-4B), on the code workload the EAGLE-3 v3 table
# was built for (instruct-coder, ~32-token prompts, T=1.0, max_num_seqs 128), so
# code128t-t4 (EAGLE-3 and no speculation) is the direct comparison.
#
#   accept      draft model K=5 on 160 code-writing prompts at T=1.0
#   costs       step costs K = 0..5 through the Dynamic SD path, ISL 32, batch sizes
#               up to 104 (KV is ~11.7k tokens: larger batches would preempt)
#   schedule    the tuner's table; predicted fixed-K losses
#   sweep       draft model K=1/3/5 and the tuned table, c = 1..128 (preemption
#               expected from ~40 requests: the formula has no memory term)
#   controller  EAGLE's v3 table as an untuned start, with (a) the controller on
#               (draft-model costs and prior) and (b) off
#
# Usage (on the instance, repo root): bash experiments/scripts/run_draft.sh [stage ...]
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
GPU=${GPU:-t4}; RS=${RUN_SUFFIX:-}   # e.g. GPU=h100 RUN_SUFFIX=-r2 (default: the T4 names)
OUT=experiments/results/draft-tuner; [[ $GPU != t4 ]] && OUT=experiments/results/draft-tuner-$GPU
export DRAFT_OUT=$OUT
mkdir -p "$OUT"
export DSD_METHOD=draft_model DRAFTER_MODEL=Qwen/Qwen3-0.6B
BATCH_SIZES="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,40,48,56,64,72,80,88,96,104"
STAGES=${*:-accept costs schedule sweep controller}
has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "=== $* ($(date +%H:%M:%S)) ==="; }

if has accept; then
  log "acceptance, draft model K=5, code prompts, T=1.0"
  DSD_K_MAX=5 python dsd_tuning/tuner/profile_acceptance.py --prompts-file experiments/results/ngram/prompts-write.jsonl \
    --temperature 1.0 --out "$OUT/acceptance.json" 2>&1 | grep -a "^\[accept\]"
fi

if has costs; then
  for k in 0 1 2 3 4 5; do
    log "costs K=$k"
    DSD_K_MAX=5 python dsd_tuning/tuner/profile_costs.py --k "$k" --isl 32 --max-num-seqs 128 \
      --batch-sizes "$BATCH_SIZES" --out "$OUT/costs_k$k.json" > "$OUT/costs_k$k.log" 2>&1 \
      || { echo "!!! costs K=$k failed, see $OUT/costs_k$k.log"; exit 1; }
    grep -aE "\[costs\] K=. B= *(1|16|64|104) " "$OUT/costs_k$k.log"
    grep -ahoE "GPU KV cache size: [0-9,]+ tokens" "$OUT/costs_k$k.log" | head -1
  done
fi

if has schedule; then
  log schedule
  python dsd_tuning/tuner/make_schedule.py --costs "$OUT/costs_k*.json" --acceptance "$OUT/acceptance.json" \
    --out "$OUT" --max-num-seqs 128 --compare-fixed-k | tee "$OUT/schedule.log" | grep -a "\[tuner\]\|K="
fi

export MAX_SEQS=128 TEMPERATURE=1.0
if has sweep; then
  log sweep
  bash experiments/bench/run_configs.sh draft-code-$GPU$RS "draft-k1 draft-k3 draft-k5 draft-tuned" \
    "1 2 4 8 16 32 64 96 128" instruct-coder
fi

if has controller; then
  C="4 16 32 40 64 96"
  log "controller (a): EAGLE's v3 table, draft-model costs and prior"
  env VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000 \
      "VLLM_DSD_COSTS=$OUT/costs_k*.json" VLLM_DSD_PRIOR=$OUT/acceptance.json \
    bash experiments/bench/run_configs.sh draft-ctrl-v3table$([[ $GPU != t4 ]] && echo -$GPU)$RS "draft-v3table" "$C" instruct-coder
  log "controller (b): EAGLE's v3 table, controller off"
  bash experiments/bench/run_configs.sh draft-ctrl-off$([[ $GPU != t4 ]] && echo -$GPU)$RS "draft-v3table" "$C" instruct-coder
fi
log "draft-model pipeline done"
