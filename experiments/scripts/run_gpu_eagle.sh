#!/usr/bin/env bash
# The EAGLE-3 tuner and controller on another GPU (e.g. the A10G), same workload
# as the T4 v3 pipeline (instruct-coder, ~32-token prompts, T=1.0, max_num_seqs 128).
#
#   accept      EAGLE-3 K=3 on 160 code prompts at T=1.0 on this GPU and dtype
#   costs       step costs K = 0..3, ISL 32, batch sizes 1..128
#   schedule    the tuner's table for this GPU
#   sweep       no spec, fixed K=1/2/3, this GPU's table, and the T4's v3 table
#               (does a table tuned on one GPU transfer to another?)
#   controller  the untuned docs tables (as in ctrlval on the T4): controller on
#               (this GPU's costs, spec-bench prior) and off
#
# Usage (on the instance, repo root; vLLM with this package installed):
#   GPU=a10g DTYPE=bfloat16 bash experiments/scripts/run_gpu_eagle.sh [stage ...]
#   GPU=a10g SUFFIX=-r2 bash experiments/scripts/run_gpu_eagle.sh sweep controller   repeat run 2 (sweeps named *-r2)
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
GPU=${GPU:?set GPU, e.g. a10g}
export DTYPE=${DTYPE:-bfloat16} DSD_DTYPE=${DTYPE:-bfloat16}
export DSD_METHOD=eagle3 DRAFTER_MODEL=AngelSlim/Qwen3-4B_eagle3
export TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}   # e.g. Qwen/Qwen3-4B-AWQ (GPU=t4-awq)
echo "[run_gpu_eagle] GPU=$GPU target=$TARGET_MODEL dtype=$DTYPE"
OUT=experiments/results/tuner-eagle-$GPU
export EAGLE_OUT=$OUT
mkdir -p "$OUT"
BATCH_SIZES="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,40,48,56,64,72,80,88,96,104,112,120,128"
SUFFIX=${SUFFIX:-}   # -r2, -r3 for repeat runs of the sweep and controller stages
STAGES=${*:-accept costs schedule sweep controller}
has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "=== $* ($(date +%H:%M:%S)) ==="; }

if has accept; then
  log "acceptance, EAGLE-3 K=3, code prompts, T=1.0, $DTYPE"
  DSD_K_MAX=3 python dsd_tuning/tuner/profile_acceptance.py --prompts-file experiments/results/ngram/prompts-write.jsonl \
    --temperature 1.0 --out "$OUT/acceptance.json" 2>&1 | grep -a "^\[accept\]"
fi

if has costs; then
  for k in 0 1 2 3; do
    log "costs K=$k"
    DSD_K_MAX=3 python dsd_tuning/tuner/profile_costs.py --k "$k" --isl 32 --max-num-seqs 128 \
      --batch-sizes "$BATCH_SIZES" --out "$OUT/costs_k$k.json" > "$OUT/costs_k$k.log" 2>&1 \
      || { echo "!!! costs K=$k failed, see $OUT/costs_k$k.log"; exit 1; }
    grep -aE "\[costs\] K=. B= *(1|16|64|128) " "$OUT/costs_k$k.log"
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
  bash experiments/bench/run_configs.sh code128t-$GPU$SUFFIX "nospec k1 k2 k3 eagle-tuned dsd-tuned-v3" \
    "1 2 4 8 16 32 64 96 128" instruct-coder
fi

if has controller; then
  C="4 16 32 40 64 96"
  log "controller on: untuned docs tables, this GPU's costs, spec-bench prior"
  env VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000 \
      "VLLM_DSD_COSTS=$OUT/costs_k*.json" VLLM_DSD_PRIOR=experiments/results/tuner-v2/acceptance.json \
    bash experiments/bench/run_configs.sh ctrlval-refit-$GPU$SUFFIX "dsd-docs dsd-rescaled" "$C" instruct-coder
  log "controller off: untuned docs tables"
  bash experiments/bench/run_configs.sh ctrlval-off-$GPU$SUFFIX "dsd-docs dsd-rescaled" "$C" instruct-coder
fi
log "EAGLE-3 pipeline on $GPU done"
