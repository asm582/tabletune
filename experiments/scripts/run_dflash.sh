#!/usr/bin/env bash
# Issue #7: the tuner and the controller with DFlash (block drafter,
# z-lab/Qwen3-4B-DFlash-b16: drafts all K tokens in one pass), on the code workload the EAGLE-3 v3 table
# was built for (instruct-coder, ~32-token prompts, T=1.0, max_num_seqs 128), so
# code128t-t4 (EAGLE-3 and no speculation) is the direct comparison.
#
#   accept      DFlash K=8 on 160 code-writing prompts at T=1.0
#   costs       step costs K = 0,1,2,3,4,6,8 through the Dynamic SD path, ISL 32, batch sizes
#               up to 128 (reduce if the KV cache is small; check the costs log)
#   schedule    the tuner's table; predicted fixed-K losses
#   sweep       DFlash K=1/3/8 and the tuned table, c = 1..128 (a smaller drafter
#               than the draft model: less KV pressure expected)
#   controller  EAGLE's v3 table as an untuned start (K <= 3 only), with (a) the
#               controller on, DFlash costs and prior, and VLLM_DSD_ANY_K=1 so it can
#               reach deeper K (MRv1, validated in #2), and (b) off
#
# GPU=a10g DTYPE=bfloat16 runs it on another GPU: results go to experiments/results/dflash-tuner-a10g
# and sweeps are named dflash-code-a10g etc. (default GPU=t4, float16).
#
# Usage (on the instance, repo root; vLLM with this package installed):
#   bash experiments/scripts/run_dflash.sh                       all stages (run 1)
#   bash experiments/scripts/run_dflash.sh accept                one stage
#   SUFFIX=-r2 bash experiments/scripts/run_dflash.sh sweep controller   repeat run 2 (sweeps named *-r2)
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
GPU=${GPU:-t4}
export DTYPE=${DTYPE:-float16} DSD_DTYPE=${DTYPE:-float16}
OUT=experiments/results/dflash-tuner; [[ $GPU != t4 ]] && OUT=experiments/results/dflash-tuner-$GPU
export DFLASH_OUT=$OUT
mkdir -p "$OUT"
export DSD_METHOD=dflash DRAFTER_MODEL=z-lab/Qwen3-4B-DFlash-b16
SUFFIX=${SUFFIX:-}   # -r2, -r3 for repeat runs of the sweep and controller stages
BATCH_SIZES="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,40,48,56,64,72,80,88,96,104,112,120,128"
STAGES=${*:-accept costs schedule sweep controller}
has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "=== $* ($(date +%H:%M:%S)) ==="; }

if has accept; then
  log "acceptance, DFlash K=8, code prompts, T=1.0"
  DSD_K_MAX=8 python dsd_tuning/tuner/profile_acceptance.py --prompts-file experiments/results/ngram/prompts-write.jsonl \
    --temperature 1.0 --out "$OUT/acceptance.json" 2>&1 | grep -a "^\[accept\]"
fi

if has costs; then
  for k in 0 1 2 3 4 6 8; do
    log "costs K=$k"
    DSD_K_MAX=8 python dsd_tuning/tuner/profile_costs.py --k "$k" --isl 32 --max-num-seqs 128 \
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
  bash experiments/bench/run_configs.sh dflash-code-$GPU$SUFFIX "dflash-k1 dflash-k3 dflash-k8 dflash-tuned" \
    "1 2 4 8 16 32 64 96 128" instruct-coder
fi

if has controller; then
  C="4 16 32 40 64 96"
  log "controller (a): EAGLE's v3 table, DFlash costs and prior"
  env VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000 \
      "VLLM_DSD_COSTS=$OUT/costs_k*.json" VLLM_DSD_PRIOR=$OUT/acceptance.json \
      VLLM_DSD_ANY_K=1 VLLM_USE_V2_MODEL_RUNNER=0 \
    bash experiments/bench/run_configs.sh dflash-ctrl-v3table-$GPU$SUFFIX "dflash-v3table" "$C" instruct-coder
  log "controller (b): EAGLE's v3 table, controller off"
  bash experiments/bench/run_configs.sh dflash-ctrl-off-$GPU$SUFFIX "dflash-v3table" "$C" instruct-coder
fi
log "DFlash pipeline done"
