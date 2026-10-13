#!/usr/bin/env bash
# Issue #3: the tuner and the controller with n-gram (prompt lookup), end to end
# on a code-apply workload (instruction + code, "reply with only the complete
# modified code"). Same method as the EAGLE-3 v3 pipeline.
#
#   prompts     160 calibration prompts (seed 1234) and a 600-prompt benchmark
#               set (seed 42) as an aiperf dataset
#   accept      n-gram K=5 and, for reference, EAGLE-3 K=3 on the calibration set
#   costs       n-gram step costs, K = 0..5, repeated-block prompts of the
#               benchmark's prompt length (drafted_per_step must equal K)
#   schedule    the tuner's table (with the proposal rate), predicted fixed-K losses,
#               and what the controller should rebuild from the untuned start
#   sweep       no spec, n-gram K=1/3/5, the tuned table, EAGLE-3 K=3; c = 1..128
#   controller  untuned start [[1,16,1],[17,64,3],[65,128,5]] (K in {1,3,5}):
#               (a) calibrated n-gram prior, (b) EAGLE-3's prior (another drafter:
#               3 positions, proposal rate 1), (c) controller off. (a) and (b) must
#               rebuild to the tuner's table restricted to K in {1,3,5}.
#               (d) the tuned v2 table + controller: no harm when the table is right
#
# Usage (on the instance, repo root): bash experiments/scripts/run_ngram.sh [stage ...]
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
AIPERF_PY=~/.local/share/uv/tools/aiperf/bin/python
D=experiments/results/ngram
GPU=${GPU:-t4}; RS=${RUN_SUFFIX:-}   # e.g. GPU=h100 RUN_SUFFIX=-r2 (default: the T4 names)
OUT=experiments/results/ngram-tuner; [[ $GPU != t4 ]] && OUT=experiments/results/ngram-tuner-$GPU
export NGRAM_OUT=$OUT
mkdir -p "$D" "$OUT"
SUFFIX="Reply with only the complete modified code, no explanation."
BATCH_SIZES="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,40,48,56,64,72,80,88,96,104,112,120,128"
CONCS="1 2 4 8 16 32 64 96 128"
export OSL=512 TEMPERATURE=0 MAX_SEQS=128
STAGES=${*:-prompts accept costs schedule sweep tuned2 controller}
has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "=== $* ($(date +%H:%M:%S)) ==="; }

if has prompts; then
  log prompts
  $AIPERF_PY dsd_tuning/tuner/sample_prompts.py --dataset likaixin/InstructCoder --column instruction \
    --append-column input --suffix "$SUFFIX" --n 160 --seed 1234 --out "$D/prompts-apply.jsonl"
  $AIPERF_PY dsd_tuning/tuner/sample_prompts.py --dataset likaixin/InstructCoder --column instruction \
    --append-column input --suffix "$SUFFIX" --n 600 --seed 42 --aiperf --out "$D/bench-apply.jsonl"
fi

if has accept; then
  for spec in "ngram 5 ngram-apply-t0" "eagle3 3 eagle3-apply-t0"; do
    set -- $spec
    log "acceptance $3"
    DSD_METHOD=$1 DSD_K_MAX=$2 python dsd_tuning/tuner/profile_acceptance.py --prompts-file "$D/prompts-apply.jsonl" \
      --temperature 0 --max-tokens $OSL --out "$D/accept-$3.json" 2>&1 | grep -a "^\[accept\]"
  done
  cp "$D/accept-ngram-apply-t0.json" "$OUT/acceptance.json"
fi

if has costs; then
  ISL=$(python -c "import json; print(round(json.load(open('$D/accept-ngram-apply-t0.json'))['isl_avg']))")
  for k in 0 1 2 3 4 5; do
    log "costs K=$k (ISL $ISL)"
    DSD_METHOD=ngram DSD_K_MAX=5 python dsd_tuning/tuner/profile_costs.py --k "$k" --isl "$ISL" --max-num-seqs 128 \
      --batch-sizes "$BATCH_SIZES" --out "$OUT/costs_k$k.json" > "$OUT/costs_k$k.log" 2>&1 \
      || { echo "!!! costs K=$k failed, see $OUT/costs_k$k.log"; exit 1; }
    grep -aE "\[costs\] K=. B= *(1|16|64|128) " "$OUT/costs_k$k.log"
  done
fi

if has schedule; then
  log schedule
  python dsd_tuning/tuner/make_schedule.py --costs "$OUT/costs_k*.json" --acceptance "$OUT/acceptance.json" \
    --out "$OUT" --max-num-seqs 128 --compare-fixed-k | tee "$OUT/schedule.log" | grep -a "\[tuner\]\|K="
  python - <<'PY'
from dsd_tuning.core import load_acceptance, load_costs, make_schedule
import os; d = os.environ["NGRAM_OUT"]
acc, p = load_acceptance(f"{d}/acceptance.json")
costs = load_costs(f"{d}/costs_k*.json")
print("[tuner] expected controller rebuild from K in {1,3,5}:",
      make_schedule(costs, acc, 128, allowed_ks=[1, 3, 5], proposal_rate=p))
PY
fi

if has sweep; then
  log sweep
  bash experiments/bench/run_configs.sh ngram-apply-$GPU$RS "nospec ngram-k1 ngram-k3 ngram-k5 ngram-tuned k3" \
    "$CONCS" "file:$D/bench-apply.jsonl"
fi

if has tuned2; then
  # The first sweep's ngram-tuned was the average-cost table [[1,128,5]], which lost
  # 5-8% to K=3 from 64 requests up; dsd_tuning/tuner/validate_costmodel.py chose the full-cost
  # model and the tuner now builds schedule.json with it.
  log "sweep: tuned table v2 (full-cost model)"
  bash experiments/bench/run_configs.sh ngram-apply-$GPU-v2$RS "ngram-tuned-v2" "$CONCS" "file:$D/bench-apply.jsonl"
fi

if has controller; then
  CTRL=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=4 VLLM_DSD_REFIT_STEPS=200
        VLLM_DSD_LOG_STEPS=1000 "VLLM_DSD_COSTS=$OUT/costs_k*.json")
  C="4 16 32 64 96 128"
  log "controller (a): untuned table, calibrated n-gram prior"
  env "${CTRL[@]}" VLLM_DSD_PRIOR=$OUT/acceptance.json \
    bash experiments/bench/run_configs.sh ngram-ctrl-calib$([[ $GPU != t4 ]] && echo -$GPU)$RS "ngram-untuned" "$C" "file:$D/bench-apply.jsonl"
  log "controller (b): untuned table, EAGLE-3's prior (3 positions, p = 1)"
  env "${CTRL[@]}" VLLM_DSD_PRIOR=experiments/results/tuner-v3-code/acceptance.json \
    bash experiments/bench/run_configs.sh ngram-ctrl-eagleprior$([[ $GPU != t4 ]] && echo -$GPU)$RS "ngram-untuned" "$C" "file:$D/bench-apply.jsonl"
  log "controller (c): untuned table, controller off (reference)"
  bash experiments/bench/run_configs.sh ngram-ctrl-off$([[ $GPU != t4 ]] && echo -$GPU)$RS "ngram-untuned" "$C" "file:$D/bench-apply.jsonl"
  log "controller (d): tuned v2 table, calibrated prior (no-harm check)"
  env "${CTRL[@]}" VLLM_DSD_PRIOR=$OUT/acceptance.json \
    bash experiments/bench/run_configs.sh ngram-ctrl-tuned$([[ $GPU != t4 ]] && echo -$GPU)$RS "ngram-tuned-v2" "$C" "file:$D/bench-apply.jsonl"
fi
log "n-gram pipeline done"
