#!/usr/bin/env bash
# Tuner v3 for the short-prompt code workload at temperature 1.0, then the
# proof sweep: does the tuned table beat every fixed K across 1-128 requests?
#
#   1. Step costs with ~32-token prompts (code prompts average 32) for
#      B = 1..128, K = 0..3. v2's costs used 320-token prompts up to B=64,
#      and the formula then missed the switch to K=1 at high load.
#   2. Acceptance: code prompts at temperature 1.0 (measured in the screening
#      step, experiments/results/screen/code_t1.0.json).
#   3. Table for max_num_seqs=128.
#   4. Sweep K=1/2/3, the v3 table and no-spec at c=1..128, T=1.0.
#
# Usage (on the instance): bash experiments/scripts/run_v3_code.sh
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
OUT=experiments/results/tuner-v3-code
mkdir -p "$OUT"
BATCH_SIZES="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,40,48,56,64,72,80,88,96,104,112,120,128"

for k in 0 1 2 3; do
  echo "=== costs K=$k ($(date +%H:%M:%S)) ==="
  python dsd_tuning/tuner/profile_costs.py --k "$k" --isl 32 --max-num-seqs 128 \
    --batch-sizes "$BATCH_SIZES" --out "$OUT/costs_k$k.json" > "$OUT/costs_k$k.log" 2>&1 \
    || { echo "!!! costs K=$k failed, see $OUT/costs_k$k.log"; exit 1; }
  grep -E "\[costs\] K=. B= *(1|16|64|128) " "$OUT/costs_k$k.log"
done

echo "=== schedule ($(date +%H:%M:%S)) ==="
cp experiments/results/screen/code_t1.0.json "$OUT/acceptance.json"
python dsd_tuning/tuner/make_schedule.py --costs "$OUT/costs_k*.json" --acceptance "$OUT/acceptance.json" \
  --out "$OUT" --max-num-seqs 128 | tee "$OUT/schedule.log" | grep "\[tuner\]"

echo "=== sweep ($(date +%H:%M:%S)) ==="
MAX_SEQS=128 TEMPERATURE=1.0 bash experiments/bench/run_configs.sh code128t-t4 \
  "k1 k2 k3 dsd-tuned-v3 nospec" "1 2 4 8 16 32 64 96 128" instruct-coder
echo "=== v3 pipeline done ($(date +%H:%M:%S)) ==="
