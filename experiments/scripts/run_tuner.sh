#!/usr/bin/env bash
# Level-1 tuner: profile step costs for each K, profile acceptance, build the
# Dynamic SD schedule. Usage (on the GPU instance):
#   bash experiments/scripts/run_tuner.sh [sweep_dir_for_calibration_and_oracle] [out_dir]
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
SWEEP=${1:-experiments/results/sweeps/baseline-t4}
OUT=${2:-experiments/results/tuner}
mkdir -p "$OUT"

for k in 0 1 2 3; do
  echo "=== costs K=$k ($(date +%H:%M:%S)) ==="
  python dsd_tuning/tuner/profile_costs.py --k "$k" --max-num-seqs 64 --out "$OUT/costs_k$k.json" \
    --batch-sizes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,36,40,44,48,52,56,60,64 2>&1 \
    | tee "$OUT/costs_k$k.log" | grep -E "\[costs\]|Error|Traceback" \
    || { echo "!!! costs K=$k failed, see $OUT/costs_k$k.log"; exit 1; }
done

echo "=== acceptance ($(date +%H:%M:%S)) ==="
python dsd_tuning/tuner/profile_acceptance.py --sweep "$SWEEP" --out "$OUT/acceptance.json" 2>&1 \
  | tee "$OUT/acceptance.log" | grep -E "\[accept\]|Error|Traceback" \
  || { echo "!!! acceptance failed, see $OUT/acceptance.log"; exit 1; }

echo "=== schedule ==="
python dsd_tuning/tuner/make_schedule.py --costs "$OUT/costs_k*.json" --max-num-seqs 64 \
  --acceptance "$OUT/acceptance.json" --out "$OUT" --oracle "$SWEEP/summary.csv"
echo "=== tuner done ($(date +%H:%M:%S)) ==="
