#!/usr/bin/env bash
# Smoke test on the GPU instance: no-spec, EAGLE-3, then EAGLE-3 + Dynamic SD.
# Usage (on the instance): bash experiments/bench/smoke_test.sh
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
mkdir -p experiments/results/smoke

for mode in nospec eagle3 dsd; do
  echo "=== $mode ==="
  python experiments/bench/smoke_test.py --mode "$mode" 2>&1 | tee "experiments/results/smoke/$mode.log" \
    | grep -E "\[smoke\]|attention backend|Using .* backend|cudagraph|CUDA graph|Overriding|Error|error|Traceback" \
    || echo "!!! $mode failed, see experiments/results/smoke/$mode.log"
done

echo "=== compare ==="
python experiments/bench/smoke_test.py --compare
