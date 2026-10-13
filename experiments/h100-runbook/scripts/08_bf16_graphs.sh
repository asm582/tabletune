#!/usr/bin/env bash
# Not in the original runbook. On the H100 the BF16 4B target with any Dynamic SD table runs ~30% below
# fixed length 3 at 1-32 requests. Attribute it: BF16, same server settings as 03_sweep.sh, 3 runs:
#   bf16pw-h100-4b : nospec and fixed k3 forced to PIECEWISE graphs
#   bf16c3-h100-4b : a table with length 3 everywhere (dsd-const3; vLLM switches to PIECEWISE)
# Compare with nospec/k3/eagle-tuned in code-h100-4b (default graphs). Time: ~1 h.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd $REPO
export TEMPERATURE=1.0
C="1 2 4 8 16 32 64"
for r in ${RUNS:-1 2 3}; do
  sfx=$([[ $r == 1 ]] && echo "" || echo "-r$r")
  echo "=== bf16 graphs run $r $(date +%T)"
  COMPILATION_CONFIG='{"cudagraph_mode": "PIECEWISE"}' bash experiments/bench/run_configs.sh bf16pw-$TAG$sfx "nospec k3" "$C" instruct-coder
  bash experiments/bench/run_configs.sh bf16c3-$TAG$sfx "dsd-const3" "$C" instruct-coder
done
echo "BF16 GRAPHS DONE $TAG"
