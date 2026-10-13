#!/usr/bin/env bash
# The paper's main comparison on the H100, three runs:
# no speculation, fixed lengths 1/2/3, the H100-tuned table, vLLM's docs table, and the A10G's table
# (to test portability), at 1-256 open requests, code prompts at T=1.0.
#   MODEL=4b bash 03_sweep.sh               (needs 02_tune.sh first; runs repeats 1 2 3)
#   MODEL=4b RUNS=2 bash 03_sweep.sh        (one repeat only: one pod per repeat, see README 'Parallel launch')
# Time: about 2-2.5 h per run (6 configs x 11 loads), so about 7 h per model for 3 runs.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd $REPO
export EAGLE_OUT=$TUNE TEMPERATURE=1.0      # eagle-tuned reads $EAGLE_OUT/schedule.json
CONFIGS="nospec k1 k2 k3 eagle-tuned dsd-docs"
for r in ${RUNS:-1 2 3}; do
  sfx=$([[ $r == 1 ]] && echo "" || echo "-r$r")
  echo "=== sweep $TAG run $r $(date +%T)"
  bash experiments/bench/run_configs.sh code-$TAG$sfx "$CONFIGS" "$CONCS" instruct-coder
done
# Portability check (4B only): the A10G's own table on the H100, one run is enough
if [[ ${MODEL:-4b} == 4b && " ${RUNS:-1 2 3} " == *" 1 "* ]]; then
  EAGLE_OUT=$REPO/tables/a10g-eagle3 bash experiments/bench/run_configs.sh code-$TAG-a10gtable "eagle-tuned" "$CONCS" instruct-coder
fi
[[ "${RUNS:-1 2 3}" == "1 2 3" ]] && python experiments/bench/aggregate_repeats.py code-$TAG   # else: 07_aggregate.sh at the end
echo "SWEEP DONE $TAG"
