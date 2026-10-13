#!/usr/bin/env bash
# Runs 2 and 3 of the spec-bench sweeps (issue #10),
# with the original settings (experiments/bench/run_configs.sh defaults: spec-bench,
# c = 1..64, max_num_seqs 64, greedy):
#   baseline-t4-rN   no spec, fixed K=1/2/3
#   dsd-t4-rN        docs table, tuned table v2   (the two other tables of dsd-t4
#                    are not repeated)
# Waits for other GPU pipelines first. Then:
#   python experiments/bench/aggregate_repeats.py baseline-t4 dsd-t4 --mlflow
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
while pgrep -f "run_quant.sh|run_repeats.sh|run_draft.sh|run_dflash.sh|run_gpu_eagle.sh" > /dev/null; do sleep 60; done
source ~/venv/bin/activate
unset MAX_SEQS TEMPERATURE OSL
for r in ${*:-2 3}; do
  echo "=== spec-bench repeat $r ($(date +%H:%M:%S)) ==="
  bash experiments/bench/run_configs.sh baseline-t4-r$r "nospec k1 k2 k3"
  bash experiments/bench/run_configs.sh dsd-t4-r$r "dsd-docs dsd-tuned-v2"
done
echo "=== spec-bench repeats done ($(date +%H:%M:%S)) ==="
