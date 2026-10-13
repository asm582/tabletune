#!/usr/bin/env bash
# Build summary.csv for every H100 sweep (run_configs.sh only does it when MLFLOW_TRACKING_URI is set),
# then the 3-run means of every repeated sweep. Safe to rerun.
source "$(dirname "$0")/env.sh"
cd $REPO
for d in experiments/results/sweeps/*h100*/; do
  s=$(basename $d)
  python experiments/bench/plot_pareto.py $s > /dev/null 2>&1 || echo "!!! summary failed: $s"
done
for s in code-h100-4b code-h100-8b bf16pw-h100-4b bf16c3-h100-4b; do
  [[ -f experiments/results/sweeps/$s/summary.csv ]] && python experiments/bench/aggregate_repeats.py $s > /dev/null
done
for p in experiments/results/sweeps/dsdov-h100-*-awq-r1; do
  [[ -d $p ]] || continue
  base=$(basename ${p%-awq-r1})
  python experiments/bench/aggregate_repeats.py $base-awq $base-awq-pw $base-fp16-pw > /dev/null || true
done
ls experiments/results/repeats | grep h100
