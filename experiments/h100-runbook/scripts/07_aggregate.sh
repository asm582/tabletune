#!/usr/bin/env bash
# After all pods are done: 3-run means for every sweep that has repeats, then pack.
source "$(dirname "$0")/env.sh"
cd $REPO
for s in code-h100-4b code-h100-8b; do
  [[ -d experiments/results/sweeps/$s ]] && python experiments/bench/aggregate_repeats.py $s
done
for p in experiments/results/sweeps/dsdov-h100-*-awq-r1; do
  [[ -d $p ]] || continue
  base=$(basename ${p%-awq-r1})
  python experiments/bench/aggregate_repeats.py $base-awq $base-awq-pw $base-fp16-pw || true
done
ls experiments/results/repeats | grep h100
bash /work/runbook/scripts/06_pack.sh
