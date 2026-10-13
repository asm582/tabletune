#!/usr/bin/env bash
# Pack every result into one file to copy off the cluster:
#   oc cp <pod>:/work/out/h100-results.tgz ./h100-results.tgz
source "$(dirname "$0")/env.sh"
cd /work
tar czf /work/out/h100-results.tgz \
  out/env-*.txt out/*.log \
  tabletune/experiments/results/tune-h100-* \
  tabletune/experiments/results/sweeps/code-h100-* \
  tabletune/experiments/results/sweeps/dsdov-h100-* \
  tabletune/experiments/results/dsd-overhead-h100-* \
  tabletune/experiments/results/repeats \
  x/acc-h100-* x/res-h100-* x/sched_*h100*.json 2>/dev/null
ls -la /work/out/h100-results.tgz
