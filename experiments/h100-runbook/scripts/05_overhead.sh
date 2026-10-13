#!/usr/bin/env bash
# The graph-mode cost (paper Section 6.8) on the H100: does a Dynamic SD table (piecewise CUDA
# graphs) cost throughput at low load, and does it depend on how many CPU cores the host gives?
# Run it in the normal pod (16 CPUs) and again in the CPU-limited pod (openshift/pod-cpu4.yaml):
#   MODEL=4b bash 05_overhead.sh            -> sweeps dsdov-h100-4b-cpu<N>-*
# Time: about 1.5 h per pod (3 runs x 3 configs groups x 4 loads, plus model starts).
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd $REPO
N=$(nproc)
# On clusters where the CPU limit is a CFS quota (not a cpuset), nproc shows every host core: use the quota
read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [[ $q != max ]] && (( q / per < N )) && N=$((q / per))
N=${CPUS:-$N}                             # e.g. CPUS=2 taskset -c 0-1 bash 05_overhead.sh
export DSDOV_PREFIX=dsdov-$TAG-cpu$N DSDOV_OUT=experiments/results/dsd-overhead-$TAG-cpu$N
# The script itself sets the AWQ 4B target, FP16, MAX_SEQS=128 and loads 1-8; for the 8B model
# edit AWQ= at the top of experiments/scripts/run_dsd_overhead.sh to Qwen/Qwen3-8B-AWQ first.
bash experiments/scripts/run_dsd_overhead.sh paths cpu
python experiments/bench/aggregate_repeats.py $DSDOV_PREFIX-awq $DSDOV_PREFIX-awq-pw $DSDOV_PREFIX-fp16-pw || true
echo "OVERHEAD DONE $TAG cpu=$N"
