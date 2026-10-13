#!/usr/bin/env bash
# What makes K=3 expensive at B=3-5 on the T4? The v2 cost grid shows the 3rd
# draft token adding 12-18% to a step at B=3,4,5 but only ~3-5% at B=2 and
# B=6+. Trace decode steps with the same setup as the tuner and compare GPU
# kernel time per step inside vs. outside the spike.
#
#   B x (K+1) = tokens the target model verifies per step
#   B=4 K=2 -> 12 (cheap)  B=3 K=3 -> 12, B=4 K=3 -> 16, B=5 K=3 -> 20 (spike)
#   B=6 K=3 -> 24 (cheap again)
#
# B=3 K=3 and B=4 K=2 verify the same 12 tokens, so if only the K=3 one is
# slow, the cause is not the target's token count (look at drafter kernels).
# Usage (on the GPU instance, with nothing else running): bash experiments/tools/nsys_spike.sh
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
NSYS=/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys
OUT=experiments/results/nsys
mkdir -p "$OUT"

export VLLM_ENABLE_V1_MULTIPROCESSING=0   # engine in-process, so nsys sees it
export NSYS_CAPTURE=1                      # profile_costs.py brackets the timed loop

for case in "4 2" "3 3" "4 3" "5 3" "6 3"; do
  set -- $case; b=$1; k=$2
  name=b${b}_k${k}
  echo "=== $name: $((b * (k + 1))) tokens/step ==="
  $NSYS profile -o "$OUT/$name" --force-overwrite true \
    --trace cuda,nvtx --cuda-graph-trace node \
    --capture-range cudaProfilerApi --capture-range-end stop \
    python dsd_tuning/tuner/profile_costs.py --k "$k" --batch-sizes "$b" --max-num-seqs 64 --out "$OUT/$name.json" \
    > "$OUT/$name.log" 2>&1 || { echo "!!! $name failed, see $OUT/$name.log"; continue; }
  grep "\[costs\]" "$OUT/$name.log"
  $NSYS stats --report cuda_gpu_kern_sum --format csv --output "$OUT/${name}" \
    "$OUT/$name.nsys-rep" > /dev/null 2>&1
done
echo "=== done: compare $OUT/*_cuda_gpu_kern_sum.csv ==="
