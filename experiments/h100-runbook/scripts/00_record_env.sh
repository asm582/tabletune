#!/usr/bin/env bash
# Record the hardware and software this run used. Keep every line of it with the results.
source "$(dirname "$0")/env.sh"
mkdir -p /work/out
{
  echo "date: $(date -u +%FT%TZ)"
  echo "pod: ${HOSTNAME:-unknown}  run: ${RUN_ID:-}  node: ${NODE_NAME:-unknown}"
  nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit,mig.mode.current --format=csv
  nvidia-smi -q | grep -iE "Product Name|CUDA Version|Persistence|Compute Mode|MIG Mode" | head
  echo "nproc (CPUs this container may use): $(nproc)"
  echo "cgroup CPU quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz"
  free -g | head -2
  python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)" 2>/dev/null
  git -C /work/vllm log -1 --format="vllm commit %H" 2>/dev/null
} | tee /work/out/env-${HOSTNAME:-pod}-$(date -u +%Y%m%d-%H%M).txt
