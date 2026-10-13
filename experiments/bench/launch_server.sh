#!/usr/bin/env bash
# Start a vLLM server for a named config and wait until it is healthy.
# Usage: bash experiments/bench/launch_server.sh <config> <log_file>
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root
source experiments/bench/configs.sh
source ~/venv/bin/activate

CONFIG=$1
LOG=$2
PORT=${PORT:-8000}
SPEC=$(spec_config "$CONFIG")

args=(serve "$TARGET_MODEL"
  --port "$PORT"
  --dtype "${DTYPE:-float16}"     # T4 has no BF16; DTYPE=bfloat16 on Ampere+ (A10G, L4)
  --max-model-len 4096
  --max-num-seqs "${MAX_SEQS:-64}"
  --gpu-memory-utilization 0.90
  --seed 0)
[[ -n "$SPEC" ]] && args+=(--speculative-config "$SPEC")
# Optional, e.g. COMPILATION_CONFIG='{"cudagraph_mode": "PIECEWISE"}' (issue #12). Unset: vLLM default.
[[ -n "${COMPILATION_CONFIG:-}" ]] && args+=(--compilation-config "$COMPILATION_CONFIG")

# Refuse to start next to a leftover server: it would keep the port (so the
# benchmark talks to the wrong server) and the GPU memory.
if curl -sf "localhost:$PORT/health" > /dev/null; then
  echo "[launch] port $PORT already serves something; stop it first" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "[launch] GPU already has processes: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')" >&2
  exit 1
fi

export VLLM_USE_V2_MODEL_RUNNER=${DSD_V2_MODEL_RUNNER:-0}   # MRv1; DSD_V2_MODEL_RUNNER=1 for the V2 study
mkdir -p "$(dirname "$LOG")"
echo "[launch] $CONFIG: vllm ${args[*]}" | tee "$LOG"
# Record controller settings (read by experiments/tracking/log_run.py).
echo "[launch] env: $(env | grep '^VLLM_DSD_' | sort | tr '\n' ' ')" >> "$LOG"
nohup vllm "${args[@]}" >> "$LOG" 2>&1 &
echo $! > "$(dirname "$LOG")/server.pid"

for _ in $(seq 1 120); do            # up to 10 minutes
  if curl -sf "localhost:$PORT/health" > /dev/null; then
    echo "[launch] $CONFIG ready"
    exit 0
  fi
  if ! kill -0 "$(cat "$(dirname "$LOG")/server.pid")" 2> /dev/null; then
    echo "[launch] $CONFIG server died, see $LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "[launch] $CONFIG timed out" >&2
exit 1
