#!/usr/bin/env bash
# For each config: start server, run the concurrency sweep, stop server.
# Usage: bash experiments/bench/run_configs.sh <run_name> "<configs>" ["<concurrencies>"] [dataset]
# Example: bash experiments/bench/run_configs.sh baseline-t4 "nospec k1 k2 k3"
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root

RUN=$1
CONFIGS=$2
CONCURRENCIES=${3:-"1 2 4 8 16 32 64"}
DATASET=${4:-spec-bench}
ROOT=experiments/results/sweeps/$RUN

stop_server() {
  local pidf=$1/server.pid
  [[ -f $pidf ]] && kill "$(cat "$pidf")" 2> /dev/null
  # vLLM runs the engine in a child process (VLLM::EngineCore) that can outlive
  # the API server and keep the GPU and port; stop both, then wait until the
  # GPU is really free so the next config starts clean.
  for _ in $(seq 1 30); do
    pgrep -f "vllm serve|VLLM::EngineCore" > /dev/null || break
    sleep 2
  done
  pkill -9 -f "vllm serve" 2> /dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2> /dev/null || true
  for _ in $(seq 1 30); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
    sleep 2
  done
  sleep 3
}

mkdir -p "$ROOT"
git -C ~/vllm log -1 --format="vllm %H" > "$ROOT/meta.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$ROOT/meta.txt"
echo "dataset $DATASET | concurrencies $CONCURRENCIES" >> "$ROOT/meta.txt"
echo "max_num_seqs ${MAX_SEQS:-64} | temperature ${TEMPERATURE:-0}" >> "$ROOT/meta.txt"

for cfg in $CONFIGS; do
  dir=$ROOT/$cfg
  mkdir -p "$dir"
  echo "=== $cfg ($(date +%H:%M:%S)) ==="
  if bash experiments/bench/launch_server.sh "$cfg" "$dir/server.log"; then
    bash experiments/bench/sweep.sh "$dir" "$CONCURRENCIES" "$DATASET" || echo "!!! $cfg sweep failed"
  else
    echo "!!! $cfg server failed to start"
  fi
  stop_server "$dir"
done
echo "=== all done ($(date +%H:%M:%S)) ==="

# Experiment tracking: with MLFLOW_TRACKING_URI set (laptop MLflow via the
# reverse tunnel, see experiments/tracking/README.md), chart the sweep and log every point.
# Needs the tracking extra in the active venv: uv pip install -e ".[tracking]".
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  python experiments/bench/plot_pareto.py "$RUN" > /dev/null 2>&1
  python experiments/tracking/log_run.py "$ROOT" 2>&1 | grep "\[track\]" \
    || echo "!!! MLflow logging failed (is the tunnel up?)"
  # Recompute gains vs. baselines and refresh the charts across all sweeps.
  python experiments/tracking/derive.py 2>&1 | grep "comparison metrics" \
    || echo "!!! MLflow derive step failed"
fi
