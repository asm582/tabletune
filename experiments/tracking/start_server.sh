#!/usr/bin/env bash
# Start the MLflow tracking server on this laptop (localhost only) and the
# reverse SSH tunnel that lets the GPU instance log to it.
#
#   UI:        http://localhost:5000
#   Data:      ~/mlflow/mlflow.db (runs, params, metrics), ~/mlflow/artifacts
#   Instance:  logs to http://localhost:5000, which the tunnel forwards here
#
# Usage: bash experiments/tracking/start_server.sh        (stop: bash experiments/tracking/stop_server.sh)
set -euo pipefail
MLFLOW_HOME=${MLFLOW_HOME:-$HOME/mlflow}
HOST_ALIAS=${HOST_ALIAS:-vllm-perf}
mkdir -p "$MLFLOW_HOME/artifacts"

if ! curl -sf localhost:5000/health > /dev/null; then
  nohup ~/.local/bin/mlflow server \
    --host 127.0.0.1 --port 5000 \
    --backend-store-uri "sqlite:///$MLFLOW_HOME/mlflow.db" \
    --artifacts-destination "$MLFLOW_HOME/artifacts" --serve-artifacts \
    > "$MLFLOW_HOME/server.log" 2>&1 &
  echo $! > "$MLFLOW_HOME/server.pid"
  for _ in $(seq 1 60); do curl -sf localhost:5000/health > /dev/null && break; sleep 1; done
fi
curl -sf localhost:5000/health > /dev/null && echo "MLflow server up: http://localhost:5000"

# Reverse tunnel: instance localhost:5000 -> laptop localhost:5000.
if ! pgrep -f "ssh -N -R 5000:localhost:5000 $HOST_ALIAS" > /dev/null; then
  nohup ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -R 5000:localhost:5000 "$HOST_ALIAS" > "$MLFLOW_HOME/tunnel.log" 2>&1 &
  echo $! > "$MLFLOW_HOME/tunnel.pid"
  sleep 3
fi
if ssh "$HOST_ALIAS" 'curl -sf localhost:5000/health > /dev/null'; then
  echo "Tunnel up: the instance reaches MLflow at http://localhost:5000"
else
  echo "Tunnel NOT working (is the instance running? see $MLFLOW_HOME/tunnel.log)" >&2
fi
