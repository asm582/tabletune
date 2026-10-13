#!/usr/bin/env bash
# Stop the laptop MLflow server and the reverse tunnel. Data in ~/mlflow is kept.
MLFLOW_HOME=${MLFLOW_HOME:-$HOME/mlflow}
for f in tunnel server; do
  [[ -f "$MLFLOW_HOME/$f.pid" ]] && kill "$(cat "$MLFLOW_HOME/$f.pid")" 2> /dev/null && echo "stopped $f"
  rm -f "$MLFLOW_HOME/$f.pid"
done
