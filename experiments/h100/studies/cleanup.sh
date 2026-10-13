#!/usr/bin/env bash
# Stop any vLLM server and load generator in this pod.
tmux kill-server 2>/dev/null
pkill -f gen.py 2>/dev/null
source "$(dirname "$0")/common.sh"
stop_server
nvidia-smi --query-compute-apps=pid --format=csv,noheader
echo cleaned
