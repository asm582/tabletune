#!/usr/bin/env bash
# Start vLLM in one configuration and wait until healthy (used by 04_shift_vary.sh).
#   serve.sh <log> nospec
#   serve.sh <log> fixed <k>
#   serve.sh <log> table '<json table>' [prior.json costs-glob]   (controller on when a prior is given)
# Extra controller options pass through the environment (VLLM_DSD_EXPLORE, VLLM_DSD_PRIOR_DECAY, ...).
set -uo pipefail
source "$(dirname "$0")/env.sh"
LOG=$1; MODE=$2
args=(serve "$TARGET_MODEL" --dtype "$DTYPE" --max-model-len 4096 --max-num-seqs "$MAX_SEQS" --port 8000 --seed 0)
spec() { echo "{\"method\":\"eagle3\",\"model\":\"$DRAFTER_MODEL\",\"num_speculative_tokens\":$1$2}"; }
case $MODE in
  nospec) ;;
  fixed) args+=(--speculative-config "$(spec "$3" "")") ;;
  table)
    args+=(--speculative-config "$(spec 3 ",\"num_speculative_tokens_per_batch_size\":$3")")
    if [[ -n "${4:-}" ]]; then
      export VLLM_DSD_CONTROLLER=1 VLLM_DSD_PRIOR="$4" VLLM_DSD_COSTS="${5:?costs glob required with a prior}"
    fi ;;
esac
pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null
for _ in $(seq 1 30); do [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break; sleep 2; done
echo "[serve] vllm ${args[*]} controller=${VLLM_DSD_CONTROLLER:-0} env=$(env | grep '^VLLM_DSD_' | sort | tr '\n' ' ')" > "$LOG"
nohup vllm "${args[@]}" >> "$LOG" 2>&1 &
for _ in $(seq 1 180); do
  curl -sf localhost:8000/health > /dev/null && { echo "[serve] ready"; exit 0; }
  sleep 5
done
echo "[serve] timeout, see $LOG"; exit 1
