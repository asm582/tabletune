# shellcheck shell=bash
# Shared setup for the H100 studies on vLLM nightly 43b4aaea with the V2 runner (adaptive baselines, open loop,
# jumping load, workload switch, mixture of experts, long prompts, DFlash profiling). Source it from a study script;
# then start_server <log> <spec-json-or-empty> [ENV=VALUE ...] and stop_server.
# Paths: WORK is the persistent volume (default /work) with the model cache, the V2 environment that
# experiments/h100-runbook/scripts/setup_v2.sh builds in $WORK/v2n, and the prompt files that make_prompts.py writes to $WORK/x.
# Results go to $OUT (default $WORK/exp-h100).
STUDIES=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$STUDIES/../../.." && pwd)
WORK=${WORK:-/work}
export HOME=$WORK/v2n VLLM_USE_V2_MODEL_RUNNER=1 DSD_V2_MODEL_RUNNER=1
export HF_HOME=$WORK/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$WORK/.cache PATH=$WORK/v2n/.local/bin:$WORK/.local/bin:$PATH
export VLLM_CACHE_ROOT=$WORK/.cache/vllm-$HOSTNAME TRITON_CACHE_DIR=$WORK/.cache/triton-$HOSTNAME
export DTYPE=bfloat16 MAX_SEQS=${MAX_SEQS:-128}
export TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
# shellcheck disable=SC1091
source "$WORK/v2n/venv/bin/activate"
BX=${OUT:-$WORK/exp-h100}
TABLES=$REPO/experiments/h100/tables       # committed inputs: tables and fitted TurboSpec-style models
RUNBOOK=$REPO/experiments/h100-runbook/scripts          # gen.py, the jumping-load and workload-switch client
X=$WORK/x                                   # prompt files from make_prompts.py and the V2 matrix's acceptance profiles
mkdir -p "$BX"
R=$REPO/experiments/results
EAGLE_TUNE=$R/tuner-eagle-h100v2
DFLASH_TUNE=$R/dflash-tuner-h100v2
EAGLE=AngelSlim/Qwen3-4B_eagle3
DFLASH=z-lab/Qwen3-4B-DFlash-b16
PROG=$BX/progress.log
log() { echo "### [${STUDY:-?}] $* $(date -u +%FT%TZ)" >> $PROG; }

eagle_spec() {  # eagle_spec <K> [table]
  local t=""; [[ -n "${2:-}" ]] && t=", \"num_speculative_tokens_per_batch_size\": $2"
  echo "{\"method\": \"eagle3\", \"model\": \"$EAGLE\", \"num_speculative_tokens\": $1$t}"
}
dflash_spec() {
  local t=""; [[ -n "${2:-}" ]] && t=", \"num_speculative_tokens_per_batch_size\": $2"
  echo "{\"method\": \"dflash\", \"model\": \"$DFLASH\", \"num_speculative_tokens\": $1$t}"
}

stop_server() {
  pkill -f "vllm serve" 2> /dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm serve|VLLM::EngineCore" > /dev/null || break; sleep 2; done
  pkill -9 -f "vllm serve" 2> /dev/null; pkill -9 -f "VLLM::EngineCore" 2> /dev/null
  for _ in $(seq 1 30); do [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break; sleep 2; done
  sleep 3
}

start_server() {  # start_server <log> <spec json or ""> [VAR=value ...]  (controller settings go in the env list)
  local logf=$1 spec=$2; shift 2
  stop_server
  local args=(serve "$TARGET_MODEL" --port 8000 --dtype "$DTYPE" --max-model-len 4096 --max-num-seqs "$MAX_SEQS"
              --gpu-memory-utilization 0.90 --seed 0)
  [[ -n "$spec" ]] && args+=(--speculative-config "$spec")
  mkdir -p "$(dirname "$logf")"
  echo "[start] vllm ${args[*]} | env: $*" > "$logf"
  env "$@" nohup vllm "${args[@]}" >> "$logf" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do
    curl -sf localhost:8000/health > /dev/null && return 0
    kill -0 $pid 2> /dev/null || { echo "[start] server died" >> "$logf"; return 1; }
    sleep 5
  done
  echo "[start] timeout" >> "$logf"; return 1
}

# Configurations: name -> spec json and controller environment. Starting tables for the baselines hold the lengths
# they may use (V2 captures full CUDA graphs only for the table's lengths).
EAGLE_TT=$(cat $EAGLE_TUNE/schedule.json)                 # TableTune, all profiled lengths: [[1,50,3],[51,128,2]]
EAGLE_TT013=$(cat $TABLES/eagle_tt013.json)            # TableTune from length-0/1/3 profiles: [[1,72,3],[73,128,1]]
EAGLE_ARMS='[[1,40,3],[41,100,1],[101,128,0]]'            # lengths 0/1/3 for the per-step baselines
DFLASH_TT=$(cat $DFLASH_TUNE/schedule.json)               # [[1,21,8],[22,33,6],[34,87,4],[88,128,2]]

config() {  # config <name>: sets SPEC and CENV (array)
  CENV=()
  local TTC=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200)
  case $1 in
    nospec)       SPEC="" ;;
    k1|k2|k3)     SPEC=$(eagle_spec ${1#k}) ;;
    tt)           SPEC=$(eagle_spec 3 "$EAGLE_TT") ;;
    tt013)        SPEC=$(eagle_spec 3 "$EAGLE_TT013") ;;
    tt013-ctrl)   SPEC=$(eagle_spec 3 "$EAGLE_TT013"); CENV=("${TTC[@]}" "VLLM_DSD_COSTS=$EAGLE_TUNE/costs_k[013].json" "VLLM_DSD_PRIOR=${PRIOR:-$EAGLE_TUNE/acceptance.json}") ;;
    ts)           SPEC=$(eagle_spec 3 "$EAGLE_ARMS"); CENV=(VLLM_TS_CONTROLLER=1 VLLM_TS_MODEL=$TABLES/ts_eagle.json "VLLM_TS_PRIOR=${PRIOR:-$EAGLE_TUNE/acceptance.json}" VLLM_TS_LOG_STEPS=5000) ;;
    nj)           SPEC=$(eagle_spec 3 "$EAGLE_ARMS"); CENV=(VLLM_NJ_CONTROLLER=1 VLLM_NJ_LOG_STEPS=5000) ;;
    df-k2|df-k3|df-k4|df-k8) SPEC=$(dflash_spec ${1#df-k}) ;;
    df-tt)        SPEC=$(dflash_spec 8 "$DFLASH_TT") ;;
    df-tt-ctrl)   SPEC=$(dflash_spec 8 "$DFLASH_TT"); CENV=("${TTC[@]}" "VLLM_DSD_COSTS=$DFLASH_TUNE/costs_k*.json" "VLLM_DSD_PRIOR=${PRIOR:-$DFLASH_TUNE/acceptance.json}") ;;
    df-ts)        SPEC=$(dflash_spec 8 "$DFLASH_TT"); CENV=(VLLM_TS_CONTROLLER=1 VLLM_TS_MODEL=$TABLES/ts_dflash.json "VLLM_TS_PRIOR=${PRIOR:-$DFLASH_TUNE/acceptance.json}" VLLM_TS_LOG_STEPS=5000) ;;
    df-nj)        SPEC=$(dflash_spec 8 "$DFLASH_TT"); CENV=(VLLM_NJ_CONTROLLER=1 VLLM_NJ_LOG_STEPS=5000) ;;
    *) echo "unknown config $1" >&2; return 1 ;;
  esac
}
