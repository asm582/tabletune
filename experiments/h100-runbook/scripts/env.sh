# shellcheck shell=bash
# Source this in every shell inside the pod:  source /work/runbook/scripts/env.sh
# Everything lives on the persistent volume mounted at /work, so a pod restart loses nothing.
export HOME=/work                         # OpenShift runs as an arbitrary UID; HOME must be writable
export HF_HOME=/work/hf                   # model cache (about 60 GB for all models below)
export XDG_CACHE_HOME=/work/.cache
# Compile caches are per pod: parallel pods on one shared volume must not write the same cache files
export VLLM_CACHE_ROOT=/work/.cache/vllm-${HOSTNAME:-local}
export TRITON_CACHE_DIR=/work/.cache/triton-${HOSTNAME:-local}
export PATH=/work/.local/bin:$PATH        # uv and aiperf
export REPO=/work/tabletune         # the project repository
export X=/work/x                          # prompts, schedules and shift/vary results
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}   # V1 by default (the V2 matrix presets 1)
export DTYPE=bfloat16
export MAX_SEQS=${MAX_SEQS:-256}          # H100 80 GB: the server's --max-num-seqs (128 on T4/A10G)
export CONCS="1 2 4 8 16 32 64 96 128 192 256"
# The tuner's batch-size grid: every size up to 16, then every 8 up to 256
export BATCH_SIZES="$(seq -s, 1 16),$(seq -s, 24 8 256)"
# Pick the model with MODEL=4b (default) or MODEL=8b before sourcing
case "${MODEL:-4b}" in
  4b) export TARGET_MODEL=Qwen/Qwen3-4B DRAFTER_MODEL=AngelSlim/Qwen3-4B_eagle3 AWQ_MODEL=Qwen/Qwen3-4B-AWQ ;;
  8b) export TARGET_MODEL=Qwen/Qwen3-8B DRAFTER_MODEL=AngelSlim/Qwen3-8B_eagle3 AWQ_MODEL=Qwen/Qwen3-8B-AWQ ;;
esac
export TAG=h100-${MODEL:-4b}              # used in every output name
export TUNE=$REPO/experiments/results/tune-$TAG   # the tuner's output folder for this model
DSD_VENV=${DSD_VENV:-/work/venv}; if [[ -f $DSD_VENV/bin/activate ]]; then source $DSD_VENV/bin/activate; fi
