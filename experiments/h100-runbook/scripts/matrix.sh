#!/usr/bin/env bash
# H100 matrix: rerun each T4/A10G study on the H100 with the same pipeline script, loads (1-128), batch cap (128),
# vLLM commit (353e491ef, V1 runner) and prompts, so the three GPUs compare one to one.
#   bash matrix.sh <study>     study: eagle awq ngram draft dflash specbench
set -uo pipefail
S=$1
# MATRIX_V2=1: vLLM nightly with the V2 runner (full CUDA graphs per draft length), in its own HOME/venv; outputs tagged h100v2
G=${MATRIX_GPU:-h100}
if [[ ${MATRIX_V2:-0} == 1 ]]; then
  G=${MATRIX_GPU:-h100v2}; export HOME=/work/v2n VLLM_USE_V2_MODEL_RUNNER=1 DSD_V2_MODEL_RUNNER=1
else
  export HOME=/work VLLM_USE_V2_MODEL_RUNNER=0
fi
export HF_HOME=/work/hf XDG_CACHE_HOME=/work/.cache PATH=/work/.local/bin:$PATH
export VLLM_CACHE_ROOT=/work/.cache/vllm-$HOSTNAME TRITON_CACHE_DIR=/work/.cache/triton-$HOSTNAME
export DTYPE=bfloat16 DSD_DTYPE=bfloat16
source $HOME/venv/bin/activate
cd /work/tabletune
L=/work/out/matrix-progress.log
log() { echo "### [$S/$G] $* $(date -u +%FT%TZ)" >> $L; }
summ() { for s in "$@"; do python experiments/bench/plot_pareto.py $s > /dev/null 2>&1 || log "!!! summary $s"; done; }
P=experiments/scripts
log START
MARK=/work/out/.matrix-start-$S-$G; touch $MARK
avail=$(df --output=avail -BG /work | tail -1 | tr -dc 0-9); log "volume free ${avail}G"
(( avail < 20 )) && { log "!!! less than 20G free on /work, stopping"; exit 1; }
case $S in
  eagle)
    GPU=$G bash $P/run_gpu_eagle.sh accept costs schedule sweep controller
    summ code128t-$G ctrlval-refit-$G ctrlval-off-$G
    for r in -r2 -r3; do GPU=$G SUFFIX=$r bash $P/run_gpu_eagle.sh sweep controller; summ code128t-$G$r ctrlval-refit-$G$r ctrlval-off-$G$r; log "repeat $r done"; done ;;
  awq)
    export DTYPE=float16 DSD_DTYPE=float16
    GPU=$G-awq TARGET_MODEL=Qwen/Qwen3-4B-AWQ bash $P/run_gpu_eagle.sh accept costs schedule sweep controller
    summ code128t-$G-awq ctrlval-refit-$G-awq ctrlval-off-$G-awq
    for r in -r2 -r3; do GPU=$G-awq TARGET_MODEL=Qwen/Qwen3-4B-AWQ SUFFIX=$r bash $P/run_gpu_eagle.sh sweep; summ code128t-$G-awq$r; log "repeat $r done"; done ;;
  ngram)
    GPU=$G bash $P/run_ngram.sh accept costs schedule sweep tuned2 controller
    summ ngram-apply-$G ngram-apply-$G-v2 ngram-ctrl-calib-$G ngram-ctrl-eagleprior-$G ngram-ctrl-off-$G ngram-ctrl-tuned-$G
    for r in -r2 -r3; do GPU=$G RUN_SUFFIX=$r bash $P/run_ngram.sh sweep tuned2; summ ngram-apply-$G$r ngram-apply-$G-v2$r; log "repeat $r done"; done ;;
  draft)
    GPU=$G bash $P/run_draft.sh accept costs schedule sweep controller
    summ draft-code-$G draft-ctrl-v3table-$G draft-ctrl-off-$G
    for r in -r2 -r3; do GPU=$G RUN_SUFFIX=$r bash $P/run_draft.sh sweep; summ draft-code-$G$r; log "repeat $r done"; done ;;
  dflash)
    GPU=$G bash $P/run_dflash.sh accept costs schedule sweep controller
    summ dflash-code-$G dflash-ctrl-v3table-$G dflash-ctrl-off-$G
    for r in -r2 -r3; do GPU=$G SUFFIX=$r bash $P/run_dflash.sh sweep controller; summ dflash-code-$G$r dflash-ctrl-v3table-$G$r dflash-ctrl-off-$G$r; log "repeat $r done"; done ;;
  specbench)   # the T4 Spec-Bench study: greedy chat, default loads 1-64, batch cap 64 (run_configs defaults)
    unset MAX_SEQS TEMPERATURE OSL
    bash experiments/bench/run_configs.sh baseline-$G "nospec k1 k2 k3"; summ baseline-$G
    bash $P/run_tuner.sh experiments/results/sweeps/baseline-$G experiments/results/tuner-v2-$G > /work/out/matrix-specbench-$G-tuner.log 2>&1 || log "!!! tuner"
    EAGLE_OUT=experiments/results/tuner-v2-$G bash experiments/bench/run_configs.sh dsd-$G "dsd-docs eagle-tuned"; summ dsd-$G
    for r in 2 3; do
      bash experiments/bench/run_configs.sh baseline-$G-r$r "nospec k1 k2 k3"
      EAGLE_OUT=experiments/results/tuner-v2-$G bash experiments/bench/run_configs.sh dsd-$G-r$r "dsd-docs eagle-tuned"
      summ baseline-$G-r$r dsd-$G-r$r; log "repeat $r done"
    done ;;
esac
# keep the volume from filling: compress this study's per-request AIPerf records (summaries stay uncompressed)
find experiments/results/sweeps -newer $MARK -name profile_export.jsonl -print0 | xargs -0 -r -P 8 -n 8 gzip -f
log "ALL DONE"
