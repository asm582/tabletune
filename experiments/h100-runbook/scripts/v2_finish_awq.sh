#!/usr/bin/env bash
# V2 matrix, finishing AWQ: (1) repeat 3 of the full-table sweep, each config capped at 25 min (the full AWQ table steps
# from length 2 to 1, which stalls this vLLM nightly's V2 runner); (2) the tuner's table restricted to lengths 0/1/3,
# built by `dsd-tuner schedule` from the same profiles, 3 runs.
set -uo pipefail
export HOME=/work/v2n VLLM_USE_V2_MODEL_RUNNER=1 DSD_V2_MODEL_RUNNER=1 HF_HOME=/work/hf XDG_CACHE_HOME=/work/.cache
export VLLM_CACHE_ROOT=/work/.cache/vllm-$HOSTNAME TRITON_CACHE_DIR=/work/.cache/triton-$HOSTNAME
export DTYPE=float16 DSD_DTYPE=float16 TARGET_MODEL=Qwen/Qwen3-4B-AWQ MAX_SEQS=128 TEMPERATURE=1.0
source $HOME/venv/bin/activate; cd /work/tabletune
L=/work/out/matrix-progress.log; log() { echo "### [awq-finish/h100v2] $* $(date -u +%FT%TZ)" >> $L; }
T=experiments/results/tuner-eagle-h100v2-awq
capped() { # capped <sweep> <config> [EAGLE_OUT]
  EAGLE_OUT=${3:-$T} timeout 25m bash experiments/bench/run_configs.sh "$1" "$2" "1 2 4 8 16 32 64 96 128" instruct-coder \
    || log "!!! $1/$2 timed out or failed"
  pkill -f "vllm serve" 2>/dev/null; sleep 10
}
log START
# (1) repeat 3, config by config into one sweep folder
for cfg in nospec k1 k2 k3 eagle-tuned dsd-tuned-v3; do capped code128t-h100v2-awq-r3-$cfg $cfg; done
# (2) restricted table
dsd-tuner schedule --costs "$T/costs_k[013].json" --acceptance $T/acceptance.json --max-num-seqs 128 --out ${T}-k013 > /dev/null 2>&1
log "restricted table $(cat ${T}-k013/schedule.json)"
for r in "" -r2 -r3; do capped code128t-h100v2-awq-k013$r eagle-tuned ${T}-k013; python experiments/bench/plot_pareto.py code128t-h100v2-awq-k013$r > /dev/null 2>&1; done
for cfg in nospec k1 k2 k3 eagle-tuned dsd-tuned-v3; do python experiments/bench/plot_pareto.py code128t-h100v2-awq-r3-$cfg > /dev/null 2>&1; done
find experiments/results/sweeps -name profile_export.jsonl -newer /work/out/.awq-finish-start -print0 | xargs -0 -r -P 8 -n 8 gzip -f
log "ALL DONE"
