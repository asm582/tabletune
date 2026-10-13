#!/usr/bin/env bash
# Experiment B (Oct 2026): with vLLM's V2 model runner, which keeps full CUDA graphs per draft length
# (PR #57053, merged 2026-10-05, commit 74c5cbcd), does a measured table beat every fixed length on the H100?
# Separate environment: HOME=/work/v2home, so ~/venv (used by launch_server.sh) is the V2 venv and the
# V1 setup in /work/venv is untouched. Qwen3-4B, code at T=1.0, loads 1-256, 3 runs.
set -uo pipefail
VC=74c5cbcd7b6fa07fa5568ba7f7dfbfedc502964e
source /work/runbook/scripts/env.sh
export HOME=/work/v2home VLLM_USE_V2_MODEL_RUNNER=1 DSD_V2_MODEL_RUNNER=1   # the latter for launch_server.sh
mkdir -p $HOME
mkdir -p $HOME/.local/bin && ln -sf /work/.local/bin/aiperf $HOME/.local/bin/aiperf   # sweep.sh runs ~/.local/bin/aiperf
L=/work/out/expB-progress.log
log() { echo "### $* $(date -u +%FT%TZ)" >> $L; }
log "expB START host=$HOSTNAME node=${NODE_NAME:-?}"
if [[ ! -x $HOME/venv/bin/vllm ]]; then
  log "install vllm $VC"
  UV=/work/venv/bin/uv               # activating the V2 venv drops the V1 venv (and its uv) from PATH
  $UV venv --python 3.12 $HOME/venv >> /work/out/expB-install.log 2>&1
  source $HOME/venv/bin/activate
  $UV pip install --prerelease=allow --torch-backend=auto vllm --extra-index-url https://wheels.vllm.ai/$VC >> /work/out/expB-install.log 2>&1 || { log "ABORT vllm install failed"; exit 1; }
  $UV pip install -e $REPO matplotlib >> /work/out/expB-install.log 2>&1
fi
source $HOME/venv/bin/activate
python -c "import vllm,torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" >> $L 2>&1
cd $REPO
export TEMPERATURE=1.0
# 1. Re-tune under V2 (full graphs); fall back to the V1-measured H100 table if the tuner fails here
TUNE2=$REPO/experiments/results/tune-h100-4b-v2
if [[ -f $TUNE2/schedule.json ]]; then log "tune already done, reusing $TUNE2"; else
log "tune START"
dsd-tuner run --target "$TARGET_MODEL" --drafter "$DRAFTER_MODEL" \
  --prompts-file docs/examples/code-prompts.jsonl --isl 32 --temperature 1.0 \
  --max-num-seqs $MAX_SEQS --batch-sizes "$BATCH_SIZES" --dtype $DTYPE --out $TUNE2 > /work/out/expB-tune.log 2>&1
fi
if [[ -f $TUNE2/schedule.json ]]; then T=$TUNE2; log "tune DONE table=$(cat $TUNE2/schedule.json)"; else T=$TUNE; log "tune FAILED, using V1 H100 table $(cat $TUNE/schedule.json)"; fi
# 2. Main sweep under V2: no spec, fixed 1/2/3, the table, the docs table
for r in ${RUNS:-1 2 3}; do
  sfx=$([[ $r == 1 ]] && echo "" || echo "-r$r")
  log "sweep run $r START"
  EAGLE_OUT=$T bash experiments/bench/run_configs.sh v2-h100-4b$sfx "nospec k1 k2 k3 eagle-tuned dsd-docs" "$CONCS" instruct-coder
  python experiments/bench/plot_pareto.py v2-h100-4b$sfx > /dev/null 2>&1 || log "!!! summary v2-h100-4b$sfx"
  grep -h "cudagraph\|CUDAGraphMode\|cudagraph_mode" experiments/results/sweeps/v2-h100-4b$sfx/eagle-tuned/server.log 2>/dev/null | head -3 >> $L
  log "sweep run $r DONE"
done
# 3. If re-tuning worked, the V1-measured table once too (does the profile's graph mode matter?)
[[ $T == $TUNE2 ]] && EAGLE_OUT=$TUNE bash experiments/bench/run_configs.sh v2-h100-4b-v1table "eagle-tuned" "$CONCS" instruct-coder \
  && python experiments/bench/plot_pareto.py v2-h100-4b-v1table > /dev/null 2>&1
python experiments/bench/aggregate_repeats.py v2-h100-4b > /dev/null 2>&1
log "expB ALL DONE"
