#!/usr/bin/env bash
# Experiment A (Oct 2026): does the controller repair a wrong Dynamic SD table on the H100?
# V1 model runner (the main venv), Qwen3-4B, code at T=1.0, loads 1-256, 3 runs, all in one pod so
# every comparison is paired:
#   ctrlh100-off      docs table and the A10G's table, controller off
#   ctrlh100-refit    the same two tables, controller on (H100 step costs, a mismatched prior:
#                     H100 summarization acceptance; h=8, re-fit every 200 steps as in Section 6.7)
#   ctrlh100-tunedref the H100's own tuned table with the controller on ("no harm" check)
set -uo pipefail
source /work/runbook/scripts/env.sh
cd $REPO
L=/work/out/expA-progress.log
export TEMPERATURE=1.0
PRIOR=$X/acc-h100-4b-summarization/acceptance.json
REFIT=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200 VLLM_DSD_LOG_STEPS=1000
       "VLLM_DSD_COSTS=$TUNE/costs_k*.json" VLLM_DSD_PRIOR=$PRIOR)
A10G=$REPO/tables/a10g-eagle3
echo "### expA START $(date -u +%FT%TZ) host=$HOSTNAME node=${NODE_NAME:-?}" >> $L
for r in ${RUNS:-1 2 3}; do
  sfx=$([[ $r == 1 ]] && echo "" || echo "-r$r")
  echo "### run $r off    $(date -u +%T)" >> $L
  EAGLE_OUT=$A10G bash experiments/bench/run_configs.sh ctrlh100-off$sfx "dsd-docs eagle-tuned" "$CONCS" instruct-coder
  echo "### run $r refit  $(date -u +%T)" >> $L
  EAGLE_OUT=$A10G env "${REFIT[@]}" bash experiments/bench/run_configs.sh ctrlh100-refit$sfx "dsd-docs eagle-tuned" "$CONCS" instruct-coder
  echo "### run $r tuned+ctrl $(date -u +%T)" >> $L
  EAGLE_OUT=$TUNE env "${REFIT[@]}" bash experiments/bench/run_configs.sh ctrlh100-tunedref$sfx "eagle-tuned" "$CONCS" instruct-coder
  for s in ctrlh100-off$sfx ctrlh100-refit$sfx ctrlh100-tunedref$sfx; do python experiments/bench/plot_pareto.py $s > /dev/null 2>&1 || echo "!!! summary $s" >> $L; done
  echo "### run $r DONE   $(date -u +%T)" >> $L
done
for s in ctrlh100-off ctrlh100-refit ctrlh100-tunedref; do python experiments/bench/aggregate_repeats.py $s > /dev/null 2>&1; done
echo "### expA ALL DONE $(date -u +%FT%TZ)" >> $L
