#!/usr/bin/env bash
# Experiment 3, step 1: DFlash acceptance on each Spec-Bench workload and code (greedy, as the workload-switch
# protocol), and the tuner's table for each from the H100 V2 DFlash step costs, lengths 0/2/4/6/8 (no length 1, so
# no 2->1 step, which stalls this vLLM nightly). Then the predicted loss of every stale table (pick_pair.py).
set -uo pipefail
STUDY=drift-screen; source "$(dirname "$0")/common.sh"
D=$BX/drift; mkdir -p $D; cd $X
log START
for w in code translation qa summarization math_reasoning rag; do
  [[ -f $D/acc-$w/schedule.json ]] && continue
  dsd-tuner recalibrate --target $TARGET_MODEL --drafter $DFLASH --method dflash --k-max 8 --dtype $DTYPE \
    --max-num-seqs 128 --temperature 0 --costs "$DFLASH_TUNE/costs_k[02468].json" --prompts-file p_$w.jsonl \
    --out $D/acc-$w > $D/acc-$w.log 2>&1 || log "!!! $w"
  log "$w: $(cat $D/acc-$w/schedule.json 2>/dev/null) tau=$(python -c "import json;print(round(1+sum(json.load(open('$D/acc-$w/acceptance.json'))['accept_by_pos']),2))" 2>/dev/null)"
done
log "ALL DONE"
