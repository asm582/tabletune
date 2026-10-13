#!/usr/bin/env bash
# Experiment 3, step 2: a mid-run workload switch with DFlash on the H100 (V2 runner), the A10G switch protocol:
# 30 s warm-up + 3 x 60 s on the old workload, then 5 x 60 s on the new one, greedy, two gen.py clients at half the
# open requests each; throughput from the server's token counter per window.
#   S1 code -> math at 96 open requests (code's table has lengths 2/4/6/8)
#   S2 translation -> math at 48 (translation's table has 2/4/8)
# Configs: the stale table, the new workload's own table (oracle), TableTune's controller (exploration + prior
# decay, the workload-switch defaults), TurboSpec-style and Nightjar-style, all on the old table's lengths.
#   RUNS="1 2 3" SCEN="s1 s2" bash drift.sh
set -uo pipefail
STUDY=drift; source "$(dirname "$0")/common.sh"
D=$BX/drift; S=$RUNBOOK
COSTS="$DFLASH_TUNE/costs_k[02468].json"
GEN_WORK="--work math=$X/p_math_reasoning.jsonl --work translation=$X/p_translation.jsonl --work code=$X/p_code.jsonl"
mksched() {  # mksched <old> <new> <conc> <file>
  python - "$@" <<'PY'
import json, sys
old, new, c, f = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
w = lambda d, x: {"dur": d, "conc": max(1, c // 2), "work": x}
json.dump([w(30, old)] + [w(60, old)] * 3 + [w(60, new)] * 5, open(f, "w"))
PY
}
run() {  # run <scenario> <name> <spec> [env ...]
  local scen=$1 name=$2 spec=$3; shift 3
  local out=$D/res-$scen; mkdir -p $out
  [[ -f $out/$name.json && -f $out/$name.b.json ]] && return
  log "run $scen $name"
  start_server $out/$name.server.log "$spec" "$@" || { log "!!! $scen $name server failed"; return; }
  python $S/gen.py --seed $SEED --schedule $D/sched_$scen.json $GEN_WORK --out $out/$name.json > $out/$name.txt 2>&1 &
  local gp=$!   # wait for this client only: the server is a background job of this shell too
  python $S/gen.py --seed $((SEED + 100)) --schedule $D/sched_$scen.json $GEN_WORK --out $out/$name.b.json > $out/$name.b.txt 2>&1
  wait $gp
  stop_server
}
log "START runs=[${RUNS:-1}] scen=[${SCEN:-s1 s2}] host=$HOSTNAME"
mksched code math 96 $D/sched_s1.json
mksched translation math 48 $D/sched_s2.json
for r in ${RUNS:-1}; do
  export SEED=$((r-1)); sfx=$([[ $r == 1 ]] && echo "" || echo "_r$r")
  for scen in ${SCEN:-s1 s2}; do
    case $scen in s1) old=code ;; s2) old=translation ;; esac
    TO=$(cat $D/acc-$old/schedule.json); TN=$(cat $D/acc-math_reasoning/schedule.json); PO=$D/acc-$old/acceptance.json
    run $scen static_old$sfx "$(dflash_spec 8 "$TO")"
    run $scen static_new$sfx "$(dflash_spec 8 "$TN")"
    run $scen ctrl$sfx "$(dflash_spec 8 "$TO")" VLLM_DSD_CONTROLLER=1 "VLLM_DSD_COSTS=$COSTS" VLLM_DSD_PRIOR=$PO \
      VLLM_DSD_EXPLORE=50 VLLM_DSD_EXPLORE_MAX_COST=2.0 VLLM_DSD_PRIOR_DECAY=0.5 VLLM_DSD_LOG_STEPS=2000
    run $scen ts$sfx "$(dflash_spec 8 "$TO")" VLLM_TS_CONTROLLER=1 VLLM_TS_MODEL=$BX/models/ts_dflash.json VLLM_TS_PRIOR=$PO VLLM_TS_LOG_STEPS=5000
    run $scen nj$sfx "$(dflash_spec 8 "$TO")" VLLM_NJ_CONTROLLER=1 VLLM_NJ_SEED=$r VLLM_NJ_LOG_STEPS=5000
  done
  log "repeat $r done"
done
log "ALL DONE"
