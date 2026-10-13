#!/usr/bin/env bash
# Experiment 1b: jumping load (open requests jump between 2 and 128 every 50 s, the A10G schedule, code prompts,
# greedy), EAGLE-3 on the H100 V2 runner, with the V2 matrix's protocol (two gen.py clients at half the requests,
# throughput from the server's counters per window): TableTune (table, and table + controller) against the
# TurboSpec-style and Nightjar-style controllers, with no speculation and fixed length 2 as references.
#   RUNS="1 2 3" bash vary.sh
set -uo pipefail
STUDY=vary; source "$(dirname "$0")/common.sh"
S=$RUNBOOK; D=$BX/vary; mkdir -p $D
SCHED=$TABLES/sched_vary_h100v2m.json   # 11 x 50 s windows, per client (2 clients)
W=$REPO/experiments/results/workload-acceptance-h100v2
TT=$(cat $W/acc-h100v2m-code/schedule.json)          # greedy code, all lengths: [[1,51,3],[52,128,2]]
TT013=$(cat $W/acc-h100v2m013-code/schedule.json)    # greedy code, lengths 0/1/3: [[1,73,3],[74,128,1]]
PRIOR=$W/acc-h100v2m-code/acceptance.json
GEN_WORK="--work code=$X/p_code.jsonl"
run() {  # run <name> <spec> [env ...]
  local name=$1 spec=$2; shift 2
  [[ -f $D/$name.json && -f $D/$name.b.json ]] && return
  log "run $name"
  start_server $D/$name.server.log "$spec" "$@" || { log "!!! $name server failed"; return; }
  python $S/gen.py --seed $SEED --schedule $SCHED $GEN_WORK --out $D/$name.json > $D/$name.txt 2>&1 &
  local gp=$!   # wait for this client only: the server is a background job of this shell too
  python $S/gen.py --seed $((SEED + 100)) --schedule $SCHED $GEN_WORK --out $D/$name.b.json > $D/$name.b.txt 2>&1
  wait $gp
  stop_server
}
log "START runs=[${RUNS:-1}] host=$HOSTNAME"
for r in ${RUNS:-1}; do
  export SEED=$((r-1)); sfx=$([[ $r == 1 ]] && echo "" || echo "_r$r")
  run nospec$sfx ""
  run k2$sfx "$(eagle_spec 2)"
  run tt$sfx "$(eagle_spec 3 "$TT")"
  run tt013$sfx "$(eagle_spec 3 "$TT013")"
  run tt013_ctrl$sfx "$(eagle_spec 3 "$TT013")" VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200 \
    "VLLM_DSD_COSTS=$EAGLE_TUNE/costs_k[013].json" VLLM_DSD_PRIOR=$PRIOR
  run ts$sfx "$(eagle_spec 3 "$EAGLE_ARMS")" VLLM_TS_CONTROLLER=1 VLLM_TS_MODEL=$BX/models/ts_eagle.json VLLM_TS_PRIOR=$PRIOR
  run nj$sfx "$(eagle_spec 3 "$EAGLE_ARMS")" VLLM_NJ_CONTROLLER=1 VLLM_NJ_SEED=$r
  log "repeat $r done"
done
log "ALL DONE"
