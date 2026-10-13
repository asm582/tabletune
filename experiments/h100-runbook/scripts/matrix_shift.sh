#!/usr/bin/env bash
# H100 matrix, traffic shift (Experiment 1) and jumping load (Experiment 2) with the A10G's protocol:
# --max-num-seqs 128, the 128-grid H100 step costs (tuner-eagle-h100), per-workload greedy acceptance, the A10G's
# jumping-load schedule (4,96,16,128,8,64,2,112,32,128,4) and shift at the load pick_conc.py names (A10G: 80).
# The load comes from two gen.py processes running the same schedule at half concurrency each (one process cannot
# keep up with no speculation on the H100); throughput is read from the server's counters, so it is the total.
set -uo pipefail
G=h100; TAG=h100m
if [[ ${MATRIX_V2:-0} == 1 ]]; then G=h100v2; TAG=h100v2m; export HOME=/work/v2n VLLM_USE_V2_MODEL_RUNNER=1 DSD_VENV=/work/v2n/venv; else export HOME=/work VLLM_USE_V2_MODEL_RUNNER=0; fi
export HF_HOME=/work/hf XDG_CACHE_HOME=/work/.cache PATH=/work/.local/bin:$PATH
export VLLM_CACHE_ROOT=/work/.cache/vllm-$HOSTNAME TRITON_CACHE_DIR=/work/.cache/triton-$HOSTNAME
export DTYPE=bfloat16 MAX_SEQS=128
export TARGET_MODEL=Qwen/Qwen3-4B DRAFTER_MODEL=AngelSlim/Qwen3-4B_eagle3
source $HOME/venv/bin/activate
REPO=/work/tabletune; S=/work/runbook/scripts; X=/work/x
COSTDIR=$REPO/experiments/results/tuner-eagle-$G; COSTS="$COSTDIR/costs_k*.json"
ACC0=$TAG   # where the per-workload acceptance profiles live
if [[ -n "${KSET:-}" ]]; then TAG=${TAG}${KSET}; COSTS="$COSTDIR/costs_k[${KSET}].json"; fi
L=/work/out/matrix-progress.log; log() { echo "### [shift/$G] $* $(date -u +%FT%TZ)" >> $L; }
log START; until ls $COSTDIR/costs_k3.json > /dev/null 2>&1; do sleep 60; done; log "costs ready"
cd $X && mkdir -p res-$TAG
RUNS=${RUNS:-1 2 3}
if [[ " $RUNS " != *" 1 "* ]]; then   # repeats 2-3 in other pods: wait for repeat 1's pod to build tables and schedules
  until [[ -f sched_vary_$TAG.json && -f res-$TAG/pick_conc.txt ]]; do sleep 60; done; sleep 30
  C=$(python -c "import json; print(2*json.load(open('sched_m2t_$TAG.json'))[0]['conc'])")
else
for w in translation summarization qa math_reasoning rag code codeapply; do
  if [[ $ACC0 != $TAG && -f acc-$ACC0-$w/acceptance.json ]]; then   # restricted lengths: reuse the acceptance profile, rebuild the table
    mkdir -p acc-$TAG-$w; cp acc-$ACC0-$w/acceptance.json acc-$TAG-$w/
    dsd-tuner schedule --costs "$COSTS" --acceptance acc-$TAG-$w/acceptance.json --max-num-seqs 128 --out acc-$TAG-$w > acc-$TAG-$w.log 2>&1
  fi
  [[ -f acc-$TAG-$w/schedule.json ]] || dsd-tuner recalibrate --target $TARGET_MODEL --drafter $DRAFTER_MODEL --dtype $DTYPE \
    --max-num-seqs 128 --temperature 0 --costs "$COSTS" --prompts-file p_$w.jsonl --out acc-$TAG-$w > acc-$TAG-$w.log 2>&1
  log "acceptance $w: $(grep -h 'table:' acc-$TAG-$w.log | tail -1)"
done
python $S/pick_conc.py --repo $REPO --costs "$COSTS" --a acc-$TAG-translation --b acc-$TAG-math_reasoning --max-b 128 > res-$TAG/pick_conc.txt 2>&1
C=$(grep -oE "Use about [0-9]+" res-$TAG/pick_conc.txt | grep -oE "[0-9]+"); C=${SHIFT_CONC:-${C:-80}}; log "shift load $C (pick_conc: $(grep -oE 'Use about [0-9]+' res-$TAG/pick_conc.txt))"
python - <<PY
import json
w = lambda d, c, x: {"dur": d, "conc": c, "work": x}
h = lambda s: [dict(e, conc=max(1, e["conc"] // 2)) for e in s]
m2t = [w(30,$C,"math")] + [w(60,$C,"math")]*3 + [w(60,$C,"translation")]*5
t2m = [w(30,$C,"translation")] + [w(60,$C,"translation")]*3 + [w(60,$C,"math")]*5
cs = [16, 4, 96, 16, 128, 8, 64, 2, 112, 32, 128, 4]   # the A10G schedule (first entry: 30 s warm-up)
vary = [w(30 if i == 0 else 50, c, "code") for i, c in enumerate(cs)]
for n, s in [("m2t", m2t), ("t2m", t2m), ("vary", vary)]:
    json.dump(h(s), open(f"sched_{n}_$TAG.json", "w"))
PY
fi
TT=$(cat acc-$TAG-translation/schedule.json); TM=$(cat acc-$TAG-math_reasoning/schedule.json); TC=$(cat acc-$TAG-code/schedule.json)
DOCS="[[1,64,3],[65,128,1],[129,512,0]]"
WORK="--work math=p_math_reasoning.jsonl --work translation=p_translation.jsonl --work code=p_code.jsonl"
run() {  # run <name> <schedule> <serve args...>
  local name=$1 sched=$2; shift 2
  [[ -f res-$TAG/$name.json ]] && return
  log "run $name"
  bash $S/serve.sh res-$TAG/$name.server.log "$@" || { log "!!! $name server failed"; return; }
  python $S/gen.py --seed $SEED --schedule $sched $WORK --out res-$TAG/$name.json > res-$TAG/$name.txt 2>&1 &
  python $S/gen.py --seed $((SEED + 100)) --schedule $sched $WORK --out res-$TAG/$name.b.json > res-$TAG/$name.b.txt 2>&1
  wait
}
for r in $RUNS; do
  export SEED=$((r-1)); sfx=$([[ $r == 1 ]] && echo "" || echo "_r$r")
  M2T=sched_m2t_$TAG.json; T2M=sched_t2m_$TAG.json; V=sched_vary_$TAG.json
  run e1_m2t_static_math$sfx  $M2T table "$TM"
  run e1_m2t_static_trans$sfx $M2T table "$TT"
  run e1_m2t_ctrl$sfx         $M2T table "$TM" acc-$TAG-math_reasoning/acceptance.json "$COSTS"
  VLLM_DSD_PRIOR_DECAY=0.5 run e1_m2t_ctrl_pd$sfx $M2T table "$TM" acc-$TAG-math_reasoning/acceptance.json "$COSTS"
  VLLM_DSD_EXPLORE=50 VLLM_DSD_EXPLORE_MAX_COST=2.0 VLLM_DSD_PRIOR_DECAY=0.5 run e1_m2t_ctrl_both$sfx $M2T table "$TM" acc-$TAG-math_reasoning/acceptance.json "$COSTS"
  run e1_t2m_static_trans$sfx $T2M table "$TT"
  run e1_t2m_static_math$sfx  $T2M table "$TM"
  run e1_t2m_ctrl$sfx         $T2M table "$TT" acc-$TAG-translation/acceptance.json "$COSTS"
  VLLM_DSD_EXPLORE=50 VLLM_DSD_EXPLORE_MAX_COST=2.0 run e1_t2m_ctrl_explore$sfx $T2M table "$TT" acc-$TAG-translation/acceptance.json "$COSTS"
  VLLM_DSD_PRIOR_DECAY=0.5 run e1_t2m_ctrl_pd$sfx $T2M table "$TT" acc-$TAG-translation/acceptance.json "$COSTS"
  VLLM_DSD_EXPLORE=50 VLLM_DSD_EXPLORE_MAX_COST=2.0 VLLM_DSD_PRIOR_DECAY=0.5 run e1_t2m_ctrl_both$sfx $T2M table "$TT" acc-$TAG-translation/acceptance.json "$COSTS"
  if [[ -z "${E1_ONLY:-}" ]]; then
  run e2_nospec$sfx     $V nospec
  run e2_k1$sfx         $V fixed 1
  run e2_k2$sfx         $V fixed 2
  run e2_tuned$sfx      $V table "$TC"
  run e2_tuned_ctrl$sfx $V table "$TC" acc-$TAG-code/acceptance.json "$COSTS"
  run e2_docs$sfx       $V table "$DOCS"
  fi
  log "repeat $r done"
done
pkill -f "vllm serve" || true
log "ALL DONE"
