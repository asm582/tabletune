#!/usr/bin/env bash
# The A10G's traffic-shift (Experiment 1) and jumping-load (Experiment 2) runs, repeated on the H100.
#   MODEL=4b SHIFT_CONC=<from pick_conc.py> bash 04_shift_vary.sh
#   RUNS=2 ...  one repeat only (parallel pods);  QUICK=1 ...  skip the single-option controllers
#   and fixed length 2 in repeats 2-3 (run 1 always runs everything)
# Time: about 7 h (Experiment 1: 22 runs x ~10 min; Experiment 2: 18 runs x ~13 min).
set -uo pipefail
source "$(dirname "$0")/env.sh"
S=/work/runbook/scripts
C=${SHIFT_CONC:?run pick_conc.py first and set SHIFT_CONC}
cd $X && mkdir -p res-$TAG
SUF=_r$(echo ${RUNS:-1 2 3} | tr -d " ")      # per-pod schedule files when pods share the volume
AT=$X/acc-$TAG-translation; AM=$X/acc-$TAG-math_reasoning; AC=$X/acc-$TAG-code
TT=$(cat $AT/schedule.json); TM=$(cat $AM/schedule.json); TC=$(cat $AC/schedule.json)
DOCS="[[1,64,3],[65,128,1],[129,512,0]]"
COSTS="$TUNE/costs_k*.json"
python - <<EOF
import json
w = lambda d, c, x: {"dur": d, "conc": c, "work": x}
json.dump([w(30,$C,"math")] + [w(60,$C,"math")]*3 + [w(60,$C,"translation")]*5, open("sched_m2t_$TAG$SUF.json","w"))
json.dump([w(30,$C,"translation")] + [w(60,$C,"translation")]*3 + [w(60,$C,"math")]*5, open("sched_t2m_$TAG$SUF.json","w"))
cs = [16, 4, 192, 16, 256, 8, 128, 2, 224, 64, 256, 4]     # the A10G schedule, scaled to max 256
json.dump([w(30 if i == 0 else 50, c, "code") for i, c in enumerate(cs)], open("sched_vary_$TAG$SUF.json","w"))
EOF
WORK="--work math=p_math_reasoning.jsonl --work translation=p_translation.jsonl --work code=p_code.jsonl"
run() {  # run <name> <schedule> <serve args...>   (SEED from the environment)
  local name=$1 sched=$2; shift 2
  echo "=== $name seed=$SEED $(date +%T)"
  bash $S/serve.sh res-$TAG/$name.server.log "$@" || { echo "!!! $name server failed"; return; }
  python $S/gen.py --seed $SEED --schedule $sched $WORK --out res-$TAG/$name.json > res-$TAG/$name.txt 2>&1 || echo "!!! $name gen failed"
  tail -n +2 res-$TAG/$name.txt | head -12
}
for r in ${RUNS:-1 2 3}; do
  export SEED=$((r-1)); sfx=$([[ $r == 1 ]] && echo "" || echo "_r$r")
  M2T=sched_m2t_$TAG$SUF.json; T2M=sched_t2m_$TAG$SUF.json; V=sched_vary_$TAG$SUF.json
  # Experiment 1: math -> translation
  run e1_m2t_static_math$sfx  $M2T table "$TM"
  run e1_m2t_static_trans$sfx $M2T table "$TT"
  run e1_m2t_ctrl$sfx         $M2T table "$TM" $AM/acceptance.json "$COSTS"
  full=$([[ $r == 1 || -z "${QUICK:-}" ]] && echo 1 || echo "")
  [[ -n $full ]] && VLLM_DSD_PRIOR_DECAY=0.5 run e1_m2t_ctrl_pd$sfx $M2T table "$TM" $AM/acceptance.json "$COSTS"
  VLLM_DSD_EXPLORE=50 VLLM_DSD_EXPLORE_MAX_COST=2.0 VLLM_DSD_PRIOR_DECAY=0.5 run e1_m2t_ctrl_both$sfx $M2T table "$TM" $AM/acceptance.json "$COSTS"
  # Experiment 1: translation -> math
  run e1_t2m_static_trans$sfx $T2M table "$TT"
  run e1_t2m_static_math$sfx  $T2M table "$TM"
  run e1_t2m_ctrl$sfx         $T2M table "$TT" $AT/acceptance.json "$COSTS"
  [[ -n $full ]] && VLLM_DSD_EXPLORE=50 VLLM_DSD_EXPLORE_MAX_COST=2.0 run e1_t2m_ctrl_explore$sfx $T2M table "$TT" $AT/acceptance.json "$COSTS"
  [[ -n $full ]] && VLLM_DSD_PRIOR_DECAY=0.5 run e1_t2m_ctrl_pd$sfx $T2M table "$TT" $AT/acceptance.json "$COSTS"
  VLLM_DSD_EXPLORE=50 VLLM_DSD_EXPLORE_MAX_COST=2.0 VLLM_DSD_PRIOR_DECAY=0.5 run e1_t2m_ctrl_both$sfx $T2M table "$TT" $AT/acceptance.json "$COSTS"
  # Experiment 2: jumping load, code prompts
  run e2_nospec$sfx     $V nospec
  run e2_k1$sfx         $V fixed 1
  [[ -n $full ]] && run e2_k2$sfx         $V fixed 2
  run e2_tuned$sfx      $V table "$TC"
  run e2_tuned_ctrl$sfx $V table "$TC" $AC/acceptance.json "$COSTS"
  run e2_docs$sfx       $V table "$DOCS"
done
pkill -f "vllm serve" || true
echo "SHIFT/VARY DONE $TAG"
