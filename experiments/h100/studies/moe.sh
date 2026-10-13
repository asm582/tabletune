#!/usr/bin/env bash
# Mixture-of-experts study on one H100 (vLLM nightly 43b4aaea, V2 runner): Qwen3-30B-A3B (3B active of 30B) with
# its EAGLE-3 drafter, the matrix protocol of the Qwen3-4B results (code prompts, T=1.0, answers up to 256 tokens,
# loads 1-128, batch cap 128):
#   tune        acceptance (160 code prompts, K=3), step costs K=0..3, the tuned table (and the length-0/1/3 table)
#   sweep       nospec, fixed 1/2/3, vLLM's documented table, the tuned table(s), TurboSpec-style and
#               Nightjar-style, RUNS repeats
#   controller  vLLM's docs table / 4 with a mismatched prior (the 4B Spec-Bench profile), controller on and off
#   STAGES="tune sweep controller" RUNS="1 2 3" bash moe.sh
set -uo pipefail
STUDY=moe; source "$(dirname "$0")/common.sh"
export TARGET_MODEL=Qwen/Qwen3-30B-A3B
EAGLE=AngelSlim/Qwen3-a3B_eagle3
M=$BX/moe; T=$M/tune; mkdir -p $T $M/results
PROG=$M/progress.log
STAGES=${STAGES:-tune sweep controller}
has() { [[ " $STAGES " == *" $1 "* ]]; }
BATCH_SIZES="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,40,48,56,64,72,80,88,96,104,112,120,128"
log "START stages=[$STAGES] runs=[${RUNS:-1}] host=$HOSTNAME"

if has tune; then
  export DSD_METHOD=eagle3 DSD_DTYPE=bfloat16 DSD_K_MAX=3
  cd $REPO
  [[ -f $T/acceptance.json ]] || TARGET_MODEL=$TARGET_MODEL DRAFTER_MODEL=$EAGLE python dsd_tuning/tuner/profile_acceptance.py \
      --prompts-file experiments/results/ngram/prompts-write.jsonl --temperature 1.0 --out $T/acceptance.json > $T/acceptance.log 2>&1 \
    || { log "!!! acceptance"; exit 1; }
  log "acceptance $(python -c "import json;print(json.load(open('$T/acceptance.json'))['accept_by_pos'])")"
  for k in 0 1 2 3; do
    [[ -f $T/costs_k$k.json ]] && continue
    TARGET_MODEL=$TARGET_MODEL DRAFTER_MODEL=$EAGLE python dsd_tuning/tuner/profile_costs.py --k $k --isl 32 --max-num-seqs 128 \
      --batch-sizes "$BATCH_SIZES" --out $T/costs_k$k.json > $T/costs_k$k.log 2>&1 || { log "!!! costs K=$k"; exit 1; }
    log "costs K=$k: $(grep -aE "\[costs\] K=. B= *(1|64|128) " $T/costs_k$k.log | awk '{print $3,$4,$5}' | tr '\n' ' ')"
  done
  python dsd_tuning/tuner/make_schedule.py --costs "$T/costs_k*.json" --acceptance $T/acceptance.json --out $T \
    --max-num-seqs 128 --compare-fixed-k > $T/schedule.log 2>&1
  mkdir -p $T/k013; python dsd_tuning/tuner/make_schedule.py --costs "$T/costs_k[013].json" --acceptance $T/acceptance.json \
    --out $T/k013 --max-num-seqs 128 > $T/k013/schedule.log 2>&1
  (cd $REPO/baselines/turbospec_style && python -m turbospec_style.fit --costs "$T/costs_k*.json" --form sequential --out $T/ts_model.json) > $T/ts_fit.log 2>&1
  log "tables: full $(cat $T/schedule.json) | 0/1/3 $(cat $T/k013/schedule.json) | $(tail -1 $T/ts_fit.log)"
fi

# Configurations for this model (the baseline study's config() with this target, drafter and tables).
TT=$(cat $T/schedule.json 2>/dev/null); TT013=$(cat $T/k013/schedule.json 2>/dev/null)
ARMS='[[1,40,3],[41,100,1],[101,128,0]]'
RESCALED='[[1,16,3],[17,32,1],[33,64,0]]'
config() {
  CENV=()
  case $1 in
    nospec)     SPEC="" ;;
    k1|k2|k3)   SPEC=$(eagle_spec ${1#k}) ;;
    tt)         SPEC=$(eagle_spec 3 "$TT") ;;
    tt013)      SPEC=$(eagle_spec 3 "$TT013") ;;
    ts)         SPEC=$(eagle_spec 3 "$ARMS"); CENV=(VLLM_TS_CONTROLLER=1 VLLM_TS_MODEL=$T/ts_model.json VLLM_TS_PRIOR=$T/acceptance.json) ;;
    nj)         SPEC=$(eagle_spec 3 "$ARMS"); CENV=(VLLM_NJ_CONTROLLER=1) ;;
    docs)       SPEC=$(eagle_spec 3 "[[1,64,3],[65,128,1],[129,512,0]]") ;;   # vLLM docs example (a dense-model table)
    rescaled)   SPEC=$(eagle_spec 3 "$RESCALED") ;;
    rescaled-ctrl) SPEC=$(eagle_spec 3 "$RESCALED"); CENV=(VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=8 VLLM_DSD_REFIT_STEPS=200
                   "VLLM_DSD_COSTS=$T/costs_k*.json" VLLM_DSD_PRIOR=$R/tuner-v2/acceptance.json VLLM_DSD_LOG_STEPS=1000) ;;
    *) echo "unknown config $1" >&2; return 1 ;;
  esac
}
sweep() {  # sweep <name> <configs> <loads>
  local name=$1 cfgs=$2 loads=$3 r cfg out
  for r in ${RUNS:-1}; do
    for cfg in $cfgs; do
      out=$M/results/$name/r$r/$cfg
      [[ -f $out/DONE ]] && continue
      config "$cfg" || exit 1
      rm -rf "$out"; mkdir -p "$out"; log "$name r$r $cfg start"
      if start_server "$out/server.log" "$SPEC" "${CENV[@]}"; then
        (cd $REPO && TEMPERATURE=1.0 OSL=256 bash experiments/bench/sweep.sh "$out" "$loads" instruct-coder > "$out/sweep.log" 2>&1) \
          && { touch "$out/DONE"; log "$name r$r $cfg done"; } || log "!!! $name r$r $cfg sweep failed"
      else log "!!! $name r$r $cfg server failed"; fi
      stop_server
      find "$out" -name profile_export.jsonl -print0 | xargs -0 -r gzip -f
    done
  done
}
# a table that steps from length 2 to 1 stalls this vLLM nightly: then run the length-0/1/3 table as TableTune's
TTCFG=tt; python - "$TT" <<'PY' || TTCFG="tt tt013"
import json, sys
t = json.loads(sys.argv[1]); ks = [r[2] for r in t]
sys.exit(1 if any(a == 2 and b == 1 for a, b in zip(ks, ks[1:])) else 0)
PY
has sweep && sweep closed "${SWEEP_CONFIGS:-nospec k1 k2 k3 docs $TTCFG ts nj}" "1 2 4 8 16 32 64 96 128"
has controller && sweep ctrl "rescaled rescaled-ctrl" "4 16 32 40 64 96"
log "ALL DONE"
