#!/usr/bin/env bash
# Long prompts on the H100 (V2), Qwen3-4B + EAGLE-3: the tuner re-profiled at the workload's prompt length against
# the table tuned for 32-token prompts, vLLM's documented table and fixed lengths; GovReport summarization at T=1.0,
# answers up to 256 tokens, 3 runs.   ISL=1024 LOADS="1 2 4 8 16 32 64 96 128" MAXB=128 bash long.sh
set -uo pipefail
STUDY=long$ISL; source "$(dirname "$0")/common.sh"
C=$BX/credib; T=$C/tune-long$ISL; P=$C/prompts/long$ISL.jsonl; mkdir -p $T $C/results; PROG=$C/progress.log
export MAX_SEQS=$MAXB
BS=$(python -c "b=[x for x in list(range(1,17))+list(range(20,25,4))+list(range(32,$MAXB+1,8))]; print(','.join(map(str,b)))")
log "START isl=$ISL loads=[$LOADS] host=$HOSTNAME"
cd $REPO
export DSD_METHOD=eagle3 DSD_K_MAX=3 DSD_DTYPE=bfloat16 DRAFTER_MODEL=$EAGLE
[[ -f $T/acceptance.json ]] || python dsd_tuning/tuner/profile_acceptance.py --prompts-file $P --temperature 1.0 \
  --out $T/acceptance.json > $T/acceptance.log 2>&1 || { log "!!! acceptance"; exit 1; }
log "acceptance $(python -c "import json;print(json.load(open('$T/acceptance.json'))['accept_by_pos'])")"
for k in 0 1 2 3; do
  [[ -f $T/costs_k$k.json ]] && continue
  python dsd_tuning/tuner/profile_costs.py --k $k --isl $ISL --max-num-seqs $MAXB --batch-sizes "$BS" --out $T/costs_k$k.json \
    > $T/costs_k$k.log 2>&1 || { log "!!! costs K=$k"; exit 1; }
done
python dsd_tuning/tuner/make_schedule.py --costs "$T/costs_k*.json" --acceptance $T/acceptance.json --out $T --max-num-seqs $MAXB > $T/schedule.log 2>&1
mkdir -p $T/k013; python dsd_tuning/tuner/make_schedule.py --costs "$T/costs_k[013].json" --acceptance $T/acceptance.json --out $T/k013 --max-num-seqs $MAXB > $T/k013/schedule.log 2>&1
TL=$(cat $T/schedule.json)
python - "$TL" <<'PY' || TL=$(cat $T/k013/schedule.json)
import json, sys
ks = [r[2] for r in json.loads(sys.argv[1])]
sys.exit(1 if any(a == 2 and b == 1 for a, b in zip(ks, ks[1:])) else 0)
PY
TS=$EAGLE_TT   # the table tuned for 32-token prompts
log "tables: long $TL | short (32-token) $TS"
export TEMPERATURE=1.0 OSL=256
for r in 1 2 3; do
  for cfg in k1 k2 k3 docs tt-short tt-long; do
    out=$C/results/long$ISL/r$r/$cfg; [[ -f $out/DONE ]] && continue
    case $cfg in k1|k2|k3) SPEC=$(eagle_spec ${cfg#k}) ;; docs) SPEC=$(eagle_spec 3 "[[1,64,3],[65,128,1],[129,512,0]]") ;;
      tt-short) SPEC=$(eagle_spec 3 "$TS") ;; tt-long) SPEC=$(eagle_spec 3 "$TL") ;; esac
    rm -rf "$out"; mkdir -p "$out"; log "r$r $cfg start"
    if start_server "$out/server.log" "$SPEC"; then
      (cd $REPO && bash experiments/bench/sweep.sh "$out" "$LOADS" "file:$P" > "$out/sweep.log" 2>&1) \
        && { touch $out/DONE; log "r$r $cfg done"; } || log "!!! r$r $cfg sweep failed"
    else log "!!! r$r $cfg server failed"; fi
    stop_server; find "$out" -name profile_export.jsonl -print0 | xargs -0 -r gzip -f
  done
done
log "ALL DONE"
