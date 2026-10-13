#!/usr/bin/env bash
# Experiment 2: open-loop arrivals. AIPerf sends requests at a target rate (Poisson, or bursty gamma arrivals with
# smoothness 0.25) instead of keeping c requests open; same prompts and answer cap as the closed-loop study.
#   STUDY=open RUNS="1 2 3" bash open.sh "nospec k1 k3 tt tt013 ts nj"
set -uo pipefail
source "$(dirname "$0")/common.sh"
CONFIGS=$1
RATES=${RATES:-"2 8 24 48 72"}
PATTERNS=${PATTERNS:-"poisson gamma"}
DUR=${DUR:-60}
EXTRA='{"temperature": 1.0, "chat_template_kwargs": {"enable_thinking": false}}'
log "START configs=[$CONFIGS] runs=[${RUNS:-1}] rates=[$RATES] host=$HOSTNAME"
for r in ${RUNS:-1}; do
  for cfg in $CONFIGS; do
    out=$BX/results/$STUDY/r$r/$cfg
    [[ -f $out/DONE ]] && continue
    config "$cfg" || exit 1
    rm -rf "$out"; mkdir -p "$out"
    log "r$r $cfg start"
    if ! start_server "$out/server.log" "$SPEC" "${CENV[@]}"; then log "!!! r$r $cfg server failed"; stop_server; continue; fi
    ok=1
    for pat in $PATTERNS; do
      for rate in $RATES; do
        n=$(( rate * DUR )); (( n < 120 )) && n=120
        w=$(( rate * 5 )); (( w < 8 )) && w=8
        sm=(); [[ $pat == gamma ]] && sm=(--arrival-smoothness 0.25)
        aiperf profile --model "$TARGET_MODEL" --url localhost:8000 --endpoint-type chat --streaming \
          --public-dataset instruct-coder --dataset-sampling-strategy shuffle --random-seed $((41 + r)) \
          --osl 256 --extra-inputs "$EXTRA" --use-server-token-count --per-chunk-usage \
          --request-rate "$rate" --arrival-pattern "$pat" "${sm[@]}" \
          --request-count "$n" --warmup-request-count "$w" \
          --artifact-dir "$out/$pat-r$rate" --no-gpu-telemetry --ui-type none \
          > "$out/$pat-r$rate.log" 2>&1 || { log "!!! r$r $cfg $pat rate $rate failed"; ok=0; }
      done
    done
    (( ok )) && { touch "$out/DONE"; log "r$r $cfg done"; }
    stop_server
    find "$out" -name profile_export.jsonl -print0 | xargs -0 -r gzip -f
  done
done
log "ALL DONE"
