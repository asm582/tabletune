#!/usr/bin/env bash
# Experiment 1: TableTune against TurboSpec-style and Nightjar-style control, closed-loop AIPerf sweep with the
# matrix protocol (InstructCoder prompts, T=1.0, answers up to 256 tokens, loads 1-128, 4c requests after c warm-up).
#   STUDY=eagle RUNS="1 2 3" bash closed.sh "nospec k1 k2 k3 tt tt013 tt013-ctrl ts nj"
#   STUDY=dflash RUNS="1 2 3" bash closed.sh "df-k3 df-k4 df-k8 df-tt df-tt-ctrl df-ts df-nj"
set -uo pipefail
source "$(dirname "$0")/common.sh"
CONFIGS=$1
CONCS=${CONCS:-"1 2 4 8 16 32 64 96 128"}
export TEMPERATURE=1.0 OSL=256
log "START configs=[$CONFIGS] runs=[${RUNS:-1}] host=$HOSTNAME"
for r in ${RUNS:-1}; do
  for cfg in $CONFIGS; do
    out=$BX/results/closed-$STUDY/r$r/$cfg
    [[ -f $out/DONE ]] && continue
    avail=$(df --output=avail -BG "$WORK" | tail -1 | tr -dc 0-9); (( avail < 20 )) && { log "!!! volume nearly full (${avail}G)"; exit 1; }
    config "$cfg" || exit 1
    rm -rf "$out"; mkdir -p "$out"
    log "r$r $cfg start"
    if start_server "$out/server.log" "$SPEC" "${CENV[@]}"; then
      if (cd $REPO && bash experiments/bench/sweep.sh "$out" "$CONCS" instruct-coder > "$out/sweep.log" 2>&1); then
        touch "$out/DONE"; log "r$r $cfg done"
      else
        log "!!! r$r $cfg sweep failed"
      fi
    else
      log "!!! r$r $cfg server failed"
    fi
    stop_server
    find "$out" -name profile_export.jsonl -print0 | xargs -0 -r gzip -f
  done
done
log "ALL DONE"
