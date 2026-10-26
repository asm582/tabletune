#!/usr/bin/env bash
# Tables built with the 2% merge tolerance (a narrow range merges only if that loses < 2% of its predicted goodput,
# the tie threshold used when comparing results) against the original tables and the fixed lengths, same session, matrix protocol, 3 runs.
#   SETTING=eagle|ngram|specbench|dflash bash mergetol.sh
set -uo pipefail
STUDY=mt-$SETTING; source "$(dirname "$0")/common.sh"
C=$BX/credib; PROG=$C/progress.log; NEWT=$TABLES/mergetol_tables.json
M=$REPO/experiments/results
new() { python -c "import json;print(json.dumps(json.load(open('$NEWT'))['$SETTING']))"; }
ngram_spec() { local t=""; [[ -n "${2:-}" ]] && t=", \"num_speculative_tokens_per_batch_size\": $2"; echo "{\"method\": \"ngram\", \"prompt_lookup_min\": 2, \"prompt_lookup_max\": 4, \"num_speculative_tokens\": $1$t}"; }
case $SETTING in
  eagle)     export TEMPERATURE=1.0 OSL=256 MAX_SEQS=128; DATA=instruct-coder; LOADS="1 2 4 8 16 32 64 96 128"
             OLD=$(cat $M/tuner-eagle-h100v2/schedule.json); CFGS="k1 k2 k3 old new" ;;
  ngram)     export TEMPERATURE=0 OSL=512 MAX_SEQS=128; DATA=file:$M/ngram/bench-apply.jsonl; LOADS="1 2 4 8 16 32 64 96 128"
             OLD=$(cat $M/ngram-tuner-h100v2/schedule.json); CFGS="k1 k3 k5 new" ;;   # old table = fixed 5
  specbench) export TEMPERATURE=0 OSL=256 MAX_SEQS=64; DATA=spec-bench; LOADS="1 2 4 8 16 32 64"
             OLD=$(cat $M/tuner-v2-h100v2/schedule.json); CFGS="k1 k2 k3 new" ;;     # old table = fixed 3
  dflash)    export TEMPERATURE=1.0 OSL=256 MAX_SEQS=128; DATA=instruct-coder; LOADS="1 2 4 8 16 32 64 96 128"
             OLD=$(cat $M/dflash-tuner-h100v2/schedule.json); CFGS="k3 k4 k8 old new" ;;
esac
NEW=$(new); log "START host=$HOSTNAME old=$OLD new=$NEW"
spec() {  # spec <cfg>
  case $SETTING:$1 in
    ngram:k*) ngram_spec ${1#k} ;; ngram:new) ngram_spec 5 "$NEW" ;;
    dflash:k*) dflash_spec ${1#k} ;; dflash:old) dflash_spec 8 "$OLD" ;; dflash:new) dflash_spec 8 "$NEW" ;;
    *:k*) eagle_spec ${1#k} ;; *:old) eagle_spec 3 "$OLD" ;; *:new) eagle_spec 3 "$NEW" ;;
  esac
}
for r in 1 2 3; do
  for cfg in $CFGS; do
    out=$C/results/mt-$SETTING/r$r/$cfg; [[ -f $out/DONE ]] && continue
    rm -rf "$out"; mkdir -p "$out"; log "r$r $cfg start"
    if start_server "$out/server.log" "$(spec $cfg)"; then
      (cd $REPO && bash experiments/bench/sweep.sh "$out" "$LOADS" "$DATA" > "$out/sweep.log" 2>&1) \
        && { touch $out/DONE; log "r$r $cfg done"; } || log "!!! r$r $cfg sweep failed"
    else log "!!! r$r $cfg server failed"; fi
    stop_server; find "$out" -name profile_export.jsonl -print0 | xargs -0 -r gzip -f
  done
done
log "ALL DONE"
