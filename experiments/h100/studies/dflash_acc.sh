#!/usr/bin/env bash
# DFlash on the H100 (V2): profile step costs with the measured acceptance (DSD_SYNTH_ACCEPT) instead of rejecting
# every guess, build the table from them, then sweep it against the forced-rejection table and fixed lengths, 3 runs.
set -uo pipefail
STUDY=dflash-acc; source "$(dirname "$0")/common.sh"
C=$BX/credib; T=$C/dflash-acc; mkdir -p $T $C/results; PROG=$C/progress.log
BS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,40,48,56,64,72,80,88,96,104,112,120,128"
log "START host=$HOSTNAME"
cd $REPO
cp $DFLASH_TUNE/costs_k0.json $T/   # no drafts at length 0: same either way
for k in 1 2 3 4 6 8; do
  [[ -f $T/costs_k$k.json ]] && continue
  DSD_SYNTH_ACCEPT=$DFLASH_TUNE/acceptance.json DSD_METHOD=dflash DSD_K_MAX=8 DSD_DTYPE=bfloat16 DRAFTER_MODEL=$DFLASH \
    python dsd_tuning/tuner/profile_costs.py --k $k --isl 32 --max-num-seqs 128 --batch-sizes "$BS" --out $T/costs_k$k.json \
    > $T/costs_k$k.log 2>&1 || { log "!!! costs K=$k"; exit 1; }
  log "costs K=$k: $(grep -aE '\[costs\] K=. B= *(1|16|64|128) ' $T/costs_k$k.log | sed 's/.*B= *//' | awk '{print $1":"$3}' | tr '\n' ' ')"
done
python dsd_tuning/tuner/make_schedule.py --costs "$T/costs_k*.json" --acceptance $DFLASH_TUNE/acceptance.json --out $T \
  --max-num-seqs 128 --compare-fixed-k > $T/schedule.log 2>&1
log "table with measured acceptance: $(cat $T/schedule.json) (forced rejection: $(cat $DFLASH_TUNE/schedule.json))"
NEW=$(cat $T/schedule.json); OLD=$(cat $DFLASH_TUNE/schedule.json)
# diagnostic: the raw argmax of the measured-acceptance profile, without the tie and minimum-width rules
python dsd_tuning/tuner/make_schedule.py --costs "$T/costs_k*.json" --acceptance $DFLASH_TUNE/acceptance.json --out $T/raw \
  --max-num-seqs 128 --eps 0 --min-width-frac 0 > $T/raw.log 2>&1
RAW=$(cat $T/raw/schedule.json); log "raw argmax table: $RAW"
export TEMPERATURE=1.0 OSL=256
for r in 1 2 3; do
  for cfg in df-k3 df-k4 df-tt df-tt-acc df-tt-raw; do
    out=$C/results/dflash-acc/r$r/$cfg; [[ -f $out/DONE ]] && continue
    case $cfg in df-k3) SPEC=$(dflash_spec 3) ;; df-k4) SPEC=$(dflash_spec 4) ;; df-tt) SPEC=$(dflash_spec 8 "$OLD") ;; df-tt-acc) SPEC=$(dflash_spec 8 "$NEW") ;; df-tt-raw) SPEC=$(dflash_spec 8 "$RAW") ;; esac
    rm -rf "$out"; mkdir -p "$out"; log "r$r $cfg start"
    if start_server "$out/server.log" "$SPEC"; then
      (cd $REPO && bash experiments/bench/sweep.sh "$out" "1 2 4 8 16 32 64 96 128" instruct-coder > "$out/sweep.log" 2>&1) \
        && { touch $out/DONE; log "r$r $cfg done"; } || log "!!! r$r $cfg sweep failed"
    else log "!!! r$r $cfg server failed"; fi
    stop_server; find "$out" -name profile_export.jsonl -print0 | xargs -0 -r gzip -f
  done
done
log "ALL DONE"
