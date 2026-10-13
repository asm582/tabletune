#!/usr/bin/env bash
# One repeat of the traffic-shift + jumping-load runs (4B), at SHIFT_CONC=88 by default. On the H100 the
# live batch runs at ~0.74x the open requests on math and ~0.4x on translation, so the picker's 231
# (aimed at the 168-256 disagreement) put neither phase there, and 100 put math (B~74) on its own 73/74
# edge; at 88 both phases sit inside the 8-72 range where math's table runs 3 and translation's 2.
#   RUNS=<n> [SHIFT_CONC=88] [WAIT_FOR=<progress-log line>] bash run_shift.sh
cd /work/runbook/scripts
L=/work/out/shift-run${RUNS}-progress.log
[[ -n "${WAIT_FOR:-}" ]] && until grep -qE "$WAIT_FOR" /work/out/*-progress.log 2>/dev/null; do sleep 60; done
echo "### shift run $RUNS START $(date -u +%FT%TZ) host=$HOSTNAME conc=${SHIFT_CONC:-88}" >> $L
MODEL=4b RUNS=$RUNS SHIFT_CONC=${SHIFT_CONC:-88} bash 04_shift_vary.sh > /work/out/04-4b-run$RUNS.log 2>&1
bash summarize.sh > /work/out/summarize-shift$RUNS.log 2>&1
echo "### shift run $RUNS DONE $(date -u +%FT%TZ)" >> $L
