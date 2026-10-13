#!/usr/bin/env bash
# Pod dsd-h100-8b: tune Qwen3-8B + its EAGLE-3 drafter, then the main sweep, 3 runs.
cd /work/runbook/scripts
L=/work/out/p8b-progress.log
bash 00_record_env.sh > /work/out/00-env-8b.log 2>&1
echo "### tune START $(date -u +%FT%TZ)" >> $L; MODEL=8b bash 02_tune.sh > /work/out/02-8b.log 2>&1
echo "### tune END $(date -u +%FT%TZ)" >> $L
grep -q "TUNE DONE" /work/out/02-8b.log || { echo "### ABORT tune failed" >> $L; exit 1; }
echo "### sweep START $(date -u +%FT%TZ)" >> $L; MODEL=8b bash 03_sweep.sh > /work/out/03-8b.log 2>&1
echo "### sweep END $(date -u +%FT%TZ)" >> $L
bash summarize.sh > /work/out/summarize-8b.log 2>&1
echo "### 8B ALL DONE $(date -u +%FT%TZ)" >> $L
