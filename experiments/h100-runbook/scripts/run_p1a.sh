#!/usr/bin/env bash
# Pod dsd-h100, after P0: summaries, BF16 graph-mode attribution, then traffic shift + jumping load (4B).
cd /work/runbook/scripts
L=/work/out/p1a-progress.log
until grep -qE "P0 ALL DONE" /work/out/p0-progress.log; do sleep 60; done
echo "### summarize $(date -u +%FT%TZ)" >> $L; bash summarize.sh > /work/out/summarize-p0.log 2>&1
echo "### bf16 START $(date -u +%FT%TZ)" >> $L; MODEL=4b bash 08_bf16_graphs.sh > /work/out/08-4b.log 2>&1
echo "### bf16 END $(date -u +%FT%TZ)" >> $L
source env.sh
python pick_conc.py --repo $REPO --costs "$TUNE/costs_k*.json" --a $X/acc-$TAG-translation --b $X/acc-$TAG-math_reasoning --max-b 256 > /work/out/pick_conc.log 2>&1
cat /work/out/pick_conc.log >> $L
C=$(grep -oE "Use about [0-9]+" /work/out/pick_conc.log | grep -oE "[0-9]+")
if [[ -z $C ]]; then   # tables agree: run only the jumping-load experiment
  echo "### tables agree: skipping Experiment 1" >> $L
  sed '/run e1_/d' 04_shift_vary.sh > 04_vary_only.sh; S=04_vary_only.sh; C=64
else S=04_shift_vary.sh; fi
echo "### shift START SHIFT_CONC=$C $(date -u +%FT%TZ)" >> $L
MODEL=4b SHIFT_CONC=$C bash $S > /work/out/04-4b.log 2>&1
echo "### shift END $(date -u +%FT%TZ)" >> $L
bash summarize.sh > /work/out/summarize-p1a.log 2>&1
echo "### P1A ALL DONE $(date -u +%FT%TZ)" >> $L
