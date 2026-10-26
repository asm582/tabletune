#!/usr/bin/env bash
# Pod dsd-h100-cpu4 (CFS quota 4 CPUs): graph-mode overhead at 4 CPUs, then pinned to 2 cores
# (taskset) to match the 2-core T4/A10G hosts. CPU manager is not static here, so the
# quota alone would let threads spread over all 224 host cores; taskset pins them.
cd /work/runbook/scripts
L=/work/out/pcpu-progress.log
bash 00_record_env.sh > /work/out/00-env-cpu4.log 2>&1
echo "### cpu4 START $(date -u +%FT%TZ)" >> $L
CPUS=4 MODEL=4b taskset -c 0-3 bash 05_overhead.sh > /work/out/05-cpu4.log 2>&1
echo "### cpu2 START $(date -u +%FT%TZ)" >> $L
CPUS=2 MODEL=4b taskset -c 0-1 bash 05_overhead.sh > /work/out/05-cpu2.log 2>&1
bash summarize.sh > /work/out/summarize-cpu.log 2>&1
echo "### CPU ALL DONE $(date -u +%FT%TZ)" >> $L
