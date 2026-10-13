#!/usr/bin/env bash
# Single-GPU P0 plan, in order: env record, setup, tune 4B, main sweep (3 runs), overhead at 16 CPUs.
cd /work/runbook/scripts
mkdir -p /work/out
step() { echo "### $1 START $(date -u +%FT%TZ)" >> /work/out/p0-progress.log; }
done_() { echo "### $1 END $(date -u +%FT%TZ) rc=$2" >> /work/out/p0-progress.log; }
step env; bash 00_record_env.sh > /work/out/00-env.log 2>&1; done_ env $?
step setup; bash 01_setup.sh > /work/out/setup.log 2>&1; rc=$?; done_ setup $rc
grep -q "SETUP DONE" /work/out/setup.log || { echo "### ABORT setup failed" >> /work/out/p0-progress.log; exit 1; }
bash 00_record_env.sh > /work/out/00-env-post.log 2>&1      # now with vllm/torch versions
step tune; MODEL=4b bash 02_tune.sh > /work/out/02-4b.log 2>&1; done_ tune $?
grep -q "TUNE DONE" /work/out/02-4b.log || { echo "### ABORT tune failed" >> /work/out/p0-progress.log; exit 1; }
step sweep; MODEL=4b bash 03_sweep.sh > /work/out/03-4b.log 2>&1; done_ sweep $?
step overhead; MODEL=4b bash 05_overhead.sh > /work/out/05-cpu16.log 2>&1; done_ overhead $?
bash 06_pack.sh > /work/out/06-pack.log 2>&1
echo "### P0 ALL DONE $(date -u +%FT%TZ)" >> /work/out/p0-progress.log
