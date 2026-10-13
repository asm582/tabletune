#!/usr/bin/env bash
# Runs the llm-d optimized-baseline experiment end to end inside the cluster (tmux session "llmd").
#   RUNS="1 2 3" CONFIGS="guide nospec k3 docs tuned tuned-ctrl docs-ctrl" bash driver.sh
set -uo pipefail
cd "$(dirname "$0")"
WORK=${WORK:-/work}
export PATH=$WORK/bin:$PATH HOME=$WORK
L=$WORK/llmd-32b/progress.log
R=$WORK/llmd-32b/results
mkdir -p "$R"
TUNE=$WORK/tabletune/experiments/results/tune-h100-32b-tp2-llmd-nightly
CONCS=${CONCS:-"4 8 16 32 64 128 256 512 1024"}
DEP=optimized-baseline-nvidia-gpu-vllm-decode
log() { echo "### $* $(date -u +%FT%TZ)" | tee -a $L; }
log "driver START runs=${RUNS:-1 2 3} configs=${CONFIGS:-all}"
for run in ${RUNS:-1 2 3}; do
  for cfg in ${CONFIGS:-guide nospec k3 docs tuned tuned-ctrl docs-ctrl}; do
    out=$R/$cfg-r$run
    [[ -f $out/DONE ]] && { log "skip $cfg r$run (done)"; continue; }
    if [[ $cfg == tuned13* ]]; then
      TABLE=$(cat ${TUNE}-k013/schedule.json)   # lengths 0/1/3 only: this vLLM nightly's V2 runner stalls at a 2->1 edge
    elif [[ $cfg == tuned* ]]; then
      until [[ -f $TUNE/schedule.json ]]; do log "waiting for the tuner"; sleep 120; done
      TABLE=$(cat $TUNE/schedule.json)
    else TABLE=""; fi
    log "deploy $cfg r$run table=${TABLE:-none}"
    bash render.sh $cfg "$TABLE" > /dev/null
    kubectl delete deploy $DEP --ignore-not-found --wait=true > /dev/null 2>&1
    until [[ -z "$(kubectl get pods -l llm-d.ai/role=decode -o name 2>/dev/null)" ]]; do sleep 5; done
    kubectl apply -k overlays/$cfg > /dev/null || { log "!!! apply failed $cfg"; continue; }
    if ! kubectl rollout status deploy/$DEP --timeout=40m > /dev/null; then
      log "!!! $cfg r$run did not become ready"; kubectl get pods -l llm-d.ai/role=decode > $R/$cfg-r$run.notready.txt 2>&1
      for p in $(kubectl get pods -l llm-d.ai/role=decode -o name); do kubectl logs $p --tail=80 >> $R/$cfg-r$run.notready.txt 2>&1; done
      continue
    fi
    IPS=$(kubectl get pods -l llm-d.ai/role=decode -o jsonpath='{.items[*].status.podIP}')
    mkdir -p $out
    kubectl get pods -l llm-d.ai/role=decode -o wide > $out/pods.txt
    for p in $(kubectl get pods -l llm-d.ai/role=decode -o name); do kubectl logs $p > $out/$(basename $p).startup.log 2>&1; done
    # warm-up through the router (graph capture is done; this fills the prefix cache and the EPP's view)
    SEED=7 bash llmd_sweep.sh $out/warmup "16" "$IPS" > /dev/null 2>&1
    log "sweep $cfg r$run START"
    SEED=$((41 + run)) bash llmd_sweep.sh $out "$CONCS" "$IPS" >> $L 2>&1
    for p in $(kubectl get pods -l llm-d.ai/role=decode -o name); do kubectl logs $p --tail=400 > $out/$(basename $p).tail.log 2>&1; done
    if [[ -f $out/FAILED ]]; then   # keep the full engine logs of a failed level
      for p in $(kubectl get pods -l llm-d.ai/role=decode -o name); do kubectl logs $p > $out/$(basename $p).full.log 2>&1; done
    fi
    touch $out/DONE; log "sweep $cfg r$run DONE"
  done
done
log "driver ALL DONE"
