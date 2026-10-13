#!/usr/bin/env bash
# Issue #12: why is the Dynamic SD path 12-17% slower than fixed K at 1-2
# requests on the AWQ target (T4), but not on FP16 or the A10G?
#
#   paths    AWQ: fixed k3 (FULL_AND_PIECEWISE) vs. dsd-const3 (table K=3 everywhere,
#            vLLM forces PIECEWISE) vs. k3 forced to PIECEWISE; nospec in both modes.
#            FP16 control: k3 and nospec forced to PIECEWISE (FP16 k3 vs. table is
#            already measured: no gap). c = 1 2 4 8, REPS runs each.
#   cpu      AWQ, c=1: CPU use of the EngineCore process during fixed k3 and
#            dsd-const3 (near 100% of one core = CPU launch-bound).
#   nsys     AWQ, c=1: 20 s Nsight Systems trace of the live server under load,
#            fixed k3 vs. dsd-const3; GPU busy %, idle gaps and launch counts
#            (experiments/tools/nsys_gpu_idle.py on the SQLite export). Not in the default stages.
#
# Usage (repo root): bash experiments/scripts/run_dsd_overhead.sh [paths] [cpu] [nsys]
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source ~/venv/bin/activate
log() { echo "=== $* ($(date +%H:%M:%S)) ==="; }
STAGES=${*:-paths cpu}
has() { [[ " $STAGES " == *" $1 "* ]]; }
AWQ=Qwen/Qwen3-4B-AWQ
C="1 2 4 8"
PW='{"cudagraph_mode": "PIECEWISE"}'
export MAX_SEQS=128 TEMPERATURE=1.0 DTYPE=float16
REPS=${REPS:-1 2 3}
P=${DSDOV_PREFIX:-dsdov}                   # sweep-name prefix, e.g. dsdov-a10g on another GPU
OUTD=${DSDOV_OUT:-experiments/results/dsd-overhead}   # cpu / nsys output folder

if has paths; then
  for r in $REPS; do
    log "run $r: AWQ, default graphs (fixed k3 FULL_AND_PIECEWISE, dsd-const3 PIECEWISE)"
    TARGET_MODEL=$AWQ bash experiments/bench/run_configs.sh $P-awq-r$r "nospec k3 dsd-const3" "$C" instruct-coder
    log "run $r: AWQ, forced PIECEWISE"
    TARGET_MODEL=$AWQ COMPILATION_CONFIG="$PW" bash experiments/bench/run_configs.sh $P-awq-pw-r$r "nospec k3" "$C" instruct-coder
    log "run $r: FP16 control, forced PIECEWISE"
    COMPILATION_CONFIG="$PW" bash experiments/bench/run_configs.sh $P-fp16-pw-r$r "nospec k3" "$C" instruct-coder
  done
fi

if has cpu; then
  for cfg in k3 dsd-const3; do
    log "cpu: AWQ $cfg at c=1"
    out=$OUTD/cpu-$cfg; mkdir -p "$out"
    TARGET_MODEL=$AWQ bash experiments/bench/launch_server.sh "$cfg" "$out/server.log" || { echo "!!! $cfg server failed"; continue; }
    mkdir -p "$out/sweep"   # sweep.sh writes its per-level logs into this folder
    TARGET_MODEL=$AWQ bash experiments/bench/sweep.sh "$out/sweep" "1" instruct-coder > "$out/sweep.log" 2>&1 &   # aiperf requests this model name
    sw=$!
    sleep 40   # aiperf takes ~35 s to start; the c=1 load then lasts ~50-60 s
    pid=$(pgrep -f "VLLM::EngineCore" | head -1)
    top -b -d 1 -n 25 -p "$pid" | awk -v p="$pid" '$1==p {print $9}' > "$out/enginecore_cpu.txt"
    wait $sw || echo "!!! $cfg: c=1 load failed, see $out/sweep.log (CPU numbers are not valid)"
    kill "$(cat "$out/server.pid")" 2> /dev/null
    for _ in $(seq 1 30); do pgrep -f "vllm serve|VLLM::EngineCore" > /dev/null || break; sleep 2; done
    pkill -9 -f "vllm serve" 2> /dev/null; pkill -9 -f "VLLM::EngineCore" 2> /dev/null
    for _ in $(seq 1 30); do [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break; sleep 2; done
    echo "[cpu] $cfg EngineCore CPU% over 25 s: $(sort -n "$out/enginecore_cpu.txt" | awk '{a[NR]=$1} END {print "median", a[int(NR/2)+1], "max", a[NR]}')"
  done
fi
if has nsys; then
  NSYS=${NSYS:-/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys}
  for cfg in k3 dsd-const3; do
    log "nsys: AWQ $cfg at c=1"
    out=$OUTD/nsys-$cfg; mkdir -p "$out"
    source experiments/bench/configs.sh
    spec=$(TARGET_MODEL=$AWQ spec_config "$cfg")
    VLLM_USE_V2_MODEL_RUNNER=0 $NSYS launch --session-new=dsdov --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
      vllm serve $AWQ --port 8000 --dtype float16 --max-model-len 4096 --max-num-seqs 128 \
      --gpu-memory-utilization 0.90 --seed 0 --speculative-config "$spec" > "$out/server.log" 2>&1 &
    for _ in $(seq 1 120); do curl -sf localhost:8000/health > /dev/null && break; sleep 5; done
    curl -sf localhost:8000/health > /dev/null || { echo "!!! $cfg nsys server not ready"; }
    mkdir -p "$out/sweep"   # sweep.sh writes its per-level logs into this folder
    TARGET_MODEL=$AWQ bash experiments/bench/sweep.sh "$out/sweep" "1" instruct-coder > "$out/sweep.log" 2>&1 &   # aiperf requests this model name
    sw=$!
    sleep 25
    $NSYS start --session=dsdov -o "$out/trace" --force-overwrite=true
    sleep 20
    $NSYS stop --session=dsdov
    wait $sw || echo "!!! $cfg: c=1 load failed, see $out/sweep.log (trace has no load)"
    # --force-overwrite: never analyze a stale .sqlite from an earlier capture
    $NSYS export --type sqlite --force-overwrite true --output "$out/trace.sqlite" "$out/trace.nsys-rep" > /dev/null 2>&1
    python experiments/tools/nsys_gpu_idle.py "$out/trace.sqlite" | tee "$out/summary.txt"
    $NSYS shutdown --session=dsdov 2> /dev/null
    for _ in $(seq 1 30); do pgrep -f "vllm serve|VLLM::EngineCore" > /dev/null || break; sleep 2; done
    pkill -9 -f "vllm serve" 2> /dev/null; pkill -9 -f "VLLM::EngineCore" 2> /dev/null
    for _ in $(seq 1 30); do [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break; sleep 2; done
  done
fi
log "issue #12 runs done"
