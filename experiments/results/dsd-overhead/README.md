# Issue #12: why Dynamic SD is slower at low load on the AWQ target (T4)

**Finding.** On Qwen3-4B-AWQ on the T4, any run with a Dynamic SD table is 12% slower at 1 request than the same K set as a fixed number. The cause is the CUDA-graph mode, not the Dynamic SD logic: on the V1 model runner vLLM switches to PIECEWISE graphs whenever a table is set (`VllmConfig._maybe_override_dynamic_sd_cudagraph_mode`), and with PIECEWISE graphs **the CPU cannot launch the GPU work fast enough** on this host (4 vCPU) for AWQ's short steps.

Run with [`experiments/scripts/run_dsd_overhead.sh`](../../scripts/run_dsd_overhead.sh); sweeps `dsdov-*` in MLflow, means in [`../repeats/`](../repeats/).

## 1. The graph mode explains the whole gap (3 runs, output tok/s, code workload)

| Requests | Fixed K=3 (full graphs) | Table, K=3 at every batch size | Fixed K=3 forced PIECEWISE | No spec: full → PIECEWISE |
|---|---|---|---|---|
| 1 | 94.6 | 82.8 (−12.4%) | 82.9 (−12.3%) | 70.0 → 50.3 (−28.1%) |
| 2 | 168.7 | 156.1 (−7.5%) | 157.0 (−7.0%) | 129.2 → 93.7 (−27.5%) |
| 4 | 270.7 | 280.0 (+3.5%) | 269.4 (−0.5%) | 229.1 → 179.2 (−21.8%) |
| 8 | 440.8 | 437.3 (−0.8%) | 441.5 (+0.2%) | 397.0 → 371.9 (−6.3%) |

The FP16 target shows no change under PIECEWISE (fixed K=3: 46.0 tok/s at 1 request either way; no spec 28.3). AWQ runs under PIECEWISE are noisier (median spread 8.5% vs. 1.1% for FP16).

## 2. The GPU waits for the CPU (nsys, 1 request, 7.2 s window of the live server)

| | Fixed K=3 (full graphs) | Table, K=3 everywhere (PIECEWISE) |
|---|---|---|
| GPU busy | **91.4%** | **59.7%** |
| GPU idle gaps > 50 µs | 344, 0.49 s in total | 13,607, **2.5 s in total** |
| CUDA-graph launches | 2,325 | 9,411 (4×) |
| CPU time waiting for the GPU (`cudaEventSynchronize`) | 2.2 s | not among the top calls |

With full graphs the CPU is ahead and waits for the GPU. With PIECEWISE graphs the CPU issues four times as many graph launches, never waits, and the GPU is idle 40% of the time: the step is CPU launch-bound. nsys adds a little CPU time per CUDA call; both configs are traced the same way. Full output: [`nsys_summary.txt`](nsys_summary.txt) (from [`experiments/tools/nsys_gpu_idle.py`](../../tools/nsys_gpu_idle.py)).

## 3. Same test on the A10G: the gap is larger, not smaller (3 runs, output tok/s)

| Requests | Fixed K=3 (full graphs) | Table, K=3 everywhere | Fixed K=3 forced PIECEWISE | No spec: full → PIECEWISE |
|---|---|---|---|---|
| 1 | 190.8 | 144.0 (−24.5%) | 144.4 (−24.3%) | 137.1 → 89.3 (−34.9%) |
| 2 | 350.8 | 270.7 (−22.8%) | 268.0 (−23.6%) | 261.6 → 165.0 (−36.9%) |
| 4 | 636.9 | 498.7 (−21.7%) | 508.2 (−20.2%) | 492.7 → 311.0 (−36.9%) |
| 8 | 1077.1 | 954.4 (−11.4%) | 969.0 (−10.0%) | 955.3 → 608.9 (−36.3%) |

Sweeps `dsdov-a10g-*`. The BF16-era A10G runs showed no gap, and the FP16 control here shows none either (fixed K=3 forced PIECEWISE 88.8 tok/s at 1 request vs. 88.6 with full graphs; no spec 52.3 vs. 52.2).

Per token without speculation, PIECEWISE adds about **5.6 ms** on the T4 (14.3 → 19.9 ms) and **3.9 ms** on the A10G (7.3 → 11.2 ms). The A10G host launches a little faster, but its AWQ GPU step is only ~7 ms, so the same kind of fixed CPU cost per step is a larger share of it. That is what a launch-bound step looks like: the penalty grows as the GPU part of the step shrinks. A faster GPU makes it worse, not better.

**Why only AWQ:** the 4-bit target's GPU step is short (~21 ms at 1 request on the T4, ~7 ms on the A10G, vs. ~36 ms and ~20 ms for the full-precision model), so the extra CPU launch work per step no longer fits inside it. It fades with load as the GPU work per step grows (on the T4 within 1% from 8 requests; on the A10G still −11% at 8).

**CPU % of the engine process is not a useful signal:** it sits near 100% of one core during load for both configs on both GPUs (the engine loop spins), see `cpu_*.txt`.

## What it means

- The tuner cannot fix this: its choices are K values inside the Dynamic SD path.
- Fixes are upstream: full CUDA graphs per K under Dynamic SD (the V2 model runner has them but ignores dynamic K for EAGLE, vllm-project/vllm#51510), or deciding K for groups of steps so full graphs can be kept. vllm-project/vllm#49986 reports the same PIECEWISE tax.
- For deployments: on a small host with a fast (e.g. quantized) target, a fixed K can beat any table at low load.
