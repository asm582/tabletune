# GPU experiment pipelines

Each script runs one study end to end on a GPU instance and writes to `experiments/results/`. They are thin drivers around the benchmark harness in [`../bench/`](../bench/) (start a server for a named config, sweep concurrency with aiperf) and the tuner in [`dsd_tuning/`](../../dsd_tuning/).

**Prerequisites:** [`install.sh`](../../install.sh), `uv tool install aiperf==0.13.0`, and nothing else on the GPU. Optional: `MLFLOW_TRACKING_URI` to log every sweep ([`../tracking/`](../tracking/README.md)).

**Running.** From anywhere: `bash experiments/scripts/<name>.sh [stage ...]`. Scripts `cd` to the repo root; most take stage names (for example `accept costs schedule sweep controller`) so a single stage can be re-run. Pipelines with repeat runs take `SUFFIX=-r2` (sweeps named `<sweep>-r2`).

| Script | Study | Writes | Time (one run) |
|---|---|---|---|
| `run_tuner.sh` | Tuner v1/v2 on spec-bench (first study) | `experiments/results/tuner` (or the folder given as 2nd argument, e.g. `experiments/results/tuner-v2`) | — |
| `run_v3_code.sh` | Tuner v3 on the short-prompt code workload, and the proof sweep | `experiments/results/tuner-v3-code`, sweep `code128t-t4` | 2.4 h (T4) |
| `run_ngram_accept.sh`, `run_ngram.sh` | N-gram drafter (issue #3) | `experiments/results/ngram`, `experiments/results/ngram-tuner` | — |
| `run_draft.sh` | Separate draft model, Qwen3-0.6B (issue #4) | `experiments/results/draft-tuner` | 3.9 h (T4) |
| `run_dflash.sh` | DFlash block drafter, BF16 GPUs only (issue #7) | `experiments/results/dflash-tuner[-GPU]` | 2.3 h (A10G) |
| `run_gpu_eagle.sh` | EAGLE-3 tuner and controller on another GPU (`GPU=a10g DTYPE=bfloat16`) | `experiments/results/tuner-eagle-GPU` | 2.5 h (A10G) |
| `run_quant.sh` | AWQ 4-bit target (issue #11) | `experiments/results/tuner-eagle-t4-awq` | 3.4 h (T4) |
| `run_ctrl_fix.sh`, `run_ctrl_items.sh` | Controller options and diagnostics (issue #2) | sweeps and `experiments/results/integration` | — |
| `run_repeats.sh`, `run_repeats_specbench.sh` | Runs 2 and 3 of the headline sweeps (issue #10) | sweeps `*-r2`, `*-r3`; mean with `python experiments/bench/aggregate_repeats.py <sweep>` | 4.7 h and 1.1 h per run (T4) |
| `run_dsd_overhead.sh` | Why Dynamic SD is slower at low load on AWQ (issue #12) | sweeps `dsdov-*`, `experiments/results/dsd-overhead` | ~1 h per run (T4) |

Times are measured from the pipeline logs (— where the log doesn't cover the whole run). Stage-by-stage times are in each pipeline's log (`=== stage (time) ===` lines, kept in `experiments/results/logs/`).
