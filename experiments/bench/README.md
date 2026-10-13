# Benchmark harness

Start a vLLM server for a named configuration, sweep concurrency with aiperf, summarise. Same prerequisites as [`../scripts/`](../scripts/README.md).

| File | What |
|---|---|
| `configs.sh` | Named server configurations (`nospec`, `k1`..`k3`, `dsd-*`, ...): `spec_config <name>` prints the `--speculative-config` |
| `launch_server.sh` | Start a server for one config and wait until it is healthy |
| `sweep.sh` | Concurrency sweep against a running server with aiperf |
| `run_configs.sh` | For each config: start the server, sweep, stop it |
| `plot_pareto.py` | `summary.csv` and `pareto.png` for a sweep (needs `.[tracking]`) |
| `aggregate_repeats.py` | Mean and spread across repeat runs of a sweep |
| `run_controller_value.sh` | Controller on vs. off, starting from misaligned tables |
| `smoke_test.sh`, `smoke_test.py` | Greedy outputs with and without speculation must match |
| `quality_gsm8k.py` | GSM8K accuracy (for the AWQ model change) |

Output goes to `experiments/results/sweeps/<run>/`. Example: step 4 of [docs/TUNING.md](../../docs/TUNING.md); the study pipelines in [`../scripts/`](../scripts/README.md) call these.
