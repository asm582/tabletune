# Experiment tracking (MLflow on the laptop)

All benchmark sweeps are recorded in an MLflow server that runs **on the laptop**, so the history survives stopping the GPU instance. The instance logs to it through a **reverse SSH tunnel**; no port is opened publicly.

```
GPU instance                                   laptop
experiments/bench/run_configs.sh ─► experiments/tracking/log_run.py ─► localhost:5000 ══ssh -R══► MLflow server (127.0.0.1:5000)
                                                                          ~/mlflow/mlflow.db, ~/mlflow/artifacts
```

## Start, view, stop

```bash
bash experiments/tracking/start_server.sh     # on the laptop: MLflow server + tunnel (idempotent)
# open http://localhost:5000  →  experiment "vllm-perf-dsd"
bash experiments/tracking/stop_server.sh      # stops server and tunnel; data in ~/mlflow is kept
```

The tunnel needs the instance to be running. After a stop/start of the instance, update its address in `~/.ssh/config` (host `vllm-perf`) and run `start_server.sh` again.

## What is recorded

One **parent run per sweep** (e.g. `code128t-t4`) with its `pareto.png`, `summary.csv` and `meta.txt`, and one **child run per point** (config × concurrency, e.g. `dsd-tuned-v3 c=64`):

| Kind | Fields |
|---|---|
| Tags (for filtering) | `sweep`, `config`, `dataset`, `speculation` (none / fixed K=n / dynamic), `controller` (on/off), `concurrency`, `run_key` |
| Params: workload | `dataset`, `concurrency`, `temperature` |
| Params: server | `target_model`, `drafter`, `spec_method`, `num_speculative_tokens`, `dsd_table`, `max_num_seqs`, `max_model_len`, `dtype`, `vllm_commit`, `gpu` |
| Params: controller | `controller_hysteresis`, `controller_refit`, `controller_allowed_k`, `controller_last_table` |
| Metrics: throughput/latency | `tok_s_gpu`, `tok_s_user`, `request_throughput`, `ttft_ms_p50/p99`, `itl_ms_p50/p99` |
| Metrics: lengths | `isl_min/p50/avg/max`, `osl_min/p50/avg/max` |
| Metrics: speculation | `acceptance_rate`, `tokens_per_step`, `accept_pos1..3`, `controller_table_changes` |
| Metrics: memory | `preemptions`, `kv_usage_max`, `running_avg` |
| Artifact | aiperf's `profile_export_aiperf.json` for the point |

aiperf only sees the client side, so the server columns come from the harness: `meta.txt` (vLLM commit, GPU, dataset, `max_num_seqs`, temperature) and each config's `server.log` (the `vllm serve` command line, the controller's startup line and its table changes).

## Logging runs

**New sweeps log automatically** when `MLFLOW_TRACKING_URI` is set on the instance. Logging needs the tracking extra in `~/venv`: `uv pip install -e ".[tracking]"` (MLflow client and matplotlib).

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000     # on the instance, tunnel up
bash experiments/bench/run_configs.sh my-sweep "k2 dsd-tuned-v3" "4 16 64" instruct-coder
# → charts the sweep and logs every point; prints "[track] my-sweep: logged N"
```

**Existing sweeps** (backfill, or re-log after a fix):

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python experiments/tracking/log_run.py experiments/results/sweeps/*
# points already logged are skipped; --force replaces them
```

Backfilled so far: 172 points from 9 sweeps (56 spec-bench, 116 instruct-coder, 22 with the controller on). The `compare-*` folders only link to other sweeps and are skipped.

## Using the UI

**Open the Runs view directly:** http://localhost:5000/#/experiments/1/runs?workflowType=machine_learning

MLflow 3 has two UI modes. In **GenAI** mode (`workflowType=genai`, e.g. the *Overview → Usage* page) classic runs are hidden and the experiment looks empty. Use the machine-learning mode (the link above, or the mode switch at the top left). The experiment is also tagged `mlflow.experimentKind=custom_model_development` so it opens in that mode.

**Charts (gains, TTFT, ITL):**

| Where | What |
|---|---|
| Run **`compare-controller`** → *Artifacts* → `controller_value.png` | Controller experiment in one view: throughput, gain vs. no speculation, TTFT p50, ITL p50 against concurrency for each table with the controller off / hysteresis only / full controller, plus no-spec and the tuned table |
| Any **sweep** run (e.g. `code128t-t4`) → *Artifacts* → `gain_latency.png` | Throughput gain vs. no speculation, TTFT p50/p99, ITL p50/p99 against concurrency, a line per config |
| Any sweep run → *Artifacts* → `pareto.png` | The throughput/latency Pareto chart |
| Runs page → **Chart** view | Build your own: e.g. scatter `concurrency` (x) vs. `gain_vs_nospec_pct` (y), grouped by `config` |

**Comparison metrics** (added to every point by `experiments/tracking/derive.py`, matched on dataset, temperature and concurrency across all sweeps): `gain_vs_nospec_pct`, `gain_vs_best_fixed_k_pct`, `gain_vs_controller_off_pct` (controller-on runs only), `ttft_p50_vs_nospec_pct`, `itl_p50_vs_nospec_pct` (negative = faster), and `concurrency` as a number for chart axes. Re-run after adding sweeps (new sweeps do this automatically):

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uv run --no-project \
  --with mlflow-skinny==3.16.1 --with matplotlib python experiments/tracking/derive.py
```

- **Filter** (search box), e.g.
  - `tags.dataset = 'instruct-coder' and params.temperature = '1.0'`
  - `tags.controller = 'on'`
  - `tags.concurrency = '64' and tags.speculation = 'dynamic'`
  - `params.dsd_table LIKE '%, 0]%'` (tables that include K=0)
- **Columns:** add `tok_s_gpu`, `tok_s_user`, `isl_avg`, `osl_avg`, `accept_pos1..3`, `preemptions` to the table view.
- **Compare:** select runs → *Compare* for side-by-side params and metrics, or the *Chart* view for scatter plots (e.g. `tok_s_user` vs. `tok_s_gpu`, the Pareto view) grouped by `config`.
- **Sweep charts:** open a parent run → *Artifacts* → `pareto.png`.

## Alternative: aiperf's own dashboard

For a quick look at one sweep without MLflow, aiperf has an interactive dashboard over run folders:

```bash
ssh -L 8050:localhost:8050 vllm-perf        # from the laptop
~/.local/bin/aiperf plot --dashboard experiments/results/sweeps/code128t-t4/*/c*   # on the instance
# open http://localhost:8050
```

aiperf can also export to MLflow itself (`--mlflow-*` flags), but it only records client-side settings; `log_run.py` is used instead so every run carries the server configuration.
