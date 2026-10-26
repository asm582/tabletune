# H100 studies

Scripts, inputs and analysis for the H100 results; [REPRODUCE.md](../../docs/REPRODUCE.md) maps each result to a command
here.

| Folder | What it holds |
|---|---|
| `studies/` | GPU study drivers on vLLM nightly `43b4aaea` with the V2 runner: adaptive baselines (`closed.sh`), open loop (`open.sh`), jumping load (`vary.sh`), DFlash workload switch (`drift_screen.sh`, `drift.sh`), mixture of experts (`moe.sh`), long prompts (`long.sh`), the DFlash profile miss (`dflash_acc.sh`), merge tolerance (`mergetol.sh`). Shared setup: `common.sh`. |
| `llmd/qwen3-32b`, `llmd/qwen3-8b` | llm-d optimized-baseline guide with configuration changes only: `render.sh` (Kustomize overlay per configuration), `driver.sh` (deploy, wait, sweep through the router), pod specs for the tuner and the AIPerf client. |
| `tables/` | Committed inputs: TableTune's 0/1/3 table for EAGLE-3, the fitted TurboSpec-style step models, the jumping-load schedule, the merge-tolerance tables and `build_mergetol.py`, which rebuilds them. |
| `pods/` | The single-GPU pod used by every study. |
| `analysis/` | Analyses and figures, reading only `experiments/results/`; figures go to `experiments/results/figures/`. |

The H100 V1/V2 matrix (the worst-loss matrix and the controller results) uses the runbook in `experiments/h100-runbook/`. The results
are in `experiments/results/`: sweeps under `sweeps/` (`*-h100` is V1, `*-h100v2` is V2), tuner profiles in the
`*-h100*` folders and the study summaries in `h100-studies/`.
