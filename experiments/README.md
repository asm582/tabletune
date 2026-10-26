# experiments

How the numbers in the top-level README and every result here were measured; [`docs/REPRODUCE.md`](../docs/REPRODUCE.md) maps each result to its data and command. Nothing here is needed to use the tuner or the controller.

| Folder | What |
|---|---|
| [`h100/`](h100/README.md) | The H100 studies (vLLM nightly `43b4aaea`, V2 runner): study drivers, llm-d deployments, committed inputs, analysis and figure scripts |
| [`h100-runbook/`](h100-runbook/README.md) | Running on OpenShift H100s: pods, setup (V1 and the V2 environment, `setup_v2.sh`), the H100 V1/V2 matrix and its workload-switch and jumping-load runs |
| [`results/`](results/) | Measured data, one folder per study, each with a README: what it holds and how it was made. H100 sweeps are `sweeps/*-h100` (V1) and `sweeps/*-h100v2` (V2), the H100 study summaries are in [`results/h100-studies/`](results/h100-studies/), and the first T4 study is in [`results/sweeps/`](results/sweeps/README.md) |
| [`scripts/`](scripts/README.md) | The Tesla T4 and A10G studies: one script per study, run end to end on a GPU instance (also used by the H100 matrix) |
| [`bench/`](bench/README.md) | Benchmark harness: start a server for a named config, sweep concurrency with aiperf |
| [`tools/`](tools/README.md) | Profiling: Nsight Systems and GEMM/attention microbenchmarks |
| [`tracking/`](tracking/README.md) | Log sweeps to MLflow |
| [`vllm_integration/`](vllm_integration/README.md) | GPU integration test for the controller plugin |
