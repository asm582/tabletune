# experiments

How the numbers in the top-level README were measured. Nothing here is needed to use the tuner or the controller.

| Folder | What |
|---|---|
| [`results/`](results/) | Measured data, one folder per study, each with a README: what it holds and how it was made (the first T4 study is in [`results/sweeps/`](results/sweeps/README.md)) |
| [`scripts/`](scripts/README.md) | One script per study, run end to end on a GPU instance |
| [`bench/`](bench/README.md) | Benchmark harness: start a server for a named config, sweep concurrency with aiperf |
| [`tools/`](tools/README.md) | Profiling: Nsight Systems and GEMM/attention microbenchmarks |
| [`tracking/`](tracking/README.md) | Log sweeps to MLflow |
| [`vllm_integration/`](vllm_integration/README.md) | GPU integration test for the controller plugin |
