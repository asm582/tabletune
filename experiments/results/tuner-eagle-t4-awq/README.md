# AWQ 4-bit target: tuner and controller on the T4 (issue #11)

Tesla T4, `Qwen/Qwen3-4B-AWQ` (4-bit weights, FP16 activations) + the unchanged EAGLE-3 drafter, code workload (instruct-coder, ~32-token prompts, T=1.0), `max_num_seqs=128`. Output tok/s per GPU, **one run** (sweep `code128t-t4-awq`; controller `ctrlval-refit-t4-awq`, `ctrlval-off-t4-awq`).

**Tuned table:** `[[1,4,3],[5,128,1]]` (FP16 target: `[[1,18,3],[19,43,2],[44,128,1]]`) · acceptance 0.59 / 0.33 / 0.15 (FP16: 0.61 / 0.34 / 0.16).

| c | K=1 | K=2 | K=3 | Tuned | No spec | FP16's table | Tuned vs. best fixed |
|---|---|---|---|---|---|---|---|
| 1 | 91.3 | 98.4 | 94.5 | 78.1 | 70.0 | 79.9 | −20.6% |
| 2 | 165.6 | 182.3 | 174.5 | 152.9 | 132.3 | 146.6 | −16.1% |
| 4 | 306.2 | 307.0 | 278.9 | 267.9 | 232.0 | 273.7 | −12.7% |
| 8 | 485.0 | 453.6 | 427.9 | 476.1 | 398.3 | 443.9 | −1.8% |
| 16 | 682.5 | 655.8 | 598.7 | 677.5 | 577.9 | 608.2 | −0.7% |
| 32 | 868.0 | 753.2 | 750.6 | 865.6 | 755.7 | 766.0 | −0.3% |
| 64 | 1025.6 | 928.0 | 838.2 | 1033.0 | 913.6 | 1027.9 | +0.7% |
| 96 | 1047.2 | 944.7 | 856.9 | 1043.3 | 910.7 | 1042.7 | −0.4% |
| 128 | 1090.9 | 992.2 | 879.3 | 1092.5 | 1006.5 | 1092.1 | +0.2% |

- **From 8 requests** the tuned table is within 2% of the best fixed K; each fixed K loses 9–19% somewhere. The FP16 table loses 10–12% at 16–32 requests on this model.
- **Below 8 requests** every Dynamic SD table is 13–21% below the same K set as a fixed number. Cause: Dynamic SD forces PIECEWISE CUDA graphs, and with the short AWQ step the CPU can't launch the work fast enough (nsys: GPU busy 91% → 60%); see [`../dsd-overhead/README.md`](../dsd-overhead/README.md) and issue #12. The tuner can't avoid it.
- **Step costs** are much lower at small batches (K=0 at 1 request: 20.6 ms vs. 35.7 ms FP16) and the KV cache is bigger (59.2k tokens with EAGLE-3 vs. ~27k): no preemption at 128 requests.
- One of 512 requests failed at K=3, 128 requests (0.2%); not re-run. Greedy outputs are not bit-exact on AWQ (numerics), so the smoke gate checks acceptance only.

**Controller** (OFF → ON, docs table / ÷4 table): +13.2% / +12.1% at 16, +17.4% / +0.1% at 32, +14.9% / +11.0% at 40, **+23.4% / +18.0% at 64**, +0.5% / +19.1% at 96 requests.

**Reproduce:** `bash experiments/scripts/run_quant.sh pipeline`. Logs: [`../logs/t4/quant2.log`](../logs/t4/quant2.log).
