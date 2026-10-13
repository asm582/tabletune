# Reproducing the paper

The paper (`docs/paper/paper.tex`, *TableTune: measured, self-correcting draft-length tables for vLLM Dynamic Speculative
Decoding*) is built from this repository alone. There are three levels of reproduction:

1. **Build the PDF** from the committed figures (no GPU, a minute).
2. **Re-derive every number and figure** from the committed results (no GPU, a minute).
3. **Re-run the experiments** on H100s (an OpenShift cluster with a shared volume; scripts below).

CI does levels 1 and 2 on every pull request (`.github/workflows/paper.yml`, the PDF is uploaded as the `paper`
artifact) and checks that the committed tables rebuild from the committed profiles (`.github/workflows/ci.yml`).

## 1. Build the PDF

The built PDF is committed as `docs/paper/paper.pdf`. To rebuild it:

```bash
bash docs/paper/build.sh        # Tectonic 0.17.0; writes docs/paper/paper.pdf
```

The figures the paper includes are committed in `docs/paper/build/` (`fig_design.tex` is TikZ).

## 2. Re-derive the numbers and figures (no GPU)

```bash
python -m pip install -e . matplotlib==3.10.8    # the paper's figures were drawn with 3.10.8; other versions render slightly differently
python experiments/h100/analysis/figs_paper.py       # Figures 1 and 4 -> docs/paper/build/ (FIG_OUT=<dir> to write elsewhere)
python experiments/h100/analysis/figs_baselines.py   # Figure 3
python experiments/h100/analysis/figs_llmd.py        # Figure 5
```

Each analysis script prints the paper's numbers for one result:

| Paper | Command (`experiments/h100/analysis/…`) | Data (`experiments/results/…`) |
|---|---|---|
| Figure 1 (no fixed length is best) | `figs_paper.py` | `sweeps/code128t-h100v2{,-r2,-r3}` |
| Table 1 (worst loss, every GPU and drafter) | `compare_gpus.py` → `h100-studies/comparison/COMPARISON.md` | `sweeps/*-h100v2*` (H100 V2), `sweeps/*-h100*` (H100 V1), `sweeps/*-t4*`, `*-a10g*` (older GPUs) |
| Q1 mixture of experts | `moe.py` | `h100-studies/moe/results.csv`, tuner `tuner-moe-h100v2` |
| Q1 long prompts (1,024 and 3,072 tokens) | `credib.py` (`long1024`, `long3072`) | `h100-studies/credib/results.csv`, tuners `tuner-eagle-h100v2-long{1024,3072}` |
| Q1 jumping load | `shift.py res-h100v2m` (e2) and `baselines.py vary` | `h100-studies/shift/res-h100v2m`, `h100-studies/baselines/vary` |
| Q1 open-loop arrivals | `baselines.py open` | `h100-studies/baselines/results.csv` (`open`) |
| Q2, Figure 3 (TurboSpec- and Nightjar-style) | `baselines.py closed`, `figs_baselines.py` | `h100-studies/baselines/results.csv` (`closed-eagle`, `closed-dflash`) |
| Q3, Table 2 top (controller on mistuned tables), Figure 4a | `compare_gpus.py` (controller section), `figs_paper.py` | `sweeps/ctrlval-*`, `ngram-ctrl-*`, `draft-ctrl-*`, `dflash-ctrl-*` |
| Q3, Table 2 bottom (workload switch) | `shift.py res-h100v2m013` (V2), `shift.py res-h100m` (V1); DFlash: `baselines.py drift` | `h100-studies/shift/`, `h100-studies/baselines/drift` |
| Q3 Figure 4b (DFlash workload switch) | `figs_paper.py` | `h100-studies/baselines/drift/res-s1` |
| Q4, Figure 5 (llm-d) | `llmd.py qwen3-32b`, `llmd.py qwen3-8b`, `figs_llmd.py` | `h100-studies/llmd/qwen3-{32b,8b}.csv`, tuners `tune-h100-32b-tp2-llmd-nightly{,-k013}` (TP=2), `tune-h100-8b-llmd-nightly{,-k013}` |
| Q6 DFlash profile miss | `credib.py` (`dflash-acc`) | `h100-studies/credib/results.csv`, tuner `dflash-tuner-h100v2-acc` |
| Q6 merge tolerance | `credib.py` (`mt-*`); tables: `experiments/h100/tables/build_mergetol.py` | `h100-studies/credib/results.csv`, `experiments/h100/tables/mergetol_tables.json` |
| Tuned tables (every setting) | `dsd-tuner schedule --costs '<dir>/costs_k*.json' --acceptance <dir>/acceptance.json --out <tmp>` | the `*-h100`, `*-h100v2` tuner folders |

The jumping-load and workload-switch results are the per-window `segments` of the `gen.py` result files
(`experiments/h100/analysis/compact.py` strips the per-request records); the llm-d CSVs come from the AIPerf exports
and vLLM metric dumps (`llmd_csv.py`). The per-request AIPerf records stay on the cluster volume.

## 3. Re-run the experiments

**Hardware.** H100 80 GB SXM GPUs on OpenShift, whole GPUs (no MIG), 16 CPUs and 128 GiB per single-GPU pod; one
read-write-many volume (`dsd-work`, mounted at `/work`, 600 GiB) shared by all pods. Results in the paper ran each
study in its own pod; the H100 V1 matrix ran one pod per otherwise idle node (piecewise CUDA graphs are sensitive to
host contention, see `experiments/h100-runbook/README.md`).

**Software.** Two environments on the volume:
- *V1* (vLLM 353e491ef, the default runner): `experiments/h100-runbook/scripts/01_setup.sh` (also installs uv and AIPerf 0.13.0).
- *V2* (vLLM nightly `43b4aaea`, V2 model runner with full CUDA graphs per draft length): `experiments/h100-runbook/scripts/setup_v2.sh`,
  which adds both baseline controllers (`baselines/`). Every study after the V1 matrix uses this environment.

### Setup

Follow `experiments/h100-runbook/README.md` sections 0-4 for the volume (`openshift/pvc-rwx.yaml`), the code copy (the
repository at `/work/tabletune`, the runbook at `/work/runbook`) and the V1 setup (`00_record_env.sh`,
`01_setup.sh`, which also writes the prompt files to `/work/x`). Then, for the V2 studies:

```bash
oc apply -f experiments/h100/pods/gpu-pod.yaml          # one pod per study: change metadata.name (h100-1, h100-2, ...)
oc exec h100-1 -- bash /work/tabletune/experiments/h100-runbook/scripts/setup_v2.sh
oc exec h100-1 -- bash /work/tabletune/experiments/h100/studies/download_models.sh   # MoE and 32B targets, drafters
```

The runbook (`experiments/h100-runbook/README.md`) covers the image, the volume, network access and the V1 setup in detail.
Run long scripts in `tmux` inside the pod; each writes a `progress.log` and skips work already done, so a restart
resumes.

### What to run for each result

| Paper | Script | Notes |
|---|---|---|
| Table 1, H100 V1 | `experiments/h100-runbook/scripts/matrix.sh <study>`, study = `eagle awq ngram draft dflash specbench` | V1 environment, 3 runs, one idle node per study (`matrix_node.sh`) |
| Table 1, H100 V2; Figure 1 | `MATRIX_V2=1 bash matrix.sh <study>`; AWQ repeat 3 and the 0/1/3 table: `v2_finish_awq.sh` | |
| Table 1, older GPUs | the T4/A10G study in `experiments/` (`experiments/README.md`) | |
| Table 2 top, Figure 4a | included in `matrix.sh` (controller stages) | |
| Table 2 bottom (switch), Q1 jumping load | `matrix_shift.sh` (V1), `MATRIX_V2=1 bash matrix_shift.sh` (V2) | two `gen.py` clients per server |
| Q1 Qwen3-8B | `experiments/h100-runbook/scripts/run_8b.sh` | V1 |
| Q1 mixture of experts | `experiments/h100/studies/moe.sh` | `STAGES="tune sweep controller" RUNS="1 2 3"` |
| Q1 long prompts | prompts (below), then `ISL=1024 LOADS="1 2 4 8 16 32 64 96 128" MAXB=128 bash long.sh` and `ISL=3072 LOADS="1 2 4 8 16 32 48 64" MAXB=64 bash long.sh` | 3,072 tokens stops at 64 (KV cache) |
| Q1 open loop | `STUDY=open RUNS="1 2 3" bash open.sh "nospec k1 k3 tt tt013 ts nj"` | |
| Q1 jumping load vs baselines | `RUNS="1 2 3" bash vary.sh` | |
| Q2, Figure 3 | `STUDY=eagle RUNS="1 2 3" bash closed.sh "nospec k1 k2 k3 tt tt013 tt013-ctrl ts nj"` and `STUDY=dflash … "df-k3 df-k4 df-k8 df-tt df-tt-ctrl df-ts df-nj"` | baseline models: `experiments/h100/tables/ts_*.json` (`python -m turbospec_style.fit`) |
| Q3 DFlash workload switch, Figure 4b | `drift_screen.sh`, then `RUNS="1 2 3" SCEN="s1 s2" bash drift.sh` | |
| Q4, Figure 5 | `experiments/h100/llmd/qwen3-32b/` (the guide as shipped) and `qwen3-8b/`; see below | |
| Q6 DFlash profile miss | `dflash_acc.sh` | uses `DSD_SYNTH_ACCEPT` (below) |
| Q6 merge tolerance | `SETTING=eagle\|ngram\|specbench\|dflash bash mergetol.sh` | one pod per setting |
| Q5/Q6 graph-mode overhead, profiling cost | `experiments/h100-runbook/scripts/05_overhead.sh`, `08_bf16_graphs.sh`; profiling time from the tuner logs | |

Long-prompt inputs (GovReport test split, cut to 1,024 and 3,072 tokens with the target's tokenizer):

```bash
mkdir -p $WORK/exp-h100/credib/prompts && cd $WORK/exp-h100/credib/prompts
curl -sL -o govreport-test.parquet https://huggingface.co/datasets/ccdv/govreport-summarization/resolve/main/document/test-00000-of-00001.parquet
HF_HOME=$WORK/hf $WORK/v2n/venv/bin/python $WORK/tabletune/experiments/h100/studies/make_long_prompts.py \
  --parquet govreport-test.parquet --out . --tokens 1024 3072
```

The study scripts in `experiments/h100/studies/` take `WORK` (the volume, default `/work`) and `OUT` (results,
default `$WORK/exp-h100`); `collect.py <results dir> <out.csv>` gathers a study's AIPerf exports into the CSV format
that the analysis scripts read.

### llm-d

`experiments/h100/llmd/<model>/` holds the configuration-only changes to llm-d's optimized-baseline guide
(llm-d `main` at `4dc1573`, the standalone router chart `llm-d-router-standalone` v0 with the guide's values):

```bash
git clone https://github.com/llm-d/llm-d $WORK/llm-d && git -C $WORK/llm-d checkout 4dc1573
cd $WORK/llm-d && helm install optimized-baseline $ROUTER_STANDALONE_CHART --version $ROUTER_CHART_VERSION -n vllm-dsd \
  -f guides/recipes/router/base.values.yaml -f guides/optimized-baseline/router/optimized-baseline.values.yaml   # the router, as in the guide
oc apply -f experiments/h100/llmd/qwen3-32b/tuner-pod.yaml                         # tuner at TP=2 (DSD_TP=2)
oc apply -f experiments/h100/llmd/qwen3-32b/bench-pod.yaml                         # CPU pod: AIPerf and kubectl
oc exec llmd32-bench -- env LLMD_GUIDES=/work/llm-d/guides NODE=<4-GPU node> RUNS="1 2 3" \
  bash /work/tabletune/experiments/h100/llmd/qwen3-32b/driver.sh
```

`driver.sh` renders each configuration with `render.sh`, rolls the decode deployment, waits for it, and runs the
sweep through the router; results go to `$WORK/llmd-32b/results`, then `analysis/llmd_csv.py` compacts them.

## Code options added for these studies

All are opt-in; the defaults reproduce the earlier T4/A10G results unchanged (the 158 unit tests pass).
- `DSD_TP=<n>`: tensor-parallel size for the tuner's profilers (Qwen3-32B at TP=2 in llm-d).
- `DSD_SYNTH_ACCEPT=<acceptance.json>`: profile step time with vLLM's synthetic rejection sampler at the measured
  acceptance instead of rejecting every guess (Q6, DFlash).
- `make_schedule.py --merge-tol 0.02` / `build_schedule(..., merge_tol=)`: merge a narrow range only if that loses
  less than 2% of its predicted goodput (Q6).
- `DSD_V2_MODEL_RUNNER=1` in `experiments/bench/launch_server.sh`: serve with vLLM's V2 model runner.

## Known gaps

- **Per-request records** (AIPerf `profile_export.jsonl`, `gen.py` request lists) and server logs are not committed
  (several GB); the committed summaries are what every table and figure uses.
- **vLLM nightly `43b4aaea`** stalls when a table steps from length 2 to 1 under the V2 runner; such settings are
  reported with the tuner's 0/1/3 table, as the paper says (Q5).
- Results the paper marks as single runs are single runs here too.
