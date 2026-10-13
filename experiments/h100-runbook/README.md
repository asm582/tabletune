# H100 runbook: what to run for the paper, on OpenShift

Reviewers' main objection: everything was measured on two older, small GPUs with 2-core hosts. This runbook repeats the paper's experiments on **one H100 80 GB** with a normal host, adds an **8B** target, and tests whether the graph-mode overhead depends on host CPUs. Every step is a script in `scripts/`; run them in order inside one pod.

**Total time on one GPU:** about 30–35 hours for everything, about 15 hours for the recommended plan, about 9 hours for the minimum. **With three H100s in parallel** (section 4b): about 12, 6–8 and 4 hours. Each script can run on its own, in `tmux` or with `nohup`, and writes to the persistent volume, so a pod restart loses at most the step in progress.

| Plan | What runs | One GPU | 3 GPUs in parallel |
|---|---|---|---|
| Everything | all rows below, 3 runs each | ~30 h | ~12 h |
| **Recommended** | P0 + shift/jumping load with `QUICK=1` + 8B with 1 run (`RUNS=1`) | **~15 h** | **~6–8 h** |
| Minimum | tune 4B, main sweep 3 runs, overhead at 16 CPUs | ~9 h | ~4 h |

## What the paper needs, in priority order

| Priority | Experiment | Script | Time | What it answers |
|---|---|---|---|---|
| **P0** | Setup, environment record, smoke test | `00`, `01` | 0.5 h + downloads | |
| **P0** | Tune Qwen3-4B on the H100 (step costs to batch 256, 7 acceptance profiles) | `02` (MODEL=4b) | 1–1.5 h | the H100's table, and whether the best length still depends on load |
| **P0** | Main sweep, 4B: no spec, lengths 1/2/3, tuned table, docs table, 1–256 requests, **3 runs**; plus the A10G's table once | `03` (MODEL=4b) | 7–8 h | the paper's Table 1 on a modern GPU; table portability A10G → H100 |
| **P0** | Graph-mode overhead, 16 CPUs | `05` (MODEL=4b) | 1.5 h | is the 12–25% piecewise-graph cost a small-host artefact? |
| P1 | Traffic shift + jumping load, 4B, 3 runs | `pick_conc.py`, then `04` | 9–10 h | the controller result, on a modern GPU |
| P1 | Graph-mode overhead, 4 CPUs (second pod) | `05` in `pod-cpu4.yaml` | 1.5 h | where the overhead appears as the host shrinks |
| P2 | Tune + main sweep, **Qwen3-8B** + `AngelSlim/Qwen3-8B_eagle3` | `02`, `03` (MODEL=8b) | 8–9 h | a larger target (reviewers asked for 8B+) |

If you only have one day of GPU time: run all of P0 (about 11 hours), then P1's shift experiment if there's time left.

## 0. Before you start (cluster side)

1. **A whole H100.** No MIG partition and no GPU time-slicing: the numbers are per GPU. Check with:
   ```bash
   oc get nodes -L nvidia.com/gpu.product,nvidia.com/mig.config,nvidia.com/gpu.count
   ```
   Put the exact `gpu.product` value (for example `NVIDIA-H100-80GB-HBM3` for SXM, `NVIDIA-H100-PCIe`, `NVIDIA-H100-NVL`) into `openshift/pod.yaml`. Record which one it is; SXM and PCIe differ a lot.
2. **Outbound network** from the pod to:
   - `huggingface.co` and `cdn-lfs.huggingface.co` (models);
   - `github.com` (vLLM source) and `raw.githubusercontent.com` (Spec-Bench prompts);
   - `pypi.org`, `files.pythonhosted.org`, `download.pytorch.org`, `astral.sh` (uv, Python);
   - vLLM's precompiled wheel host (`wheels.vllm.ai`).

   If the cluster needs a proxy, set `HTTPS_PROXY`/`HTTP_PROXY`/`NO_PROXY` in the pod's `env`.
3. **Quota in your namespace:** 1 `nvidia.com/gpu`, 16 CPUs, 128 GiB memory, and a 200 GiB volume.
4. **Tooling:** the `oc` CLI logged in, and permission to run a build (or push an image to the cluster registry).

## 1. Build the image (once)

OpenShift runs pods as a random non-root user, so `apt` can't run inside the pod. The image adds `git`, `curl`, a C compiler (vLLM's Triton kernels compile at run time) and `procps`/`tmux`:

```bash
oc project <namespace>
oc new-build --name dsd-bench --binary --strategy docker
oc start-build dsd-bench --from-dir openshift/ --follow
```
Then put the resulting image (`image-registry.openshift-image-registry.svc:5000/<namespace>/dsd-bench:latest`) into `openshift/pod.yaml` and `openshift/pod-cpu4.yaml`.

## 2. Create the volume and the pod

For several GPUs at once, skip sections 2–4 and follow **4b. Parallel launch**.

```bash
oc apply -f openshift/pvc.yaml          # set storageClassName first if your cluster needs one
oc apply -f openshift/pod.yaml
oc wait --for=condition=Ready pod/dsd-h100 --timeout=20m
```

## 3. Copy the code in

The repository is private, so copy a snapshot instead of cloning inside the pod. Use **this branch's latest commit**:
```bash
# on your laptop, in a clone of the repo
git fetch && git archive --format=tar.gz -o /tmp/tabletune.tgz origin/main
oc cp /tmp/tabletune.tgz dsd-h100:/work/tabletune.tgz
oc cp experiments/h100-runbook dsd-h100:/work/runbook          # this folder
```

## 4. Run, inside the pod

```bash
oc rsh dsd-h100
tmux new -s bench            # reattach later with: tmux attach -t bench
cd /work/runbook/scripts
```

**Step 1: record the environment and set up** (P0):
```bash
bash 00_record_env.sh                       # GPU model, driver, clocks, power cap, CPUs, memory
bash 01_setup.sh 2>&1 | tee /work/out/setup.log
```
- **Expect:** `dsd-tuner 0.0.1`, `plugin registered`, `156 passed`, a short completion, a line `Dynamic SD controller enabled: …`, and `SETUP DONE`.
- **If the vLLM install fails** on the precompiled wheel, the wheel host is blocked; see step 0.2.

**Step 2: tune the 4B model** (P0, 1–1.5 h):
```bash
MODEL=4b bash 02_tune.sh 2>&1 | tee /work/out/02-4b.log
cat /work/tabletune/experiments/results/tune-h100-4b/schedule.json
```
- **Expect:** a table over batch sizes 1–256. Compare it with the A10G's, which switches to length 0 at 91/92 and which we can't predict for the H100. Then seven `=== acceptance …` blocks, each printing a table.
- **Watch:** the step-cost logs should show no "preempted" lines up to 256 (80 GB is plenty for 4B).

**Step 3: main sweep, 4B** (P0, 7–8 h):
```bash
MODEL=4b bash 03_sweep.sh 2>&1 | tee /work/out/03-4b.log
cat /work/tabletune/experiments/results/repeats/code-h100-4b.md
```
- **Expect:** 6 configurations × 11 loads × 3 runs, then one run of the A10G's table.
- **Check after run 1:** `experiments/results/sweeps/code-h100-4b/summary.csv` should show the `errors` column at 0 and `preemptions` at 0 at most loads.

**Step 4: graph-mode overhead with 16 CPUs** (P0, 1.5 h):
```bash
MODEL=4b bash 05_overhead.sh 2>&1 | tee /work/out/05-cpu16.log
```
- **What it measures, on the 4-bit (AWQ) target at 1–8 requests:**
  - fixed length 3 with full CUDA graphs;
  - a table with length 3 everywhere (piecewise graphs);
  - fixed length 3 forced to piecewise graphs;
  - no speculation in both graph modes.

  The paper found 12–25% lost at 1 request on 2-core hosts.

**Step 5: traffic shift and jumping load** (P1, 9–10 h). First pick the load where the translation and math tables disagree on this GPU:
```bash
source env.sh
python pick_conc.py --repo $REPO --costs "$TUNE/costs_k*.json" \
  --a $X/acc-$TAG-translation --b $X/acc-$TAG-math_reasoning --max-b 256
```
- **If it prints "agree everywhere":** skip the shift part, and record that result (it's still a finding). Experiment 2 can still run: comment out the `e1_` lines in `04_shift_vary.sh`.
- **Otherwise:**
  ```bash
  MODEL=4b SHIFT_CONC=<number it printed> bash 04_shift_vary.sh 2>&1 | tee /work/out/04-4b.log
  ```
  In the first run's output, check the `B=` column: the live batch should sit inside the range where the tables disagree. If not, stop, adjust `SHIFT_CONC`, and restart.

**Step 6: overhead with 4 CPUs** (P1, 1.5 h). This needs a new pod, because the volume can only attach to one pod at a time:
```bash
exit                                         # leave the pod
oc delete pod dsd-h100
oc apply -f openshift/pod-cpu4.yaml && oc wait --for=condition=Ready pod/dsd-h100-cpu4 --timeout=20m
oc rsh dsd-h100-cpu4
cd /work/runbook/scripts && bash 00_record_env.sh && MODEL=4b bash 05_overhead.sh 2>&1 | tee /work/out/05-cpu4.log
```
`nproc` must print 4. Sweep names include the CPU count, so the 16-CPU results stay separate.

**Step 7: the 8B model** (P2, 8–9 h), back in the 16-CPU pod:
```bash
MODEL=8b bash 02_tune.sh 2>&1 | tee /work/out/02-8b.log
MODEL=8b bash 03_sweep.sh 2>&1 | tee /work/out/03-8b.log
```

**Step 8: pack and copy the results out:**
```bash
bash 06_pack.sh
exit
oc cp dsd-h100:/work/out/h100-results.tgz ./h100-results.tgz
```


## 4b. Parallel launch (several H100s)

The three repeats of each experiment are independent, so each can run on its own GPU at the same time. All pods share one volume, so setup and model downloads happen once. The scripts take two switches:
- `RUNS=<n>`: run only repeat *n* (default `1 2 3`);
- `QUICK=1` (shift script only): in repeats 2–3, skip the single-option controllers and fixed length 2. Repeat 1 always runs everything.

**Requirements:**
- **One H100 per worker,** all of the same variant (for example all SXM). Mixing variants makes the repeats incomparable.
- **A ReadWriteMany storage class** for the shared volume, for example CephFS on OpenShift Data Foundation (`oc get storageclass`). Without one, give each pod its own `pvc.yaml` volume (each pod then installs and downloads separately, about 20 minutes extra each), and copy results out of every pod at the end.
- **Isolated CPUs:** workers on one node share its CPUs unless the kubelet's CPU manager is `static`. If you're not sure, uncomment the `podAntiAffinity` block in `openshift/pod-run.yaml` to put each worker on its own node.

**Phase A: shared volume, workers, setup and tuning (about 1.5 h).**
```bash
oc apply -f openshift/pvc-rwx.yaml                       # set storageClassName first
for r in 1 2 3; do sed "s/RUNID/$r/g" openshift/pod-run.yaml | oc apply -f -; done
oc wait --for=condition=Ready pod -l role=worker --timeout=20m
oc cp /tmp/tabletune.tgz dsd-h100-run1:/work/tabletune.tgz
oc cp experiments/h100-runbook dsd-h100-run1:/work/runbook              # the volume is shared: copy once

# setup once, from worker 1 (installs into /work, downloads all models)
oc exec dsd-h100-run1 -- bash -c 'cd /work/runbook/scripts && bash 00_record_env.sh && bash 01_setup.sh > /work/out/setup.log 2>&1; tail -3 /work/out/setup.log'
oc exec dsd-h100-run2 -- bash /work/runbook/scripts/00_record_env.sh      # record every worker's hardware
oc exec dsd-h100-run3 -- bash /work/runbook/scripts/00_record_env.sh

# tune the 4B model on worker 1 and the 8B model on worker 2, at the same time
oc exec dsd-h100-run1 -- bash -c 'cd /work/runbook/scripts && MODEL=4b setsid nohup bash 02_tune.sh > /work/out/02-4b.log 2>&1 &'
oc exec dsd-h100-run2 -- bash -c 'cd /work/runbook/scripts && MODEL=8b setsid nohup bash 02_tune.sh > /work/out/02-8b.log 2>&1 &'
```
Wait until both logs end with `TUNE DONE` (`oc exec dsd-h100-run1 -- tail -2 /work/out/02-4b.log`). The sweeps need the tuned table, so don't start Phase B earlier.

**Phase B: main sweep, one repeat per worker (about 2.5 h).**
```bash
for r in 1 2 3; do
  oc exec dsd-h100-run$r -- bash -c "cd /work/runbook/scripts && MODEL=4b RUNS=$r setsid nohup bash 03_sweep.sh > /work/out/03-4b-run$r.log 2>&1 &"
done
```
Worker 1 also runs the A10G table once at the end (about 25 minutes more). Each log ends with `SWEEP DONE`.

**Phase C: traffic shift and jumping load (about 2.5–3 h).** Pick the load once, then one repeat per worker:
```bash
oc exec dsd-h100-run1 -- bash -c 'source /work/runbook/scripts/env.sh && python /work/runbook/scripts/pick_conc.py --repo $REPO --costs "$TUNE/costs_k*.json" --a $X/acc-$TAG-translation --b $X/acc-$TAG-math_reasoning --max-b 256'
C=<number it printed>
for r in 1 2 3; do
  oc exec dsd-h100-run$r -- bash -c "cd /work/runbook/scripts && MODEL=4b RUNS=$r QUICK=1 SHIFT_CONC=$C setsid nohup bash 04_shift_vary.sh > /work/out/04-4b-run$r.log 2>&1 &"
done
```
After worker 1's first run, check the `B=` column in its log (README step 5) before trusting the other two.

**Phase D, on any free worker: the rest of the recommended plan.**
```bash
# graph-mode overhead with 16 CPUs (1.5 h)
oc exec dsd-h100-run2 -- bash -c 'cd /work/runbook/scripts && MODEL=4b setsid nohup bash 05_overhead.sh > /work/out/05-cpu16.log 2>&1 &'
# 8B main sweep, one run (about 2.5 h)
oc exec dsd-h100-run3 -- bash -c 'cd /work/runbook/scripts && MODEL=8b RUNS=1 setsid nohup bash 03_sweep.sh > /work/out/03-8b-run1.log 2>&1 &'
```
With four or more GPUs, start Phase D at the same time as Phase B on a fourth worker (`sed "s/RUNID/4/g" …`): it only needs Phase A.

**Optional: overhead with 4 CPUs.** On the shared volume this doesn't need any other pod deleted:
```bash
oc apply -f openshift/pod-cpu4.yaml && oc wait --for=condition=Ready pod/dsd-h100-cpu4 --timeout=20m
oc exec dsd-h100-cpu4 -- bash -c 'cd /work/runbook/scripts && bash 00_record_env.sh && MODEL=4b setsid nohup bash 05_overhead.sh > /work/out/05-cpu4.log 2>&1 &'
```

**Finish: means and the results file.** When every log ends with `DONE`:
```bash
oc exec dsd-h100-run1 -- bash /work/runbook/scripts/07_aggregate.sh
oc cp dsd-h100-run1:/work/out/h100-results.tgz ./h100-results.tgz
oc delete pod -l app=dsd-h100                            # free the GPUs; the volume keeps everything
```

**Watching progress:** each worker writes its own log in `/work/out` (`02-4b.log`, `03-4b-run2.log`, …):
```bash
for r in 1 2 3; do echo "== run$r"; oc exec dsd-h100-run$r -- bash -c 'tail -1 /work/out/0*-run*.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'; done
```

## 5. What to send back

`h100-results.tgz` contains everything I need to update the paper:
- the `env-*.txt` records;
- the tuner folders;
- the sweeps and their 3-run means;
- the overhead runs at 16 and 4 CPUs;
- the shift and jumping-load runs.

Also note anything the files can't show: the GPU variant (SXM, PCIe or NVL), whether another job shared the node, and any pod restarts.

## 6. Things that would make the numbers wrong

- **A shared or partitioned GPU** (MIG, time-slicing, another pod on the same GPU): invalid. Check that `nvidia-smi` shows only vLLM's process during runs.
- **CPU throttling:** keep requests equal to limits (as in the manifests). `00_record_env.sh` records `cpu.max`; it should match the CPU limit.
- **Power or clock caps:** `00_record_env.sh` records the power limit and max SM clock. Report them if they're below the GPU's default (700 W for SXM).
- **Changing anything between runs** (vLLM commit, image, flags): the three runs of a configuration must be identical except for prompt order.
- **Stopping mid-sweep:** `run_configs.sh` restarts a whole sweep, so rerun the step. Partial sweeps are labelled by name; delete a half-finished folder before rerunning it.

## 7. Settings used, for the record

| Setting | Value |
|---|---|
| vLLM | commit `353e491ef`, V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`), installed by the repo's `install.sh` |
| Precision | BF16 targets; the AWQ overhead study runs FP16 activations (as on T4/A10G) |
| Server | `--max-num-seqs 256` (T4/A10G used 128), `--max-model-len 4096`, GPU memory 0.90 (sweeps) / default (shift runs) |
| Tuner | grid 1–16 then every 8 to 256; 32-token prompts; lengths 0–3; acceptance on 160 code prompts at temperature 1.0 |
| Shift/jumping-load workloads | greedy; 7 workloads (5 Spec-Bench categories, code, code-apply); schedule as on the A10G, loads scaled to 256 |
| Controller | defaults (hold 4, rebuild every 500 steps, prior weight 2000, decay 0.5); prior decay 0.5; exploration every 50 steps, cost cap 2.0 |
