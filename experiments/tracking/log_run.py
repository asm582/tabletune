"""Log benchmark sweeps to MLflow: one parent run per sweep, one child run per
(config, concurrency) point, with the server config aiperf can't see.

Reads what the harness already writes, so it works for old runs (backfill)
and new ones (called by experiments/bench/run_configs.sh when MLFLOW_TRACKING_URI is set):

  experiments/results/sweeps/<sweep>/meta.txt                      vLLM commit, GPU, dataset,
                                                       max_num_seqs, temperature
  experiments/results/sweeps/<sweep>/<config>/server.log           speculative config, controller
  experiments/results/sweeps/<sweep>/<config>/c<N>/*.json          aiperf + vLLM /metrics

Each point is keyed by the tag run_key = <sweep>/<config>/c<N>; existing points
are skipped unless --force (then replaced).

Usage (on the instance, tunnel to the laptop MLflow up, uv pip install -e ".[tracking]" done):
  MLFLOW_TRACKING_URI=http://localhost:5000 python experiments/tracking/log_run.py \\
      experiments/results/sweeps/code128t-t4 [more sweep dirs] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from plot_pareto import kv_pressure, metric, spec_counters  # noqa: E402

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "vllm-perf-dsd")


# ------------------------------------------------------------------ parsing
def parse_meta(root: Path) -> dict:
    meta = {"max_num_seqs": "64", "temperature": "0"}   # defaults before these were recorded
    f = root / "meta.txt"
    if not f.exists():
        return meta
    for line in f.read_text().splitlines():
        if line.startswith("vllm "):
            meta["vllm_commit"] = line.split()[1][:12]
        elif line.startswith("dataset "):
            meta["dataset"] = line.split()[1]
        elif line.startswith("max_num_seqs "):
            m = re.match(r"max_num_seqs (\S+) \| temperature (\S+)", line)
            if m:
                meta["max_num_seqs"], meta["temperature"] = m.groups()
        elif "," in line and "vllm" not in line:
            meta["gpu"] = line.split(",")[0].strip()
    return meta


def parse_server_log(cfg_dir: Path) -> dict:
    out: dict = {"speculation": "none", "controller": "off"}
    log = cfg_dir / "server.log"
    if not log.exists():
        return out
    text = log.read_text(errors="replace")
    launch = next((l for l in text.splitlines() if l.startswith("[launch]")), "")
    for flag, key in (("--dtype", "dtype"), ("--max-model-len", "max_model_len"),
                      ("--max-num-seqs", "max_num_seqs")):
        m = re.search(rf"{flag} (\S+)", launch)
        if m:
            out[key] = m.group(1)
    m = re.search(r"serve (\S+)", launch)
    if m:
        out["target_model"] = m.group(1)
    if "--speculative-config " in launch:
        # raw_decode: other flags (e.g. --compilation-config) may follow the JSON
        spec, _ = json.JSONDecoder().raw_decode(launch.split("--speculative-config ", 1)[1])
        out["spec_method"] = spec.get("method")
        out["drafter"] = spec.get("model")
        out["num_speculative_tokens"] = spec.get("num_speculative_tokens")
        table = spec.get("num_speculative_tokens_per_batch_size")
        out["speculation"] = "dynamic" if table else f"fixed K={spec.get('num_speculative_tokens')}"
        if table:
            out["dsd_table"] = json.dumps(table)
    if "--compilation-config " in launch:
        comp, _ = json.JSONDecoder().raw_decode(launch.split("--compilation-config ", 1)[1])
        out["compilation_config"] = json.dumps(comp)
    m = re.search(r"CUDAGraphMode\.([A-Z_]+)", text)
    if m:
        out["cudagraph_mode"] = m.group(1)
    m = re.search(r"Dynamic SD controller enabled: hysteresis=(\d+), refit=([^,]+), "
                  r"allowed K=(\[[^\]]*\])", text)
    if m:
        out["controller"] = "on"
        out["controller_hysteresis"] = m.group(1)
        out["controller_refit"] = m.group(2)
        out["controller_allowed_k"] = m.group(3)
        out["controller_table_changes"] = len(re.findall(r"Dynamic SD controller: new table", text))
        tables = re.findall(r"Dynamic SD controller: new table (\[\[.*?\]\])", text)
        if tables:
            out["controller_last_table"] = tables[-1]
    env = re.search(r"\[launch\] env: (.*)", text)
    if env:
        out["controller_env"] = env.group(1)[:500]
    return out


def point_metrics(cdir: Path) -> dict:
    d = json.loads((cdir / "profile_export_aiperf.json").read_text())
    m = {
        "tok_s_gpu": metric(d, "output_token_throughput"),
        "tok_s_user": metric(d, "output_token_throughput_per_user"),
        "request_throughput": metric(d, "request_throughput"),
        "ttft_ms_p50": metric(d, "time_to_first_token", "p50"),
        "ttft_ms_p99": metric(d, "time_to_first_token", "p99"),
        "itl_ms_p50": metric(d, "inter_token_latency", "p50"),
        "itl_ms_p99": metric(d, "inter_token_latency", "p99"),
        "requests": metric(d, "request_count"),
        "errors": len(d.get("error_summary") or []),
    }
    for name, key in (("isl", "input_sequence_length"), ("osl", "output_sequence_length")):
        for stat in ("min", "p50", "avg", "max"):
            m[f"{name}_{stat}"] = metric(d, key, stat)
    sc = spec_counters(cdir / "server_metrics_export.json")
    drafts = sc.get("vllm:spec_decode_num_drafts")
    acc = sc.get("vllm:spec_decode_num_accepted_tokens")
    drafted = sc.get("vllm:spec_decode_num_draft_tokens")
    if acc and drafted and drafts:
        m["acceptance_rate"] = acc / drafted
        m["tokens_per_step"] = 1 + acc / drafts
        for i, a in enumerate(sc.get("vllm:spec_decode_num_accepted_tokens_per_pos") or [], 1):
            m[f"accept_pos{i}"] = a / drafts
    m.update(kv_pressure(cdir / "server_metrics_export.json"))
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}


# ------------------------------------------------------------------ logging
def existing(client: MlflowClient, exp_id: str, key: str, value: str):
    runs = client.search_runs([exp_id], f"tags.{key} = '{value}'", max_results=10)
    return runs


def log_sweep(root: Path, force: bool) -> tuple[int, int]:
    client = MlflowClient()
    exp_id = mlflow.set_experiment(EXPERIMENT).experiment_id
    meta = parse_meta(root)
    group = root.name
    cfg_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                      and any(p.glob("c*/profile_export_aiperf.json")))
    if not cfg_dirs:
        print(f"[track] {group}: no config folders with results (symlink-only compare dir?), skipped")
        return 0, 0

    parents = existing(client, exp_id, "group_key", group)
    if parents and force:
        for r in parents:
            for child in existing(client, exp_id, "mlflow.parentRunId", r.info.run_id):
                client.delete_run(child.info.run_id)
            client.delete_run(r.info.run_id)
        parents = []
    if parents:
        parent_id = parents[0].info.run_id
    else:
        with mlflow.start_run(run_name=group, tags={"group_key": group, "kind": "sweep",
                                                    **{k: v for k, v in meta.items()}}) as p:
            parent_id = p.info.run_id
            for f in ("pareto.png", "summary.csv", "meta.txt", "README.md"):
                if (root / f).exists():
                    mlflow.log_artifact(str(root / f))

    logged = skipped = 0
    for cfg_dir in cfg_dirs:
        server = parse_server_log(cfg_dir)
        for cdir in sorted(cfg_dir.glob("c*/"), key=lambda p: int(p.name[1:])):
            if not (cdir / "profile_export_aiperf.json").exists():
                continue
            key = f"{group}/{cfg_dir.name}/{cdir.name}"
            old = existing(client, exp_id, "run_key", key)
            if old and not force:
                skipped += 1
                continue
            for r in old:
                client.delete_run(r.info.run_id)
            conc = int(cdir.name[1:])
            params = {**meta, **{k: v for k, v in server.items() if not k.startswith("controller_") or k in (
                "controller_hysteresis", "controller_refit", "controller_allowed_k", "controller_last_table")},
                "config": cfg_dir.name, "concurrency": conc}
            tags = {"run_key": key, "sweep": group, "config": cfg_dir.name,
                    "dataset": meta.get("dataset", "?"), "speculation": server["speculation"],
                    "controller": server["controller"], "concurrency": str(conc)}
            with mlflow.start_run(run_name=f"{cfg_dir.name} c={conc}", nested=True,
                                  parent_run_id=parent_id, tags=tags):
                mlflow.log_params({k: str(v)[:6000] for k, v in params.items() if v is not None})
                mlflow.log_metrics(point_metrics(cdir))
                if "controller_table_changes" in server:
                    mlflow.log_metric("controller_table_changes", server["controller_table_changes"])
                mlflow.log_artifact(str(cdir / "profile_export_aiperf.json"))
            logged += 1
    print(f"[track] {group}: logged {logged}, skipped {skipped} (already in MLflow)")
    return logged, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="+", help="experiments/results/sweeps/<name> folders")
    ap.add_argument("--force", action="store_true", help="replace points already logged")
    a = ap.parse_args()
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        sys.exit("set MLFLOW_TRACKING_URI (e.g. http://localhost:5000 through the tunnel)")
    total = 0
    for s in a.sweeps:
        total += log_sweep(Path(s), a.force)[0]
    print(f"[track] done: {total} points logged to experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
