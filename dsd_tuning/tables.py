"""The table catalog: every tuned table in tables/<name>/, ready to serve with vLLM.

A table is a folder with a ``table.json`` describing where it was measured and
how to serve it, next to the tuner's output: ``schedule.json`` (the table),
``acceptance.json`` and ``costs_k*.json`` (what the controller re-fits from).
Adding a folder is all it takes: ``dsd-tuner tables`` lists it, ``dsd-tuner serve <name>`` serves it, and
with ``"default": true`` it is picked on a matching GPU. ``dsd-tuner add-table``
creates the folder from a tuning run; see CONTRIBUTING.md.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from .controller import controller_from_env
from .core import expand, load_acceptance, load_costs, speculative_config

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
REQUIRED = ("gpu", "gpu_match", "target", "method", "drafter", "num_speculative_tokens",
            "dtype", "max_num_seqs", "max_model_len", "measured", "contributor")
# Optional fields: "model_runner" ("v1", the default, or "v2": vLLM's V2 model runner, which keeps full CUDA graphs
# for every draft length), "tensor_parallel_size" (default 1), "vllm" (the vLLM it was measured with), "source".
RUNNERS = ("v1", "v2")


def tables_dir() -> Path:
    """The catalog folder: $DSD_TABLES, else tables/ in this clone, else ./tables."""
    if env := os.environ.get("DSD_TABLES"):
        return Path(env)
    for d in (REPO / "tables", Path.cwd() / "tables"):
        if d.is_dir():
            return d
    raise ValueError("no tables/ folder found: run from a clone of the repository "
                     "(install.sh), or set DSD_TABLES to the catalog folder")
NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def load_all() -> dict[str, dict]:
    """Every table in tables/, by folder name, with the path of its folder."""
    out = {}
    for f in sorted(tables_dir().glob("*/table.json")):
        t = json.loads(f.read_text())
        t["path"] = f.parent
        out[f.parent.name] = t
    return out


def get(name: str) -> dict:
    tables = load_all()
    if name not in tables:
        raise ValueError(f"unknown table {name!r}; choose from: {', '.join(tables)}")
    return tables[name]


def check(name: str, t: dict) -> list[str]:
    """Problems that would make the table fail to serve or the controller refuse it."""
    problems = [f"table.json: missing {k!r}" for k in REQUIRED if k not in t]
    if not NAME.match(name):
        problems.append("name: use lower-case letters, digits and dashes")
    if problems:
        return problems
    p = t["path"]
    try:
        schedule = json.loads((p / "schedule.json").read_text())
        load_acceptance(p / "acceptance.json")
        costs = load_costs(str(p / "costs_k*.json"))
    except (OSError, ValueError, KeyError) as e:
        return [f"data in {p}: {e}"]
    if not schedule:
        return ["schedule.json is empty"]
    ks = {k for _, _, k in schedule}
    if max(ks) > t["num_speculative_tokens"]:
        problems.append(f"schedule uses K={max(ks)} > num_speculative_tokens={t['num_speculative_tokens']}")
    covered = [b for s, e, _ in sorted(schedule) for b in range(s, e + 1)]
    if covered != list(range(1, t["max_num_seqs"] + 1)):
        problems.append(f"schedule must cover batch sizes 1..{t['max_num_seqs']} once, in order")
    if missing := sorted(ks - set(costs)):
        problems.append(f"no costs_k*.json for K={missing} (the controller needs them)")
    runner, tp = t.get("model_runner", "v1"), t.get("tensor_parallel_size", 1)
    if runner not in RUNNERS:
        problems.append(f"model_runner must be one of {RUNNERS}, not {runner!r}")
    if not isinstance(tp, int) or isinstance(tp, bool) or tp < 1:
        problems.append(f"tensor_parallel_size must be a positive integer, not {tp!r}")
    order = [k for _, _, k in sorted(schedule)]
    if runner == "v2" and any(a == 2 and b == 1 for a, b in zip(order, order[1:])):
        problems.append("schedule steps from K=2 to K=1, which can stall vLLM's V2 model runner (nightly 43b4aaea); "
                        "build it from the K=0, 1 and 3 profiles (dsd-tuner schedule --costs '.../costs_k[013].json')")
    if not problems:
        env = {"VLLM_DSD_CONTROLLER": "1", "VLLM_DSD_COSTS": str(p / "costs_k*.json"),
               "VLLM_DSD_PRIOR": str(p / "acceptance.json")}
        try:
            controller_from_env(expand(schedule, t["max_num_seqs"]), t["num_speculative_tokens"],
                                env=env, method=t["method"])
        except ValueError as e:
            problems.append(f"controller refuses it: {e}")
    return problems


def default_clashes(tables: dict[str, dict]) -> dict[str, list[str]]:
    """GPU match word -> the default tables for it, where there is more than one."""
    out = {}
    for gpu in sorted({m for t in tables.values() for m in t.get("gpu_match", [])}):
        defaults = [n for n, t in tables.items() if t.get("default") and gpu in t.get("gpu_match", [])]
        if len(defaults) > 1:
            out[gpu] = defaults
    return out


def check_all(tables: dict[str, dict] | None = None) -> dict[str, list[str]]:
    """Problems per table, for the catalog (or the given tables): what `tables --check` runs."""
    tables = load_all() if tables is None else tables
    out = {name: check(name, t) for name, t in tables.items()}
    for gpu, defaults in default_clashes(tables).items():
        for n in defaults:
            out[n].append(f"more than one default table for GPU {gpu!r}: {defaults}")
    return out


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        raise ValueError("no NVIDIA GPU found (nvidia-smi failed); name a table explicitly")


def detect() -> str:
    """The default table for this machine's GPU."""
    gpu = gpu_name()
    for name, t in load_all().items():
        if t.get("default") and any(m in gpu.split() for m in t["gpu_match"]):
            return name
    raise ValueError(f"no default table for {gpu!r}. Tune one (docs/TUNING.md) and add it (CONTRIBUTING.md)")


def serve_command(name: str, controller: bool = True, port: int = 8000) -> tuple[dict, list[str]]:
    """(environment variables, argv) for `vllm serve` with the table."""
    t = get(name)
    schedule = json.loads((t["path"] / "schedule.json").read_text())
    spec = speculative_config(schedule, t["drafter"], t["method"], k_max=t["num_speculative_tokens"])
    # V1 model runner unless the table was measured on the V2 runner (full CUDA graphs for every draft length)
    env = {"VLLM_USE_V2_MODEL_RUNNER": "1" if t.get("model_runner", "v1") == "v2" else "0"}
    if controller:
        env |= {"VLLM_DSD_CONTROLLER": "1",
                "VLLM_DSD_COSTS": str(t["path"] / "costs_k*.json"),
                "VLLM_DSD_PRIOR": str(t["path"] / "acceptance.json")}
    argv = ["vllm", "serve", t["target"], "--dtype", t["dtype"],
            "--max-model-len", str(t["max_model_len"]), "--max-num-seqs", str(t["max_num_seqs"]),
            "--port", str(port), "--speculative-config", json.dumps(spec)]
    if (tp := t.get("tensor_parallel_size", 1)) > 1:
        argv += ["--tensor-parallel-size", str(tp)]
    return env, argv


def serve(name: str | None, controller: bool, port: int, dry_run: bool) -> None:
    if name is None:
        name = detect()
        logger.info("table %s for this GPU", name)
    env, argv = serve_command(name, controller, port)
    if vllm := get(name).get("vllm"):
        logger.info("table %s was measured with vLLM %s (see docs/RUN.md)", name, vllm)
    shown = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + shlex.join(argv)
    if dry_run:
        print(shown)
        return
    logger.info("%s", shown)
    os.execvpe(argv[0], argv, os.environ | env)


def print_tables() -> None:
    rows = [(n + ("*" if t.get("default") else ""), t["gpu"], t["target"], t["drafter"], t["measured"])
            for n, t in load_all().items()]
    rows.insert(0, ("table", "GPU", "model", "drafter", "measured"))
    w = [max(len(r[i]) for r in rows) for i in range(4)]
    for r in rows:
        print("  ".join(f"{c:<{w[i]}}" for i, c in enumerate(r[:4])) + "  " + r[4])
    print("\n* = picked automatically on that GPU.  Serve one:  dsd-tuner serve <table>")


def add(src: str, name: str, gpu: str, match: list[str], dtype: str, measured: str,
        contributor: str, default: bool, max_model_len: int = 4096, model_runner: str = "v1",
        tensor_parallel_size: int = 1, vllm: str | None = None, source: str | None = None) -> Path:
    """Create tables/<name>/ from a `dsd-tuner run` output folder, if `tables --check` would pass."""
    src_path, dest = Path(src), tables_dir() / name
    if dest.exists():
        raise ValueError(f"{dest} already exists")
    cost_files = sorted(src_path.glob("costs_k*.json"))
    if not cost_files:
        raise ValueError(f"no costs_k*.json in {src_path}: pass the --out folder of `dsd-tuner run`")
    acc = json.loads((src_path / "acceptance.json").read_text())
    cost0 = json.loads(cost_files[0].read_text())
    method = acc.get("method") or "eagle3"
    k_max = acc.get("k_max") or max(k for _, _, k in json.loads((src_path / "schedule.json").read_text()))
    table = {"gpu": gpu, "gpu_match": match, "default": default, "target": acc["target"],
             "method": method, "drafter": acc.get("drafter", ""), "num_speculative_tokens": k_max,
             "dtype": dtype, "max_num_seqs": cost0["max_num_seqs"], "max_model_len": max_model_len,
             "measured": measured,
             "contributor": contributor}
    if model_runner != "v1":
        table["model_runner"] = model_runner
    if tensor_parallel_size != 1:
        table["tensor_parallel_size"] = tensor_parallel_size
    if vllm:
        table["vllm"] = vllm
    if source:
        table["source"] = source
    _check_before_adding(name, dict(table, path=src_path))
    dest.mkdir(parents=True)
    for f in [src_path / "schedule.json", src_path / "acceptance.json", *cost_files]:
        shutil.copy(f, dest)
    (dest / "table.json").write_text(json.dumps(table, indent=2) + "\n")
    t = dict(table, path=dest)
    if problems := check(name, t):
        shutil.rmtree(dest)
        raise ValueError("table not added: " + "; ".join(problems))
    return dest


def _check_before_adding(name: str, t: dict) -> None:
    """Run `tables --check` on the catalog plus the new table (its files still in the
    run folder), so nothing is written when the result would fail it."""
    catalog = load_all()
    for gpu, defaults in default_clashes(catalog | {name: t}).items():
        if name in defaults:
            old = [n for n in defaults if n != name]
            raise ValueError(
                f"table not added: {', '.join(old)} is already the default for GPU {gpu!r} (one default per GPU). "
                f"Drop --default, or first set \"default\": false in {tables_dir() / old[0] / 'table.json'}")
    problems = {n: p for n, p in check_all(catalog | {name: t}).items() if p}
    if problems:
        raise ValueError("table not added, `dsd-tuner tables --check` would fail: "
                         + "; ".join(f"{n}: {', '.join(p)}" for n, p in problems.items()))
