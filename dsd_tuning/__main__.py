"""dsd-tuner: measure step time, measure acceptance, apply one formula.

  dsd-tuner run          full tuning on a GPU: step costs + acceptance + table
  dsd-tuner recalibrate  new workload, existing step costs: acceptance + table (~2 min)
  dsd-tuner schedule     table from existing profiles (no GPU)
  dsd-tuner screen       how much would each fixed K lose? (no GPU)
  dsd-tuner tables       list the tuned tables in tables/ (--check to validate them)
  dsd-tuner serve [NAME] start vllm serve with a table and the controller (default table: this GPU's)
  dsd-tuner add-table    add a tuning run to tables/ as a new table (see CONTRIBUTING.md)

`run` and `recalibrate` write <out>/speculative_config.json and print the
`--speculative-config` value to pass to `vllm serve`.

Examples:
  dsd-tuner run --target Qwen/Qwen3-4B --drafter AngelSlim/Qwen3-4B_eagle3 \\
      --prompts-file docs/examples/code-prompts.jsonl --isl 32 --out my-tune
  dsd-tuner recalibrate --costs 'my-tune/costs_k*.json' \\
      --prompts-file docs/examples/code-edit-prompts.jsonl --out my-tune-edit
  dsd-tuner schedule --costs 'tables/t4-eagle3/costs_k*.json' \\
      --acceptance tables/t4-eagle3/acceptance.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from . import tables
from .tuner.profile_costs import BATCH_SIZES
from .core import (EPS, MIN_WIDTH_FRAC, build_schedule, compress, fixed_k_losses,
                   goodput_table, load_costs, load_tokens_per_step, speculative_config)

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
DEFAULT_BATCH_SIZES = ",".join(map(str, BATCH_SIZES))


# ------------------------------------------------------------------ helpers
def _ranged(kind: type, lo: float, hi: float | None = None, hi_open: bool = False):
    """argparse type: a number in [lo, hi] (or [lo, hi) with hi_open)."""
    def parse(text: str):
        try:
            v = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not {'an integer' if kind is int else 'a number'}: "
                                             f"{text!r}") from None
        if v != v or v < lo or (hi is not None and (v >= hi if hi_open else v > hi)):
            rng = f">= {lo}" if hi is None else f"in [{lo}, {hi}{')' if hi_open else ']'}"
            raise argparse.ArgumentTypeError(f"must be {rng}, got {text}")
        return v
    return parse


def _batch_sizes(text: str) -> str:
    """argparse type: comma-separated, increasing batch sizes >= 1 (kept as text)."""
    try:
        sizes = [int(x) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from None
    if not sizes or sizes[0] < 1 or any(b <= a for a, b in zip(sizes, sizes[1:])):
        raise argparse.ArgumentTypeError(f"batch sizes must be >= 1 and increasing, got {text!r}")
    return text


def _warn_extrapolation(costs: dict, max_num_seqs: int) -> None:
    profiled = min(xs[-1] for xs, _ in costs.values())
    if max_num_seqs > profiled:
        logger.warning("step costs are profiled up to batch size %d; batch sizes %d..%d are "
                       "extrapolated. Profile with --batch-sizes up to %d for a measured table.",
                       profiled, profiled + 1, max_num_seqs, max_num_seqs)


def _env(a: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"TARGET_MODEL": a.target, "DRAFTER_MODEL": a.drafter,
                "DSD_K_MAX": str(a.k_max), "DSD_DTYPE": a.dtype, "DSD_METHOD": a.method})
    return env


def _run(cmd: list[str], env: dict[str, str], log: Path) -> None:
    logger.info("$ %s  (log: %s)", " ".join(cmd), log)
    with log.open("w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f"[dsd-tuner] failed (exit {rc}), see {log}")


def _profile_costs(a: argparse.Namespace, out: Path) -> None:
    # One process per K so each one gets the whole GPU.
    for k in range(a.k_max + 1):
        _run([sys.executable, str(HERE / "tuner" / "profile_costs.py"), "--k", str(k),
              "--isl", str(a.isl), "--max-num-seqs", str(a.max_num_seqs),
              "--batch-sizes", a.batch_sizes, "--out", str(out / f"costs_k{k}.json")],
             _env(a), out / f"costs_k{k}.log")


def _profile_acceptance(a: argparse.Namespace, out: Path) -> Path:
    src = ["--prompts-file", a.prompts_file] if a.prompts_file else ["--sweep", a.sweep]
    dest = out / "acceptance.json"
    _run([sys.executable, str(HERE / "tuner" / "profile_acceptance.py"), *src,
          "--temperature", str(a.temperature), "--out", str(dest)],
         _env(a), out / "acceptance.log")
    return dest


def _schedule(costs_glob: str, acceptance: Path, max_num_seqs: int, out: Path | None,
              drafter: str | None, method: str, eps: float, min_width_frac: float) -> list:
    costs = load_costs(costs_glob)
    _warn_extrapolation(costs, max_num_seqs)
    tps, rate = load_tokens_per_step(acceptance)
    goodput = goodput_table(costs, tps, max_num_seqs)
    if rate < 1.0:
        logger.info("proposal rate %.3f: the drafter proposes on that share of steps (n-gram); "
                    "tokens per step scale with p, step cost is taken at the full K", rate)
    raw, best = build_schedule(goodput, eps, min_width_frac)
    schedule = compress(best)
    print(f"[dsd-tuner] tokens per step by K: {{{', '.join(f'{k}: {v:.3f}' for k, v in tps.items())}}}")
    print(f"[dsd-tuner] raw per-batch best: {compress(raw)}")
    print(f"[dsd-tuner] table: {schedule}")
    drafter = drafter or json.loads(Path(acceptance).read_text()).get("drafter")
    if drafter:
        cfg = speculative_config(schedule, drafter, method)
        print(f"\n--speculative-config '{json.dumps(cfg)}'\n")
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "schedule.json").write_text(json.dumps(schedule))
        if drafter:
            (out / "speculative_config.json").write_text(json.dumps(cfg, indent=2))
        logger.info("wrote %s%s", out / "schedule.json",
                    f" and {out / 'speculative_config.json'}" if drafter else "")
    return schedule


# ----------------------------------------------------------------- commands
def cmd_run(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _profile_costs(a, out)
    acc = _profile_acceptance(a, out)
    _schedule(str(out / "costs_k*.json"), acc, a.max_num_seqs, out, a.drafter, a.method,
              a.eps, a.min_width_frac)


def cmd_recalibrate(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = _profile_acceptance(a, out)
    _schedule(a.costs, acc, a.max_num_seqs, out, a.drafter, a.method, a.eps, a.min_width_frac)


def cmd_schedule(a: argparse.Namespace) -> None:
    _schedule(a.costs, Path(a.acceptance), a.max_num_seqs, Path(a.out) if a.out else None,
              a.drafter, a.method, a.eps, a.min_width_frac)


def cmd_screen(a: argparse.Namespace) -> None:
    costs = load_costs(a.costs)
    _warn_extrapolation(costs, a.max_num_seqs)
    tps, rate = load_tokens_per_step(a.acceptance)
    goodput = goodput_table(costs, tps, a.max_num_seqs)
    losses = fixed_k_losses(goodput)
    print(f"[dsd-tuner] predicted loss of each fixed K vs. the best K at each batch size "
          f"(B = 1..{a.max_num_seqs}):")
    for k, v in losses.items():
        print(f"  K={k}: worst {v['worst_loss']:5.1%} at B={v['worst_b']:3d}, "
              f"average {v['avg_loss']:5.1%}")
    best_fixed = min((v["worst_loss"], k) for k, v in losses.items() if k > 0)
    print(f"[dsd-tuner] best single fixed K = {best_fixed[1]} (worst loss {best_fixed[0]:.1%}). "
          f"A table is worth it when this is large (e.g. >5-10%).")
    print("[dsd-tuner] note: the formula ignores prefill, queueing and KV-cache pressure; "
          "profile step costs at the workload's prompt length.")


def cmd_tables(a: argparse.Namespace) -> None:
    if not a.check:
        tables.print_tables()
        return
    results = tables.check_all()
    for name, problems in results.items():
        print(f"{'ok  ' if not problems else 'FAIL'} {name}" + "".join(f"\n     - {p}" for p in problems))
    if any(results.values()):
        sys.exit(1)


def _version() -> str:
    """The installed package version (pyproject.toml), or "unknown" when run from source."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("dsd-tuning")
    except PackageNotFoundError:
        return "unknown"


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="dsd-tuner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--target", default="Qwen/Qwen3-4B")
        p.add_argument("--drafter", default="AngelSlim/Qwen3-4B_eagle3")
        p.add_argument("--method", default="eagle3")
        p.add_argument("--k-max", type=_ranged(int, 0), default=3)
        p.add_argument("--dtype", default="float16")

    def table_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-num-seqs", type=_ranged(int, 1), default=128,
                       help="the server's --max-num-seqs (default 128)")
        p.add_argument("--eps", type=_ranged(float, 0, 1, hi_open=True), default=EPS)
        p.add_argument("--min-width-frac", type=_ranged(float, 0), default=MIN_WIDTH_FRAC)

    def prompt_args(p: argparse.ArgumentParser) -> None:
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--prompts-file", help="JSONL, one {\"prompt\": ...} per line")
        g.add_argument("--sweep", help="previous sweep dir: use its unused prompts")
        p.add_argument("--temperature", type=_ranged(float, 0), default=0.0)

    p = sub.add_parser("run", help="full tuning on a GPU")
    model_args(p); table_args(p); prompt_args(p)
    p.add_argument("--isl", type=_ranged(int, 1), required=True,
                   help="your workload's typical prompt length in tokens (step cost depends on it)")
    p.add_argument("--batch-sizes", type=_batch_sizes, default=DEFAULT_BATCH_SIZES)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("recalibrate", help="new workload, reuse step costs")
    model_args(p); table_args(p); prompt_args(p)
    p.add_argument("--costs", required=True, help="glob of costs_k*.json")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_recalibrate)

    p = sub.add_parser("schedule", help="table from existing profiles (no GPU)")
    table_args(p)
    p.add_argument("--costs", required=True)
    p.add_argument("--acceptance", required=True)
    p.add_argument("--drafter", help="default: read from the acceptance file")
    p.add_argument("--method", default="eagle3")
    p.add_argument("--out")
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("screen", help="predicted loss of each fixed K (no GPU)")
    table_args(p)
    p.add_argument("--costs", required=True)
    p.add_argument("--acceptance", required=True)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("tables", help="list the tuned tables in tables/")
    p.add_argument("--check", action="store_true", help="validate every table (what CI runs)")
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("serve", help="start vllm serve with a table from tables/")
    p.add_argument("table", nargs="?", help="name from `dsd-tuner tables`; default: this GPU's table")
    p.add_argument("--controller", action=argparse.BooleanOptionalAction, default=True,
                   help="the in-engine controller (hold + re-fit); on by default, "
                        "--no-controller serves the table alone")
    p.add_argument("--port", type=_ranged(int, 1), default=8000)
    p.add_argument("--dry-run", action="store_true", help="print the command, don't run it")
    p.set_defaults(func=lambda a: tables.serve(a.table, a.controller, a.port, a.dry_run))

    p = sub.add_parser("add-table", help="add a `dsd-tuner run` output to tables/ (see CONTRIBUTING.md)")
    p.add_argument("--from", dest="src", required=True, help="the --out folder of `dsd-tuner run`")
    p.add_argument("--name", required=True, help="e.g. h100-llama3-8b-eagle3 (lower case, dashes)")
    p.add_argument("--gpu", required=True, help='GPU as nvidia-smi names it, e.g. "NVIDIA H100 80GB HBM3"')
    p.add_argument("--match", nargs="+", required=True,
                   help="word(s) in the GPU name that select this table, e.g. H100")
    p.add_argument("--dtype", required=True, help="the --dtype used for tuning (float16 or bfloat16)")
    p.add_argument("--measured", required=True, help="one line: workload and result, e.g. from a benchmark")
    p.add_argument("--contributor", required=True, help="your GitHub handle")
    p.add_argument("--default", action="store_true", help="serve this table by default on a matching GPU")
    p.add_argument("--max-model-len", type=_ranged(int, 1), default=4096,
                   help="vllm serve --max-model-len for this table (default 4096)")
    p.add_argument("--model-runner", choices=tables.RUNNERS, default="v1",
                   help="the vLLM model runner it was tuned and measured on (default v1; v2 keeps full CUDA graphs per K)")
    p.add_argument("--tensor-parallel-size", type=_ranged(int, 1), default=1,
                   help="vllm serve --tensor-parallel-size (tune with DSD_TP set to the same value)")
    p.add_argument("--vllm", help="the vLLM version or commit it was measured with, e.g. nightly-43b4aaea")
    p.add_argument("--source", help="optional: where the table came from (a study folder or a link)")
    p.set_defaults(func=lambda a: print(
        f"[dsd-tuner] added {tables.add(a.src, a.name, a.gpu, a.match, a.dtype, a.measured, a.contributor, a.default, a.max_model_len, a.model_runner, a.tensor_parallel_size, a.vllm, a.source)}; "
        "next: git add it and open a pull request"))

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        a.func(a)
    except (OSError, ValueError) as e:  # bad or missing profile: one line, no traceback
        sys.exit(f"[dsd-tuner] error: {e}")


if __name__ == "__main__":
    main()
