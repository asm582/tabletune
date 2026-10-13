"""Fit TurboSpec's linear step-time model to profiled step times.

Reads step-time profiles in the TableTune tuner's format (costs_k<K>.json: rows of batch_size, k,
step_ms) and least-squares fits the 4-coefficient model of controller.StepTimeModel. TurboSpec
profiles target and draft passes separately; the TableTune profiles time whole steps, so the model is
fitted to the step as a whole. By default the fit uses a sparse grid (batch sizes 1, 2, 4, ..., 128),
the kind of offline profile TurboSpec describes, not every measured point.

  python -m turbospec_style.fit --costs 'tuner-out/costs_k*.json' --form sequential --out model.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .controller import StepTimeModel


def load_rows(pattern: str, ks: set[int] | None, batch_sizes: set[int] | None) -> list[tuple[int, int, float]]:
    rows = []
    for f in sorted(glob.glob(pattern)):
        for r in json.loads(Path(f).read_text())["rows"]:
            if (ks is None or r["k"] in ks) and (batch_sizes is None or r["batch_size"] in batch_sizes):
                rows.append((r["batch_size"], r["k"], r["step_ms"]))
    if not rows:
        raise SystemExit(f"no profile rows match {pattern}")
    return rows


def solve(a: list[list[float]], y: list[float]) -> list[float]:
    """Least squares via the normal equations and Gaussian elimination (no numpy needed)."""
    n = len(a[0])
    m = [[sum(r[i] * r[j] for r in a) for j in range(n)] + [sum(r[i] * v for r, v in zip(a, y))] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-12:
            raise SystemExit("singular fit: profile more draft lengths or batch sizes")
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * z for x, z in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit(rows: list[tuple[int, int, float]], form: str) -> tuple[StepTimeModel, float]:
    a = [StepTimeModel.features(b, k, form) for b, k, _ in rows]
    model = StepTimeModel(solve(a, [t for _, _, t in rows]), form)
    err = max(abs(model.step_ms(b, k) / t - 1) for b, k, t in rows)
    return model, err


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--costs", required=True, help="glob of costs_k*.json files")
    ap.add_argument("--form", choices=["sequential", "block"], required=True,
                    help="sequential: one draft pass per token (EAGLE, draft model); block: one pass (DFlash)")
    ap.add_argument("--ks", default="", help="comma-separated draft lengths to fit on (default: all)")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128",
                    help="profiled batch sizes to fit on ('all' for every row)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ks = {int(x) for x in args.ks.split(",")} if args.ks else None
    bs = None if args.batch_sizes == "all" else {int(x) for x in args.batch_sizes.split(",")}
    rows = load_rows(args.costs, ks, bs)
    model, err = fit(rows, args.form)
    all_rows = load_rows(args.costs, ks, None)
    err_all = max(abs(model.step_ms(b, k) / t - 1) for b, k, t in all_rows)
    Path(args.out).write_text(json.dumps({"form": model.form, "coef": model.coef, "fit_points": len(rows),
                                          "max_rel_err_fit": err, "max_rel_err_all": err_all}, indent=1))
    print(f"[fit] {args.form}: coef {[round(c, 5) for c in model.coef]}  {len(rows)} points, "
          f"max error {err:.1%} on them, {err_all:.1%} on all {len(all_rows)} profiled points -> {args.out}")


if __name__ == "__main__":
    main()
