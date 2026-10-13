"""Write a random sample of a HuggingFace dataset's prompts to JSONL.

Takes prompts from the column aiperf uses for that dataset, so calibration and
benchmark prompts have the same shape; use a seed other than the sweep's (42).
Needs the `datasets` package:

  python dsd_tuning/tuner/sample_prompts.py --dataset likaixin/InstructCoder \\
      --column instruction --n 160 --out prompts.jsonl

--append-column input adds the code to edit (n-gram workloads); --aiperf
writes aiperf's {"text": ...} format for benchmarking.
"""

import argparse
import json
import logging
import random
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--column", required=True)
    ap.add_argument("--append-column", default=None,
                    help="append this column after a blank line, e.g. InstructCoder's `input` "
                         "(the code to edit) for a code-editing workload")
    ap.add_argument("--suffix", default="",
                    help="appended after a blank line, e.g. an instruction on the answer format")
    ap.add_argument("--aiperf", action="store_true",
                    help='write {"text": ...} lines for aiperf --custom-dataset-type single_turn')
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    ds = load_dataset(a.dataset, split=a.split)
    idx = random.Random(a.seed).sample(range(len(ds)), a.n)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i in idx:
            prompt = ds[i][a.column]
            if a.append_column:
                prompt = f"{prompt}\n\n{ds[i][a.append_column]}"
            if a.suffix:
                prompt = f"{prompt}\n\n{a.suffix}"
            row = {"text": prompt} if a.aiperf else {"prompt": prompt, "row": i}
            f.write(json.dumps(row) + "\n")
    logger.info(f"[prompts] {a.n} of {len(ds)} rows from {a.dataset}:{a.column} -> {out}")


if __name__ == "__main__":
    main()
