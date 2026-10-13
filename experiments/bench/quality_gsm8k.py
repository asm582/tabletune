"""Quality check for a model change (e.g. AWQ quantization): GSM8K accuracy.

Greedy, thinking off, the first N test problems. The answer is the number after
'####' in the reply (the prompt asks for it), else the last number in the reply.
Speculative decoding is lossless, so this is run without it: it measures the
model change only.

  # data (once, with any Python that has `datasets`):
  python experiments/bench/quality_gsm8k.py --export experiments/results/quality/gsm8k_test.jsonl
  # evaluate:
  TARGET_MODEL=Qwen/Qwen3-4B-AWQ python experiments/bench/quality_gsm8k.py \
      --data experiments/results/quality/gsm8k_test.jsonl --n 500 --out experiments/results/quality/gsm8k_awq.json
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

PROMPT = "{q}\nSolve it step by step, then give the final answer as a number after '####'."


def number(text: str) -> str | None:
    tail = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    if not nums:
        return None
    return nums[0 if "####" in text else -1].replace(",", "").rstrip(".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export")
    ap.add_argument("--data")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out")
    ap.add_argument("--spec", choices=["eagle3"], help="also run with speculative decoding (EAGLE-3, K=3): "
                    "checks that speculation leaves quality unchanged where greedy outputs are not "
                    "bit-exact (e.g. AWQ kernels)")
    a = ap.parse_args()
    if a.export:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        Path(a.export).parent.mkdir(parents=True, exist_ok=True)
        with open(a.export, "w") as f:
            for r in ds:
                f.write(json.dumps({"question": r["question"], "answer": r["answer"].split("####")[-1].strip()}) + "\n")
        print(f"[gsm8k] exported {len(ds)} problems -> {a.export}")
        return

    rows = [json.loads(l) for l in open(a.data)][:a.n]
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    from vllm import LLM, SamplingParams
    target = os.environ.get("TARGET_MODEL", "Qwen/Qwen3-4B")
    spec = ({"method": "eagle3", "model": "AngelSlim/Qwen3-4B_eagle3", "num_speculative_tokens": 3}
            if a.spec == "eagle3" else None)
    llm = LLM(model=target, dtype=os.environ.get("DSD_DTYPE", "float16"), max_model_len=4096,
              max_num_seqs=64, gpu_memory_utilization=0.90, seed=0, speculative_config=spec)
    outs = llm.chat([[{"role": "user", "content": PROMPT.format(q=r["question"])}] for r in rows],
                    SamplingParams(temperature=0.0, max_tokens=768),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    correct = [number(o.outputs[0].text) == r["answer"].replace(",", "") for o, r in zip(outs, rows)]
    n, k = len(correct), sum(correct)
    p = k / n
    half = 1.96 * math.sqrt(p * (1 - p) / n)
    res = {"model": target, "speculation": a.spec or "none", "n": n, "correct": k, "accuracy": round(p, 4),
           "ci95": [round(p - half, 4), round(p + half, 4)],
           "sample": [{"answer": r["answer"], "reply_tail": o.outputs[0].text[-200:]} for o, r in zip(outs[:3], rows)]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[gsm8k] {target} (spec: {a.spec or 'none'}): {k}/{n} = {p:.1%} (95% CI {p - half:.1%}..{p + half:.1%}) -> {a.out}")


if __name__ == "__main__":
    main()
