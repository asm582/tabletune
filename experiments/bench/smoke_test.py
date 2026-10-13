"""Smoke test: does vLLM run on this GPU, with and without EAGLE-3 / Dynamic SD?

Runs one mode per process (so each gets the whole GPU), writes greedy outputs
and spec-decode counters to experiments/results/smoke/<mode>.json. Compare modes with
`python experiments/bench/smoke_test.py --compare`.

Modes:
  nospec  - target model only
  eagle3  - EAGLE-3, fixed K
  dsd     - EAGLE-3 + Dynamic SD table (MRv1)
  ngram      - n-gram prompt lookup, fixed K (no drafter model)
  ngram-dsd  - n-gram + the same Dynamic SD table (issue #3)
  draft      - separate draft model (Qwen3-0.6B), fixed K (issue #4)
  draft-dsd  - draft model + the same Dynamic SD table
  dflash     - DFlash block drafter (z-lab/Qwen3-4B-DFlash-b16), fixed K (issue #7)
  dflash-dsd - DFlash + the same Dynamic SD table
"""

import argparse
import json
import os
import time
from pathlib import Path

TARGET = os.environ.get("TARGET_MODEL", "Qwen/Qwen3-4B")  # e.g. Qwen/Qwen3-4B-AWQ
DRAFTER = "AngelSlim/Qwen3-4B_eagle3"
K = 3
# Batch sizes 1-4 use K=3, 5-8 use K=1, larger batches K=0. The smoke batch
# (8 prompts, then single prompts) crosses both boundaries.
DSD_TABLE = [[1, 4, 3], [5, 8, 1], [9, 256, 0]]
NGRAM_MIN, NGRAM_MAX = 2, 4  # n-gram match length searched in the prompt + output so far
DRAFT_MODEL = "Qwen/Qwen3-0.6B"  # same tokenizer and vocabulary as the target
DFLASH_MODEL = "z-lab/Qwen3-4B-DFlash-b16"  # drafts a block of up to 16 tokens in one pass

PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain in two sentences why the sky is blue.",
    "List the first ten prime numbers.",
    "What is the capital of France? Answer in one word.",
    "Write a haiku about GPUs.",
    "Translate 'good morning' into Spanish, French and German.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "def fibonacci(n):\n    # complete this function",
]

OUT_DIR = Path(os.environ.get("SMOKE_OUT") or Path(__file__).resolve().parent.parent / "results" / "smoke")


def spec_config(mode: str) -> dict | None:
    if mode == "nospec":
        return None
    if mode.startswith("ngram"):
        cfg = {"method": "ngram", "num_speculative_tokens": K,
               "prompt_lookup_min": NGRAM_MIN, "prompt_lookup_max": NGRAM_MAX}
    elif mode.startswith("dflash"):
        cfg = {"method": "dflash", "model": DFLASH_MODEL, "num_speculative_tokens": K}
    elif mode.startswith("draft"):
        cfg = {"method": "draft_model", "model": DRAFT_MODEL, "num_speculative_tokens": K}
    else:
        cfg = {"method": "eagle3", "model": DRAFTER, "num_speculative_tokens": K}
    if mode.endswith("dsd"):
        cfg["num_speculative_tokens_per_batch_size"] = DSD_TABLE
    return cfg


def spec_counters(llm) -> dict:
    """Pull vllm:spec_decode_* counters from the offline metrics API."""
    out = {}
    try:
        for m in llm.get_metrics():
            if m.name.startswith("vllm:spec_decode"):
                out[m.name] = getattr(m, "value", None) or getattr(m, "values", None)
    except Exception as e:  # metrics API shape varies across versions
        out["error"] = repr(e)
    return out


def run(mode: str, tag: str = "") -> None:
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")  # MRv1
    import torch
    from vllm import LLM, SamplingParams

    cap = torch.cuda.get_device_capability()
    print(f"[smoke] torch={torch.__version__} gpu={torch.cuda.get_device_name()} cc={cap}")

    t0 = time.time()
    llm = LLM(
        model=TARGET,
        dtype=os.environ.get("DSD_DTYPE", "float16"),  # T4 (sm75) has no BF16; bfloat16 on Ampere+
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_seqs=64,
        speculative_config=spec_config(mode),
        disable_log_stats=False,
        seed=0,
    )
    load_s = time.time() - t0

    params = SamplingParams(temperature=0.0, max_tokens=128)
    chat = [[{"role": "user", "content": p}] for p in PROMPTS]
    kwargs = {"chat_template_kwargs": {"enable_thinking": False}}

    # Batched pass (batch 8) then sequential passes (batch 1) so a DSD table
    # sees more than one tier.
    t0 = time.time()
    batched = llm.chat(chat, params, **kwargs)
    single = [llm.chat([c], params, **kwargs)[0] for c in chat[:3]]
    gen_s = time.time() - t0

    n_tokens = sum(len(o.outputs[0].token_ids) for o in batched + single)
    result = {
        "mode": mode,
        "target": TARGET,
        "spec_config": spec_config(mode),
        "load_seconds": round(load_s, 1),
        "gen_seconds": round(gen_s, 2),
        "output_tokens": n_tokens,
        "tokens_per_second": round(n_tokens / gen_s, 1),
        "outputs": [o.outputs[0].text for o in batched],
        "token_ids": [list(o.outputs[0].token_ids) for o in batched],
        "single_token_ids": [list(o.outputs[0].token_ids) for o in single],
        "spec_counters": spec_counters(llm),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{mode}{tag}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[smoke] {mode}: {n_tokens} tokens in {gen_s:.1f}s "
          f"({result['tokens_per_second']} tok/s), load {load_s:.0f}s -> {path}")
    print(f"[smoke] spec counters: {result['spec_counters']}")


def compare() -> None:
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("*.json"))}
    if "nospec" not in runs:
        raise SystemExit("run --mode nospec first")
    ref = runs["nospec"]
    for mode, r in runs.items():
        if mode == "nospec":
            continue
        same = sum(a == b for a, b in zip(ref["token_ids"], r["token_ids"]))
        same1 = sum(a == b for a, b in zip(ref["single_token_ids"], r["single_token_ids"]))
        print(f"{mode:10s} greedy match vs nospec: batched {same}/{len(ref['token_ids'])}, "
              f"single {same1}/{len(ref['single_token_ids'])} | "
              f"{r['tokens_per_second']} tok/s (nospec {ref['tokens_per_second']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nospec", "eagle3", "dsd", "ngram", "ngram-dsd", "draft", "draft-dsd", "dflash", "dflash-dsd"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. _ctrl")
    a = ap.parse_args()
    compare() if a.compare else run(a.mode, a.tag)
