"""Measure per-position draft acceptance on a set of calibration prompts.

Runs the drafter at K=K_MAX with real rejection sampling and reads vLLM's
spec-decode counters: a_i = accepted at position i / drafts, the probability
that the first i drafts are all accepted. For drafters that don't propose on
every step (n-gram) it also writes proposal_rate = drafts / decode steps.

Prompts come from --prompts-file (JSONL, one {"prompt": ...} per line) or
--sweep DIR: the prompts a previous benchmark sweep did not send, so the tuner
isn't fitted on its evaluation set.

  python dsd_tuning/tuner/profile_acceptance.py --prompts-file prompts.jsonl --out my-tune/acceptance.json
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


TARGET = os.environ.get("TARGET_MODEL", "Qwen/Qwen3-4B")
DRAFTER = os.environ.get("DRAFTER_MODEL", "AngelSlim/Qwen3-4B_eagle3")
K_MAX = int(os.environ.get("DSD_K_MAX", "3"))
DTYPE = os.environ.get("DSD_DTYPE", "float16")  # T4 (sm75) has no BF16
METHOD = os.environ.get("DSD_METHOD", "eagle3")
NGRAM_MIN = int(os.environ.get("NGRAM_MIN", "2"))
NGRAM_MAX = int(os.environ.get("NGRAM_MAX", "4"))


def spec_config() -> dict:
    if METHOD == "ngram":
        return {"method": "ngram", "num_speculative_tokens": K_MAX,
                "prompt_lookup_min": NGRAM_MIN, "prompt_lookup_max": NGRAM_MAX}
    return {"method": METHOD, "model": DRAFTER, "num_speculative_tokens": K_MAX}


def calibration_prompts(sweep: Path, n: int) -> tuple[list[str], int, int]:
    dataset: dict[str, str] = {}
    for f in glob.glob(str(sweep / "*/c*/inputs.json")):
        for s in json.loads(Path(f).read_text())["data"]:
            dataset[s["session_id"]] = s["payloads"][0]["messages"][-1]["content"]
    used = set()
    for f in glob.glob(str(sweep / "*/c*/profile_export.jsonl")):
        for line in open(f):
            used.add(json.loads(line)["metadata"]["conversation_id"])
    unused = sorted(sid for sid in dataset if sid not in used)
    # All unused prompts by default: a 64-prompt sample came out ~2% low.
    picks = unused[:n] if n else unused
    return [dataset[s] for s in picks], len(dataset), len(used)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sweep")
    src.add_argument("--prompts-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="cap on prompts (0 = all unused)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature (0 = greedy); acceptance drops with sampling")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if a.prompts_file:
        prompts = [json.loads(line)["prompt"] for line in open(a.prompts_file)]
        prompts = prompts[:a.n] if a.n else prompts
        logger.info(f"[accept] prompts file {a.prompts_file}: calibration={len(prompts)}")
    else:
        prompts, n_total, n_used = calibration_prompts(Path(a.sweep), a.n)
        logger.info(f"[accept] dataset={n_total} used_by_sweeps={n_used} calibration={len(prompts)}")
    if not prompts:
        raise SystemExit("no unused prompts left for calibration")

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(model=TARGET, dtype=DTYPE, max_model_len=4096, max_num_seqs=64,
              gpu_memory_utilization=0.90, seed=0, disable_log_stats=False,
              tensor_parallel_size=int(os.environ.get("DSD_TP", "1")),
              speculative_config=spec_config())
    chats = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(chats, SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens, seed=0),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    isl = [len(o.prompt_token_ids) for o in outs]
    osl = [len(o.outputs[0].token_ids) for o in outs]

    c = {m.name: m for m in llm.get_metrics() if m.name.startswith("vllm:spec_decode")}
    drafts = c["vllm:spec_decode_num_drafts"].value
    per_pos = c["vllm:spec_decode_num_accepted_tokens_per_pos"].values
    rates = [round(x / drafts, 4) for x in per_pos]
    steps = sum(osl) - sum(per_pos) - len(outs)  # decode steps (request-steps)
    per_step = [round(x / steps, 4) for x in per_pos]
    result = {"target": TARGET, "method": METHOD,
              "drafter": DRAFTER if METHOD != "ngram" else f"ngram {NGRAM_MIN}-{NGRAM_MAX}",
              "k_max": K_MAX, "max_tokens": a.max_tokens,
              "isl_avg": round(sum(isl) / len(isl), 1), "osl_avg": round(sum(osl) / len(osl), 1),
              "decode_steps": steps, "proposal_rate": round(drafts / steps, 4),
              "accept_by_pos_per_step": per_step,
              "tokens_per_step_by_k_per_step": {k: round(1 + sum(per_step[:k]), 3)
                                                for k in range(K_MAX + 1)},
              "temperature": a.temperature,
              "calibration_prompts": len(prompts), "drafts": drafts,
              "accepted_per_pos": per_pos, "accept_by_pos": rates,
              "tokens_per_step_by_k": {k: round(1 + sum(rates[:k]), 3)
                                       for k in range(K_MAX + 1)}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info(f"[accept] by position {rates} -> tokens/step {result['tokens_per_step_by_k']}")
    logger.info(f"[accept] proposal rate {result['proposal_rate']}, per decode step {per_step} -> "
                f"tokens/step {result['tokens_per_step_by_k_per_step']}; ISL {result['isl_avg']} OSL {result['osl_avg']}")
    logger.info(f"[accept] wrote {out}")


if __name__ == "__main__":
    main()
