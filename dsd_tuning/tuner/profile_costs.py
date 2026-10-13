"""Measure decode step time T(B, K) for one K over a grid of batch sizes.

Uses the serving code path: the drafter at num_speculative_tokens=K_MAX with a
Dynamic SD table pinned to K ([[1, max_num_seqs, K]]), so PIECEWISE CUDA graphs
and the K=0 drafter forward are part of the cost. Synthetic zero acceptance
makes every step emit one token per request, so the batch stays at B while the
drafter and verifier still do the full K work:

  step_time(B, K) = (t(max_tokens=N+1) - t(max_tokens=1)) / N

N-gram drafts only when recent tokens occurred earlier, so its prompts are one
line of code repeated to the prompt length (the model copies it, so every step
has a match); each row records drafted_per_step to confirm it equals K.

Usage (one K per process, so each gets the whole GPU):
  python dsd_tuning/tuner/profile_costs.py --k 2 --out my-tune/costs_k2.json
"""

import argparse
import json
import logging
import os
import random
import statistics
import time
from pathlib import Path

logger = logging.getLogger(__name__)


TARGET = os.environ.get("TARGET_MODEL", "Qwen/Qwen3-4B")
DRAFTER = os.environ.get("DRAFTER_MODEL", "AngelSlim/Qwen3-4B_eagle3")
K_MAX = int(os.environ.get("DSD_K_MAX", "3"))
DTYPE = os.environ.get("DSD_DTYPE", "float16")  # T4 (sm75) has no BF16
METHOD = os.environ.get("DSD_METHOD", "eagle3")
NGRAM_MIN = int(os.environ.get("NGRAM_MIN", "2"))
NGRAM_MAX = int(os.environ.get("NGRAM_MAX", "4"))
MAX_NUM_SEQS = 128
# Dense at small B: a coarser grid let interpolation smear a spike at B=4 over B=3-7.
BATCH_SIZES = list(range(1, 17)) + list(range(24, MAX_NUM_SEQS + 1, 8))
ISL = 320   # default when run directly; dsd-tuner passes the workload's --isl
N_DECODE = 64
REPEATS = 3
# DSD_SYNTH_ACCEPT=<acceptance.json>: profile with the measured acceptance instead of rejecting every guess,
# so the step also does the work that grows with accepted tokens (drafters that read the accepted context, KV
# writes). Steps are then counted from vLLM's spec_decode_num_drafts metric instead of assumed equal to the
# decode tokens, and answers are longer so that requests finish close together.
SYNTH_ACCEPT = os.environ.get("DSD_SYNTH_ACCEPT", "")
N_DECODE_ACCEPT = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-sizes", default=",".join(map(str, BATCH_SIZES)))
    ap.add_argument("--isl", type=int, default=ISL,
                    help="prompt tokens; match the workload (spec-bench ~370, code ~32)")
    ap.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ngram = METHOD == "ngram"
    spec = {
        "method": METHOD,
        "model": DRAFTER,
        "num_speculative_tokens": K_MAX,
        "num_speculative_tokens_per_batch_size": [[1, a.max_num_seqs, a.k]],
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_rates": [0.0] * K_MAX,
    }
    accept = SYNTH_ACCEPT and a.k > 0 and METHOD != "ngram"
    n_decode = N_DECODE_ACCEPT if accept else N_DECODE
    if accept:
        rates = json.loads(Path(SYNTH_ACCEPT).read_text())["accept_by_pos"][:K_MAX]
        rates = rates + [0.0] * (K_MAX - len(rates))
        spec["synthetic_acceptance_rates"] = [min(rates[:i + 1]) for i in range(K_MAX)]  # non-increasing
    extra = {}
    if ngram:
        del spec["model"]
        spec.update(prompt_lookup_min=NGRAM_MIN, prompt_lookup_max=NGRAM_MAX)
        extra["disable_log_stats"] = False  # draft counters, to check drafts per step
    if accept:
        extra["disable_log_stats"] = False  # draft counters, to count decode steps
    llm = LLM(
        model=TARGET,
        dtype=DTYPE,
        max_model_len=4096,
        max_num_seqs=a.max_num_seqs,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=int(os.environ.get("DSD_TP", "1")),  # e.g. DSD_TP=2: profile the model as served at TP=2
        enable_prefix_caching=False,  # every timed prompt must really prefill
        seed=0,
        speculative_config=spec,
        **extra,
    )
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(a.k)

    tok = llm.get_tokenizer()

    def prompts(b):
        if ngram:  # a repeated line of code: the model copies it, n-gram always matches
            out = []
            for _ in range(b):
                name = f"v{rng.randrange(10**6)}"
                block = tok.encode(f"{name} = {name} + 1\n", add_special_tokens=False)
                out.append(TokensPrompt(prompt_token_ids=(block * (a.isl // len(block) + 1))[:a.isl]))
            return out
        return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab - 1000)
                                               for _ in range(a.isl)]) for _ in range(b)]

    def num_drafts():
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_drafts":
                return m.value
        return 0

    def drafted_tokens():
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_draft_tokens":
                return m.value
        return 0

    def timed(b, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        ps = prompts(b)
        t0 = time.perf_counter()
        outs = llm.generate(ps, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        assert all(len(o.outputs[0].token_ids) >= max_tokens for o in outs)
        return dt

    timed(4, 16)  # warm-up (compile caches, allocator)
    # Under nsys (--capture-range=cudaProfilerApi), record only the timed loop.
    # Needs VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs in this process.
    nsys = os.environ.get("NSYS_CAPTURE") == "1"
    if nsys:
        import torch
        torch.cuda.profiler.start()
    rows = []
    for b in map(int, a.batch_sizes.split(",")):
        samples, drafted = [], []
        for _ in range(REPEATS):
            t_prefill = timed(b, 1)
            d0 = drafted_tokens() if ngram else 0
            s0 = num_drafts() if accept else 0
            t_full = timed(b, n_decode + 1)
            if ngram:
                drafted.append((drafted_tokens() - d0) / (b * N_DECODE))
            steps = (num_drafts() - s0) / b if accept else n_decode   # decode steps per request
            samples.append((t_full - t_prefill) / steps * 1000)
            if accept:
                drafted.append(n_decode / steps)                       # tokens per step
        step_ms = statistics.median(samples)
        rows.append({"batch_size": b, "k": a.k, "step_ms": round(step_ms, 3),
                     "samples_ms": [round(s, 3) for s in samples]})
        if ngram:
            rows[-1]["drafted_per_step"] = round(statistics.mean(drafted), 3)
        if accept:
            rows[-1]["tokens_per_step"] = round(statistics.mean(drafted), 3)
        logger.info(f"[costs] K={a.k} B={b:3d} step={step_ms:7.2f} ms  samples={rows[-1]['samples_ms']}"
                    + (f"  drafted/step={rows[-1]['drafted_per_step']}" if ngram else ""))

    if nsys:
        torch.cuda.profiler.stop()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"target": TARGET, "method": METHOD,
                               "drafter": f"ngram {NGRAM_MIN}-{NGRAM_MAX}" if ngram else DRAFTER,
                               "k_max": K_MAX,
                               "isl": a.isl, "max_num_seqs": a.max_num_seqs, "n_decode": n_decode,
                               "acceptance": SYNTH_ACCEPT if accept else "none (every guess rejected)",
                               "rows": rows}, indent=2))
    logger.info(f"[costs] wrote {out}")


if __name__ == "__main__":
    main()
