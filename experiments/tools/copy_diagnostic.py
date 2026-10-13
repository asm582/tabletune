"""How much of each answer copies the prompt? (issue #3, n-gram)

Generates greedy answers without speculation and measures, per output token,
whether the 3 tokens before it also occur earlier in prompt + output (n-gram
could propose there) and whether the next token is what followed that earlier
occurrence. Also reports how many answers contain a fenced code block and how
much of the answer is outside it (explanation).

  python experiments/tools/copy_diagnostic.py --prompts-file experiments/results/ngram/prompts-edit.jsonl --n 40 \
      --out experiments/results/ngram/copy-edit.json
"""

import argparse
import json
import os
import re
from pathlib import Path


def ngram_stats(ctx: list[int], n_prompt: int, n: int = 3) -> tuple[int, int]:
    """(positions where the last n tokens occurred before, of those where the next token matched)."""
    seen: dict[tuple, int] = {}
    for i in range(n, n_prompt + 1):
        seen[tuple(ctx[i - n:i])] = i
    found = hit = 0
    for i in range(n_prompt, len(ctx)):
        key = tuple(ctx[i - n:i])
        j = seen.get(key)
        if j is not None and j < i:
            found += 1
            hit += ctx[j] == ctx[i]
        seen[key] = i
    return found, hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts_file)][:a.n]

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3-4B", dtype="float16", max_model_len=4096, max_num_seqs=64,
              gpu_memory_utilization=0.90, seed=0)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                    SamplingParams(temperature=0.0, max_tokens=512),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    found = hit = total = 0
    code_frac, samples = [], []
    for o in outs:
        out_ids = list(o.outputs[0].token_ids)
        f, h = ngram_stats(list(o.prompt_token_ids) + out_ids, len(o.prompt_token_ids))
        found, hit, total = found + f, hit + h, total + len(out_ids)
        text = o.outputs[0].text
        code = "".join(re.findall(r"```.*?\n(.*?)```", text, re.S))
        code_frac.append(len(code) / max(len(text), 1))
        samples.append(text)
    res = {"answers": len(outs), "output_tokens": total,
           "match_rate": round(found / total, 4), "match_correct_rate": round(hit / total, 4),
           "next_token_right_when_matched": round(hit / max(found, 1), 4),
           "code_fraction_of_answer_chars": round(sum(code_frac) / len(code_frac), 3),
           "answers_with_code_block": sum(c > 0 for c in code_frac),
           "sample_answers": samples[:3]}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("[copy]", {k: v for k, v in res.items() if k != "sample_answers"})


if __name__ == "__main__":
    main()
