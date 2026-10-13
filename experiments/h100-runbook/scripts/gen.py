"""Closed-loop load generator with a time schedule of concurrency and workload.

Each schedule segment: {"dur": seconds, "conc": concurrent requests, "work": workload name}.
Requests stream from vLLM's OpenAI server; per request we record start, first-token
time, end, output tokens and the segment it started in. At every segment boundary
we scrape vLLM's speculative-decoding counters from /metrics.

  python gen.py --schedule sched.json --work code=p_code.jsonl --work translation=p_translation.jsonl --out run.json
"""
import argparse, asyncio, json, random, re, time

import aiohttp
from openai import AsyncOpenAI

COUNTERS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total", "vllm:generation_tokens_total",
            "vllm:num_preemptions_total")


async def scrape(session, url):
    try:
        async with session.get(url) as r:
            text = await r.text()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for c in COUNTERS:
            if line.startswith(c):
                out[c] = out.get(c, 0.0) + float(line.rsplit(" ", 1)[1])
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="(\d+)".*\} (\S+)', line)
        if m:
            out[f"accepted_pos_{m.group(1)}"] = out.get(f"accepted_pos_{m.group(1)}", 0.0) + float(m.group(2))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--work", action="append", required=True, help="name=prompts.jsonl")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sched = json.load(open(a.schedule))
    prompts = {}
    for w in a.work:
        name, path = w.split("=", 1)
        ps = [json.loads(l)["prompt"] for l in open(path)]
        random.Random(a.seed).shuffle(ps)
        prompts[name] = ps
    idx = {k: 0 for k in prompts}
    client = AsyncOpenAI(base_url=a.base + "/v1", api_key="x", timeout=600)
    records, active = [], set()
    t0 = time.perf_counter()
    bounds = []

    async def one(seg_i, work):
        p = prompts[work][idx[work] % len(prompts[work])]
        idx[work] += 1
        rec = {"seg": seg_i, "work": work, "start": time.perf_counter() - t0, "ttft": None, "tokens": 0}
        try:
            stream = await client.chat.completions.create(
                model=a.model, messages=[{"role": "user", "content": p}], max_tokens=a.max_tokens,
                temperature=a.temperature, stream=True, stream_options={"include_usage": True},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "seed": a.seed + len(records)})
            async for ch in stream:
                if rec["ttft"] is None and ch.choices and ch.choices[0].delta.content:
                    rec["ttft"] = time.perf_counter() - t0 - rec["start"]
                if ch.usage:
                    rec["tokens"] = ch.usage.completion_tokens
        except Exception as e:
            rec["error"] = repr(e)[:200]
        rec["end"] = time.perf_counter() - t0
        records.append(rec)

    running = []   # (t, num_requests_running) sampled every second

    async def sampler(http):
        while True:
            try:
                async with http.get(a.base + "/metrics") as r:
                    for line in (await r.text()).splitlines():
                        if line.startswith("vllm:num_requests_running"):
                            running.append((time.perf_counter() - t0, float(line.rsplit(" ", 1)[1])))
            except Exception:
                pass
            await asyncio.sleep(1.0)

    async with aiohttp.ClientSession() as http:
        samp = asyncio.create_task(sampler(http))
        metrics = [await scrape(http, a.base + "/metrics")]
        seg_start = 0.0
        for i, seg in enumerate(sched):
            seg_end = seg_start + seg["dur"]
            bounds.append((seg_start, seg_end))
            while time.perf_counter() - t0 < seg_end:
                active = {t for t in active if not t.done()}
                while len(active) < seg["conc"]:
                    active.add(asyncio.create_task(one(i, seg["work"])))
                await asyncio.sleep(0.02)
            metrics.append(await scrape(http, a.base + "/metrics"))
            seg_start = seg_end
        await asyncio.gather(*active)
        samp.cancel()

    segs = []
    for i, (s, e) in enumerate(bounds):
        fin = [r for r in records if s <= r["end"] < e and "error" not in r]
        m0, m1 = metrics[i], metrics[i + 1]
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in set(m0) | set(m1)}
        drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
        segs.append({
            "seg": i, **sched[i], "t0": s, "t1": e,
            "tok_s_finished": sum(r["tokens"] for r in fin) / (e - s),
            "tok_s_engine": d.get("vllm:generation_tokens_total", 0) / (e - s),
            "requests_finished": len(fin),
            "ttft_p50": sorted(r["ttft"] for r in fin if r["ttft"])[len(fin) // 2] if fin else None,
            "mean_draft_len": d.get("vllm:spec_decode_num_draft_tokens_total", 0) / drafts if drafts else 0.0,
            "accept_by_pos": [round(d.get(f"accepted_pos_{p}", 0) / drafts, 3) for p in range(8)] if drafts else [],
            "preemptions": d.get("vllm:num_preemptions_total", 0),
            "batch_mean": (lambda v: sum(v) / len(v) if v else None)([n for t, n in running if s <= t < e]),
        })
    json.dump({"segments": segs, "requests": records,
               "errors": sum("error" in r for r in records)}, open(a.out, "w"), indent=1)
    for s in segs:
        print(f"seg {s['seg']:2d} {s['work']:>14} c={s['conc']:3d} tok/s={s['tok_s_engine']:7.1f} "
              f"draft_len={s['mean_draft_len']:.2f} acc={s['accept_by_pos'][:3]} B={s['batch_mean']} ttft50={s['ttft_p50']}")


if __name__ == "__main__":
    asyncio.run(main())
