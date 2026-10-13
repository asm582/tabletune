"""Long-prompt workload: GovReport reports (ccdv/govreport-summarization, test split) cut to N tokens of the target's
tokenizer, followed by a summarization instruction. Writes <out>/long<N>.jsonl with {"text", "prompt"} per line
(AIPerf reads "text", the tuner reads "prompt").
  python make_long_prompts.py --parquet test.parquet --out DIR --tokens 1024 3072"""
import argparse, json, os
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tokens", type=int, nargs="+", required=True); ap.add_argument("--model", default="Qwen/Qwen3-4B")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
docs = [d for d in pq.read_table(a.parquet).column("report").to_pylist() if d]
ask = "\n\nSummarize the report above in one paragraph."
os.makedirs(a.out, exist_ok=True)
for n in a.tokens:
    rows = []
    for d in docs:
        ids = tok.encode(d, add_special_tokens=False)
        if len(ids) < n:
            continue
        text = tok.decode(ids[:n]) + ask
        rows.append({"text": text, "prompt": text})
    with open(f"{a.out}/long{n}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"long{n}: {len(rows)} prompts")
