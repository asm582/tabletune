"""Write the prompt files used on the A10G: five Spec-Bench categories (80 prompts each)
and the repository's two code sets (160 each). Format: one {"prompt": ...} per line."""
import argparse, json, shutil, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
url = "https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl"
rows = [json.loads(l) for l in urllib.request.urlopen(url).read().decode().splitlines() if l.strip()]
(out / "specbench.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
for cat in ["translation", "summarization", "qa", "math_reasoning", "rag"]:
    with open(out / f"p_{cat}.jsonl", "w") as f:
        for r in rows:
            if r["category"] == cat:
                f.write(json.dumps({"prompt": r["turns"][0]}) + "\n")
shutil.copy(Path(a.repo) / "docs/examples/code-prompts.jsonl", out / "p_code.jsonl")
shutil.copy(Path(a.repo) / "docs/examples/code-edit-prompts.jsonl", out / "p_codeapply.jsonl")
for p in sorted(out.glob("p_*.jsonl")):
    print(p.name, sum(1 for _ in open(p)))
