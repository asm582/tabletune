"""Fail if a relative link in a tracked Markdown file points at a missing file.

Checks [text](path) links that are not URLs or in-page anchors, resolved from
the linking file's folder (anchors and query strings are ignored). Keeps
READMEs, docs and results write-ups consistent when files move.

Usage: python .github/ci/check_links.py
"""
import re
import subprocess
import sys
from pathlib import Path

LINK = re.compile(r"\]\(([^)\s]+)\)")
files = subprocess.run(["git", "ls-files", "*.md"], capture_output=True, text=True,
                       check=True).stdout.split()
broken = []
for f in files:
    text = Path(f).read_text(errors="replace")
    text = re.sub(r"```.*?```", "", text, flags=re.S)       # skip code blocks
    for target in LINK.findall(text):
        if re.match(r"^[a-z]+:|^#|^mailto:", target):
            continue
        path = target.split("#")[0].split("?")[0]
        if path and not (Path(f).parent / path).exists():
            broken.append(f"{f}: {target}")
print("\n".join(broken) or f"all relative links resolve ({len(files)} files)")
sys.exit(1 if broken else 0)
