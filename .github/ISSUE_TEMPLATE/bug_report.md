---
name: Bug report
about: Something doesn't work as documented
labels: bug
---

**What happened**

**What you expected** (link the doc or line if it's documented)

**How to reproduce**

```bash
# exact commands
```

**Output** (relevant log lines; for `dsd-tuner serve`, the lines with `dsd_tuning` or `Dynamic SD controller`)

**Environment**
- `dsd-tuner --version`:
- GPU (`nvidia-smi --query-gpu=name --format=csv,noheader`):
- vLLM commit (install.sh pins 353e491ef):
- Table served (`dsd-tuner tables`):
