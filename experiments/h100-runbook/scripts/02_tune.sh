#!/usr/bin/env bash
# Step costs and acceptance on the H100, then the table. Run once per model:
#   MODEL=4b bash 02_tune.sh      (then MODEL=8b bash 02_tune.sh)
# Time: step costs ~40-60 min per model (grid 1..256, lengths 0-3); acceptance ~3 min per workload.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd $REPO
mkdir -p $TUNE /work/out

# 1. Full tune: step costs (forced rejection, 32-token prompts) + code acceptance at T=1.0 + table.
#    Same settings as the T4/A10G tables, except the batch grid goes to 256.
dsd-tuner run --target "$TARGET_MODEL" --drafter "$DRAFTER_MODEL" \
  --prompts-file docs/examples/code-prompts.jsonl --isl 32 --temperature 1.0 \
  --max-num-seqs $MAX_SEQS --batch-sizes "$BATCH_SIZES" --dtype $DTYPE --out $TUNE 2>&1 | tee /work/out/tune-$TAG.log
# The profiler's wall time is the tuner's cost (the A10G took 22 min up to 128)
grep -h "took\|elapsed" $TUNE/costs_k*.log 2>/dev/null | tail -4

# 2. Acceptance and table for the seven workloads, greedy (the A10G shift study used these)
for w in translation summarization qa math_reasoning rag code codeapply; do
  echo "=== acceptance $w $(date +%T)"
  dsd-tuner recalibrate --target "$TARGET_MODEL" --drafter "$DRAFTER_MODEL" --dtype $DTYPE \
    --max-num-seqs $MAX_SEQS --temperature 0 --costs "$TUNE/costs_k*.json" \
    --prompts-file $X/p_$w.jsonl --out $X/acc-$TAG-$w > $X/acc-$TAG-$w.log 2>&1 || echo "!!! $w failed"
  grep -E "tokens per step|table:" $X/acc-$TAG-$w.log | tail -2
done
echo "TUNE DONE $TAG"
