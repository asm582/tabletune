#!/usr/bin/env bash
# Run matrix studies one after another on one otherwise idle node (no other pod on it), then shift/vary repeats.
#   bash matrix_node.sh "<studies>" "<shift repeats>"
set -uo pipefail
for s in $1; do bash /work/runbook/scripts/matrix.sh $s > /work/out/matrix-$s.log 2>&1; done
[[ -n "${2:-}" ]] && RUNS="$2" bash /work/runbook/scripts/matrix_shift.sh > /work/out/matrix-shift-$(echo $2 | tr -d ' ').log 2>&1
echo "### [node $HOSTNAME] NODE DONE $(date -u +%FT%TZ)" >> /work/out/matrix-progress.log
