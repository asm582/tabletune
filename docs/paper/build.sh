#!/usr/bin/env bash
# Build docs/paper/paper.pdf from paper.tex (figures in build/). SOURCE_DATE_EPOCH makes the build reproducible on one
# machine. Commit the PDF together with any change to the paper's sources; CI checks that they agree.
set -euo pipefail
cd "$(dirname "$0")"
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-0} tectonic -X compile paper.tex "$@"
