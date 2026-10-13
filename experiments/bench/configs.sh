# shellcheck shell=bash
# Named server configurations. Source this file; spec_config <name>
# prints the --speculative-config JSON ("" means no speculation).

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"   # separate draft model (issue #4)
DFLASH_MODEL="${DFLASH_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"   # DFlash block drafter (issue #7)
DRAFTER_MODEL="${DRAFTER_MODEL:-AngelSlim/Qwen3-4B_eagle3}"

eagle3() {  # eagle3 <K> [dynamic-sd-table-json]
  local k=$1 table=${2:-}
  local extra=""
  [[ -n "$table" ]] && extra=", \"num_speculative_tokens_per_batch_size\": $table"
  echo "{\"method\": \"eagle3\", \"model\": \"$DRAFTER_MODEL\", \"num_speculative_tokens\": $k$extra}"
}

ngram() {  # ngram <K> [dynamic-sd-table-json]  (prompt lookup, no drafter model; issue #3)
  local k=$1 table=${2:-}
  local extra=""
  [[ -n "$table" ]] && extra=", \"num_speculative_tokens_per_batch_size\": $table"
  echo "{\"method\": \"ngram\", \"prompt_lookup_min\": 2, \"prompt_lookup_max\": 4, \"num_speculative_tokens\": $k$extra}"
}

draftm() {  # draftm <K> [dynamic-sd-table-json]  (separate draft model; issue #4)
  local k=$1 table=${2:-}
  local extra=""
  [[ -n "$table" ]] && extra=", \"num_speculative_tokens_per_batch_size\": $table"
  echo "{\"method\": \"draft_model\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $k$extra}"
}

dflashm() {  # dflashm <K> [dynamic-sd-table-json]  (DFlash block drafter; issue #7)
  local k=$1 table=${2:-}
  local extra=""
  [[ -n "$table" ]] && extra=", \"num_speculative_tokens_per_batch_size\": $table"
  echo "{\"method\": \"dflash\", \"model\": \"$DFLASH_MODEL\", \"num_speculative_tokens\": $k$extra}"
}

spec_config() {
  case "$1" in
    nospec) echo "" ;;
    k1) eagle3 1 ;;
    k2) eagle3 2 ;;
    k3) eagle3 3 ;;
    # Config C: the example table from the vLLM Dynamic SD docs page, unchanged.
    # The docs present it as an example; they don't state the GPU it was tuned
    # for. With max_num_seqs=64 only its first range ([1, 64] -> K=3) is ever
    # used, so on this server it behaves like fixed K=3 through the DSD path.
    dsd-docs) eagle3 3 "[[1, 64, 3], [65, 128, 1], [129, 512, 0]]" ;;
    # Hand-written table: the docs example with every boundary divided by 4
    # (my rescaling to max_num_seqs=64, NOT from the docs). Kept as an example
    # of a hand table whose boundaries fall inside the operating range.
    dsd-rescaled) eagle3 3 "[[1, 16, 3], [17, 32, 1], [33, 64, 0]]" ;;
    # Issue #12: K=3 at every batch size through the Dynamic SD path, so any
    # difference from fixed k3 is the path (CUDA-graph mode), not the table.
    dsd-const3) eagle3 3 "[[1, 128, 3]]" ;;
    # Config D: tables produced by the tuner (dsd_tuning/tuner/make_schedule.py writes them)
    dsd-tuned) eagle3 3 "$(cat experiments/results/tuner/schedule.json)" ;;        # v1
    dsd-tuned-v2) eagle3 3 "$(cat experiments/results/tuner-v2/schedule.json)" ;;  # v2
    # v3: short-prompt costs up to 128 requests, code acceptance at T=1.0
    dsd-tuned-v3) eagle3 3 "$(cat experiments/results/tuner-v3-code/schedule.json)" ;;
    # EAGLE-3 table tuned on another GPU (experiments/scripts/run_gpu_eagle.sh sets EAGLE_OUT)
    eagle-tuned) eagle3 3 "$(cat ${EAGLE_OUT:?set EAGLE_OUT}/schedule.json)" ;;
    # N-gram (issue #3): fixed K, the tuner's table, and EAGLE's v3 table as an
    # untuned starting point for the controller. K max 5: n-gram drafting is free.
    ngram-k1) ngram 1 ;;
    ngram-k2) ngram 2 ;;
    ngram-k3) ngram 3 ;;
    ngram-k4) ngram 4 ;;
    ngram-k5) ngram 5 ;;
    # v1 (average-cost model, [[1,128,5]]): what the first sweep ran as ngram-tuned
    ngram-tuned) ngram 5 "$(cat ${NGRAM_OUT:-experiments/results/ngram-tuner}/schedule_mean.json)" ;;
    # v2 (full-cost model, the tuner's current table)
    ngram-tuned-v2) ngram 5 "$(cat ${NGRAM_OUT:-experiments/results/ngram-tuner}/schedule.json)" ;;
    ngram-v3table) ngram 5 "$(cat experiments/results/tuner-v3-code/schedule.json)" ;;
    # Untuned start for the controller: K in {1,3,5}, deep K at high load (backwards)
    ngram-untuned) ngram 5 "[[1, 16, 1], [17, 64, 3], [65, 128, 5]]" ;;
    # Draft model (issue #4): fixed K, the tuner's table, EAGLE's v3 table (untuned start)
    draft-k1) draftm 1 ;;
    draft-k2) draftm 2 ;;
    draft-k3) draftm 3 ;;
    draft-k4) draftm 4 ;;
    draft-k5) draftm 5 ;;
    draft-tuned) draftm 5 "$(cat ${DRAFT_OUT:-experiments/results/draft-tuner}/schedule.json)" ;;
    draft-v3table) draftm 5 "$(cat experiments/results/tuner-v3-code/schedule.json)" ;;
    # DFlash (issue #7): fixed K, the tuner's table, EAGLE's v3 table (untuned start)
    dflash-k1) dflashm 1 ;;
    dflash-k3) dflashm 3 ;;
    dflash-k4) dflashm 4 ;;
    dflash-k8) dflashm 8 ;;
    dflash-tuned) dflashm 8 "$(cat ${DFLASH_OUT:-experiments/results/dflash-tuner}/schedule.json)" ;;
    dflash-v3table) dflashm 8 "$(cat experiments/results/tuner-v3-code/schedule.json)" ;;
    *) echo "unknown config: $1" >&2; return 1 ;;
  esac
}
