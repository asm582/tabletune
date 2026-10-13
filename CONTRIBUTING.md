# Add a tuned table

The [`tables/`](tables/) folder is a catalog: one folder per tuned table, for one GPU, model and drafter. Every table in it can be served with one command, and a GPU's default table is picked automatically:

```bash
dsd-tuner tables
dsd-tuner serve <table>
```

To add yours:

**1. Tune it** on your GPU, model and drafter: [docs/TUNING.md](docs/TUNING.md). The result is a folder, e.g. `my-tune`, with `schedule.json`, `acceptance.json` and `costs_k*.json`.

**2. Add it to the catalog.** This copies the files into `tables/<name>/`, writes `table.json`, and runs the same checks as CI.

```bash
dsd-tuner add-table --from my-tune --name h100-qwen3-4b-eagle3 --gpu "NVIDIA H100 80GB HBM3" --match H100 --dtype bfloat16 --measured "code, 1-128 requests: +9% over the best fixed K at 128" --contributor your-github-handle
```

Add `--default` to serve it automatically on a matching GPU (one default per GPU), and `--max-model-len` if 4096 doesn't fit your use.

**3. Try it.**

```bash
dsd-tuner serve h100-qwen3-4b-eagle3
```

**4. Open a pull request** with the new `tables/<name>/` folder. Put the benchmark behind your `--measured` line in the description (for example the summary from step 4 of [docs/TUNING.md](docs/TUNING.md)).

## What CI checks (`dsd-tuner tables --check`)

- `table.json` has the fields below (`default` and `source` are optional), and the name is lower case with dashes.
- `schedule.json` covers batch sizes 1 to `max_num_seqs` exactly once, and never uses a K above `num_speculative_tokens`.
- There is a `costs_k<K>.json` for every K the schedule uses, and the in-engine controller accepts the costs and acceptance.
- At most one default table per GPU.

## `table.json`

| Field | Example | Meaning |
|---|---|---|
| `gpu` | `NVIDIA H100 80GB HBM3` | The GPU it was measured on |
| `gpu_match` | `["H100"]` | Words in `nvidia-smi`'s GPU name that select this table |
| `default` | `false` | Serve it when no table is named, on a matching GPU |
| `target`, `drafter`, `method` | `Qwen/Qwen3-4B`, `AngelSlim/Qwen3-4B_eagle3`, `eagle3` | What `vllm serve` runs (method: `eagle3`, `dflash`, `draft_model` or `ngram`) |
| `num_speculative_tokens` | `3` | The largest K |
| `dtype`, `max_num_seqs` | `bfloat16`, `128` | Serving settings the table was tuned for |
| `max_model_len` | `4096` | `vllm serve --max-model-len` (4096 fits Qwen3-4B's KV cache on a 16 GB GPU) |
| `measured` | one line | Workload and result |
| `contributor` | GitHub handle | Who measured it |
| `source` | `experiments/results/tuner-v3-code` | Optional: where the table came from (a study folder or a link) |

## Releasing (maintainers)

1. On `main`, set the new version in `pyproject.toml`, through a pull request.
2. Create the release branch and an annotated tag from that commit, and push them:

   ```bash
   git switch -c release/vX.Y.Z && git push origin release/vX.Y.Z
   git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
   ```

3. The tag runs `.github/workflows/release.yml`. It checks that the tag matches the version (`vX.Y.Z` for version `X.Y.Z`), runs the tests and `dsd-tuner tables --check`, builds the wheel and source archive (`dsd_tuning-X.Y.Z-py3-none-any.whl`, `dsd_tuning-X.Y.Z.tar.gz`) with SHA-256 checksums (`SHA256SUMS`), installs the wheel in a clean environment to check that `dsd-tuner --version` runs and `dsd_tuning.controller.vllm_plugin` imports, and attaches the files to the GitHub release (created with generated notes if it doesn't exist yet). Add the user-visible changes to the release notes.

   To rebuild the files for a tag that already exists (for example after moving it), run the workflow by hand with that tag: `gh workflow run release.yml -f tag=vX.Y.Z`. It replaces files of the same name on the release.
