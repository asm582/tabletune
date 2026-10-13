# Example prompts

JSONL, one `{"prompt": ...}` per line, for `dsd-tuner run` / `recalibrate --prompts-file`. Both are 160 prompts from [InstructCoder](https://huggingface.co/datasets/likaixin/InstructCoder) (`row` is the dataset row), the code workload behind the T4 and A10G tables:

- `code-prompts.jsonl`: the instructions alone (short prompts, ~32 tokens; the model writes code).
- `code-edit-prompts.jsonl`: instruction plus the code to edit (the model copies much of the prompt; n-gram drafting does well).

For your own traffic, sample prompts from it, or from a HuggingFace dataset with [`dsd_tuning/tuner/sample_prompts.py`](../../dsd_tuning/tuner/sample_prompts.py) (needs `uv pip install -e ".[prompts]"`; `--help` lists its options).
