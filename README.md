# autoresearch_calm

`autoresearch_calm` constrains autonomous model research to one editable file, one fixed five-minute training budget, and one score: lower `val_bpb`.

## What It Is

This repo is a CALM-inspired fork of an autonomous LLM pretraining experiment harness. The idea is to let an agent iterate on `train.py`, run short fixed-budget experiments, and keep or discard changes based on validation bits per byte.

The interesting part is the research protocol. `program.md` is unusually explicit about falsification: start with the baseline, modify only `train.py`, log crashes as crashes, discard ideas that do not improve `val_bpb`, and accept that a faithful CALM-style design may lose under this benchmark.

## Current Status

The repo contains:

- `prepare.py` for data/tokenizer preparation and evaluation utilities
- `train.py` with model, optimizer, and training loop
- `program.md` with the CALM-inspired experiment ladder and keep/discard policy
- `analysis.ipynb` for reading a future `results.tsv`

No `results.tsv` file was present in the repo snapshot, so this README does not claim any completed CALM result.

## Tech Stack

- Python
- PyTorch 2.9.1
- uv
- NumPy, pandas, pyarrow
- tiktoken/rustbpe tokenizer dependencies

## Limitations

This is an experiment harness plus instructions, not a finished research result. Any future README should report exact `val_bpb`, memory, status, commit, and description from `results.tsv` or run logs, not from memory.
