# autoresearch

This repo is for strict H100-scale CALM-K2 research.

## H100 Path

Use the Colab runtime repository at `/content/autoresearch_calm`.

Use the protected frontier branch:

- `autoresearch/2026-03-26-h100`
- base commit: `a3af35317310c7eded443c2d2d86c64c87c068ad`

The working rules below are strict. Do not improvise outside them.

## Setup

To start a new H100 experiment run, do this in the terminal only:

1. `git fetch origin autoresearch/2026-03-26-h100`
2. `git switch autoresearch/2026-03-26-h100`
3. Read these files for context:
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `program.md`
4. Verify the Colab runtime has the expected data and tokenizer in `~/.cache/autoresearch/`.
5. Keep `results.tsv` untracked unless the user explicitly asks otherwise.

If the runtime is missing data, stop and ask for `uv run prepare.py`.

## Experiment Rules

The H100 validation setup is fixed:

- exact H100 environment
- equal wall-clock budget: `1200 s`
- external sequence length: `2048`
- internal backbone length: `1024`
- keep `K=2`
- keep the winning gated low-rank interface from `95a6e5b`
- keep the pre/post-backbone LayerNorm path from `95a6e5b`
- keep the streaming suffix split from `a3af353`
- keep all changes causal and deterministic
- keep parameter count flat or within `+1 %`

Only `train.py` may be edited for experiments.

Do not modify:

- `prepare.py`
- the tokenizer
- the data pipeline
- the evaluation harness
- package dependencies

## Phase 6

This class is limited to exactly 4 official experiments.

The four priority probes are:

1. Rolling-window online phase only
2. Rolling-window + mild online weight decay (`1e-5`, online only)
3. Rolling-window + compressor-only online clip (`norm=0.75`)
4. Rolling-window + tiny previous-chunk carry, zero-init

Rules for this class:

- keep the rolling online window causal-safe
- do not let current data read the next chunk
- log gradient norm statistics, online `val_bpb` drift, and MFU
- if any candidate shows positive `bpb` drift or exploding gradients (`> 2x` prefix max), stop exploring that subdirection
- promotion requires at least `0.0010` improvement over `0.471874`
- near-ties require repeat verification
- if all 4 experiments lose, stop this class and do not invent random variants

## Terminal Workflow

Use the terminal, not notebook cells.

For each experiment:

1. Inspect the git state.
2. Make one isolated change in `train.py`.
3. Commit the change.
4. Run `uv run train.py > run.log 2>&1`.
5. Read `val_bpb`, `peak_vram_mb`, gradient norms, and MFU from the log.
6. Decide keep, discard, or crash.
7. Record the result in `results.tsv` if the user wants a record.

If the run crashes, fix only obvious bugs. If the idea is fundamentally wrong, log it as a crash and move on.

## Reporting

Report immediately on any provisional or real win.

Otherwise, after 4 experiments, report whether the class advanced or stalled.

If stalled, restore the H100 frontier and propose the next search class explicitly before continuing.
