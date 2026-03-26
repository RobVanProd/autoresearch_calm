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

## Phase 6 Results

**Verdict: CLASS STALLED**  no experiment improved over frontier 0.471874. Best was Exp2 at 0.529464 (delta +0.057590 worse). Per spec: stop this class, restore frontier, propose next class explicitly.

| Exp | Variant | val_bpb | online_prefix_bpb | online_final_bpb | online_drift | grad_clip | peak_vram_mb |
|-----|---------|---------|-------------------|-----------------|--------------|-----------|--------------|
| 1 | rolling-window baseline (variant=1) | 0.533767 | 0.582464 | 0.572964 | -0.009500 | 0 | 7038.8 |
| 2 | + online weight decay 1e-5 (variant=2) **BEST** | 0.529464 | 0.581876 | 0.568871 | -0.013005 | 0 | 7038.8 |
| 3 | + compressor-only online clip norm=0.75 (variant=3) | 0.530624 | 0.583070 | 0.569732 | -0.013338 | 0 | 7038.8 |
| 4 | + zero-init carry (variant=4) | 0.532282 | 0.582961 | 0.571391 | -0.011570 | 0 | 7038.8 |

Observations:
- All variants converge to val_bpb 0.529-0.534, far above frontier 0.471874
- Online drift is consistently negative (online adaptation is working, ~1% BPB improvement)
- No gradient explosions; all variants numerically stable
- Weight decay (Exp2) gave marginal best  mild L2 regularisation during online phase helps
- Zero-init carry (Exp4) had smallest drift, suggesting it stabilises carry but reduces adaptation speed
- Gap to frontier (~0.058 BPB) is too large to close with single-axis online-phase stability tweaks

## Phase 7

**New search class: Online-phase learning-rate schedule**

Hypothesis: the 300s budget online phase uses the same LR as the tail of prefix training. A tailored LR regime during the online phase could unlock faster adaptation and push val_bpb toward the frontier.

This class is limited to exactly 4 official experiments.

Baseline: Phase 6 Exp2, val_bpb = 0.529464 (variant=2, weight decay 1e-5).
Promotion threshold: val_bpb < 0.528464 (0.001 improvement over Phase 6 best) OR < 0.470874 (frontier beat).

The four priority probes are:

1. Online LR x0.1 only  hard LR step-down to 10% of tail-LR at online phase start, no weight decay (ONLINE_LR_MULT=0.1, variant=1)
2. Online LR x0.1 + weight decay 1e-5  combine LR step-down with best Phase 6 regulariser (ONLINE_LR_MULT=0.1, variant=2)
3. Longer online phase  LIVE_PREFIX_FRAC=0.7, giving 30% online budget instead of 20% (variant=1, no LR change)
4. Frozen compressor online  freeze compressor weights at online phase start, only backbone adapts (ONLINE_FREEZE_COMPRESSOR=1, variant=2)

Rules:
- implement ONLINE_LR_MULT env var (default "1.0") that scales all param-group LRs at online phase start
- implement ONLINE_FREEZE_COMPRESSOR env var (default "0") that freezes compressor params at online phase start
- do not change architecture, K, sequence lengths, or TIME_BUDGET
- promotion requires 0.001 improvement over 0.529464 or frontier beat
- if all 4 experiments lose, stop and propose a new architectural class


