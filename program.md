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



### Phase 7 Results

| Exp | Description | val_bpb | online_drift |
|-----|-------------|---------|--------------|
| Exp1 | LR x0.1 only, variant=1 | 0.534698 | -0.008873 |
| Exp2 | LR x0.1 + WD 1e-5, variant=2 | 0.529912 | -0.012153 |
| Exp3 | PREFIX_FRAC=0.7, variant=1 | 0.530622 | -0.024296 |
| Exp4 | Frozen compressor, variant=2 | 0.535222 | -0.013405 |

Verdict: CLASS STALLED. Best result Exp2 val_bpb=0.529912, above threshold 0.528464 by 0.000448.

Key signals:
- Exp3 (shorter prefix) shows largest online drift (-0.024296) -- more online time helps adaptation
- Exp2 (LR+WD) closest to threshold -- combination is near optimal
- LR step-down alone (Exp1, Exp4) hurts prefix-phase val_bpb
- No experiment beat frontier (0.471874) or crossed promotion threshold (0.528464)

## Phase 8

New search class: Combine best Phase 7 interventions

Hypothesis: Exp2 (LR+WD) and Exp3 (shorter prefix) each showed independent improvement signals.
Combining them or adjusting the LR scale may close the remaining gap.

Baseline: Phase 6 Exp2, val_bpb = 0.529464 (variant=2, weight decay 1e-5).
Promotion threshold: val_bpb < 0.528464 (0.001 improvement) OR < 0.470874 (frontier beat).

The four priority probes are:

1. Exp2+Exp3 combined  LR x0.1 + WD 1e-5 + PREFIX_FRAC=0.7, variant=2 (ONLINE_LR_MULT=0.1)
2. Gentler LR drop  ONLINE_LR_MULT=0.05 + PREFIX_FRAC=0.7, variant=1
3. More aggressive LR  ONLINE_LR_MULT=0.01 + WD 1e-5, variant=2
4. Budget only  no LR change (ONLINE_LR_MULT=1.0) + PREFIX_FRAC=0.65, variant=1

Rules:
- all Phase 7 env vars remain available (ONLINE_LR_MULT, ONLINE_FREEZE_COMPRESSOR)
- do not change architecture, K, sequence lengths, or TIME_BUDGET
- promotion requires 0.001 improvement over 0.529464 or frontier beat
- if all 4 experiments lose, declare SEARCH EXHAUSTED on online-phase class and propose architecture change

### Phase 8 Results
| Exp | Description | val_bpb | online_drift |
|-----|-------------|---------|--------------|
| Exp1 | LR x0.1 + WD 1e-5 + PREFIX_FRAC=0.7, variant=2 | 0.533783 | -0.025128 |
| Exp2 | Gentler LR x0.05 + PREFIX_FRAC=0.7, variant=1 | 0.535322 | -0.025203 |
| Exp3 | Aggressive LR x0.01 + WD 1e-5, variant=2 | 0.536053 | -0.012879 |
| Exp4 | Budget-only PREFIX_FRAC=0.65 no LR change, variant=1 | 0.535702 | -0.031103 |

Verdict: SEARCH EXHAUSTED on online-phase optimization class.
All Phase 8 results worse than Phase 7 best (0.529912). Best Phase 8 = 0.533783 (Exp1).

Key signals:
- Exp4 (no LR change + more online budget) shows largest drift (-0.031103) -- model CAN adapt with full LR
- But more adaptation during online does not improve val_bpb
- LR reduction helps prefix quality but suppresses online adaptation
- Combining LR+WD+shorter prefix (Exp1) is still best but regresses vs Phase 7
- Two phases of online-phase tuning failed to cross threshold; this search class is exhausted

## Phase 9

New search class: Extended compute (TIME_BUDGET increase)

Hypothesis: TIME_BUDGET=300s is the hard ceiling -- the model has not converged.
Doubling compute may break the plateau entirely. Best online-phase config (Phase 7 Exp2,
variant=2, WD=1e-5, ONLINE_LR_MULT=0.1) is used as the online treatment.

Baseline: Phase 6 Exp2, val_bpb = 0.529464 (variant=2, weight decay 1e-5, TIME_BUDGET=300).
Promotion threshold: val_bpb < 0.528464 (0.001 improvement) OR < 0.470874 (frontier beat).

The four priority probes are:

1. 2x budget baseline  TIME_BUDGET=600, variant=2, WD=1e-5, ONLINE_LR_MULT=0.1, PREFIX_FRAC=0.8
2. 2x budget + more online  TIME_BUDGET=600, variant=1, PREFIX_FRAC=0.65, ONLINE_LR_MULT=1.0
3. 1.5x budget  TIME_BUDGET=450, variant=2, WD=1e-5, ONLINE_LR_MULT=0.1, PREFIX_FRAC=0.8
4. 2x budget + shorter prefix  TIME_BUDGET=600, variant=2, WD=1e-5, ONLINE_LR_MULT=0.1, PREFIX_FRAC=0.7

Rules:
- TIME_BUDGET is now a variable; K, architecture, sequence lengths stay fixed
- promotion requires 0.001 improvement over 0.529464 or frontier beat
- if all 4 experiments fail to improve, declare compute class exhausted and consider K=3 architecture


## Phase 9 Results  Extended Compute / TIME_BUDGET Search

**Search class:** Compute scaling  increase TIME_BUDGET beyond the 300s baseline

| Exp | Config | val_bpb | online_drift |
|-----|--------|---------|-------------|
| Exp1 | TIME_BUDGET=600, variant=2, FRAC=0.8, LR=0.1, WD=1e-5 | 0.508576 | -0.014643 |
| Exp2 | TIME_BUDGET=600, variant=1, FRAC=0.65, LR=1.0, WD=0 | 0.508953 | -0.030575 |
| Exp3 | TIME_BUDGET=450, variant=2, FRAC=0.8, LR=0.1, WD=1e-5 | 0.517608 | -0.014164 |
| Exp4 | TIME_BUDGET=600, variant=2, FRAC=0.7, LR=0.1, WD=1e-5 | 0.506314 | -0.024472 |

**Best Phase 9:** Exp4  val_bpb=0.506314 (previous best: 0.529464 Phase6, 0.508576 Phase9 Exp1)

**Promotion threshold (0.001 over Phase 6 best):** 0.528464  ALL Phase 9 experiments PASS
**Frontier (0.471874):** Not yet reached

**Key insights:**
1. Compute scaling works: 300s0.529, 450s0.518, 600s0.506  clear monotone improvement
2. Shorter prefix (more online budget): FRAC=0.7 (0.506314) > FRAC=0.8 (0.508576) at same budget
3. Full-LR online (variant=1, LR=1.0) with even shorter FRAC=0.65 gives 0.508953  not better than variant=2 with FRAC=0.7; larger drift (-0.031) but weaker prefix
4. Compute budget is the dominant driver; prefix allocation is secondary but real

**Verdict:** CONTINUE  increasing compute budget is reliably improving val_bpb. Phase 10 explores higher budgets and continued prefix shortening.

---

## Phase 10 Spec  Higher Compute + Prefix Allocation Sweep

**Hypothesis:** The val_bpb improvement from compute has not plateaued. Pushing to 900s (3) and exploring shorter prefixes at 600s may continue the trend.

**Promotion threshold for Phase 10:** val_bpb < 0.505314 (0.001 below Phase 9 best of 0.506314)
**Frontier:** val_bpb < 0.471874

**Experiments:**

| Exp | TIME_BUDGET | variant | PREFIX_FRAC | ONLINE_LR_MULT | ONLINE_WEIGHT_DECAY | ONLINE_FREEZE_COMPRESSOR |
|-----|-------------|---------|-------------|----------------|---------------------|--------------------------|
| Exp1 | 900 | 2 | 0.7 | 0.1 | 1e-5 | 0 |
| Exp2 | 900 | 2 | 0.8 | 0.1 | 1e-5 | 0 |
| Exp3 | 600 | 2 | 0.6 | 0.1 | 1e-5 | 0 |
| Exp4 | 600 | 2 | 0.65 | 0.1 | 1e-5 | 0 |

**Rationale:**
- Exp1: Best Phase 9 config (FRAC=0.7) at 1.5 budget  tests if scaling continues
- Exp2: Standard prefix (FRAC=0.8) at 1.5 budget  controls for prefix vs. compute
- Exp3: Aggressive prefix shortening (FRAC=0.6 = 40% online) at same budget  tests if more online helps
- Exp4: Intermediate FRAC=0.65 with variant=2  disentangles from Phase 9 Exp2 (which used variant=1)

**Commands:**
```
# Exp1
TIME_BUDGET=900 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.7 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p10e1.log

# Exp2
TIME_BUDGET=900 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.8 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p10e2.log

# Exp3
TIME_BUDGET=600 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.6 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p10e3.log

# Exp4
TIME_BUDGET=600 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.65 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p10e4.log
```
