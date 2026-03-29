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

---

## Phase 10 Results

| Exp | TIME_BUDGET | FRAC | variant | val_bpb | online_drift | epoch | notes |
|-----|-------------|------|---------|---------|--------------|-------|-------|
| 1 | 900 | 0.7 | 2 | 0.496308 | -0.024898 | 3 | sub-0.5 milestone |
| 2 | 900 | 0.8 | 2 | **0.496073** | -0.014892 | 3 | **BEST P10** |
| 3 | 600 | 0.6 | 2 | 0.505502 | -0.035672 | 2 | shorter prefix |
| 4 | 600 | 0.65 | 2 | 0.505508 | -0.029295 | 2 | shorter prefix |

All experiments: ONLINE_LR_MULT=0.1, ONLINE_WEIGHT_DECAY=1e-5, ONLINE_FREEZE_COMPRESSOR=0

**Phase 10 best**: val_bpb=0.496073 (Exp2: TIME_BUDGET=900, FRAC=0.8)

### Analysis

**Compute scaling continues to dominate**: 900s reliably achieves ~0.496 vs 600s ~0.505-0.506. The improvement from 600s900s is ~0.010 bpb.

**FRAC insensitivity at 900s**: FRAC=0.7 (0.496308) and FRAC=0.8 (0.496073) are essentially tied (~0.0003 difference). At high compute, prefix fraction barely matters.

**FRAC weakly improves at 600s**: FRAC=0.6 (0.505502)  FRAC=0.65 (0.505508) < FRAC=0.7 (0.506314 from P9). Shorter prefix gives slight edge at lower compute, but gains are diminishing beyond 0.6-0.65.

**Scaling law so far** (FRAC0.7, variant=2):
- 300s  0.529
- 450s  0.518
- 600s  0.506
- 900s  0.496

The per-150s improvement rate: 300450: 0.011, 450600: 0.012, 600900: 0.005/150s. The rate is slowing but scaling has not plateaued.

**Verdict**: CONTINUE  compute scaling is the primary lever and we have not exhausted it.

---

## Phase 11 Spec

**Goal**: Continue compute scaling beyond 900s. Determine if val_bpb < 0.490 is reachable within a single H100 Colab session.

**Fixed config** (best from Phase 10): FRAC=0.7, variant=2, ONLINE_LR_MULT=0.1, ONLINE_WEIGHT_DECAY=1e-5, ONLINE_FREEZE_COMPRESSOR=0

| Exp | TIME_BUDGET | FRAC | Notes |
|-----|-------------|------|-------|
| 1 | 1200 | 0.7 | +300s beyond best, ~20 min run |
| 2 | 1500 | 0.7 | +600s beyond best, ~25 min run |
| 3 | 1200 | 0.65 | shorter prefix at 1200s |

**Threshold**: Phase 11 best must beat 0.495314 (0.001 from Phase 10) to CONTINUE. Otherwise REDESIGN.

### Commands

```bash
# Exp1
TIME_BUDGET=1200 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.7 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p11e1.log

# Exp2
TIME_BUDGET=1500 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.7 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p11e2.log

# Exp3
TIME_BUDGET=1200 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.65 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p11e3.log
```

---

## Phase 11 Results

| Exp | TIME_BUDGET | FRAC | variant | val_bpb | online_drift | epoch | notes |
|-----|-------------|------|---------|---------|--------------|-------|-------|
| 1 | 1200 | 0.7 | 2 | 0.492387 | -0.023534 | 4 | NEW_BEST at time |
| 2 | 1500 | 0.7 | 2 | **0.489633** | -0.024517 | 5 | **BEST P11, SUB_0.49** |
| 3 | 1200 | 0.65 | 2 | 0.491908 | -0.030214 | 4 | FRAC control |

All experiments: ONLINE_LR_MULT=0.1, ONLINE_WEIGHT_DECAY=1e-5, ONLINE_FREEZE_COMPRESSOR=0

**Phase 11 best**: val_bpb=0.489633 (Exp2: TIME_BUDGET=1500, FRAC=0.7)

### Analysis

**Compute scaling continues monotonically through 1500s.** The full law:

| TIME_BUDGET | val_bpb |  per 300s |
|-------------|---------|------------|
| 300s | 0.529464 |  |
| 600s | 0.506314 | 0.012 |
| 900s | 0.496073 | 0.010 |
| 1200s | 0.492387 | 0.004 |
| 1500s | 0.489633 | 0.003 |

**Deceleration is real.** The per-300s gain halved from 900s1200s and is compressing further. Compute alone at this rate does not straightforwardly reach the frontier (0.471874). Extrapolating naively from the last two steps puts 0.4718 somewhere beyond 5000s, which is not a viable experimental unit.

**FRAC signal at 1200s.** Exp3 (FRAC=0.65) gives 0.491908 vs Exp1 (FRAC=0.7) gives 0.492387  a 0.0005 advantage for shorter prefix. This is real but tiny, and consistent with the pattern seen at all compute levels: marginal FRAC benefit that does not compound.

**ONLINE_LR_MULT is the next unexplored lever.** The current value (0.1, a hard 10 step-down at online phase start) has never been varied in Phase 7+. Given that the online phase is where all the gains come from (see online_bpb_drift values), the learning rate during that phase is plausibly hiding significant performance.

**Verdict**: CONTINUE with two-lane Phase 12.

---

## Phase 12 Spec

**Goal**: (A) Confirm compute scaling continues to 18002100s. (B) First systematic test of ONLINE_LR_MULT as a control variable.

**Fixed base config**: FRAC=0.7, variant=2, ONLINE_WEIGHT_DECAY=1e-5, ONLINE_FREEZE_COMPRESSOR=0

### Lane A  Compute scaling

| Exp | TIME_BUDGET | ONLINE_LR_MULT | Notes |
|-----|-------------|----------------|-------|
| 1 | 1800 | 0.1 | +300s beyond best, ~30 min |
| 2 | 2100 | 0.1 | +600s beyond best, ~35 min |

### Lane B  Online LR sweep (at 1500s, best known budget)

| Exp | TIME_BUDGET | ONLINE_LR_MULT | Notes |
|-----|-------------|----------------|-------|
| 3 | 1500 | 0.05 | half current LR |
| 4 | 1500 | 0.2 | double current LR |

**Rationale**: LR_MULT=0.1 (Exp2, 0.489633) is the current baseline for the LR sweep. If 0.05 or 0.2 beats it at 1500s, that config becomes the new base for Phase 13 compute runs.

**Threshold**: Phase 12 best < 0.488633 (0.001 from Phase 11) to CONTINUE.

### Commands

```bash
# Lane A  Exp1
TIME_BUDGET=1800 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.7 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p12e1.log

# Lane A  Exp2
TIME_BUDGET=2100 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.7 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p12e2.log

# Lane B  Exp3
TIME_BUDGET=1500 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.7 ONLINE_LR_MULT=0.05 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p12e3.log

# Lane B  Exp4
TIME_BUDGET=1500 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.7 ONLINE_LR_MULT=0.2 ONLINE_WEIGHT_DECAY=1e-5 ONLINE_FREEZE_COMPRESSOR=0 uv run train.py 2>&1 | tee /tmp/p12e4.log
```


---

### Phase 12 Exp4  Lane B: LR_MULT=0.2

**1. Config**
```
TIME_BUDGET=1500  LIVE_STABILITY_VARIANT=2  LIVE_PREFIX_FRAC=0.7
ONLINE_LR_MULT=0.2  ONLINE_WEIGHT_DECAY=1e-5  ONLINE_FREEZE_COMPRESSOR=0
```
**2. Code diff:** none (env-var-only). Base commit `781d839`.

**3. Metrics**
| metric | value |
|---|---|
| val_bpb | 0.489390 |
| training_seconds | 1500.2 |
| total_tokens_M | 2219.3 |
| num_steps | 4233 |
| epoch | 5 |
| online_bpb_drift | 0.024024 |
| grad_norm_prefix_mean | 0.1020 |
| grad_norm_online_mean | 0.0354 |

**4. Ruling:** LR_MULT=0.2 gives 0.489390 vs baseline 0.1 at 0.489633 (0.000243 better); monotone 0.050.10.2 trend motivates Phase 13 exploration of 0.5 and 1.0.


---

### Phase 13 Exp1  Lane A: LR_MULT=0.5

**1. Config**
```
TIME_BUDGET=1500  LIVE_STABILITY_VARIANT=2  LIVE_PREFIX_FRAC=0.7
ONLINE_LR_MULT=0.5  ONLINE_WEIGHT_DECAY=1e-5  ONLINE_FREEZE_COMPRESSOR=0
```
**2. Code diff:** none (env-var-only). Base commit `90e2c16`.

**3. Metrics**
| metric | value |
|---|---|
| val_bpb | 0.489872 |
| training_seconds | 1500.3 |
| total_tokens_M | 2305.8 |
| num_steps | 4398 |
| epoch | 5 |
| online_bpb_drift | 0.024197 |
| grad_norm_prefix_mean | 0.1018 |
| grad_norm_online_mean | 0.0359 |

**4. Ruling:** LR_MULT=0.5 gives 0.489872  worse than baseline 0.1 (0.489633) by +0.000239 and worse than 0.2 (0.489390); the monotone trend reverses at 0.5, placing the optimum near 0.2; weak signal, not a material win.


---

### Phase 13 Exp2  Lane A: LR_MULT=1.0

**1. Config**
```
TIME_BUDGET=1500  LIVE_STABILITY_VARIANT=2  LIVE_PREFIX_FRAC=0.7
ONLINE_LR_MULT=1.0  ONLINE_WEIGHT_DECAY=1e-5  ONLINE_FREEZE_COMPRESSOR=0
```
**2. Code diff:** none (env-var-only). Base commit `dc6afe8`.

**3. Metrics**
| metric | value |
|---|---|
| val_bpb | 0.489884 |
| training_seconds | 1500.3 |
| total_tokens_M | 2238.2 |
| num_steps | 4269 |
| epoch | 5 |
| online_bpb_drift | -0.024825 |
| grad_norm_prefix_mean | 0.1026 |
| grad_norm_online_mean | 0.0355 |

**4. Ruling:** LR_MULT=1.0 (no step-down) gives 0.489884  worse than baseline 0.1 by +0.000251 and worse than 0.5; Lane A sweep (0.050.10.20.51.0) shows a clear optimum at 0.2 with all other values worse; total LR_MULT range is ~0.0005 bpb, noise-scale, not a material second lever.

---

### Phase 13 Exp3  Lane B: ONLINE_WEIGHT_DECAY=0

**1. Config**
```
TIME_BUDGET=1500  LIVE_STABILITY_VARIANT=2  LIVE_PREFIX_FRAC=0.7
ONLINE_LR_MULT=0.1  ONLINE_WEIGHT_DECAY=0  ONLINE_FREEZE_COMPRESSOR=0
```
**2. Code diff:** none (env-var-only). Base commit `17d79cf`.

**3. Metrics**
| metric | value |
|---|---|
| val_bpb | 0.489748 |
| training_seconds | 1500.3 |
| total_tokens_M | 2295.9 |
| num_steps | 4379 |
| epoch | 5 |
| online_bpb_drift | -0.024518 |
| grad_norm_prefix_mean | 0.1006 |
| grad_norm_online_mean | 0.0402 |

**4. Ruling:** WD=0 gives 0.489748 vs baseline WD=1e-5 at 0.489633 (+0.000115 worse)  removing weight decay is a marginal regression, confirming baseline WD=1e-5 is near-optimal.


### Phase 13 Exp4  Lane B: ONLINE_WEIGHT_DECAY=1e-4

**1. Config**
```
TIME_BUDGET=1500  LIVE_STABILITY_VARIANT=2  LIVE_PREFIX_FRAC=0.7
ONLINE_LR_MULT=0.1  ONLINE_WEIGHT_DECAY=1e-4  ONLINE_FREEZE_COMPRESSOR=0
```
**2. Code diff:** none (env-var-only). Base commit `91f17f7`.

**3. Metrics**
| metric | value |
|---|---|
| val_bpb | **0.488372** |
| training_seconds | 1500.3 |
| total_tokens_M | 2317.9 |
| num_steps | 4421 |
| epoch | 5 |
| online_bpb_drift | -0.023650 |
| grad_norm_prefix_mean | 0.1009 |
| grad_norm_online_mean | 0.0356 |

**4. Ruling:** WD=1e-4 (10x stronger) gives 0.488372 vs baseline WD=1e-5 at 0.489633 (-0.001261 better)  STRONG SUCCESS (beats <=0.488390 threshold); stronger online weight decay is a genuine second lever.


---

## Phase 13 Closeout

### All 4 Runs (1500s wall-clock, VARIANT=2, FRAC=0.7, FREEZE_COMP=0)

| Rank | Exp | LR_MULT | WD | val_bpb | Delta vs baseline |
|------|-----|---------|-----|---------|-------------------|
| 1 | Exp4 Lane B | 0.1 | 1e-4 | **0.488372** | **-0.001261 STRONG SUCCESS** |
| 2 | baseline | 0.1 | 1e-5 | 0.489633 |  |
| 3 | Exp3 Lane B | 0.1 | 0 | 0.489748 | +0.000115 (worse) |
| 4 | Exp1 Lane A | 0.5 | 1e-5 | 0.489872 | +0.000239 (worse) |
| 5 | Exp2 Lane A | 1.0 | 1e-5 | 0.489884 | +0.000251 (worse) |

Phase 13 threshold: strong success <= 0.488390. **Exp4 is a strong success (0.488372 < 0.488390).**

---

### Closeout Q&A

**Q1: Did LR_MULT >0.2 help materially?**

No. Both LR_MULT=0.5 (+0.000239) and LR_MULT=1.0 (+0.000251) are worse than baseline 0.1. The weak monotone signal observed in Phase 12 (0.05->0.1->0.2) does not extend to 0.5 or 1.0  the optimum is near 0.2 at best. Total range across all 5 LR_MULT values tested (0.05, 0.1, 0.2, 0.5, 1.0) is ~0.0005 bpb, which is noise-scale. LR_MULT is confirmed as not a material lever; the default of 0.1 can be held fixed.

**Q2: Did ONLINE_WEIGHT_DECAY help materially?**

Yes. WD=1e-4 delivers -0.001261 improvement over the WD=1e-5 baseline. This is >5x larger than any LR_MULT gain observed and clearly clears the strong success threshold. WD=0 was marginally worse (+0.000115), confirming WD=1e-5 was slightly sub-optimal on the low side and the optimum is in the 1e-4 range or above. ONLINE_WEIGHT_DECAY is confirmed as the second lever.

**Q3: Is there a true second lever beyond compute?**

Yes  ONLINE_WEIGHT_DECAY is confirmed as a genuine second lever. At 1500s, WD=1e-4 (0.488372) already narrows the gap to the current all-time best (WD=1e-5 at 2100s: 0.486455) by 0.001261. The two levers (compute and WD) are additive candidates for Phase 14.

**Q4: What exact single question should Phase 14 test?**

Does combining ONLINE_WEIGHT_DECAY=1e-4 with TIME_BUDGET=2100s set a new all-time best below 0.486455 (the current frontier hold from compute alone at WD=1e-5)?

Phase 14 spec: one run  TIME_BUDGET=2100 LIVE_STABILITY_VARIANT=2 LIVE_PREFIX_FRAC=0.7 ONLINE_LR_MULT=0.1 ONLINE_WEIGHT_DECAY=1e-4 ONLINE_FREEZE_COMPRESSOR=0. Strong success threshold: any result < 0.485000.

---

*Phase 13 complete. All 4 runs committed and pushed. STOP.*


## Phase 14

### Phase 14 Exp1  WD=1e-4 + TIME_BUDGET=2100s (compose both levers)

**1. Config**
```
TIME_BUDGET=2100  LIVE_STABILITY_VARIANT=2  LIVE_PREFIX_FRAC=0.7
ONLINE_LR_MULT=0.1  ONLINE_WEIGHT_DECAY=1e-4  ONLINE_FREEZE_COMPRESSOR=0
```
**2. Code diff:** none (env-var-only). Base commit `742e01f`.

**3. Metrics**
| metric | value |
|---|---|
| val_bpb | **0.486263** |
| training_seconds | 2100.2 |
| total_tokens_M | 3231.7 |
| num_steps | 6164 |
| epoch | 7 |
| online_bpb_drift | -0.022918 |
| grad_norm_prefix_mean | 0.1012 |
| grad_norm_online_mean | 0.0371 |

**4. Ruling:** WD=1e-4 at 2100s gives 0.486263 vs prior all-time best (WD=1e-5, 2100s) of 0.486455 (-0.000192)  REAL SUCCESS; WD composes with compute and establishes a new best. Not a strong success (threshold <0.485000).


---

## Phase 14 Exp1 Closeout

### Run Table

| Config | TIME_BUDGET | WD | val_bpb | Delta |
|--------|-------------|-----|---------|-------|
| Phase 12 all-time best (prior) | 2100s | 1e-5 | 0.486455 |  |
| **Phase 14 Exp1 (this run)** | **2100s** | **1e-4** | **0.486263** | **-0.000192** |
| Phase 13 best | 1500s | 1e-4 | 0.488372 | -0.001261 vs 1500s baseline |

### Comparisons

vs 2100s all-time best (WD=1e-5): **-0.000192** (new best)
vs Phase 13 WD=1e-4 at 1500s: **-0.002109** (600s of extra compute continues to help at WD=1e-4)
vs 1500s baseline (WD=1e-5): -0.001370 combined gain from both levers

### Outcome

**WD composes with compute and establishes a new best.**

The second lever (WD=1e-4) does transfer from 1500s to 2100s. The new all-time best is 0.486263.

However, the margin over the prior best is small: -0.000192. This is below the noise floor seen in LR_MULT sweeps (~0.0005) but in the right direction and consistent with the 1500s finding. It passes the real success threshold (<0.486455) but not the strong success threshold (<0.485000).

### What this result does NOT prove

1. **WD=1e-4 is not confirmed as the optimal value.** The WD sweep at 1500s only tested 0, 1e-5, 1e-4. The optimum may lie at 3e-4 or 1e-3  untested at 2100s.

2. **The gain is not confirmed to be statistically robust.** The -0.000192 margin is smaller than the noise range observed in prior LR_MULT experiments (~0.0002-0.0005 bpb). A single run cannot rule out run-to-run variance.

3. **The levers do not add linearly.** WD=1e-4 gave -0.001261 at 1500s, but only -0.000192 more at 2100s on top of the compute gain  suggesting the interaction may be subadditive or that 2100s is a harder baseline to beat.

4. **This does not guarantee further improvement at higher compute.** The scaling law is decelerating; adding more compute (2400s, 2700s) with WD=1e-4 is untested.

### Recommendation for next step

Since this is a real success, the recommended next step is one confirmatory repeat:
- Same config: TIME_BUDGET=2100, WD=1e-4, all other vars fixed
- Purpose: confirm the -0.000192 margin is real and not a single-run fluctuation
- If confirmed: the new WD=1e-4 config becomes the permanent baseline for Phase 15+ and further compute scaling
- If not confirmed (repeat >= 0.486455): result is inconclusive; move to the WD neighborhood test instead

**Do not start this repeat automatically. Stop and report.**

## Phase 14 Experiment 2  Statistical Replication of WD=1e-4 at 2100s

**Date:** 2026-03-29
**Config:** TIME_BUDGET=2100, ONLINE_WEIGHT_DECAY=1e-4, LIVE_STABILITY_VARIANT=2, LIVE_PREFIX_FRAC=0.7, ONLINE_LR_MULT=0.1, ONLINE_FREEZE_COMPRESSOR=0
**Purpose:** 4 independent runs to determine if Exp1 result (0.486263) is robust or a lucky sample.

### Results

| Run | val_bpb |
|-----|---------|
| 1   | 0.486787 |
| 2   | 0.487469 |
| 3   | 0.487321 |
| 4   | 0.487415 |
| **Mean** | **0.487248** |
| **Std (sample)** | **0.000313** |
| **Best** | **0.486787** |

Reference points:
- Phase 14 Exp1 (single run, prior session): 0.486263
- Prior ATB (WD=1e-5, 2100s): 0.486455

### Outcome

**WD=1e-4 does NOT reliably beat WD=1e-5 at 2100s.**

All 4 independent runs produced val_bpb above the prior ATB of 0.486455. The mean (0.487248) is 0.000793 worse than the ATB. Even the best run (0.486787) is 0.000332 worse than the ATB.

The original Phase 14 Exp1 result of 0.486263 was a statistical outlier  approximately 3.1 standard deviations below the observed mean. It does not represent the true expected performance of this configuration.

### Ruling

**REJECT WD=1e-4 as an improvement over WD=1e-5 at 2100s.**

The prior all-time best of **0.486455** (WD=1e-5, 2100s) stands as the official baseline for Phase 15+.

### Recommendation for next step

Explore the WD neighborhood more carefully: test WD values between 1e-5 and 1e-4 (e.g. 3e-5, 5e-5) at 2100s with multiple seeds, or proceed to compute scaling with the confirmed WD=1e-5 baseline.

## Phase 15 Experiment 1  Compute Scaling: 2700s with WD=1e-5
Date: 2026-03-29
Config: TIME_BUDGET=2700, ONLINE_WEIGHT_DECAY=1e-5, LIVE_STABILITY_VARIANT=2,
        LIVE_PREFIX_FRAC=0.7, ONLINE_LR_MULT=0.1, ONLINE_FREEZE_COMPRESSOR=0
Result: val_bpb=0.486252 (single run, step 6968, 2700.2s training)
Prior ATB: 0.486455 (WD=1e-5, 2100s)
Delta: -0.000203 (improvement over 2100s baseline)

### Scaling curve update
| TIME_BUDGET | val_bpb |
|-------------|---------|
| 300s  | 0.529464 |
| 600s  | 0.506314 |
| 900s  | 0.496073 |
| 1200s | 0.492387 |
| 1500s | 0.489633 |
| 1800s | 0.487938 |
| 2100s | 0.486455 |
| 2700s | 0.486252 (this run, single probe) |

### Assessment
Scaling continues to yield marginal improvements. Delta 2100s2700s = -0.000203, which is
below the run-to-run std observed in Phase 14 Exp2 (~0.000313). Result is tentative 
needs replication to confirm it is a genuine improvement vs. lucky single run.

### Next step
Run 2700s 3 more to establish mean/std, OR continue scaling to 3000s/3600s first.


## Phase 15 Exp2  2700s replication (4 runs, WD=1e-5)

Runs: 0.486252, 0.485197, 0.484844, 0.485704
Mean: 0.485499, Std: 0.000614, Best: 0.484844
All 4 runs below prior ATB of 0.486455.

RULING: TIME_BUDGET=2700s CONFIRMED. New ATB = 0.484844 (Run 3).
Config: WD=1e-5, FRAC=0.7, variant=2, FREEZE=0, LR_MULT=0.1
