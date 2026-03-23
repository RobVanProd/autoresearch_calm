# autoresearch — CALM-inspired next-vector research

This is an experiment to have the LLM do its own research.

## Mission

Your mission is to explore whether a **CALM-inspired architecture** can beat the baseline in this repo under the existing harness.

You are **not** trying to reproduce the paper literally. You are trying to translate its core ideas into this codebase while respecting the repo constraints.

The core hypothesis to explore is:

- standard next-token autoregression is bottlenecked by sequence length
- we may improve the performance/compute trade-off by increasing the semantic bandwidth of each autoregressive step
- the most promising direction is to compress **K tokens -> 1 latent vector**, model sequences over those latents, then decode back to tokens
- robust latent spaces matter; brittle compression is not enough
- one-step latent generation is preferred over slow iterative latent samplers
- in this repo, **the only thing that matters is lower `val_bpb`**

You are allowed to be pragmatic. If a pure CALM implementation underperforms, you should pivot to **hybrid CALM-style models** that preserve the core idea but optimize for this benchmark.

---

## Hard constraints

These are non-negotiable:

- Modify **only** `train.py`
- Do **not** modify `prepare.py`
- Do **not** install packages or add dependencies
- Do **not** change the evaluation harness
- Ground-truth metric is **`val_bpb`**
- Every experiment must run within the existing fixed time budget
- Keep VRAM reasonable
- Prefer simpler solutions when gains are small

This means:

- do **not** optimize for paper fidelity over benchmark performance
- do **not** spend time implementing paper-only evaluation machinery if it does not help `val_bpb`
- do **not** obsess over exact likelihood-free temperature sampling or exact BrierLM unless it is trivially useful
- do **not** build giant multi-stage infrastructure outside `train.py`

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the only file you modify
4. **Verify data exists**: check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize `results.tsv`** with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once setup is complete, begin experimentation.

---

## What “CALM-inspired” means in this repo

Treat the paper as a source of architectural principles, not a strict implementation spec.

### Principles to preserve when possible

1. **Chunked modeling**
   - group tokens into chunks of size `K`
   - model chunk-level states instead of only token-level states
   - start with small `K` values such as `2` or `4`
   - only try `K=8` if earlier chunked variants are stable and promising

2. **Latent bottleneck**
   - compress each chunk to a latent representation
   - the latent can be deterministic or stochastic
   - the latent should be compact enough to change the information flow, not just act like a trivial reshaping

3. **Robust latent space**
   - brittle compression is bad
   - test VAE-style or noise-robust latents:
     - small KL regularization
     - KL floor / clipping if needed
     - latent dropout
     - token masking / chunk corruption
   - robustness matters more than elegant theory

4. **One-step next-latent prediction**
   - prefer a cheap, single-step latent head
   - avoid diffusion or any iterative sampler unless an extremely tiny version is obviously worth trying
   - a small residual MLP latent head conditioned on transformer state is in-scope

5. **Discrete-token grounding**
   - prefer feeding the model compressed information from recent **tokens/chunks**
   - do not assume feeding previous latents directly is best
   - chunk-token input compression is a first-class direction

6. **Hybrid rescue is allowed**
   - if pure latent-only CALM variants lag badly, use hybrid objectives or hybrid pathways
   - examples:
     - token CE + latent auxiliary loss
     - token decoder with chunk latent bottleneck
     - shared transformer with chunk-level auxiliary prediction
     - grouped multi-token heads with latent regularization
   - the benchmark decides what lives

---

## Research priorities

Prioritize experiments in this order.

### Priority 0 — baseline and understanding

The first run must always be the unmodified baseline.

Then inspect the existing architecture and identify:
- where token embeddings are formed
- where sequence mixing happens
- where logits are produced
- where a chunk bottleneck could be inserted with minimal damage
- where auxiliary losses can be added safely

### Priority 1 — minimal viable CALM

Implement the smallest plausible chunked latent variant that can still compete under the 5-minute budget.

Good first ideas:
- chunk tokens in groups of `K`
- create a chunk encoder that compresses `K` token embeddings to one latent
- run the backbone over chunk states
- decode chunk latents back to token logits for the `K` target tokens
- train end-to-end with token cross-entropy
- keep it simple and cheap

This first CALM-like version does **not** need to be stochastic.

### Priority 2 — robust latent variants

Once a minimal chunked latent model works, try making the latent space more robust:

- add a VAE-style posterior with `mu` and `logvar`
- use small `beta` KL penalties
- try per-dimension KL flooring/clipping if collapse appears
- add latent dropout
- add token masking or chunk corruption in the encoder input
- compare deterministic vs stochastic latent bottlenecks

Do not overcommit. If stochastic variants destabilize training or hurt `val_bpb`, revert quickly.

### Priority 3 — lightweight generative latent heads

Explore a small next-latent head conditioned on transformer hidden state.

Examples:
- residual MLP head
- SwiGLU-based latent head
- noise-conditioned latent predictor
- mixture-style or multi-sample latent predictor if cheap

If you try energy-style or diversity-aware latent training:
- keep it tiny
- use very small sample counts
- do not let auxiliary latent losses dominate token CE
- the token metric is the scoreboard

### Priority 4 — better chunk input adaptation

The paper’s idea of compressing chunk tokens into a single input state is important.

Explore:
- 2-layer chunk compression MLP
- gated chunk pooling
- learned weighted pooling over the `K` token embeddings
- tiny local attention within chunk before compression
- residual projection from chunk summary into model width

This may be one of the highest-value areas because it is cheap and directly changes information bandwidth.

### Priority 5 — hybrid and pragmatic variants

If pure CALM-style next-vector approaches are not winning, keep the spirit but get more pragmatic.

Allowed pivots:
- grouped multi-token prediction heads
- latent bottleneck plus direct token shortcut
- chunk autoencoding auxiliary loss on top of a strong token LM
- chunk prediction branches that regularize a normal transformer
- hierarchical local/global sequence compression
- low-rank or bottleneck sequence summaries
- semantic bandwidth increases without full paper faithfulness

Do not get religious about purity.

---

## What is probably NOT worth it here

Avoid sinking many runs into these unless they are nearly free:

- exact reproduction of paper evaluation
- exact BrierLM implementation
- exact likelihood-free temperature sampling
- expensive diffusion or flow-matching latent heads
- large separate pretraining stages that consume most of the 5-minute budget
- giant autoencoders that starve the main LM of compute
- elaborate math that does not move `val_bpb`

This harness is not the paper’s original environment. Respect that.

---

## Experimental ladder

When you need ideas, follow this ladder:

1. Baseline
2. Deterministic chunk bottleneck (`K=2`)
3. Deterministic chunk bottleneck (`K=4`)
4. Better chunk compressor
5. Shared decoder from latent back to `K` token logits
6. Add auxiliary reconstruction loss if separate from CE
7. Add stochastic latent (`mu/logvar`)
8. Add tiny KL
9. Add KL flooring/clipping if collapse appears
10. Add latent dropout
11. Add token masking in chunk encoder
12. Add lightweight residual latent head
13. Add noise-conditioned latent head
14. Try hybrid token+latent loss weighting
15. Scale width/depth/batch around the best chunked design
16. Simplify aggressively if complexity is not paying rent

---

## Keep / discard rules

Use the same philosophy as the stock loop, plus these CALM-specific rules:

- A CALM-ish change only survives if it improves `val_bpb` enough to justify its complexity
- Small gains from ugly code are usually not worth keeping
- Small gains from a cleaner or more principled chunked design may be worth keeping
- If a variant is interesting but clearly slower, more fragile, and not better, discard it
- If the latent branch collapses or becomes a no-op, either fix it quickly or kill it
- If the architecture only works with a direct token shortcut doing all the work, be honest about that in the description
- Prefer `K=2` or `K=4` early because they are more likely to survive the tight time budget

---

## First-run policy

Your very first run must be the original baseline.

Only after that should you begin the CALM-inspired line.

Do not skip the baseline.

---

## Output format

Once the script finishes it prints a summary like this:

```text
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8

## Extract the key metrics with:

grep "^val_bpb:\|^peak_vram_mb:" run.log

If the grep output is empty, the run crashed.

Logging results

When an experiment is done, log it to results.tsv (tab-separated, NOT comma-separated).

Header:

commit	val_bpb	memory_gb	status	description

Columns:

git commit hash (short, 7 chars)
val_bpb achieved — use 0.000000 for crashes
peak memory in GB, round to .1f — use 0.0 for crashes
status: keep, discard, or crash
short description of what the experiment tried

Use descriptions that are actually informative.

Good examples:

baseline
calm-k2 deterministic chunk bottleneck
calm-k4 better chunk compressor
calm-k4 vae beta1e-4 latent dropout
hybrid chunk latent + token shortcut
noise-conditioned latent head
chunk bottleneck worse than baseline

Be brutally honest in descriptions.

The experiment loop

The experiment runs on a dedicated branch such as autoresearch/<tag>.

LOOP FOREVER:

Check current branch / commit
Choose one concrete experimental change in train.py
Implement it directly
git commit

Run:

uv run train.py > run.log 2>&1

Read results:

grep "^val_bpb:\|^peak_vram_mb:" run.log

If empty, inspect:

tail -n 50 run.log
Log the result in results.tsv (do not commit results.tsv)
If val_bpb improved, keep the commit and continue from there
If val_bpb is equal or worse, reset back to the previous good commit
Timeout / crash policy
Each experiment should take about 5 minutes plus startup/eval overhead
If a run exceeds 10 minutes, kill it and treat it as failure
If the crash is a trivial bug, fix and re-run once
If the idea itself is broken, log crash, revert, move on

Do not get stuck debugging dead ideas for too long.

Strategic mindset

Think like an autonomous research engineer, not a paper replicator.

Your actual job is:

translate the CALM paper’s strongest ideas into this benchmark
preserve the semantic-bandwidth hypothesis where possible
discover which parts are real and which parts do not pay off here
keep the branch advancing only when the metric earns it

Be willing to learn the negative result:
a fully faithful CALM-style design may not win under this harness.
That is acceptable.
If so, the goal becomes extracting the strongest CALM-derived improvements that do win.

Never stop

Once the loop has started, do not pause to ask the human whether to continue.

Do not ask:

“should I keep going?”
“want me to try more?”
“is this a good stopping point?”

You are autonomous and should continue until manually interrupted.

If you run out of ideas:

re-read train.py
re-read this file
simplify
combine near-misses
revisit chunk size
revisit latent dimension
revisit loss weighting
revisit whether the chunk compressor is the real win
revisit whether a hybrid beats purity

Keep going until stopped.