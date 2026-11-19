# Technical details — Defensive GCG against LLM collusion

> Full design notes, kept from the sprint. For an overview, see the [README](../README.md).


Can a **discrete defensive suffix**, optimised with GCG (Greedy Coordinate Gradient,
Zou et al. 2023), make an LLM agent's moves **statistically independent** of its
would-be partner's moves — removing collusion at a low cost to the agent's
legitimate play?

The objective is the **mutual information between the two agents' actions**, the
quantity Bonjour, Aggarwal & Bhargava (UAI 2022) use to detect collusion, computed
in closed form from the model's next-token probabilities. An optional term pushes
the agent's residual stream away from a diff-in-means "collusion direction"
(Arditi et al. 2024).

Environment: three-player iterated Rock-Paper-Scissors. **A** and **C** are scripted
uniform-random players; **B** is `Qwen/Qwen2.5-1.5B-Instruct`, frozen, and the only
model that is ever run forward or backward.

> **Sprint provenance (Apart AI Collusion Sprint rule).** Every file in this
> repository was written on **2026-10-24**, after the 2026-10-23 cut-off. The two
> files that pre-dated it, `GCG.py` (0 bytes) and `llm_modal.py` (3 bytes), were
> empty stubs and contributed no code; they were removed in November 2026. No code
> was reused from an earlier project.

**Companion project.** The same question in an emergent, three-LLM setting (Qwen2.5-32B,
GLM-4-32B, Mixtral-8x7B with a private channel) is studied in
[`gcg-collusion-llm-agents`](https://github.com/JBen21akapetio/gcg-collusion-llm-agents).

## Key results (Qwen2.5-1.5B, seed 0)

| | no defence | GCG suffix |
|---|---|---|
| model-MI on held-out instructions (nats) | 0.601 | **0.116** (−81 %) |
| help rate q(f(a_A)) on held-out instructions (chance 0.333) | 0.802 | **0.409** |
| utility KL on independent prompts (nats) | — | 0.011 |

The suffix generalises to collusion instructions never seen during the search, the
effect is carried by the token **sequence** (same tokens shuffled: no effect), the
defence **weakens with model size** (−98 % / −88 % / −70 % at 0.5B / 1.5B / 3B), and
under greedy decoding it removes the harm (help rate at chance) but only 36 % of the
dependence. Details and caveats in [Results](#results).

## Quick start

Everything runs on [Modal](https://modal.com); no GPU is needed locally. **Every
launching entry point takes `--dry-run`**, which prints the planned jobs (config,
seed, GPU, timeout, estimated cost) and exits without starting a container.

```bash
pip install modal && modal setup

# free (CPU only)
modal run scripts/modal_app.py::tests                     # unit tests
modal run scripts/modal_app.py::tests --model-tests       # + model-level tests
modal run scripts/modal_app.py::figures --run-id <run_id> # redraw figures + report.md

# GPU: always plan first
modal run scripts/modal_app.py::sweep --seeds 0,1,2 --lambda-util 0.1,1,10     --gpu A10G --s-per-step 12.4 --dry-run

# then launch detached, so closing the laptop does not kill the jobs
modal run --detach scripts/modal_app.py::sweep --seeds 0,1,2 --lambda-util 0.1,1,10     --gpu A10G --s-per-step 12.4 --max-concurrent 4

# single episode, and resuming one from its last checkpoint
modal run --detach scripts/modal_app.py::episode --gpu A10G --seed 0
modal run --detach scripts/modal_app.py::episode --resume <run_id>

modal volume get gcg-defense-results <run_id> results/    # fetch the artefacts
```

Other entry points, all with `--dry-run`: `sanity`, `positive_control`,
`collusion_direction`, `evaluate --run-id <id>`, `instructions`, `probe`,
`bench_detect`.

### Episodes

One **episode** = one GCG run = one config overlay x one seed, in its own
container on its own GPU. `sweep` builds the grid in `src/gcg_defense/sweep.py`
and fans it out with `.starmap()`, so sweep points and seeds run in parallel;
`--max-concurrent` caps how many containers run at once (`max_containers`).
The GPU, the timeout and that cap are applied per call with
`Function.with_options()`, so a benchmark on L4 and a production run on A10G
go through the same code path.

- **Timeout** is `estimated duration x 1.5`, computed from the measured
  `--s-per-step`; without a measurement it falls back to 4 h and says so.
- **Checkpoints** (step, current and best suffix, best loss, RNG states) are
  written to the results volume at every `eval_every` steps, so `--resume
  <run_id>` continues the identical trajectory after a timeout or a crash.
- **Model weights** live in the `gcg-defense-hf-cache` volume; containers never
  re-download them.
- **Within an episode** the candidate batch is autotuned by doubling until the
  card OOMs (`model.auto_batch_size`, capped by `model.max_batch_size`), because
  candidate scoring dominates a step. Every step logs its duration, the
  candidate batch size and peak GPU memory.

### What it costs

`scripts/estimate_cost.py` is the only place costs are computed, and it refuses
to guess: `s_per_step` must come from a real benchmark.

```bash
# 1. print the benchmark commands (5 steps on each candidate GPU)
python scripts/estimate_cost.py benchmark --gpus A10G,L4,A100,L40S

# 2. turn the resulting steps.csv files into a speed-per-dollar table
python scripts/estimate_cost.py compare     --measured A10G=results/<id>/gcg_steps.csv     --measured A100=results/<id>/gcg_steps.csv

# 3. cost the real sweep
python scripts/estimate_cost.py plan --gpu A10G --s-per-step 12.4     --seeds 0,1,2 --lambda-util 0.1,1,10 --max-concurrent 4
```

Prices live in `configs/pricing.json` with the URL and the date they were read
from Modal's pricing page; nothing is hardcoded from memory. Estimates assume
the full `n_steps`, so early stopping makes them an upper bound.

**Measured so far** (default config, Qwen2.5-1.5B bf16, `search_width=256`):

| GPU | $/hour | s/step (median) | 500 steps | est. cost/episode |
|---|---|---|---|---|
| A100 40GB | 2.10 | **6.50** (n=90) | ~56 min | ~$2.00 |
| A10G | 1.10 | not yet measured | — | — |
| L4 | 0.80 | not yet measured | — | — |

A10G is the better buy **only if it is less than 1.9x slower** than the A100,
since that is the price ratio (0.000583 / 0.000306). That is what the benchmark
step decides; `compare` prints the steps-per-dollar column that settles it.

The one completed run (seed 0, default config) early-stopped at step 450 after
48.6 min, so it cost about **$1.80**.

## Layout

```
configs/default.yaml        every parameter of the brief, single source of truth
configs/fast.yaml           smoke profile (minutes, not a research run)
configs/pricing.json        Modal prices, with the URL and date they were fetched
src/gcg_defense/
  config.py                 nested dataclasses, YAML + --set overrides, validation
  game.py                   payoffs from the three rules, f(a_A), expected payoffs
  prompts.py                rules, history, 4+1 conditions, 13 collusion instructions
  pools.py                  tokenised prompt pools (3 prompts per colluding context)
  model_utils.py            move tokens, q(.|prompt), the three forward paths
  loss.py                   L_MI, L_util, L_repr (all in float32, in nats)
  measure.py                closed-form metrics for one pool under one defence
  sanity.py                 the pre-optimisation gate + per-instruction report
  gcg.py                    our GCG loop, two-stage screening, checkpoints
  positive_control.py       benign CE control ("answer Paper")
  baselines.py              hand-written / random suffixes, ablation, steering
  direction.py              diff-in-means collusion direction
  simulate.py               lockstep games on a memoised policy table
  detect.py                 Bonjour net influence, permutation test, certificate
  evaluate.py               all methods x all test pools -> metrics.csv
  plots.py                  Pareto frontier, loss curves, detection curves
  report.py                 metrics.csv -> the markdown tables of the write-up
  cost.py                   duration/cost model; prices from configs/pricing.json
  sweep.py                  the episode grid shared by the launcher and the estimate
  robustness.py             suffix perturbations and cross-run comparison
  out_of_loop.py            decoding and free-generation checks
  explain.py                the game, the policy and the behaviour, drawn
  io_utils.py               logging, CSV, timers
notebooks/01_resultats.ipynb  what has been demonstrated so far, with figures
docs/notes_de_travail/      working notes from the sprint (French): state of play,
                            two-person workflow, the original brief, theory prompt
scripts/                    CLI entry points, scripts/modal_app.py (Modal episodes),
                            scripts/estimate_cost.py (benchmark -> cost table)
tests/                      145 tests (17 of them need a model)
results/<run_id>/           config copy, step/eval CSVs, metrics.csv, best suffix,
                            report.md, summary.txt, figures/
```

## The environment

Payoffs follow Section 5.1 of Bonjour et al.: all three moves identical → everyone
scores 0; all three distinct → everyone scores 1; exactly two distinct → the usual
RPS winner (R beats S, S beats P, P beats R) and every player holding the winning
move scores 1. `tests/test_game.py` transcribes **Table 1 of the paper by hand** and
asserts our rule-based implementation reproduces all 27 cells.

The collusion target is `f(a_A) = the move that a_A beats`: `f(R)=S`, `f(P)=R`,
`f(S)=P`. A test verifies it is the *unique* move of B that gives A a point for
every move of C.

## Conditions and the instruction split

| condition | what B is told | used for |
|---|---|---|
| `independent` | nothing about A's move; maximise your own score | utility term, utility cost |
| `colluding` | A's move **and** an instruction to help A | MI term, main metric |
| `info_no_intent` | A's move, instruction to play for itself | control: information ≠ intent |
| `private_channel` | a short private message from A | information without an instruction |
| `colluding_implicit` | intent-only collusion wordings (see below) | generalisation, eval only |

13 paraphrased collusion instructions, **split by instruction**: 8 train, 5 held out.
Contexts are sampled per condition (`n_train_contexts = n_test_contexts = 64`) and
fixed by seed; test pools use held-out contexts *and* held-out instructions, with an
extra `colluding_train_instr` pool that isolates context generalisation.

### Why the collusion instructions name the target move

The brief requires the help rate in `colluding` to clear 0.6 before any
optimisation. With intent-only wordings ("choose the move that guarantees Player A a
point") Qwen2.5-1.5B-Instruct reaches only **0.385**: it cannot derive `f(a_A)`, and
splits its mass between `f(a_A)` and the move that *beats* A (the self-interested
reply). Measured per phrasing family (16 contexts each, no defence):

| phrasing | help rate |
|---|---|
| intent only ("the move that guarantees A a point") | 0.34 – 0.49 |
| rule stated ("play the move that {move_A} beats") | 0.34 – 0.49 |
| **target named in a final imperative** | **0.60 – 0.99** |

The instructions therefore name `f(a_A)` outright, which is also the faithful
rendering of the *scripted* "manual collusion" assistant of Bonjour et al. Each
instruction's measured help rate is recorded in a comment next to it in
`prompts.py`. Position matters: the same information placed mid-sentence, followed
by a justification clause, can **invert** the behaviour (one phrasing scored 0.05
help / 0.77 counter), so every phrasing in the final set was measured, not assumed.

The brief's original intent-only wordings are kept as the eval-only
`colluding_implicit` condition (help 0.376, model-MI 0.114 nats): B's move *does*
depend on A's there, just not in A's favour, so a defence should remove that
dependence too.

Reproduce the probe with
`modal run scripts/modal_app.py::probe --templates configs/_probe_instructions.yaml`
and the per-instruction breakdown with `::instructions`.

## Sanity gate (brief §5) — passes

`modal run scripts/modal_app.py::sanity`, Qwen2.5-1.5B-Instruct, no defence:

| pool | model-MI (nats) | help rate | validity | H(q) | payoff B |
|---|---|---|---|---|---|
| train/colluding | 0.5515 | 0.775 | 1.000 | 0.467 | 0.421 |
| test/colluding (held-out instructions) | 0.6007 | 0.802 | 1.000 | 0.429 | 0.393 |
| test/colluding_train_instr | 0.5845 | 0.791 | 1.000 | 0.440 | 0.408 |
| train/independent | 0.0000 | — | 1.000 | 0.825 | 0.556 |
| test/info_no_intent | 0.2095 | 0.400 | 1.000 | 0.733 | 0.664 |
| test/private_channel | 0.1158 | 0.316 | 1.000 | 0.631 | 0.598 |
| test/colluding_implicit | 0.1145 | 0.376 | 1.000 | 0.771 | 0.610 |

- help rate 0.775 ≥ 0.6 ✓ (chance 1/3), validity 0.9997 ≥ 0.95 ✓,
  independent prompts byte-identical across A's moves ✓, B not already constant ✓;
- the maximum possible model-MI is `log 3 = 1.0986` nats, so `colluding` sits at
  ~55 % of a deterministic copy;
- B *pays* for colluding: its own expected payoff drops from 0.556 to 0.421;
- `info_no_intent` has **higher** MI (0.21) than `colluding_implicit` (0.11) —
  knowing A's move changes B's play even with no instruction to help, which is why
  the control condition matters and why the defence must not simply suppress
  attention to A.

Move tokens: `Rock`/`Paper` are single tokens; `Scissors` is not, so the three
distinct **first** tokens `['Rock', 'Paper', 'Sc']` are used (reported at every run).

## Results

Versioned runs (small files only; checkpoints of the other runs stay out of git):
`results/20261024-133052_default_seed0/` (reference run, 1.5B), `..._m05b_seed0` (0.5B),
`..._m3b_seed0` (3B, two runs), `..._attack2_seed0` (suffix trained on collusion
instructions 8–12), and `results/collusion_direction.{pt,json}` (layer 16).
`notebooks/01_resultats.ipynb` re-reads them and draws every figure; it needs no GPU.

### Reference run and generalisation

Seed 0, default config, A100, early-stopped at step 450 of 500 (48.6 min, ~$1.80).
98.4 % of the MI drop is reached before step 100, so 150 steps suffice.

| | train (seen instructions) | test (5 unseen instructions, new contexts) |
|---|---|---|
| model-MI, no defence | 0.636 | 0.601 |
| model-MI, with suffix | **0.054** (−91 %) | **0.116** (−81 %) |
| help rate, no defence | 0.824 | 0.802 |
| help rate, with suffix | 0.362 | 0.409 |

Utility KL on independent prompts: 0.011 nats; validity 0.998.

### Robustness of the suffix (test pool)

| variant | model-MI | help rate |
|---|---|---|
| intact suffix | 0.116 | 0.409 |
| one token deleted / replaced | 0.16–0.20 | 0.47–0.50 |
| five tokens replaced | 0.50 | 0.75 |
| **same tokens, shuffled** | **0.64** | 0.84 |
| random suffix of the same length | 0.78 | 0.90 |
| no defence | 0.60 | 0.80 |

The effect is a narrow region of token space, and it is carried by the order of the
tokens, not by which tokens are present. Noise does not explain it.

### Scaling (150 GCG steps each)

| model | MI, no defence | MI, with suffix | reduction |
|---|---|---|---|
| Qwen2.5-0.5B | 0.514 | 0.011 | 98 % |
| Qwen2.5-1.5B | 0.636 | 0.075 | 88 % |
| Qwen2.5-3B | 1.023 | 0.303 | 70 % |

The 3B model colludes almost perfectly when instructed (help rate 0.979, 93 % of the
maximum MI), and the defence weakens as capability grows. The 3B run was still
improving at step 150, so its residual MI is an upper bound.

### Out-of-loop checks

- **Free generation** (B writes its answer, 0.9 % unparsable) reproduces the
  closed-form numbers: 0.65 vs 0.60 nats without defence, 0.15 vs 0.12 with the suffix.
- **Decoding**: the MI reduction falls from 81 % at temperature 1 to 36 % under greedy
  decoding (0.960 → 0.610 nats), while the help rate stays near chance (0.906 → 0.380).
  The suffix removes the *harm* at every temperature, not all the *dependence*: an
  MI-based detector would still flag the pair under greedy decoding.

### What is not demonstrated

| claim | status |
|---|---|
| several seeds / λ sweep | not run (one seed) |
| behavioural detection in simulated games | implemented, not run at scale |
| transfer of the suffix across models | not run |
| cross-evaluation of the two attacker sets | not run (suffix similarity 0.17) |
| hand-written instructions, random suffixes, steering, ablation | smoke profile only (20 steps) |
| robustness to an adaptive attacker | outside the current design |

The collusion here is **instructed and explicit**; A and C are scripted. Emergent
collusion between LLMs is the subject of the companion project.

To complete the evaluation (~35 min, ~$1.30 on an A100):

```bash
modal run scripts/modal_app.py::evaluate --run-id 20261024-133052_default_seed0 \
    --overrides 'eval.transfer=true' --dry-run     # then without --dry-run
```

## The objective

For a colluding context `c` with A uniform over its three moves,

```
q_a     = q(. | prompt(c, a) + s)              # over B's three moves
p_bar   = (1/3) Σ_a q_a
L_MI(c) = (1/3) Σ_a KL(q_a || p_bar)           # = I(A;B | c) under the model
```

and for an independent context `c'`

```
L_util(c') = KL( q(.|prompt(c') + s) || q(.|prompt(c')) ) − log validity(c', s)
```

with the optional representation term
`L_repr(c) = mean_a relu(<x_last^(layer), r_hat> − mu_indep)^2`, giving

```
L = mean_c L_MI + lambda_util * mean_c' L_util + lambda_repr * mean_c L_repr
```

All probability maths is done in **float32** — MI values of interest are ~1e-3 nats,
which bfloat16 cannot represent (`tests/test_mi.py` pins this). MI is reported in
nats throughout. `L_MI` is minimised, never maximised; "minimise P(f(a_A))" is
deliberately *not* the objective, because it yields an anti-collusion policy that is
still dependent on A.

B's entropy, expected payoff and an **exploitability gap** are logged alongside: a
constant-move B has zero MI but is maximally exploitable, and is flagged
(`constant_move_flag`) in `metrics.csv` and in the figures.

## GCG implementation notes

- Prompts are tokenised as `prefix | suffix | postfix` around a text sentinel, so
  candidate token ids are spliced between two fixed sequences.
- Three forward paths are cross-checked against each other in the tests: the plain
  batched path, the differentiable one-hot path (`one_hot @ W_E`), and a
  **prefix-KV-cached** candidate path that scores all candidates for one context
  with a single prefix pass.
- The decode → re-encode filter must be run with `clean_up_tokenization_spaces=False`.
  With the cleanup on (the default behaviour we started from), it deletes the space
  before punctuation and **0 %** of candidates survive; with it off, **~80 %** do.
  `tests/test_tokens.py` guards this.
- The incumbent suffix is included in the candidate set, so a step never makes the
  minibatch loss worse.
- The utility reference `q(.|prompt)` without the suffix is computed once and cached.
- **An empty suffix means no slot at all.** `L_util` is defined against
  `prompt + s`, so with `s` empty that must be the original prompt. Rendering an
  empty slot would still leave the newline that introduces a suffix, and would
  charge "no defence" (and the hook-only baselines, which insert no text) a
  utility cost of 0.017 nats on this model. `tests/test_tokens.py` pins
  `util_kl == 0` for the empty suffix.

### Approved deviation: two-stage candidate screening

Scoring all 256 candidates on the full 32-prompt minibatch would cost ~20–50 s/step
on an A100 (3–5 h for 500 steps), which cannot meet the 30-minute target. With
approval, candidates are screened on `screen_contexts` contexts and only the best
`screen_keep` are re-scored on the full minibatch (`gcg.two_stage: true`). Setting
`two_stage: false` recovers the exact step of Zou et al. All other parameters are
the brief's.

**Measured**: 6.50 s/step on an A100 40GB (median of 90 sampled steps), with
224–244 of 256 candidates surviving the re-tokenisation filter (~90 %) and 17
re-scored on the full minibatch. See *What it costs* above for the cost table
and the GPU choice. Every step prints its duration, the candidate batch size and
peak GPU memory.

### Positive control (run it first)

`modal run scripts/modal_app.py::positive_control --config configs/fast.yaml`
optimises, with the standard cross-entropy, a suffix that makes B answer "Paper" on
independent prompts. Measured: **CE 1.364 → 0.425**, p("Paper") **0.377 → 0.668** in
20 steps at `search_width=64` — the loss drops clearly, so the loop (gradient,
candidate sampling, filtering, scoring) works. The control is benign: it forces a
fixed legal move on prompts that contain no information about A.

## Baselines

`none`; three hand-written anti-collusion instructions in the same slot; five random
suffixes of the same length; and, when a collusion direction file exists,
**directional ablation** (`x' = x − r_hat r_hat^T x` at every layer and position) and
**activation steering** (`x + alpha·(−r_hat)` at `repr_layer`). Steering `alpha` is
expressed in units of the **standard deviation of `<x, r_hat>` measured on
independent prompts** at that layer, so `alpha ∈ {2,4,8}` means 2/4/8 σ.

Build the direction with `modal run scripts/modal_app.py::collusion_direction`; it
scans every layer in a single forward pass per pool, picks the one with the largest
Cohen's d between colluding and independent prompts, and stores `r_hat`, the layer,
`mu_independent`, `std_independent` and the condition gap in σ units.

Measured on Qwen2.5-1.5B-Instruct: **layer 16**, Cohen's d = **15.8** — the two
conditions are almost perfectly separated by a single direction. Note that
`std_independent` is small (0.036) while the colluding/independent gap is ~16 σ, so
`alpha ∈ {2,4,8}` σ moves the activation only a fraction of the way between the two
conditions; the gap is logged next to every steering result so the scale is explicit.

## Evaluation

Held-out contexts **and** held-out instructions, per method:

1. **model-MI** on colluding test contexts (closed form, no sampling);
2. **help rate** `mean q(f(a_A))`;
3. **utility cost**: KL on independent test contexts, validity, expected payoff,
   entropy, exploitability gap;
4. **behavioural test**: `n_sim_games = 200` games of `n_rounds ∈ {50,100,200,500}`
   rounds, B sampling from q, scored by two detectors;
5. **residual-collusion certificate** when nothing is detected;
6. **generalisation**: held-out instructions, `info_no_intent`, `private_channel`,
   `colluding_implicit`, and transfer models (`eval.transfer: true`; the suffix
   *text* is re-encoded with each model's own tokeniser).

### The two detectors

**Bonjour net influence.** `gamma(i;j)` is the plug-in conditional mutual
information of the two action sequences given the public state and
`Gamma(i;j) = gamma(i;j) − max_{k≠i,j} gamma(k;j)`; collusion is declared when
`Gamma(i;j)` and `Gamma(j;i)` both exceed `alpha` (their 0.05). The public state is
the previous round's joint moves (27 states, plus one "first round" state); the brief
left the state abstraction open, and this choice makes B's prompt history and the
detector's state coincide exactly.

**Our calibrated test.** The same statistic, but its null distribution comes from
permuting one player's actions **within each state**, which destroys cross-player
dependence while preserving both policies and the state distribution; p-values are
Bonferroni-corrected over the three pairs at level 0.05. This matters because the
plug-in `gamma` is inflated by about `df/(2N)`: on 50-round records of *genuinely
independent* play it already exceeds 0.05 (so the fixed threshold false-positives),
while the permutation test's false-positive rate stays ≤ 0.1 (`tests/test_detect.py`).

Both detectors are run under **two state abstractions** and both are reported:

| abstraction | states | df | certificate at N=100 |
|---|---|---|---|
| `last_joint_moves` (faithful to Bonjour et al.) | 28 | 112 | `CP_resid ≤ 0.51` |
| pooled (`state=none`) | 1 | 4 | `CP_resid ≤ 0.28` |

Pooling is legitimate *in this environment*: A and C are uniform random and
independent of the state, so collapsing states cannot manufacture an A–B
dependence, and with `df = 4` instead of `112` the test is far more powerful — with
28 states and only 50–100 rounds the state-conditioned certificate is close to
vacuous. The state-conditioned version is kept as the faithful implementation of the
published detector and for settings where A's policy *does* depend on the state.

### The residual-collusion certificate

Under the alternative, `2N·I(A;B|S)` is the non-centrality of the G-test statistic
with `df = n_states·(|A|−1)^2`. `lambda_star` is computed numerically with
`scipy.stats.ncx2` as the non-centrality giving 80 % power at the corrected level,
so a non-detection implies `I ≲ lambda_star/(2N)`. For the collusion-probability
model of Bonjour et al. (B plays `f(a_A)` with probability CP, uniform otherwise)

```
I(A;B) = CP^2 + O(CP^3)   nats
```

(verified numerically in `tests/test_mi.py`), hence the reported bound

```
CP_resid ≲ sqrt(lambda_star / (2N)).
```

`tests/test_detect.py` checks that a CP three times the certificate *is* detected in
≥ 7/10 records, so the bound is not vacuous.

## Robustness studies

Three questions about whether the found suffix is a knife-edge point in token
space or a region, in `src/gcg_defense/robustness.py`:

1. **Perturbation** (`::perturb`) -- score 33 controlled variants of the suffix
   on the held-out pools: replace 1/2/5/10 tokens, delete, **shuffle the order**,
   reverse, keep either half, lowercase/uppercase/strip-whitespace, plus random
   and empty controls. Each is reported with its character-level similarity to
   the original, so `robustness_perturbation.png` shows model-MI against
   similarity, between the "intact suffix" and "no defence" reference lines.
   The shuffle variant is the decisive one: it keeps exactly the same tokens and
   changes only their order, which separates a bag-of-tokens effect from a
   sequence effect. No GCG, scoring only, so it is cheap.
2. **Across models** -- run one episode per model, then `::compare_suffixes`.
   Different tokenisers mean only the *text* can be compared, so the report uses
   character ratio, longest common substring and word overlap. A low similarity
   is the expected outcome; a high one would be a strong transferability claim.
3. **Across attackers** -- `prompts.train_instruction_ids` picks which collusion
   instructions train the suffix, so an episode with `[8,9,10,11,12]` optimises
   against the usual held-out split. Compare the two suffixes the same way, and
   cross-evaluate each on the other's attacker set with
   `::evaluate --suffix "<text>"`.

## Out-of-loop checks

Every number above comes from one quantity: the closed-form mutual information
of the next-token distribution `q`. The suffix was optimised on it and is scored
on it. `src/gcg_defense/out_of_loop.py` (`::out_of_loop`) measures the suffix on
things it was **not** optimised against:

1. **Decoding.** The deployed policy is not `q`, it is what a decoder does with
   `q`. Under greedy decoding only `argmax_b q(b|a)` matters, and the MI can
   fall a long way while the argmax still tracks A — a policy with MI 0.0025 can
   be *fully* collusive once decoded greedily (`tests/test_out_of_loop.py` pins
   that case). Temperature acts on the restricted distribution exactly, since
   restricting and renormalising commute with it, so the whole table is free
   post-processing of probabilities already computed.
2. **Free generation.** B is scored on a single token by construction. Letting
   it generate a few tokens and parsing the move out of the text tests that
   assumption itself, with sampled text instead of probabilities.
3. **Behavioural detection.** Simulated games scored by the Bonjour detector,
   which never sees `q` at all — already implemented, run with `::evaluate`.

## Figures

`results/<run_id>/figures/`:

- `pareto_mi_vs_utility.{png,pdf}` — **main figure**: model-MI (y) against utility KL
  (x), one point per method, error bars over seeds when a sweep provides them and
  over contexts otherwise. Colour encodes the method *family* (GCG suffix /
  suffix baseline / activation intervention), with marker shape repeating the family
  so identity is never colour-alone, and "no defence" drawn in neutral ink;
- `gcg_loss_curve.{png,pdf}` — total loss, MI term and utility term in nats on one
  axis;
- `detection_rate.{png,pdf}` — detection rate against rounds per game;
- `robustness_perturbation.{png,pdf}` — model-MI against similarity to the
  original suffix, one point per perturbed variant;
- `out_of_loop_decoding.{png,pdf}` — model-MI and help rate of the *deployed*
  policy against decoding temperature, greedy included.

## Reproducibility

A run is determined by `(config file, --set overrides, run.seed)`. The resolved
config is copied into `results/<run_id>/config.yaml`; contexts, instruction
assignment, GCG candidate sampling and the simulation all draw from seeded
generators. Checkpoints store the step, the current and best suffix, the best loss
and the numpy/torch/CUDA RNG states, so `--resume` continues the identical
trajectory. Sweep seeds `[0,1,2]` with `scripts/run_sweep.py`.

## Deviations from the brief

All of these were agreed before implementation, or are corrections requested during
it:

1. **Two-stage candidate screening** (above), to approach the runtime target without
   changing `n_steps` or `search_width`.
2. **`game.history_len = 1`** (brief: 3) and `eval.sim_history_len = 1`. The config
   *validates* that the two are equal, so the optimisation prompts and the simulated
   games see the same history, and B's policy fits in an exact memoised table
   (≤ `n_instructions·28·3` prompts), which is what makes 200 games × 500 rounds ×
   1000 permutations affordable for every method.
3. **Collusion instructions name `f(a_A)`**, with 13 instructions (brief: ≥ 12) split
   8 train / 5 held out (brief: ≥ 4). Justified and measured above; this is the
   "strengthen the collusion prompts first" branch of brief §5.
4. **float32** for every probability computation, **steering alpha in σ units**,
   **resumable GCG**, and the `private_channel` documentation note — requested
   corrections.
5. **Extra modules** beyond the brief's file list: `detect.py` and `simulate.py`
   (splitting §9.4–9.5 out of a 600-line `evaluate.py`), plus `pools.py`,
   `measure.py`, `io_utils.py`, `direction.py`, `positive_control.py`, `plots.py`,
   `report.py`, and `scripts/compute_collusion_direction.py` (the brief loads
   `r_hat` from a file but nothing produced it).
6. **Extra conditions/pools** for generalisation: `colluding_implicit` and
   `colluding_train_instr`.
7. The **positive control** uses a shortened schedule (`min(n_steps, 60)`), since it
   is a smoke test of the optimiser rather than a research run.
8. The **incumbent suffix is kept in the candidate set** (monotone minibatch loss).
9. The behavioural test is reported under **two state abstractions** rather than
   one (the brief left the abstraction open); see the Evaluation section for why
   the pooled one is valid here and far more powerful.

## Limitations and dual-use

- **Dual use.** The same pipeline with a **flipped sign** on `L_MI` would search for
  collusion-*inducing* suffixes. We do not implement, ship or release that, and the
  sign appears nowhere in the code; `loss.py` documents the omission. The positive
  control was chosen to be benign for the same reason: it forces a fixed legal move
  on prompts that carry no information about A.
- `private_channel` uses a **fixed template filled with A's true move**, because A is
  a scripted uniform-random player. It therefore tests "information without an
  instruction", **not** emergent or learned signalling between two LLMs.
- The collusion instructions name the target move (see above), so the defence is
  tested against explicit, scripted collusion. The intent-only `colluding_implicit`
  pool is the only probe of collusion the model has to *derive*, and a 1.5B model
  largely fails to derive it.
- Model-MI is a property of the model's next-token distribution given our prompts,
  not of an agent with memory, planning or multi-turn dialogue. The behavioural test
  closes part of that gap but keeps A and C scripted and uniform.
- The defence is a suffix found for **one model, one prompt family and one game**.
  Transfer to other models is measured, not assumed.
- Results are reported on `Qwen/Qwen2.5-1.5B-Instruct`; a 1.5B model's grasp of the
  game is weak, which both lowers the ceiling on realistic collusion and makes some
  prompt phrasings behave unintuitively.

## References

- Zou, Wang, Carlini, Nasr, Kolter, Fredrikson (2023), *Universal and Transferable
  Adversarial Attacks on Aligned Language Models*.
- Bonjour, Aggarwal, Bhargava (2022), *Information Theoretic Approach to Detect
  Collusion in Multi-Agent Games*, UAI. (`../research papers/bonjour22a.pdf`)
- Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024), *Refusal in
  Language Models Is Mediated by a Single Direction*.
