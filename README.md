# Trace the Ace — final solution (team **appelswim**)

Our submission to the DrivenData **Trace the Ace** challenge, hosted by Digital Promise and
the National Tutoring Observatory.

**Task.** Given a 1:1 student–tutor session transcript and a learning objective, predict the
probability that the student then answers the next question on that objective correctly. Each
sample is one `(session, learning_objective)` pair; the target is binary; the metric is **log
loss** (AUROC is reported but does not affect ranking).

**Result.**

| split | log loss | AUROC | rank |
|---|---:|---:|---|
| public leaderboard | 0.5945 | 0.6477 | — |
| **final private split** | **0.59241** | **0.64735** | **#2 by log loss** (0.0001 behind first) |

This repository contains the exact bytes that produced that score. Nothing has been rebuilt or
re-exported: the tree is the submitted `submission.zip` unpacked, plus documentation.

```
submission zip sha256  4ab1450b1b720e1dc625e73af8c51b4dd3ad331d32937d8059c14628088dba07
LoRA adapter  sha256   3d326de39d1ad91b580a28cf6d31e0fab179b97c1d9b88a75265473b29304910
XGBoost model sha256   ef853901694077d4c0c476a6e715010a280b192ad4bc44e250e0975b27a9db59
```

---

## 1. Architecture

A two-model ensemble, blended in probability space at a frozen weight:

```
             +-----------------------------------------------+
transcript --| LLM half:  Qwen3-8B + LoRA (r=16)             |-- p_llm --+
   + LO      |            -> frozen Platt calibrator          |           |  p = 0.57*p_llm
             +-----------------------------------------------+           +-  + 0.43*p_gbm
             +-----------------------------------------------+           |
transcript --| GBM half:  XGBoost over 177 numeric columns   |-- p_gbm --+
   + LO      |            + LO TF-IDF + LO id one-hot        |
             +-----------------------------------------------+
```

`p = (1 - w) * p_llm + w * p_gbm`, with **w = 0.4300** read from `assets/blend_weight.txt`.

Two design points that are easy to get wrong:

- **Both halves are Platt-mapped *before* they meet.** The optimal weight depends on that; a
  raw-vs-raw blend puts the optimum somewhere else.
- **There is no post-blend calibration.** We measured it: a post-blend Platt costs +0.00307 log
  loss out-of-fold and isotonic costs +0.00371. Both halves are already calibrated, and
  re-calibrating the mixture only adds variance.

The weight is frozen, not fitted per fold. Log loss is flat to within 0.0002 for w in
[0.49, 0.66], and per-fold optima scatter widely ({0.22, 0.32, 0.80, 0.48, 0.82}) — so a fitted
weight is noise. An honest leave-one-fold-out selection scores **0.00035 worse** than the frozen
0.43.

### Why an ensemble at all

The two halves are wrong about different rows. Within-class correlation between them is 0.637,
and the Jensen gap `A` that the blend harvests is about 0.010 log loss. A *better* GBM does not
reliably make a better ensemble: arms that beat the shipped booster standalone tend to lose once
blended, because they converge on the LLM and spend exactly the diversity the blend was
exploiting. The quantity worth optimising is `corr(p_gbm, p_llm)`, not GBM accuracy.

---

## 2. Repository layout

```
main.py                              entry point; the platform ran this verbatim
main_llmkt_lora.py                   LLM-only variant, kept for provenance
assets/
  model_kind.txt                     "ensemble"
  blend_weight.txt                   "0.4300"
  manifest_sha256.txt                sha256 of every asset; main.py verifies on startup
  lora_llm/                          the LoRA adapter (175 MB) + its peft config
  tokenizer/                         Qwen3 tokenizer, pinned so prompts are byte-stable
  llmkt_calibrator.joblib            frozen Platt for the LLM half (87 bytes)
  baseline.joblib                    the XGBoost model: 5 seeds + frozen Platt
  v2t_clf.joblib, v2s_clf.joblib     distilled per-turn tutor / student move classifiers
  doc29_clf.joblib                   distilled interview-codebook classifier
  stu_states_clf.joblib              distilled student-state classifier
  turn_correctness_clf.joblib        distilled per-turn answer-correctness classifier
  nc_curriculum.csv                  National Curriculum reference table
  magnificat/                        feature-engineering package (25 modules)
FEATURES.md                          all 177 booster columns, one line each
tools/build_feature_table.py         regenerates FEATURES.md from the model itself
requirements.txt                     pinned to the environment the model was scored in
```

> **`assets/` is byte-identical to the submitted zip and must stay that way.** `main.py`
> sha256-verifies all 46 assets against `assets/manifest_sha256.txt` on startup and aborts on
> any mismatch, so editing even a comment breaks the model. A consequence: source comments in
> `assets/magnificat/` occasionally cite internal working documents (`docs/EXPERIMENTS.md`, an
> internal rules file, cluster job ids) that are not part of this repository. Those references
> are dangling by design — we preferred a verifiable artifact over a tidier one.

Large binaries are tracked with **Git LFS**. Install it before cloning, or the joblibs and the
adapter arrive as pointer files:

```bash
git lfs install
git clone https://github.com/LCB0B/trace-the-ace_challenge_appelswim.git
```

---

## 3. Running inference

```bash
git lfs install && git clone <this repo> && cd trace-the-ace_challenge_appelswim
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt               # pinned to the scoring environment
ln -s /path/to/competition/data data          # read-only; must contain test_features.csv
huggingface-cli download Qwen/Qwen3-8B --local-dir huggingface_models/Qwen/Qwen3-8B
python main.py                                # writes submission.csv
```

Requirements: 1x A100-80GB, Python 3.12, no internet needed at run time. The full test set takes
about 3 h 40 min.

The pins in `requirements.txt` are the competition runtime's own, and they matter: the booster
and the five distilled children are pickled joblib artifacts, so a different `scikit-learn` or
`xgboost` minor version can warn, change behaviour, or refuse to load.

`main.py` verifies every asset against `assets/manifest_sha256.txt`, merges the LoRA onto the
base model, builds all feature blocks from the raw transcripts, scores both halves, Platt-maps
each, blends at w=0.43, and writes `response_id,probability`.

**The base weights are not shipped.** The competition runtime mounts them at
`/code_execution/huggingface_models/Qwen/Qwen3-8B/`, so the submission carries only a 175 MB
LoRA adapter instead of 16 GB of merged weights. We verified this is lossless: identical smoke
score (0.4546 either way) and 399/399 tensors bit-identical, at 99x smaller.

> **Merge the LoRA on CPU.** `peft` only upcasts the `B @ A` product to fp32 on CPU; a GPU-side
> merge silently moves the weights. `main.py` does this correctly — do not "optimise" it.

---

## 4. Training the LLM half

> **The training code is not in this repository.** This repo holds the *inference* bundle -- the
> exact artifact the platform ran. The commands in sections 4 and 5 are the recipes that produced
> the shipped weights, given for reproducibility and review; they will not run from a clone of
> this repo as-is. The LLM half is trained with our fork of
> [`umass-ml4ed/dialogue-kt`](https://github.com/umass-ml4ed/dialogue-kt), and the GBM half with
> our own research tree. The trained artifacts themselves are all here, so you do not need either
> to run the model.


A two-stage LoRA fine-tune of `Qwen/Qwen3-8B`, built on the
[LLMKT / dialogue-kt](https://github.com/umass-ml4ed/dialogue-kt) architecture: the model is
asked a yes/no question about whether the student holds the knowledge component, and the
probability is read off the `True`/`False` logits rather than generated.

### LoRA configuration (both stages)

| | |
|---|---|
| base | `Qwen/Qwen3-8B` (Apache-2.0) |
| rank `r` | 16 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| `bias` | none |
| `task_type` | `CAUSAL_LM` |
| target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (all seven) |
| trainable params | 43,646,976 of 8,234,382,336 (**0.53%**) |
| peft | 0.19.1 |

### Stage 1 — fit on the competition data

Trains on all 35,062 dialogues (174,855 training points) with no held-out fold.

```bash
LLMKT_MAX_SEQ_LEN=8192 \
python -m dialogue_kt.main train \
  --dataset traceace --model_type lmkt --train_all \
  --pack_kcs 0 --quantize 0 --inc_first_label \
  --base_model Qwen/Qwen3-8B \
  --model_name lmkt_traceace_qwen3_8b_full_final
```

Resolved hyperparameters: lr **2e-4**, weight decay 1e-2, grad-clip 1.0, batch size 1,
gradient accumulation 64 (global batch 64), AdamW, **constant LR with no warmup**, bf16,
gradient checkpointing, full transcript with no turn cut, `enable_thinking=False`.

> **Reproduction caveat, stated honestly.** The launcher for this stage did not survive. Our
> ablation notes record **2 epochs**; the closest surviving sibling launcher used **1**. We
> cannot resolve which produced the shipped parent, so anyone reproducing from scratch should
> treat the epoch count as ambiguous and check both. Every other flag above is confirmed.

### Stage 2 — brief continuation on MRBench

This is what produced the shipped adapter. It is deliberately tiny: **1 epoch over ~1k MRBench
turns, 17 optimizer steps total**, at half the stage-1 learning rate. The point is a light nudge
toward a second tutoring corpus for out-of-domain robustness, not further fitting.

```bash
TRACEACE_PKL=data/annotated/mrbench_train.pkl \
LLMKT_MAX_SEQ_LEN=8192 \
python -m dialogue_kt.main train \
  --dataset traceace --model_type lmkt --train_all \
  --pt_model_name lmkt_traceace_qwen3_8b_full_final \
  --pack_kcs 0 --quantize 0 --inc_first_label \
  --epochs 1 --lr 1e-4 \
  --base_model Qwen/Qwen3-8B \
  --model_name lmkt_mrft_sa_8b_all35k
```

Fully resolved: `epochs=1, lr=1e-4, wd=1e-2, gc=1.0, batch_size=1, grad_accum_steps=64, r=16,
lora_alpha=16, optim=adamw, schedule=constant, warmup_ratio=0.0, agg='mean-ar', pack_kcs=False,
quantize=False, inc_first_label=True, train_all=True`, seed `LLMKT_SEED=221`.
Training loss over the 17 steps ran 0.666 -> 0.845.

### Prompt format

System prompt (122 tokens under the Qwen3 chat template):

```
You are an experienced math teacher. You are given a dialogue between a student and teacher
where the student is learning about math concepts. Your job is to predict if the student has
a particular knowledge component at the current point in the dialogue. Please follow these
instructions carefully when making your prediction:
- The student will need to possess this knowledge component in order to respond correctly to
  the teacher's most recent question.
- Use previous information in the dialogue to determine if the student has this knowledge
  component or not.
- Only respond with a single word, "True" or "False".
```

User message:

```
[BEGIN DIALOGUE]
Teacher Turn 1: ...
Student Turn 1: ...
[END DIALOGUE]

Knowledge Component: <learning objective text>
```

Applied with `apply_chat_template(add_generation_prompt=True, enable_thinking=False)`. The
prediction is `softmax([logit_True, logit_False])[0]` at the last real token — one forward pass,
no generation.

### Inference settings

`LLMKT_MAX_SEQ_LEN=8192`, batch size 8, bf16, flash-attention-2 with an SDPA fallback,
`padding_side="right"`, `truncation_side="left"`.

**Header-preserving truncation (`LLMKT_KEEP_HEADER=1`) matters more than it looks.** Prompts over
the cap keep the first **122** tokens (the system prompt) plus the last **8,070**. Before this
fix, a plain left-cut deleted the task instruction on the longest ~15% of transcripts — the model
was being asked nothing at all on those rows, and it reached the leaderboard that way for two
months.

### Calibrator

Frozen Platt scaling, fit on out-of-fold predictions and never refit:

```
a = 0.6876178756942771    b = 0.45650491437726304
```

---

## 5. Training the XGBoost half

About 40 minutes, CPU only, 12 cores.

```bash
python scripts/train_model_e3l.py \
  --n-seeds 5 --n-jobs 12 --par 4 --no-loknn --swap-student --extra-no-ablate \
  --drop-cols ntodst_ \
  --extra-csv "d29dst_=artifacts/doc29_features_distilled.csv" \
  --extra-csv "v2h=artifacts/v2fam_shrunk_122bC.csv" \
  --out artifacts/baseline_v2shrunkD.joblib
```

### Booster hyperparameters

| parameter | value |
|---|---|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 0 (unlimited) |
| `grow_policy` | `lossguide` |
| `max_leaves` | 31 |
| `min_child_weight` | 126.0 |
| `reg_lambda` | 80.0 |
| `reg_alpha` | 10.0 |
| `colsample_bytree` | 0.35 |
| `subsample` | 0.6 |
| `gamma` | 0.02 |
| `tree_method` | `hist` |
| `objective` | `binary:logistic` |
| `eval_metric` | `logloss` |
| `missing` | `np.nan` |

`min_child_weight = 126` is **derived, not tuned**: 600 minimum samples x p(1-p) at the 0.70 base
rate, in hessian units. A five-point sweep later put the empirical optimum exactly there.

Heavy regularisation (`reg_lambda=80`, `reg_alpha=10`, `colsample_bytree=0.35`) is deliberate.
With objectives disjoint between train and test, the failure mode is memorising the training
objectives, and shallow wide-leaf trees with aggressive column subsampling resist that.

**Seed ensemble:** 5 boosters at seeds `42 + 101*s` = 42, 143, 244, 345, 446. Their
`predict_proba` outputs are averaged, then a single frozen Platt is applied:

```
a = 0.9410532717942149    b = -0.011745382464934489
```

### Pipeline shape

A `ColumnTransformer` with three live branches, expanding to **2,032** features:

| branch | input | transformer | expanded |
|---|---|---|---:|
| `obj_text` | `learning_objective` | `TfidfVectorizer(max_features=5000, ngram_range=(1,2))` | 1,457 |
| `obj_id` | `learning_objective_id` | `OneHotEncoder(handle_unknown="ignore")` | 398 |
| `dlg` | 177 numeric columns | passthrough, densified | 177 |
| `remainder` | 92 further columns | **`drop`** | 0 |

**Do not compress the objective text.** Replacing the sparse TF-IDF with an SVD projection is
catastrophic: 32 components costs −1.87 AUROC, 16 costs −2.74, 8 costs −5.15. The objective
channel needs its sparse vocabulary, not a low-rank summary.

### The 177 feature columns

Full per-column table in **[FEATURES.md](FEATURES.md)**, generated directly from
`assets/baseline.joblib` so it cannot drift from the model.

| family | n | what it is |
|---|---:|---|
| Handcrafted | 54 | regex, counts and timestamps: turn balance, pacing, objective/transcript lexical overlap, curriculum metadata, regex tutor moves |
| Interview codes (`d29dst_`) | 34 | LLM-distilled interview-codebook tutor moves and student states, each as a session rate and a final-third rate |
| Tutor moves (`v2ht_`) | 34 | taxonomy-v2 tutor move rates, k-shrunk |
| Answer log (`recdst_`) | 14 | distilled per-turn correctness: accuracy, last-1/3/5, recency weighting, run lengths, trend |
| Student moves (`v2hs_`) | 12 | taxonomy-v2 student move rates, k-shrunk |
| Student states (`studst_`) | 12 | distilled student states: explains-why, confused, guesses, self-corrects, ... |
| Contingency (`ctg_`) | 10 | tutor tell/elicit rates split by whether the student was at an impasse |
| Repair (`rep_`) | 7 | arcs following a tutor correction, and whether the student recovers |

Six of these families come from **distillation**: an open LLM annotates each turn against a
codebook, a small classifier is trained on those labels, and the classifier's per-turn
probabilities are aggregated into session-level rates. That keeps inference cheap while still
using LLM-quality labels, and it leaves per-turn probabilities available for interpretation.

### k-shrinkage — the one non-obvious transform

Per-turn probabilities averaged into a session rate have sampling standard deviation
`sigma/sqrt(n)`, so a short transcript's rate is much noisier than a long one's at the same
nominal scale. The v2 blocks correct for this by shrinking toward a frozen per-code prior:

```
r'_c = (n * r_c + k * pi_c) / (n + k)      k = 25, pi_c = frozen corpus mean for code c
```

`pi_c` is a hardcoded constant measured over 2.36M scored turns; it is never recomputed at test
time. The strength barely matters (k of 10/25/50/100 are flat) — *that* you shrink is what
counts.

---

## 6. Validation protocol

**Group by `learning_objective_id`, never by session.** There are two leakage axes:

1. *Transcript leakage.* One session yields several responses sharing a transcript, so a random
   split leaks badly. Grouping by `session_id` fixes this — necessary, but not sufficient.
2. *Objective memorisation.* Folds that share objectives let the model learn per-objective base
   rates that do not transfer. DrivenData confirmed in the competition forum that not every
   objective in the test set appears in training.

The two views disagree completely, and only one is honest:

| view | log loss | AUROC | |
|---|---:|---:|---|
| session-grouped (objectives shared) | 0.552 | 0.707 | fantasy |
| **objective-disjoint `GroupKFold`** | **0.6296** | **0.572** | matches the leaderboard (0.6221 / 0.581) |

The leaderboard lands on the objective-disjoint view. We therefore select and report on it, and
keep the fold partition frozen (sha `59f0c807dad3ac08`) so every experiment is comparable.

Consequences we followed throughout:

- Report **log loss and AUROC together**, always. Off-corpus especially, AUROC ranks while log
  loss is confounded by how a frozen calibrator transports to a different base rate.
- Score on external corpora (MRBench, QATD, MathDial, CoMTA, Bridge, TutorChat) as well as
  in-domain. In-domain gates were 0-for-4 at predicting leaderboard movement; the off-corpus
  head-to-head was the better signal.
- Any train-fit statistic keyed on the objective id degrades to a global-mean fallback on unseen
  objectives. Per-objective difficulty columns are excluded from every model from the fourth rung
  of the ladder onward. Objective *text* generalises; objective *identity* does not.
- Test samples are processed independently. No cross-sample features, no test-set normalisation,
  no transductive tricks — the competition treats these as disqualifying, and frozen priors are
  read from shipped constants rather than recomputed on the batch.

---

## 7. What did not work

Recorded because negative results were most of the information.

- **Every XGBoost knob.** 57 one-variable arms on this exact base, on the final day: lambda,
  alpha, `min_child_weight`, all three `colsample_*`, `subsample`, `max_leaves`, `gamma`,
  `max_delta_step`, `scale_pos_weight`, depthwise regime, lr x trees. Every curve is shallow with
  an interior optimum at or beside the incumbent, and stacked combinations were sub-additive —
  every pair and the triple did worse than the best single change.
- **Compressing the objective text** (see above): −1.87 to −5.15 AUROC.
- **Alternative v2 aggregations:** k=35, per-code empirical-Bayes k, bounded-share ratios, and
  all four narrow collapses. All refuted.
- **k-shrinking the other distilled blocks:** `d29dst_` resolves in-domain (−0.00292) but costs
  about 1.6 AUROC on QATD; `studst_` is null.
- **Row weighting** by objective frequency: null to harmful.
- **Contingency-style interaction features:** 6 of 6 null. The blocker is that session-level
  aggregation makes a feature constant across all of a session's objectives, which is precisely
  the axis the task asks you to discriminate along.
- **Adding width.** Three of four width-increasing changes that resolved *better* on clean
  held-out rows went on to score *worse* on the leaderboard. In-domain improvement was close to
  uninformative about generalisation.
- **Pruning columns by distribution shift.** We ranked blocks by how far their inputs sat from
  the training distribution and dropped the worst; every ablation hurt, and the block furthest
  off-distribution was the second most damaging to remove. A block can read far off-distribution
  and still carry signal. Judge by ablation, never by a z-score.

A structural note worth carrying forward: **this configuration sits on an isolated peak.** Eleven
perturbations by unrelated mechanisms all land 1.2–2.0 AUROC lower on out-of-corpus transfer,
while the shipped model sits about 0.8 above that shelf. The size of an in-domain gain does not
predict the transfer cost (r = −0.09, p = 0.78) — it is a cliff, not a gradient.

---

## 8. Licence and data

Code in this repository is released under the **MIT Licence** (see [LICENSE](LICENSE)).

- **The competition data is not redistributed here.** Transcripts and labels are the property of
  the challenge organisers and their data providers, and are available only through the
  competition. `data/` is gitignored.
- **`Qwen/Qwen3-8B`** is Apache-2.0 and must be downloaded separately. Only our LoRA adapter is
  distributed here.
- Every model and dataset used in the solution was checked for commercial licensability, as the
  competition rules require; no non-commercial (NC / CC-BY-NC) model or corpus is used.
