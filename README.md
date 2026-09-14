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

The table below is generated directly from `assets/baseline.joblib` by
`tools/build_feature_table.py`, so it cannot drift from the model that was scored. The same
177 columns grouped by family, plus the 92 columns that are computed and then dropped, are in
**[FEATURES.md](FEATURES.md)**.

Summary first, then every column.

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

<!-- BEGIN FEATURE TABLE (generated by tools/build_feature_table.py) -->

All 177 columns, in the order the booster receives them.

| # | column | family | description |
|---:|---|---|---|
| 1 | `stu_words_final_third_share` | Handcrafted | Share of all student words falling in the final third of the session. |
| 2 | `stu_words_trend` | Handcrafted | Linear slope of per-turn student word count across the session. |
| 3 | `final_third_stu_q_rate` | Handcrafted | Student question rate (turns ending in '?') within the final third. |
| 4 | `stu_q_rate_delta` | Handcrafted | Late-third minus early-third student question rate. |
| 5 | `tutor_elicit_rate` | Handcrafted | Fraction of tutor turns eliciting reasoning ('why', 'explain', 'walk me through'). |
| 6 | `tutor_tell_rate` | Handcrafted | Fraction of tutor turns telling the answer or the next step directly. |
| 7 | `elicit_to_tell` | Handcrafted | Elicit rate divided by tell rate: how Socratic the tutor is overall. |
| 8 | `tutor_q_rate` | Handcrafted | Fraction of tutor turns ending in a question mark. |
| 9 | `uptake` | Handcrafted | Token overlap between a tutor turn and the student turn before it: does the tutor build on what the student said. |
| 10 | `revoicing_rate` | Handcrafted | Fraction of tutor turns with high uptake, i.e. the tutor restates the student's contribution. |
| 11 | `confirm_rate` | Handcrafted | Fraction of tutor turns carrying praise or confirmation. |
| 12 | `corrective_rate` | Handcrafted | Fraction of tutor turns correcting the student. |
| 13 | `wait_time_s` | Handcrafted | Mean gap in seconds after a tutor question before the student replies. |
| 14 | `gap_cv` | Handcrafted | Coefficient of variation of inter-utterance gaps: how uneven the session's rhythm is. |
| 15 | `lo_cov_session` | Handcrafted | Fraction of the objective's content tokens appearing anywhere in the session. |
| 16 | `lo_cov_student` | Handcrafted | Fraction of the objective's tokens appearing in student turns specifically. |
| 17 | `lo_cov_end` | Handcrafted | Fraction of the objective's tokens appearing in the final third of the session. |
| 18 | `student_tutor_word_ratio` | Handcrafted | Student words divided by tutor words. |
| 19 | `n_long_pauses` | Handcrafted | Count of gaps over 30 seconds, normalised per turn. |
| 20 | `student_words_per_turn` | Handcrafted | Mean student words per turn. |
| 21 | `student_short_turn_ratio` | Handcrafted | Fraction of student turns of 3 words or fewer. |
| 22 | `student_ttr` | Handcrafted | Student type-token ratio: lexical diversity over the whole student text. |
| 23 | `tutor_words_per_turn` | Handcrafted | Mean tutor words per turn. |
| 24 | `student_unclear_only` | Handcrafted | Ratio of student turns that consist only of an '[unclear]' marker. |
| 25 | `unclear_per_student_turn` | Handcrafted | '[unclear]' transcription markers per student turn. |
| 26 | `late_student_short_ratio` | Handcrafted | Fraction of short student turns in the last 20% of the session. |
| 27 | `lo_keyword_overlap` | Handcrafted | Fraction of objective words found anywhere in the transcript. |
| 28 | `lo_keyword_in_student` | Handcrafted | Fraction of objective words found in student turns. |
| 29 | `lo_keyword_in_tutor` | Handcrafted | Fraction of objective words found in tutor turns. |
| 30 | `lo_student_tutor_coverage_ratio` | Handcrafted | Student objective-vocabulary production divided by the tutor's. |
| 31 | `lo_first_mention_frac` | Handcrafted | Position (0-1) of the first utterance mentioning an objective keyword. |
| 32 | `lo_mention_density` | Handcrafted | Utterances mentioning the objective, divided by total utterances. |
| 33 | `feedback_positive_ratio` | Handcrafted | Praise divided by (praise + incorrect-feedback). |
| 34 | `elicit_vs_tell_ratio` | Handcrafted | (elicit + prompt moves) divided by (tell + procedural moves). |
| 35 | `scaffold_vs_tell_ratio` | Handcrafted | (hint + example) divided by gives-answer. |
| 36 | `error_response_quality` | Handcrafted | Self-correction prompts divided by (incorrect-feedback + 1). |
| 37 | `lo_ks_level` | Handcrafted | UK key stage (1-4) inferred from the objective text. |
| 38 | `lo_topic_strand` | Handcrafted | Curriculum strand (Number, Algebra, Geometry, ...) inferred from the objective text. |
| 39 | `lo_bloom_level` | Handcrafted | Bloom's taxonomy cognitive-demand level inferred from the objective text. |
| 40 | `tutor_feedback_incorrect_rate` | Handcrafted | Rate at which the tutor flags a response as incorrect. |
| 41 | `tutor_backchannels_rate` | Handcrafted | Rate of tutor turns that are a bare neutral acknowledgement ('ok', 'got it'). |
| 42 | `tutor_prompts_explanation_rate` | Handcrafted | Rate at which the tutor asks why or how the student reasoned. |
| 43 | `tutor_prompts_next_step_rate` | Handcrafted | Rate at which the tutor asks what comes next procedurally. |
| 44 | `tutor_prompts_self_correction_rate` | Handcrafted | Rate at which the tutor prompts the student to fix their own error. |
| 45 | `tutor_prompts_alt_repr_rate` | Handcrafted | Rate at which the tutor asks for an alternative representation (draw, sketch). |
| 46 | `tutor_tag_questions_rate` | Handcrafted | Rate of tutor turns ending in a tag question ('right?', 'ok?'). |
| 47 | `tutor_gives_hint_rate` | Handcrafted | Rate at which the tutor gives a partial hint. |
| 48 | `tutor_gives_example_rate` | Handcrafted | Rate at which the tutor gives an analogy or worked example. |
| 49 | `tutor_explains_procedural_rate` | Handcrafted | Rate at which the tutor explains the steps ('how'). |
| 50 | `tutor_explains_conceptual_rate` | Handcrafted | Rate at which the tutor explains the underlying reason ('why'). |
| 51 | `tutor_gives_answer_rate` | Handcrafted | Rate at which the tutor states the answer outright. |
| 52 | `tutor_restating_rate` | Handcrafted | Rate at which the tutor restates the student's prior turn (content-word recall >= 60%). |
| 53 | `tutor_prior_knowledge_check_rate` | Handcrafted | Rate at which the tutor probes what the student already knows. |
| 54 | `student_turn_share` | Handcrafted | Student turns divided by total utterances. |
| 55 | `recdst_accuracy` | Answer log | Mean distilled correctness over all graded student turns in the session. |
| 56 | `recdst_last_1` | Answer log | Distilled correctness of the most recent graded turn. |
| 57 | `recdst_last_3` | Answer log | Mean distilled correctness over the last 3 graded turns. |
| 58 | `recdst_last_5` | Answer log | Mean distilled correctness over the last 5 graded turns. |
| 59 | `recdst_recency_acc` | Answer log | Exponentially recency-weighted accuracy (half-life about 3.5 turns). |
| 60 | `recdst_run_correct_share` | Answer log | Length of the longest correct run, as a share of graded turns. |
| 61 | `recdst_run_incorrect_share` | Answer log | Length of the longest incorrect run, as a share of graded turns. |
| 62 | `recdst_half_delta` | Answer log | Second-half accuracy minus first-half accuracy. |
| 63 | `recdst_trend` | Answer log | OLS slope of correctness against turn position. |
| 64 | `recdst_pos_last_incorrect` | Answer log | Relative position (0-1) of the last incorrect answer. |
| 65 | `recdst_early_acc` | Answer log | Accuracy over the first 30% of the session. |
| 66 | `recdst_late_acc` | Answer log | Accuracy over the last 30% of the session. |
| 67 | `recdst_answered_share` | Answer log | Share of student turns that were graded as an answer at all. |
| 68 | `recdst_log_answered` | Answer log | log1p of the number of graded turns: how much evidence the log rests on. |
| 69 | `ctg_tell_imp` | Contingency | Tutor tell-rate while the student is at an impasse. |
| 70 | `ctg_tell_flu` | Contingency | Tutor tell-rate while the student is fluent. |
| 71 | `ctg_elicit_imp` | Contingency | Tutor elicit-rate while the student is at an impasse. |
| 72 | `ctg_elicit_flu` | Contingency | Tutor elicit-rate while the student is fluent. |
| 73 | `ctg_tell_contrast` | Contingency | Tell-rate at impasse minus tell-rate when fluent: does the tutor switch to telling when the student is stuck. |
| 74 | `ctg_elicit_contrast` | Contingency | Elicit-rate at impasse minus elicit-rate when fluent. |
| 75 | `ctg_imp_share` | Contingency | Share of the session spent in the impasse state. |
| 76 | `ctg_imp_utt_share` | Contingency | Share of utterances occurring during an impasse. |
| 77 | `ctg_late_imp_share` | Contingency | Share of the final third spent at an impasse. |
| 78 | `ctg_episodes_per_utt` | Contingency | Impasse episodes per utterance: how often the student gets stuck. |
| 79 | `rep_prompt_share` | Repair | Share of repair arcs where the tutor responds to an error by prompting. |
| 80 | `rep_explain_share` | Repair | Share of repair arcs where the tutor responds by explaining. |
| 81 | `rep_answer_share` | Repair | Share of repair arcs where the tutor responds by giving the answer. |
| 82 | `rep_recovery_rate` | Repair | Share of repair arcs where the next graded answer is correct. |
| 83 | `rep_re_error_rate` | Repair | Share of repair arcs followed by another error within 6 utterances. |
| 84 | `rep_arcs_per_tutor_turn` | Repair | Repair arcs per tutor turn. |
| 85 | `rep_has_arc` | Repair | Whether the session contains any repair arc at all. |
| 86 | `d29dst_NORMALIZING_DIFFICULTY_rate` | Interview codes | Session rate of 'normalises the difficulty'. |
| 87 | `d29dst_NORMALIZING_DIFFICULTY_late` | Interview codes | Rate of 'normalises the difficulty' within the final third of the session. |
| 88 | `d29dst_RELEASE_HANDOFF_rate` | Interview codes | Session rate of 'hands the work back to the student'. |
| 89 | `d29dst_RELEASE_HANDOFF_late` | Interview codes | Rate of 'hands the work back to the student' within the final third of the session. |
| 90 | `d29dst_TRANSFER_PROBE_rate` | Interview codes | Session rate of 'probes transfer to a new case'. |
| 91 | `d29dst_TRANSFER_PROBE_late` | Interview codes | Rate of 'probes transfer to a new case' within the final third of the session. |
| 92 | `d29dst_SIMPLIFYING_TO_SUBPROBLEM_rate` | Interview codes | Session rate of 'breaks the task into a subproblem'. |
| 93 | `d29dst_SIMPLIFYING_TO_SUBPROBLEM_late` | Interview codes | Rate of 'breaks the task into a subproblem' within the final third of the session. |
| 94 | `d29dst_MOTIVATING_RELEVANCE_rate` | Interview codes | Session rate of 'motivates why it matters'. |
| 95 | `d29dst_MOTIVATING_RELEVANCE_late` | Interview codes | Rate of 'motivates why it matters' within the final third of the session. |
| 96 | `d29dst_SUMMARIZING_PROGRESS_rate` | Interview codes | Session rate of 'summarises progress so far'. |
| 97 | `d29dst_SUMMARIZING_PROGRESS_late` | Interview codes | Rate of 'summarises progress so far' within the final third of the session. |
| 98 | `d29dst_OFFERING_CHOICE_rate` | Interview codes | Session rate of 'offers the student a choice'. |
| 99 | `d29dst_OFFERING_CHOICE_late` | Interview codes | Rate of 'offers the student a choice' within the final third of the session. |
| 100 | `d29dst_any_move_rate` | Interview codes | Rate at which a tutor turn carries any of the seven interview-codebook moves. |
| 101 | `d29dst_answer_only_rate` | Interview codes | Session rate of 'gives a bare answer with no working'. |
| 102 | `d29dst_answer_only_late` | Interview codes | Rate of 'gives a bare answer with no working' within the final third of the session. |
| 103 | `d29dst_reasoning_shown_rate` | Interview codes | Session rate of 'shows their working'. |
| 104 | `d29dst_reasoning_shown_late` | Interview codes | Rate of 'shows their working' within the final third of the session. |
| 105 | `d29dst_insight_delight_rate` | Interview codes | Session rate of 'expresses insight or delight'. |
| 106 | `d29dst_insight_delight_late` | Interview codes | Rate of 'expresses insight or delight' within the final third of the session. |
| 107 | `d29dst_frustration_rate` | Interview codes | Session rate of 'frustration'. |
| 108 | `d29dst_frustration_late` | Interview codes | Rate of 'frustration' within the final third of the session. |
| 109 | `d29dst_hedged_rate` | Interview codes | Session rate of 'hedged'. |
| 110 | `d29dst_hedged_late` | Interview codes | Rate of 'hedged' within the final third of the session. |
| 111 | `d29dst_unmarked_rate` | Interview codes | Session rate of 'shows no confidence marker'. |
| 112 | `d29dst_unmarked_late` | Interview codes | Rate of 'shows no confidence marker' within the final third of the session. |
| 113 | `d29dst_assertive_rate` | Interview codes | Session rate of 'assertive'. |
| 114 | `d29dst_assertive_late` | Interview codes | Rate of 'assertive' within the final third of the session. |
| 115 | `d29dst_answering_rate` | Interview codes | Rate at which a student turn is an answer of any kind. |
| 116 | `d29dst_reasoning_share` | Interview codes | Share of answering student turns that show reasoning rather than a bare answer. |
| 117 | `d29dst_confidence_contrast` | Interview codes | Assertive rate minus hedged rate: net student confidence. |
| 118 | `d29dst_delight_pos_mean` | Interview codes | Mean insight/delight probability over the turns where it fires. |
| 119 | `d29dst_delight_any` | Interview codes | Whether any turn shows insight or delight. |
| 120 | `v2ht_asking_question_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor asking question. |
| 121 | `v2ht_explaining_conceptual_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor explaining conceptual. |
| 122 | `v2ht_explaining_procedural_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor explaining procedural. |
| 123 | `v2ht_explaining_tool_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor explains the platform. |
| 124 | `v2ht_feedback_correct_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor feedback correct. |
| 125 | `v2ht_feedback_incorrect_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor feedback incorrect. |
| 126 | `v2ht_feedback_neutral_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor feedback neutral. |
| 127 | `v2ht_giving_answer_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor giving answer. |
| 128 | `v2ht_giving_example_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor giving example. |
| 129 | `v2ht_giving_hint_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor giving hint. |
| 130 | `v2ht_giving_praise_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor giving praise. |
| 131 | `v2ht_guiding_session_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor steers the lesson. |
| 132 | `v2ht_motivating_relevance_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor motivates why it matters. |
| 133 | `v2ht_normalizing_difficulty_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor normalises the difficulty. |
| 134 | `v2ht_offering_choice_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor offers the student a choice. |
| 135 | `v2ht_prompting_alternative_representation_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor asks for another representation. |
| 136 | `v2ht_prompting_next_step_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor asks for the next step. |
| 137 | `v2ht_prompting_related_concepts_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor links to a related concept. |
| 138 | `v2ht_prompting_self_correction_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor asks the student to fix their own error. |
| 139 | `v2ht_prompting_self_explanation_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor asks the student to explain their thinking. |
| 140 | `v2ht_release_handoff_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor hands the work back to the student. |
| 141 | `v2ht_restating_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor restating. |
| 142 | `v2ht_revoicing_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor revoicing. |
| 143 | `v2ht_simplifying_to_subproblem_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor breaks the task into a subproblem. |
| 144 | `v2ht_summarizing_progress_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor summarises progress so far. |
| 145 | `v2ht_transfer_probe_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor probes transfer to a new case. |
| 146 | `v2ht_tutor_no_move_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor makes no codeable move. |
| 147 | `v2ht_tutor_other_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor tutor other. |
| 148 | `v2ht_tutor_social_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor tutor social. |
| 149 | `v2ht_tutor_technical_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor tutor technical. |
| 150 | `v2ht_tutor_unintelligible_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor tutor unintelligible. |
| 151 | `v2ht_worked_demonstration_rate` | Tutor move (v2) | Session rate (k-shrunk) of tutor turns where the tutor works the problem aloud. |
| 152 | `v2ht_elicit_to_tell` | Tutor move (v2) | Eliciting move mass divided by telling move mass over the v2 taxonomy (capped at 10). |
| 153 | `v2ht_feedback_positive_ratio` | Tutor move (v2) | Positive feedback divided by all feedback, over the v2 taxonomy. |
| 154 | `v2hs_answer_only_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student gives a bare answer with no working. |
| 155 | `v2hs_confusion_expressed_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student expresses confusion. |
| 156 | `v2hs_frustration_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student frustration. |
| 157 | `v2hs_insight_delight_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student expresses insight or delight. |
| 158 | `v2hs_reasoning_shown_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student shows their working. |
| 159 | `v2hs_seeking_clarification_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student asks for clarification. |
| 160 | `v2hs_self_correcting_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student corrects themselves. |
| 161 | `v2hs_student_no_move_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student makes no codeable move. |
| 162 | `v2hs_student_other_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student student other. |
| 163 | `v2hs_student_social_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student student social. |
| 164 | `v2hs_student_technical_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student student technical. |
| 165 | `v2hs_student_unintelligible_rate` | Student move (v2) | Session rate (k-shrunk) of student turns where the student student unintelligible. |
| 166 | `studst_explains_why_rate` | Student state | Mean probability that a student turn shows that the student explains why. |
| 167 | `studst_shows_method_rate` | Student state | Mean probability that a student turn shows that the student shows their method. |
| 168 | `studst_answer_only_rate` | Student state | Mean probability that a student turn shows that the student gives a bare answer with no working. |
| 169 | `studst_confused_rate` | Student state | Mean probability that a student turn shows that the student confused. |
| 170 | `studst_hedged_rate` | Student state | Mean probability that a student turn shows that the student hedged. |
| 171 | `studst_insight_rate` | Student state | Mean probability that a student turn shows that the student insight. |
| 172 | `studst_asks_conceptual_rate` | Student state | Mean probability that a student turn shows that the student asks a conceptual question. |
| 173 | `studst_asks_verification_rate` | Student state | Mean probability that a student turn shows that the student asks whether they are right. |
| 174 | `studst_self_corrects_rate` | Student state | Mean probability that a student turn shows that the student self corrects. |
| 175 | `studst_guesses_rate` | Student state | Mean probability that a student turn shows that the student guesses. |
| 176 | `studst_minimal_rate` | Student state | Mean probability that a student turn shows that the student gives a minimal response. |
| 177 | `studst_off_task_rate` | Student state | Mean probability that a student turn shows that the student goes off task. |

<!-- END FEATURE TABLE -->

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
