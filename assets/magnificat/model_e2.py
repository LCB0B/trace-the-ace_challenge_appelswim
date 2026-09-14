"""Model E2 feature spec — SINGLE SOURCE OF TRUTH for both the trainer and
``submission_src/main.py`` (the classic train/inference matrix-mismatch bug
surface; see the project's rules).

E2 = Model E's full feature set (125 numeric cols), minus:
  - compliance drops (4)   -> session_task, cross-sample rule-3 DQ risk
  - source-artifact drops (9), verified-noise drops (6), redundant raw-count
    drops (~8) -> nothing left to preserve on either regime
  - LOC + LOC-delta drops (19) -> null on honest CV, degrades off-source
plus:
  - derived twin `student_turn_share` (turn balance, no existing equivalent)
  - Sandpiper's 15 raw counts -> per-turn rates (keep every move for per-move
    write-up analysis; rate ~= raw on TSL so transfer-safe at ~0 cost)

Design doc: Kay_model_E2.md. Deliberately takes ANY "full-E-shaped" matrix —
built from cached artifacts at train time (see scripts/model_e2.py) or built
live from transcripts at inference time (see submission_src/main.py's
`model_e2` branch) — and returns the identical 71-col E2 subset either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- compliance: cross-sample, must go ---------------------------------------
DROP_COMPLIANCE = ["session_n_responses", "session_n_distinct_los",
                   "session_lo_diversity", "session_lo_same"]
# --- verified noise: ~zero signal raw OR normalized (audit measurements) ------
DROP_NOISE = ["student_questions", "late_student_questions", "student_words_var",
              "student_max_turn_words", "total_chars", "n_unclear"]
# --- raw counts whose rate twin already exists (or is derived below) ----------
DROP_RAW_HAS_TWIN = ["n_utterances", "n_student", "n_tutor", "student_words",
                     "tutor_words", "n_question_marks", "student_short_turns",
                     "tutor_questions", "late_student_words", "late_tutor_questions"]
# --- source artifacts: no transfer value, near-dead weight on TSL ------------
DROP_SOURCE = ["is_tsl", "n_background", "has_tutor_ph", "tutor_ph_count",
               "student_says_tutor_name", "tutor_says_tutor_name",
               "tutor_name_calls", "tutor_name_calls_per_turn", "has_name_call"]
# --- extra drops: twin stronger / weak+timestamp (measured) -------------------
# stu_ttr dropped 2026-07-19: near-duplicate of student_ttr (both type-token
# ratio on student words) -- kept student_ttr (standard whole-text TTR formula,
# easier to justify in the write-up) over stu_ttr (content-word-only, unusual
# per-turn-dedup-then-union method).
DROP_EXTRA = ["tutor_praise", "stu_content_words", "longest_stu_utt_words", "pace_delta",
              "stu_ttr"]
# --- flat timing: no signal + source-coupled (timestamp) ----------------------
DROP_FLAT_TIMING = ["session_duration_secs", "avg_pause_secs", "avg_student_pause_secs"]

# --- Sandpiper 15 counts -> per-turn rates (keep every move for analysis) -----
SANDPIPER_COUNTS = ["tutor_feedback_incorrect", "tutor_backchannels",
    "tutor_prompts_explanation", "tutor_prompts_next_step",
    "tutor_prompts_self_correction", "tutor_prompts_alt_repr", "tutor_tag_questions",
    "tutor_gives_hint", "tutor_gives_example", "tutor_explains_procedural",
    "tutor_explains_conceptual", "tutor_gives_answer", "tutor_restating",
    "tutor_prior_knowledge_check", "student_explains"]
NEW_TWINS = ["student_turn_share"]  # turn balance; no existing equivalent


def _loc_drop_cols() -> list[str]:
    # imported lazily to avoid import-order coupling; dialogue_features never
    # imports this module, so this is safe at module scope too, but lazy keeps
    # this module importable standalone (e.g. from a bare script) either way.
    from . import dialogue_features as dfeat
    return list(dfeat.LOC_COLS) + list(dfeat.LOC_DELTA_COLS)


def drop_cols() -> list[str]:
    """Full E2 drop list (compliance + noise + raw-with-twin + source + LOC + extra + timing)."""
    return (DROP_COMPLIANCE + DROP_NOISE + DROP_RAW_HAS_TWIN + DROP_SOURCE
            + _loc_drop_cols() + DROP_EXTRA + DROP_FLAT_TIMING)


def add_e2_derived(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add `student_turn_share` + Sandpiper per-turn rate cols to X (copy).

    Requires `n_student`/`n_utterances`/`n_tutor` and whichever SANDPIPER_COUNTS
    columns are present in X (any subset is fine — inference and training both
    have the full set, but this stays defensive). Returns (X_with_derived,
    sandpiper_rate_col_names).
    """
    X = X.copy()
    X["student_turn_share"] = (X["n_student"] / X["n_utterances"]).where(
        X["n_utterances"] > 0, 0.0)
    sand_rate_cols = []
    for c in SANDPIPER_COUNTS:
        if c not in X.columns:
            continue
        denom = X["n_student"] if c == "student_explains" else X["n_tutor"]
        X[c + "_rate"] = (X[c] / denom).where(denom > 0, 0.0)
        sand_rate_cols.append(c + "_rate")
    return X, sand_rate_cols


def e2_columns(full_cols: list[str], sand_rate_cols: list[str]) -> list[str]:
    """Given the full-E column list and the derived Sandpiper-rate names,
    return the final E2 column selection (drops applied, twins + rates added)."""
    drop = set(drop_cols() + SANDPIPER_COUNTS)
    return [c for c in full_cols if c not in drop] + sand_rate_cols + NEW_TWINS


def build_e2_matrix(X_full: pd.DataFrame, full_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Single entry point for both trainer and main.py: given ANY full-E-shaped
    matrix (cached-artifact-built at train time, live-built at inference time)
    and its column list, return (X_with_derived_cols, e2_cols)."""
    X, sand_rate_cols = add_e2_derived(X_full)
    e2_cols = e2_columns(full_cols, sand_rate_cols)
    return X, e2_cols
