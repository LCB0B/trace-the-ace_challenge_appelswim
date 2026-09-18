#!/usr/bin/env python3
"""Regenerate FEATURES.md from the shipped booster itself.

The column list is read out of ``assets/baseline.joblib`` -- the model is the
authority, so the documentation cannot drift from what is actually scored. Most
columns are formulaic (one rate per codebook code) and get their description
derived from the code name; the irregular ones are in IRREGULAR below. The
script fails loudly if the booster carries a column it cannot describe.

Usage:
    python tools/build_feature_table.py            # rewrite FEATURES.md
    python tools/build_feature_table.py --check    # verify it is in sync
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "FEATURES.md"

# --------------------------------------------------------------------------
# Families, in the order they are presented.
# prefix -> (family label, instrument)
# --------------------------------------------------------------------------
#: Columns are grouped by the CONSTRUCT they measure, not by the prefix that happens to
#: carry it. Tutor moves arrive through two instruments (the v2 pass as `v2ht_`, the
#: interview pass as `d29dst_`) and student signals through three, so a prefix-per-family
#: split scattered one construct across several tables. Within tutor moves the codebook a
#: code comes from is reported separately, because 15 of them are NTO's, verbatim.
FAMILIES: list[tuple[str, str, str]] = [
    ("__tutor__", "Tutor move",
     "LLM-distilled - NTO codebook, interview codes, v2 additions"),
    ("__student__", "Student signal",
     "LLM-distilled - interview codes, taxonomy v2, 12-code student states"),
    ("recdst_", "Answer log", "LLM-distilled - per-turn correctness"),
    ("ctg_", "Contingency", "regex state machine"),
    ("rep_", "Repair", "regex state machine"),
]

NTO_TAXONOMY_URL = "https://arxiv.org/abs/2603.05778"
NTO_REPO_URL = "https://github.com/National-Tutoring-Observatory/sandpiper/blob/main/app/modules/prompts/helpers/defaultPrompts.ts"

#: NTO's 15 tutor codes, verbatim (docs/sandpiper_codebooks.md).
NTO_CODES = {
    "prompting_related_concepts", "prompting_alternative_representation",
    "prompting_self_explanation", "prompting_next_step", "prompting_self_correction",
    "feedback_correct", "feedback_incorrect", "feedback_neutral", "revoicing",
    "restating", "giving_hint", "giving_example", "explaining_conceptual",
    "explaining_procedural", "giving_answer",
}
#: The 7 codes written from the practising-tutor interviews. Uppercase in `d29dst_`.
DOC29_TUTOR = {
    "NORMALIZING_DIFFICULTY", "RELEASE_HANDOFF", "TRANSFER_PROBE",
    "SIMPLIFYING_TO_SUBPROBLEM", "MOTIVATING_RELEVANCE", "SUMMARIZING_PROGRESS",
    "OFFERING_CHOICE",
}
#: Added in v2 after hand-scoring showed the vocabulary could not express them.
V2_NEW_TUTOR = {"asking_question", "guiding_session", "explaining_tool", "giving_praise",
                "worked_demonstration"}
TUTOR_RESIDUAL = {"tutor_unintelligible", "tutor_technical", "tutor_social", "tutor_other",
                  "tutor_no_move"}
DERIVED_RATIOS = {"elicit_to_tell", "feedback_positive_ratio"}


def source_of(col: str) -> str:
    """Which codebook a tutor-move or student-signal column's code comes from."""
    if col.startswith("v2ht_"):
        code = col[len("v2ht_"):]
        if code in DERIVED_RATIOS:
            return "derived ratio"
        code = code.removesuffix("_rate")
        if code in NTO_CODES:
            return "NTO codebook"
        if code in V2_NEW_TUTOR:
            return "added in v2"
        if code in TUTOR_RESIDUAL:
            return "residual"
        return "interview (doc29)"
    if col.startswith("d29dst_"):
        return "interview (doc29)"
    if col.startswith("v2hs_"):
        return "taxonomy v2"
    if col.startswith("studst_"):
        return "12-code student states"
    return ""


def is_tutor(col: str) -> bool:
    if col.startswith("v2ht_"):
        return True
    if col.startswith("d29dst_"):
        stem = col[len("d29dst_"):]
        # the interview pass labels BOTH speakers; its tutor codes are the uppercase ones,
        # plus any_move_rate, which is the share of tutor turns carrying any of them.
        code = stem.removesuffix("_late").removesuffix("_rate")
        return code in DOC29_TUTOR or stem == "any_move_rate"
    return False


def is_student(col: str) -> bool:
    return col.startswith(("v2hs_", "studst_")) or (
        col.startswith("d29dst_") and not is_tutor(col))


HANDCRAFTED = ("Handcrafted", "regex / counts / timestamps")

# --------------------------------------------------------------------------
# Descriptions for every column that is not a plain "<code>_rate".
# Sources: docs/18_model_E2_features.md (handcrafted), evidence_features.py
# (recdst_/ctg_/rep_), doc29_features.py, v2_agg.py, student_features.py.
# --------------------------------------------------------------------------
IRREGULAR: dict[str, str] = {
    # --- windowed student engagement --------------------------------------
    "stu_words_final_third_share": "Share of all student words falling in the final third of the session.",
    "stu_words_trend": "Linear slope of per-turn student word count across the session.",
    "final_third_stu_q_rate": "Student question rate (turns ending in '?') within the final third.",
    "stu_q_rate_delta": "Late-third minus early-third student question rate.",
    # --- tutor discourse moves (regex lexicons) ---------------------------
    "tutor_elicit_rate": "Fraction of tutor turns eliciting reasoning ('why', 'explain', 'walk me through').",
    "tutor_tell_rate": "Fraction of tutor turns telling the answer or the next step directly.",
    "elicit_to_tell": "Elicit rate divided by tell rate: how Socratic the tutor is overall.",
    "tutor_q_rate": "Fraction of tutor turns ending in a question mark.",
    "uptake": "Token overlap between a tutor turn and the student turn before it: does the tutor build on what the student said.",
    "revoicing_rate": "Fraction of tutor turns with high uptake, i.e. the tutor restates the student's contribution.",
    "confirm_rate": "Fraction of tutor turns carrying praise or confirmation.",
    "corrective_rate": "Fraction of tutor turns correcting the student.",
    # --- timing and audio quality -----------------------------------------
    "wait_time_s": "Mean gap in seconds after a tutor question before the student replies.",
    "gap_cv": "Coefficient of variation of inter-utterance gaps: how uneven the session's rhythm is.",
    "n_long_pauses": "Count of gaps over 30 seconds, normalised per turn.",
    "student_unclear_only": "Ratio of student turns that consist only of an '[unclear]' marker.",
    "unclear_per_student_turn": "'[unclear]' transcription markers per student turn.",
    # --- LO <-> transcript alignment --------------------------------------
    "lo_cov_session": "Fraction of the objective's content tokens appearing anywhere in the session.",
    "lo_cov_student": "Fraction of the objective's tokens appearing in student turns specifically.",
    "lo_cov_end": "Fraction of the objective's tokens appearing in the final third of the session.",
    "lo_keyword_overlap": "Fraction of objective words found anywhere in the transcript.",
    "lo_keyword_in_student": "Fraction of objective words found in student turns.",
    "lo_keyword_in_tutor": "Fraction of objective words found in tutor turns.",
    "lo_student_tutor_coverage_ratio": "Student objective-vocabulary production divided by the tutor's.",
    "lo_first_mention_frac": "Position (0-1) of the first utterance mentioning an objective keyword.",
    "lo_mention_density": "Utterances mentioning the objective, divided by total utterances.",
    # --- turn balance and lexical diversity -------------------------------
    "student_tutor_word_ratio": "Student words divided by tutor words.",
    "student_words_per_turn": "Mean student words per turn.",
    "tutor_words_per_turn": "Mean tutor words per turn.",
    "student_short_turn_ratio": "Fraction of student turns of 3 words or fewer.",
    "late_student_short_ratio": "Fraction of short student turns in the last 20% of the session.",
    "student_ttr": "Student type-token ratio: lexical diversity over the whole student text.",
    "student_turn_share": "Student turns divided by total utterances.",
    # --- objective curriculum metadata ------------------------------------
    "lo_ks_level": "UK key stage (1-4) inferred from the objective text.",
    "lo_topic_strand": "Curriculum strand (Number, Algebra, Geometry, ...) inferred from the objective text.",
    "lo_bloom_level": "Bloom's taxonomy cognitive-demand level inferred from the objective text.",
    # --- Sandpiper/NTO regex tutor-move rates -----------------------------
    "tutor_feedback_incorrect_rate": "Rate at which the tutor flags a response as incorrect.",
    "tutor_backchannels_rate": "Rate of tutor turns that are a bare neutral acknowledgement ('ok', 'got it').",
    "tutor_prompts_explanation_rate": "Rate at which the tutor asks why or how the student reasoned.",
    "tutor_prompts_next_step_rate": "Rate at which the tutor asks what comes next procedurally.",
    "tutor_prompts_self_correction_rate": "Rate at which the tutor prompts the student to fix their own error.",
    "tutor_prompts_alt_repr_rate": "Rate at which the tutor asks for an alternative representation (draw, sketch).",
    "tutor_tag_questions_rate": "Rate of tutor turns ending in a tag question ('right?', 'ok?').",
    "tutor_gives_hint_rate": "Rate at which the tutor gives a partial hint.",
    "tutor_gives_example_rate": "Rate at which the tutor gives an analogy or worked example.",
    "tutor_explains_procedural_rate": "Rate at which the tutor explains the steps ('how').",
    "tutor_explains_conceptual_rate": "Rate at which the tutor explains the underlying reason ('why').",
    "tutor_gives_answer_rate": "Rate at which the tutor states the answer outright.",
    "tutor_restating_rate": "Rate at which the tutor restates the student's prior turn (content-word recall >= 60%).",
    "tutor_prior_knowledge_check_rate": "Rate at which the tutor probes what the student already knows.",
    "feedback_positive_ratio": "Praise divided by (praise + incorrect-feedback).",
    "elicit_vs_tell_ratio": "(elicit + prompt moves) divided by (tell + procedural moves).",
    "scaffold_vs_tell_ratio": "(hint + example) divided by gives-answer.",
    "error_response_quality": "Self-correction prompts divided by (incorrect-feedback + 1).",
    # --- R: the within-session answer log ---------------------------------
    "recdst_accuracy": "Mean distilled correctness over all graded student turns in the session.",
    "recdst_last_1": "Distilled correctness of the most recent graded turn.",
    "recdst_last_3": "Mean distilled correctness over the last 3 graded turns.",
    "recdst_last_5": "Mean distilled correctness over the last 5 graded turns.",
    "recdst_recency_acc": "Exponentially recency-weighted accuracy (half-life about 3.5 turns).",
    "recdst_run_correct_share": "Length of the longest correct run, as a share of graded turns.",
    "recdst_run_incorrect_share": "Length of the longest incorrect run, as a share of graded turns.",
    "recdst_half_delta": "Second-half accuracy minus first-half accuracy.",
    "recdst_trend": "OLS slope of correctness against turn position.",
    "recdst_pos_last_incorrect": "Relative position (0-1) of the last incorrect answer.",
    "recdst_early_acc": "Accuracy over the first 30% of the session.",
    "recdst_late_acc": "Accuracy over the last 30% of the session.",
    "recdst_answered_share": "Share of student turns that were graded as an answer at all.",
    "recdst_log_answered": "log1p of the number of graded turns: how much evidence the log rests on.",
    # --- C: contingency (move x student state) ----------------------------
    "ctg_tell_imp": "Tutor tell-rate while the student is at an impasse.",
    "ctg_tell_flu": "Tutor tell-rate while the student is fluent.",
    "ctg_elicit_imp": "Tutor elicit-rate while the student is at an impasse.",
    "ctg_elicit_flu": "Tutor elicit-rate while the student is fluent.",
    "ctg_tell_contrast": "Tell-rate at impasse minus tell-rate when fluent: does the tutor switch to telling when the student is stuck.",
    "ctg_elicit_contrast": "Elicit-rate at impasse minus elicit-rate when fluent.",
    "ctg_imp_share": "Share of the session spent in the impasse state.",
    "ctg_imp_utt_share": "Share of utterances occurring during an impasse.",
    "ctg_late_imp_share": "Share of the final third spent at an impasse.",
    "ctg_episodes_per_utt": "Impasse episodes per utterance: how often the student gets stuck.",
    # --- P: repair arcs ----------------------------------------------------
    "rep_prompt_share": "Share of repair arcs where the tutor responds to an error by prompting.",
    "rep_explain_share": "Share of repair arcs where the tutor responds by explaining.",
    "rep_answer_share": "Share of repair arcs where the tutor responds by giving the answer.",
    "rep_recovery_rate": "Share of repair arcs where the next graded answer is correct.",
    "rep_re_error_rate": "Share of repair arcs followed by another error within 6 utterances.",
    "rep_arcs_per_tutor_turn": "Repair arcs per tutor turn.",
    "rep_has_arc": "Whether the session contains any repair arc at all.",
    # --- doc29 derived (non rate/late) ------------------------------------
    "d29dst_any_move_rate": "Rate at which a tutor turn carries any of the seven interview-codebook moves.",
    "d29dst_answering_rate": "Rate at which a student turn is an answer of any kind.",
    "d29dst_reasoning_share": "Share of answering student turns that show reasoning rather than a bare answer.",
    "d29dst_confidence_contrast": "Assertive rate minus hedged rate: net student confidence.",
    "d29dst_delight_pos_mean": "Mean insight/delight probability over the turns where it fires.",
    "d29dst_delight_any": "Whether any turn shows insight or delight.",
    # --- v2 derived --------------------------------------------------------
    "v2ht_elicit_to_tell": "Eliciting move mass divided by telling move mass over the v2 taxonomy (capped at 10).",
    "v2ht_feedback_positive_ratio": "Positive feedback divided by all feedback, over the v2 taxonomy.",
}

# Human-readable names for codebook codes that do not read well de-underscored.
CODE_GLOSS: dict[str, str] = {
    "guiding_session": "steers the lesson",
    "worked_demonstration": "works the problem aloud",
    "explaining_tool": "explains the platform",
    "release_handoff": "hands the work back to the student",
    "transfer_probe": "probes transfer to a new case",
    "normalizing_difficulty": "normalises the difficulty",
    "simplifying_to_subproblem": "breaks the task into a subproblem",
    "motivating_relevance": "motivates why it matters",
    "summarizing_progress": "summarises progress so far",
    "offering_choice": "offers the student a choice",
    "prompting_self_explanation": "asks the student to explain their thinking",
    "prompting_self_correction": "asks the student to fix their own error",
    "prompting_alternative_representation": "asks for another representation",
    "prompting_related_concepts": "links to a related concept",
    "prompting_next_step": "asks for the next step",
    "tutor_no_move": "makes no codeable move",
    "student_no_move": "makes no codeable move",
    "answer_only": "gives a bare answer with no working",
    "reasoning_shown": "shows their working",
    "insight_delight": "expresses insight or delight",
    "seeking_clarification": "asks for clarification",
    "self_correcting": "corrects themselves",
    "confusion_expressed": "expresses confusion",
    "explains_why": "explains why",
    "shows_method": "shows their method",
    "asks_conceptual": "asks a conceptual question",
    "asks_verification": "asks whether they are right",
    "minimal": "gives a minimal response",
    "off_task": "goes off task",
    "unmarked": "shows no confidence marker",
}


def _gloss(code: str) -> str:
    if code in CODE_GLOSS:
        return CODE_GLOSS[code]
    return code.replace("_", " ")


def describe(col: str) -> str:
    """Return a one-line description, deriving the formulaic cases."""
    if col in IRREGULAR:
        return IRREGULAR[col]

    if col.startswith("v2ht_") and col.endswith("_rate"):
        return f"Session rate (k-shrunk) of tutor turns where the tutor {_gloss(col[5:-5])}."
    if col.startswith("v2hs_") and col.endswith("_rate"):
        return f"Session rate (k-shrunk) of student turns where the student {_gloss(col[5:-5])}."
    if col.startswith("studst_") and col.endswith("_rate"):
        return f"Mean probability that a student turn shows that the student {_gloss(col[7:-5])}."
    if col.startswith("d29dst_"):
        stem = col[7:]
        if stem.endswith("_late"):
            return f"Rate of '{_gloss(stem[:-5].lower())}' within the final third of the session."
        if stem.endswith("_rate"):
            return f"Session rate of '{_gloss(stem[:-5].lower())}'."
    raise KeyError(col)


def family_of(col: str) -> tuple[str, str]:
    if is_tutor(col):
        return FAMILIES[0][1], FAMILIES[0][2]
    if is_student(col):
        return FAMILIES[1][1], FAMILIES[1][2]
    for prefix, label, instrument in FAMILIES[2:]:
        if col.startswith(prefix):
            return label, instrument
    return HANDCRAFTED


def booster_columns() -> tuple[list[str], list[str]]:
    """Return (columns fed to the booster, columns built then dropped)."""
    sys.path.insert(0, str(ROOT / "assets"))
    import joblib  # noqa: PLC0415

    model = joblib.load(ROOT / "assets" / "baseline.joblib")
    pre = model.bases[0].steps[0][1]
    used: list[str] | None = None
    dropped: list[str] = []
    for name, transformer, cols in pre.transformers_:
        if name == "dlg":
            used = list(cols)
        elif name == "remainder" and transformer == "drop":
            dropped = list(cols)
    if used is None:
        raise SystemExit("no 'dlg' branch found in the shipped ColumnTransformer")
    return used, dropped


def render(cols: list[str], dropped: list[str]) -> str:
    order = [label for _p, label, _i in FAMILIES]
    order.insert(0, HANDCRAFTED[0])
    grouped: dict[str, list[str]] = {label: [] for label in order}
    for col in cols:
        grouped[family_of(col)[0]].append(col)

    lines = [
        "# The 177 feature columns of the shipped XGBoost model",
        "",
        "Generated by `tools/build_feature_table.py` directly from",
        "`assets/baseline.joblib`, so this table cannot drift from the model that",
        "was actually scored. Regenerate with:",
        "",
        "```bash",
        "python tools/build_feature_table.py",
        "```",
        "",
        "These 177 numeric columns are the `dlg` branch of the pipeline's",
        "`ColumnTransformer`. Two further branches expand the objective itself",
        "(TF-IDF over the objective text, and a one-hot of the objective id) for",
        "**2,032** features in total. A further 92 columns are computed by the",
        "feature code but dropped by the transformer; they are listed at the end.",
        "",
        "## Summary",
        "",
        "Columns are grouped by the construct they measure. The write-up groups the same 177",
        "columns by construct too, but counts the regex tutor and student columns under tutor",
        "moves and student signals rather than under Handcrafted, so its per-construct totals",
        "are larger. Same matrix, one classification decision apart.",
        "",
        "| family | n | instrument |",
        "|---|---:|---|",
    ]
    for label in order:
        instrument = HANDCRAFTED[1] if label == HANDCRAFTED[0] else next(
            i for _p, lbl, i in FAMILIES if lbl == label
        )
        lines.append(f"| {label} | {len(grouped[label])} | {instrument} |")
    lines += [f"| **Total** | **{len(cols)}** | |", ""]

    tutor = grouped[FAMILIES[0][1]]
    by_source: dict[str, int] = {}
    for col in tutor:
        by_source[source_of(col)] = by_source.get(source_of(col), 0) + 1
    lines += [
        "Our tutor moves build on the National Tutoring Observatory's Tutor Move Taxonomy",
        f"([Zhou et al., 2026]({NTO_TAXONOMY_URL})). Concretely, we use its 15 learning-support",
        "codes, with the definitions from the tutor-move prompt in NTO's",
        f"[Sandpiper]({NTO_REPO_URL}) repository. They are marked **NTO** below. The rest",
        "are ours: codes from practising-tutor interviews, codes added after human validation,",
        "and residual categories.",
        "",
        "| tutor-move source | n |",
        "|---|---:|",
    ]
    for label in ("NTO codebook", "interview (doc29)", "added in v2", "residual",
                  "derived ratio"):
        if by_source.get(label):
            lines.append(f"| {label} | {by_source[label]} |")
    lines += [
        "",
        "The 7 interview codes are counted once per instrument that measures them: as",
        "`v2ht_*` session rates and again as `d29dst_*` rate/late pairs. Both blocks are in",
        "the booster, so both are listed.",
        "",
    ]

    for label in order:
        if not grouped[label]:
            continue
        annotated = label in (FAMILIES[0][1], FAMILIES[1][1])
        head = ("| column | source | description |", "|---|---|---|") if annotated else (
            "| column | description |", "|---|---|")
        lines += [f"## {label} ({len(grouped[label])})", "", *head]
        for col in grouped[label]:
            if annotated:
                src = source_of(col)
                src = "**NTO**" if src == "NTO codebook" else src
                lines.append(f"| `{col}` | {src} | {describe(col)} |")
            else:
                lines.append(f"| `{col}` | {describe(col)} |")
        lines.append("")

    lines += [
        f"## Built but dropped ({len(dropped)})",
        "",
        "The feature code computes these, but the pipeline's `remainder='drop'`",
        "keeps them out of the booster. Three groups matter:",
        "",
        "- the 21 `ntodst_` NTO move rates, deliberately swapped out for the 46 v2",
        "  columns in the final model (`--drop-cols ntodst_`);",
        "- the 3 `loknn_` objective-neighbourhood columns (`--no-loknn`);",
        "- raw counts whose per-turn rate twin is kept instead, session-level",
        "  aggregates that would breach the no-cross-sample rule, and the 14 student",
        "  regex columns replaced by distilled `studst_` states (`--swap-student`).",
        "",
        "Per-objective label statistics (`lo_mean_correct`, `lo_difficulty_rank`, ...)",
        "are excluded further upstream and never reach the matrix at all: the hidden",
        "test contains objectives absent from training, so a train-fit statistic keyed",
        "on the objective id would silently fall back to a global mean.",
        "",
        "```",
    ]
    for i in range(0, len(dropped), 3):
        lines.append("  ".join(f"{c:<38}" for c in dropped[i:i + 3]).rstrip())
    lines += ["```", ""]
    return "\n".join(lines)


README = ROOT / "README.md"
BEGIN = "<!-- BEGIN FEATURE TABLE (generated by tools/build_feature_table.py) -->"
END = "<!-- END FEATURE TABLE -->"


def render_readme_table(cols: list[str]) -> str:
    """One flat table, one row per booster column, for inlining in the README."""
    lines = [
        BEGIN,
        "",
        f"All {len(cols)} columns, in the order the booster receives them.",
        "",
        "| # | column | family | description |",
        "|---:|---|---|---|",
    ]
    for i, col in enumerate(cols, 1):
        fam = family_of(col)[0]
        if source_of(col) == "NTO codebook":
            fam += " (NTO)"
        lines.append(f"| {i} | `{col}` | {fam} | {describe(col)} |")
    lines += ["", END]
    return "\n".join(lines)


def splice_readme(cols: list[str]) -> str:
    """Return README.md with the generated table swapped in between markers."""
    text = README.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(
            f"README.md is missing the {BEGIN!r} / {END!r} markers; add them "
            "around the feature table before regenerating."
        )
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    return head + render_readme_table(cols) + tail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify without rewriting")
    args = parser.parse_args()

    cols, dropped = booster_columns()
    missing = []
    for col in cols:
        try:
            describe(col)
        except KeyError:
            missing.append(col)
    if missing:
        print(f"ERROR: {len(missing)} columns have no description:", file=sys.stderr)
        for col in missing:
            print(f"  {col}", file=sys.stderr)
        return 1

    text = render(cols, dropped)
    readme = splice_readme(cols)

    if args.check:
        stale = []
        if not OUT.exists() or OUT.read_text() != text:
            stale.append("FEATURES.md")
        if README.read_text() != readme:
            stale.append("README.md")
        if stale:
            print(
                f"out of sync with assets/baseline.joblib: {', '.join(stale)}",
                file=sys.stderr,
            )
            return 1
        print(f"FEATURES.md and README.md in sync ({len(cols)} columns)")
        return 0

    OUT.write_text(text)
    README.write_text(readme)
    print(f"wrote {OUT} and the README table ({len(cols)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
