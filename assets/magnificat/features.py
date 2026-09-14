"""Turn raw transcripts into model-ready features.

Two outputs per session:
- ``transcript_text``: a single concatenated string (for TF-IDF / text models).
- a set of cheap structured features (turn counts, lengths, question marks...).

The structured features are intentionally simple — they are a baseline to beat,
not the final feature set.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # keep the vendored submission package importable without tqdm
    def tqdm(it, **_kwargs):
        return it

from . import data

_UNCLEAR = re.compile(r"\[unclear\]", flags=re.IGNORECASE)
_WORD = re.compile(r"\w+")
_PRAISE = re.compile(
    r"\b(well done|good job|great job|excellent|fantastic|brilliant|amazing|"
    r"perfect|superb|wonderful|well said|nice work|good work|that'?s? (right|correct|good|great))\b",
    flags=re.IGNORECASE,
)
_TUTOR_NAME_PH = re.compile(r"\[tutor_name\]", flags=re.IGNORECASE)

# Counts how many times the tutor addresses the student by name.
# Pattern: greeting/affirmation cue word → capitalized name.
# Inline (?i:...) makes only the cue group case-insensitive; the capture
# group [A-Z][a-z]{1,14} stays strict so "okay" / "what" aren't grabbed.
_NAME_CALL_RE = re.compile(
    r"""(?:
        (?i:hello|hi|hey|okay|ok|alright|all\s+right|welcome|great|good|
            nice|yes|yep|yeah|come\s+on|so|now|right|remember|well\s+done|
            perfect|excellent|wonderful|brilliant|fantastic|well)
        [,!?.\s]+
    )
    ([A-Z][a-z]{1,14})
    (?:[,!?.\s]|$)""",
    re.VERBOSE,
)
_STOP_WORDS = frozenset({
    "okay", "what", "now", "job", "effort", "hello", "please",
    "explanation", "amazing", "thank", "see", "yes", "yeah", "no",
    "great", "good", "nice", "right", "well", "so", "come", "all",
})

# ── Sandpiper-codebook-derived patterns ──────────────────────────────────────
# Each pattern targets one Sandpiper Tutor Moves / Rubric category that can be
# detected without semantic understanding.  Applied to tutor utterances only
# (the caller filters by role before calling .sum()).

_FEEDBACK_INCORRECT = re.compile(
    r"\b(not quite|that'?s? not (right|correct|quite right)|"
    r"not (exactly|right|correct)|no[,.]?\s+that'?s?|almost but|"
    r"close but not|not quite right|that'?s? (wrong|incorrect)|"
    r"no[,.]?\s+(it'?s?|the answer|that) (is |was )?(not|wrong|incorrect))\b",
    flags=re.IGNORECASE,
)

# Backchannel utterances: short neutral acknowledgments that are the *whole*
# utterance (fullmatch so "okay let me explain" is NOT a backchannel).
_BACKCHANNEL = re.compile(
    r"^\s*(i see|okay|ok|right|mm.?hmm|uh.?huh|yes|yeah|indeed|"
    r"alright|all right|sure|of course|absolutely|certainly|"
    r"got it|cool|nice|great|yep)\s*[.,!?]?\s*$",
    flags=re.IGNORECASE,
)

# Prompting self-explanation: tutor asks WHY / HOW student reasoned.
# Bare "why"/"how" + terse typed-chat prompts added 2026-07-19 (thought-experiment
# widening, unvalidated — see Kay_model_E2.md decision #6): typed tutoring reads
# as terse standalone prompts ("why?", "how?") rather than voice's full clauses.
_PROMPT_EXPLANATION = re.compile(
    r"\b(how did you (get|do|work out|arrive at|figure)|"
    r"why (did|do|is|are|does|would|have)\b|"
    r"can you explain|walk (me )?through|"
    r"explain (why|how|that|your (thinking|reasoning|answer|working))|"
    r"what made you|tell me (why|how)|"
    r"why is that\b|how do you know|"
    r"what'?s? your (reasoning|thinking)|"
    r"what were you thinking|"
    r"show (your )?working|show (your )?work|your turn)\b|"
    r"(why|how)\?\s*$",
    flags=re.IGNORECASE,
)

# Prompting next step: tutor asks student what comes next procedurally.
# Terse "next?"/"then?" added 2026-07-19 (thought-experiment widening,
# unvalidated) — anchored to end-of-utterance so bare "then" mid-sentence
# (a common TSL filler) doesn't false-positive.
_PROMPT_NEXT_STEP = re.compile(
    r"\b(what (is|are|do|would|will|'?s) (the |you |we |your )?(next( step)?|do next)|"
    r"what comes next|"
    r"so what (do we|should we|would you|are we) (do|doing)\b|"
    r"and then (what|where)\??|"
    r"what (should|do) (we|you) do (now|next)\b|"
    r"what'?s? (the )?next (step|thing))\b|"
    r"(next|then)\?\s*$",
    flags=re.IGNORECASE,
)

# Prompting self-correction: tutor points to an error and asks student to fix.
_PROMPT_SELF_CORRECTION = re.compile(
    r"\b(try (that )?again|have (another|a second|a closer) look|"
    r"check (your|that) (work|answer|calculation|solution)\b|"
    r"look at that again|are you sure( about that)?\??|"
    r"double.?check|go back (and|to)\b|"
    r"re.?check (that|it|your)\b|"
    r"does that (look|seem) right\??|"
    r"have (a look|a think) at|let'?s? (re.?check|look again))\b",
    flags=re.IGNORECASE,
)

# Prompting alternative representation: draw, sketch, write differently.
_PROMPT_ALT_REPR = re.compile(
    r"\b(draw (a |the )?(picture|diagram|sketch|it out?|that)\b|"
    r"show (me )?(another|a different) way|"
    r"write (it )?(as|out|differently|another way)\b|"
    r"represent (it|that)\b|sketch (it|that|a)\b|"
    r"can you (draw|sketch|show me))\b",
    flags=re.IGNORECASE,
)

# Giving a hint: partial information to guide without full explanation.
_GIVING_HINT = re.compile(
    r"\b(remember (that|when|how|what|the|to)\b|"
    r"think about\b|"
    r"what do you (know|remember) about|"
    r"here'?s? (a |one )?(clue|hint)|"
    r"think back (to|about)\b|don'?t forget\b|"
    r"have you (thought|considered) about|"
    r"what (else )?do you know (about)?\b|"
    r"clue[: ]|hint[: ])\b",
    flags=re.IGNORECASE,
)

# Giving an example: analogy or concrete illustration.
_GIVING_EXAMPLE = re.compile(
    r"\b(for (example|instance)\b|"
    r"think of it (like|as)\b|"
    r"imagine (if|you|that|a |an )|"
    r"let'?s? say\b|such as\b|"
    r"as an example\b|"
    r"picture (a |the |this |that )|"
    r"like (a |an )\w)\b",
    flags=re.IGNORECASE,
)

# Explaining procedurally: step-by-step sequencing language.
_EXPLAINING_PROCEDURAL = re.compile(
    r"\b(first[,. ]+(you|we|let'?s?|thing)\b|"
    r"step (one|two|three|four|five|1|2|3|4|5)( is)?[: ]|"
    r"(the )?first (step|thing) (is |to )|"
    r"next[, ]+(you|we)\b|"
    r"after (that|this)[, ]+(you|we)\b|"
    r"start (by|with) |begin (by|with) |"
    r"then[, ]+(you|we|divide|multiply|add|subtract|simplify|expand|factorise?)\b|"
    r"finally[, ]+(you|we)\b)\b",
    flags=re.IGNORECASE,
)

# Explaining conceptually: the "why" behind a mathematical idea (not step-by-step how-to).
# Distinguishes from _EXPLAINING_PROCEDURAL (which is the "how" / step sequence).
_EXPLAINING_CONCEPTUAL = re.compile(
    r"\b("
    r"the reason (is|why|we do|this works|that works)\b|"
    r"that'?s? why\b|"
    r"what (this|that) means (is )?\b|"
    r"the (key |underlying |core |main )?(idea|concept|principle) (is|here|behind)\b|"
    r"(this|that) works because\b|"
    r"(we|you) (use|need|do this) because\b|"
    r"mathematically[,. ]\b|"
    r"the rule is (that|because|since)\b|"
    r"this is (true|correct|right) because\b"
    r")\b",
    flags=re.IGNORECASE,
)

# Prior knowledge check: tutor probes what the student already knows.
# Maps to Sandpiper rubric dimension 3 (determining_what_students_know).
_PRIOR_KNOWLEDGE_CHECK = re.compile(
    r"\b("
    r"what do you (already )?(know|remember) (about|from)\b|"
    r"have you (seen|done|learned|come across|studied|covered|heard of) (this|that|these)\b|"
    r"do you remember (when|how|what|learning|doing)\b|"
    r"what can you tell me about\b|"
    r"what (have you|did you) (learn|do|cover) (about|in)\b|"
    r"(have|did) you (ever )?(see|do|learn|hear about) (this|that|these)\b|"
    # terse typed forms added 2026-07-19 (thought-experiment, unvalidated)
    r"remember (this|that)\b|seen this before\b|done this before\b"
    r")\b",
    flags=re.IGNORECASE,
)

# Giving the answer directly.
_GIVING_ANSWER = re.compile(
    r"\b(the (answer|solution|result|value) is\b|"
    r"it'?s? equals? \d|it'?s? equal to\b|"
    r"the final (answer|result)\b|"
    r"so (it'?s?|that'?s?|x|y|the answer) (is |=\s*|equals? )\d|"
    r"which (gives|equals?) \d|"
    r"that (gives|equals?) (us |you )?\d|"
    r"equals? \d+\.?\d*)\b",
    flags=re.IGNORECASE,
)
# Bare numeric/symbolic answer with NO surrounding phrase — the ENTIRE tutor
# utterance is just a number or a short "var = number" expression (e.g. "6",
# "x = 2", "= 12"). Added 2026-07-19 (thought-experiment, unvalidated): typed
# chat plausibly gives answers this tersely far more than voice does, where
# _GIVING_ANSWER's phrase-anchored forms ("the answer is...") dominate.
# fullmatch (like _BACKCHANNEL) so it only fires when that's the WHOLE turn,
# not a number appearing incidentally inside a longer sentence.
_GIVING_ANSWER_BARE = re.compile(
    r"^\s*[a-z]?\s*=?\s*[-+]?\d+\.?\d*\s*[.!]?\s*$",
    flags=re.IGNORECASE,
)

# Tag questions at end of tutor turn: "keeping everyone together" move.
_TAG_QUESTION = re.compile(
    r"(right\?|yes\?|okay\?|ok\?|"
    r"isn'?t (it|that)\?|doesn'?t (it|that)\?|"
    r"aren'?t (they|we)\?|don'?t (you|we)\?|"
    r"can'?t (you|we)\?|"
    r"do you (see|understand|follow|agree|get (it|that))\?|"
    r"(make|makes) sense\?|got it\?|"
    r"(does|do) that make sense\?)\s*$",
    flags=re.IGNORECASE,
)

# Student produces reasoning language: "because", "so that", "therefore"…
# cos/cus/coz + which gives/leaves added 2026-07-19 (thought-experiment
# widening, unvalidated, twin of self_explanation_rate's SELF_EXPL regex —
# see Kay_model_E2.md decision #6). Deliberately no bare "so" beyond the
# already-narrow "so (it|the|we|that|this)" form, for the same TSL-filler
# false-positive reason SELF_EXPL avoided it.
_STUDENT_EXPLAINS = re.compile(
    r"\b(because\b|since\b|so that\b|"
    r"that'?s? (why|because)\b|"
    r"the reason (is|why)\b|"
    r"which means\b|therefore\b|"
    r"cos\b|cus\b|coz\b|which gives\b|which leaves\b|giving us\b|"
    r"so (it|the|we|that|this)\b)\b",
    flags=re.IGNORECASE,
)


def _count_restating(df: pd.DataFrame) -> int:
    """Count tutor turns that restate the immediately preceding student turn.

    Uses content-word recall: if >60 % of the student's non-stop words reappear
    in the tutor's next turn AND the student said ≥3 content words, it's a
    restatement.  This catches both verbatim repeats and near-verbatim ones.
    """
    roles = df["role"].fillna("").tolist()
    contents = df["content"].fillna("").astype(str).tolist()
    count = 0
    for i in range(1, len(df)):
        if roles[i] != "tutor" or roles[i - 1] != "student":
            continue
        sw = {w for w in _WORD.findall(contents[i - 1].lower()) if w not in _STOPS}
        tw = {w for w in _WORD.findall(contents[i].lower()) if w not in _STOPS}
        if len(sw) < 3:
            continue
        if len(sw & tw) / len(sw) > 0.6:
            count += 1
    return count


# Stopwords for LO keyword extraction — common function words that won't
# appear meaningfully in a learning objective's topical vocabulary.
_STOPS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "whether", "if", "as", "while", "when",
    "where", "which", "who", "whom", "that", "this", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "what", "how",
})

LO_TRANSCRIPT_COLS = [
    "lo_keyword_overlap", "lo_keyword_in_student", "lo_keyword_in_tutor",
    "lo_student_tutor_coverage_ratio", "lo_first_mention_frac", "lo_mention_density",
]
SESSION_TASK_COLS = [
    "session_n_responses", "session_n_distinct_los",
    "session_lo_diversity", "session_lo_same",
]
LO_DIFFICULTY_COLS = [
    "lo_mean_correct", "lo_n_sessions", "lo_difficulty_rank",
    "lo_difficulty_vs_session_mean", "lo_is_hardest_in_session",
]


def _parse_ts(s: str) -> float:
    """Parse HH:MM:SS elapsed offset to seconds. Returns nan on failure."""
    try:
        parts = str(s).strip().split(":")
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        return float(h * 3600 + m * 60 + sec)
    except Exception:
        return float("nan")


def _lo_transcript_stats(tdf: pd.DataFrame, lo_text: str) -> dict:
    """Per-response features: how well does the LO topic appear in the transcript?"""
    zero = dict.fromkeys(LO_TRANSCRIPT_COLS, 0.0)
    if tdf.empty or not lo_text:
        return zero
    lo_words = {w.lower() for w in _WORD.findall(lo_text)
                if w.lower() not in _STOPS and len(w) > 2}
    if not lo_words:
        return zero

    role    = tdf["role"].fillna("")
    content = tdf["content"].fillna("").astype(str)
    sm = role.eq("student")
    tm = role.eq("tutor")

    all_words  = {w.lower() for w in _WORD.findall(" ".join(content))}
    st_words   = {w.lower() for w in _WORD.findall(" ".join(content[sm]))}
    tut_words  = {w.lower() for w in _WORD.findall(" ".join(content[tm]))}

    n             = len(lo_words)
    lo_in_all     = len(lo_words & all_words) / n
    lo_in_student = len(lo_words & st_words)  / n
    lo_in_tutor   = len(lo_words & tut_words) / n
    lo_ratio      = lo_in_student / (lo_in_tutor + 1e-9)

    # Which utterance index first mentions an LO keyword?
    first = next(
        (i for i, c in enumerate(content)
         if lo_words & {w.lower() for w in _WORD.findall(c)}),
        None,
    )
    lo_first_frac = (first / len(tdf)) if first is not None else 1.0

    n_mentioning = sum(1 for c in content
                       if lo_words & {w.lower() for w in _WORD.findall(c)})
    # per-utterance, NOT per-minute: timestamps are TSL-only (elapsed-offset
    # convention absent/unreliable off-source), so a duration-based rate would
    # blow up (or need NaN-ablation) on typed transcripts. Utterance count is
    # available identically in both regimes -- source-invariant by construction.
    lo_mention_density = n_mentioning / max(len(content), 1)
    return {
        "lo_keyword_overlap":              lo_in_all,
        "lo_keyword_in_student":           lo_in_student,
        "lo_keyword_in_tutor":             lo_in_tutor,
        "lo_student_tutor_coverage_ratio": lo_ratio,
        "lo_first_mention_frac":           lo_first_frac,
        "lo_mention_density":              lo_mention_density,
    }


def compute_session_task_features(meta: pd.DataFrame) -> pd.DataFrame:
    """Pure-metadata features: how many / how varied are the LOs per session.

    No transcript reading required; call before or after building transcripts.
    Returns a DataFrame indexed by session_id.
    """
    sess = (
        meta.groupby("session_id")
        .agg(
            session_n_responses   =("response_id",          "count"),
            session_n_distinct_los=("learning_objective_id", "nunique"),
        )
        .reset_index()
    )
    sess["session_lo_diversity"] = (sess["session_n_distinct_los"]
                                    / sess["session_n_responses"])
    sess["session_lo_same"] = (sess["session_n_distinct_los"] == 1).astype(int)
    return sess.set_index("session_id")


def compute_lo_stats(meta: pd.DataFrame, labels) -> pd.DataFrame:
    """Per-LO difficulty stats from training labels.

    ``labels`` must be aligned with ``meta`` (same row order or a Series
    with matching response_id index).  Call this on training data only —
    or out-of-fold subsets for proper CV.
    Returns a DataFrame indexed by learning_objective_id.
    """
    joined = meta[["response_id", "learning_objective_id"]].copy()
    joined["is_correct"] = (
        labels.values if hasattr(labels, "values") else np.asarray(labels)
    )
    stats = (
        joined.groupby("learning_objective_id")["is_correct"]
        .agg(lo_mean_correct="mean", lo_n_sessions="count")
        .reset_index()
    )
    stats["lo_difficulty_rank"] = stats["lo_mean_correct"].rank(pct=True)
    return stats.set_index("learning_objective_id")


def add_lo_difficulty_features(
    X: pd.DataFrame, meta: pd.DataFrame, lo_stats: pd.DataFrame
) -> pd.DataFrame:
    """Join per-LO difficulty features onto a response-indexed design matrix.

    ``X`` must be indexed by response_id.
    ``meta`` must contain response_id, learning_objective_id, session_id.
    ``lo_stats`` is the output of :func:`compute_lo_stats`.
    """
    m = meta.copy()
    if m.index.name != "response_id":
        m = m.set_index("response_id")
    m = m[["learning_objective_id", "session_id"]].loc[X.index]
    m = m.join(
        lo_stats[["lo_mean_correct", "lo_n_sessions", "lo_difficulty_rank"]],
        on="learning_objective_id", how="left",
    )
    global_mean = float(lo_stats["lo_mean_correct"].mean())
    m["lo_mean_correct"].fillna(global_mean, inplace=True)
    m["lo_n_sessions"].fillna(0, inplace=True)
    m["lo_difficulty_rank"].fillna(0.5, inplace=True)

    sess_mean = m.groupby("session_id")["lo_mean_correct"].transform("mean")
    m["lo_difficulty_vs_session_mean"] = m["lo_mean_correct"] - sess_mean
    sess_min  = m.groupby("session_id")["lo_mean_correct"].transform("min")
    m["lo_is_hardest_in_session"] = (m["lo_mean_correct"] == sess_min).astype(int)

    result = X.copy()
    for col in LO_DIFFICULTY_COLS:
        result[col] = m[col].values
    return result


def transcript_to_text(df: pd.DataFrame, roles=("student", "tutor")) -> str:
    """Concatenate utterances as ``role: content`` lines, keeping order."""
    if df.empty:
        return ""
    keep = df[df["role"].isin(roles)]
    content = keep["content"].fillna("").astype(str)
    return "\n".join(f"{r}: {c}" for r, c in zip(keep["role"], content))


def transcript_stats(df: pd.DataFrame) -> dict:
    """Structured features from one transcript — cheap, no models required."""
    _ZERO: dict = {
        # --- original features ---
        "n_utterances": 0, "n_student": 0, "n_tutor": 0, "n_background": 0,
        "student_words": 0, "tutor_words": 0, "student_tutor_word_ratio": 0.0,
        "total_chars": 0, "n_question_marks": 0, "n_unclear": 0,
        # --- pacing / timing ---
        "session_duration_secs": 0.0, "avg_pause_secs": 0.0, "n_long_pauses": 0,
        "avg_student_pause_secs": 0.0,
        # --- student turn quality ---
        "student_words_per_turn": 0.0, "student_words_var": 0.0,
        "student_max_turn_words": 0, "student_short_turns": 0,
        "student_short_turn_ratio": 0.0, "student_ttr": 0.0,
        "student_questions": 0,
        # --- tutor behaviour ---
        "tutor_questions": 0, "tutor_praise": 0, "tutor_words_per_turn": 0.0,
        # --- unclear signals ---
        "student_unclear_only": 0, "unclear_per_student_turn": 0.0,
        # --- session type / provenance ---
        "is_tsl": 0, "has_tutor_ph": 0,
        # --- name-call features ---
        "tutor_name_calls": 0, "tutor_name_calls_per_turn": 0.0,
        "has_name_call": 0,
        "tutor_ph_count": 0,
        "student_says_tutor_name": 0, "tutor_says_tutor_name": 0,
        # --- late-session (last 20 %) ---
        "late_student_words": 0, "late_student_questions": 0,
        "late_tutor_questions": 0, "late_student_short_ratio": 0.0,
        # --- Sandpiper tutor feedback ---
        "tutor_feedback_incorrect": 0, "tutor_backchannels": 0,
        "feedback_positive_ratio": 0.0,
        # --- Sandpiper tutor prompting (eliciting moves) ---
        "tutor_prompts_explanation": 0, "tutor_prompts_next_step": 0,
        "tutor_prompts_self_correction": 0, "tutor_prompts_alt_repr": 0,
        "tutor_tag_questions": 0,
        # --- Sandpiper tutor teaching (telling/scaffolding moves) ---
        "tutor_gives_hint": 0, "tutor_gives_example": 0,
        "tutor_explains_procedural": 0, "tutor_explains_conceptual": 0,
        "tutor_gives_answer": 0,
        "tutor_restating": 0,
        # --- Sandpiper tutor knowledge probing ---
        "tutor_prior_knowledge_check": 0,
        # --- Sandpiper student ---
        "student_explains": 0,
        # --- derived ratios ---
        "elicit_vs_tell_ratio": 0.0, "scaffold_vs_tell_ratio": 0.0,
        "error_response_quality": 0.0,
    }
    if df.empty:
        return _ZERO

    role    = df["role"].fillna("")
    content = df["content"].fillna("").astype(str)
    words   = content.map(lambda s: len(_WORD.findall(s)))

    student_mask = role.eq("student")
    tutor_mask   = role.eq("tutor")
    bg_mask      = role.eq("background")

    student_words = int(words[student_mask].sum())
    tutor_words   = int(words[tutor_mask].sum())

    # ---- pacing / timing ------------------------------------------------
    ts = df["timestamp"].map(_parse_ts)
    ts_valid = ts.dropna()
    if len(ts_valid) >= 2:
        session_duration = float(ts_valid.iloc[-1] - ts_valid.iloc[0])
        gaps             = ts_valid.diff().dropna()
        avg_pause        = float(gaps.mean())
        n_long_pauses    = int((gaps > 30).sum())
    else:
        session_duration = avg_pause = 0.0
        n_long_pauses    = 0

    student_ts = ts[student_mask].dropna()
    avg_student_pause = float(student_ts.diff().dropna().mean()) if len(student_ts) >= 2 else 0.0

    # ---- student turn quality -------------------------------------------
    sw = words[student_mask]
    n_student = int(student_mask.sum())
    student_words_per_turn = float(sw.mean())   if n_student else 0.0
    student_words_var      = float(sw.var())    if n_student else 0.0
    student_max_turn       = int(sw.max())      if n_student else 0
    student_short          = int((sw <= 3).sum())
    student_short_ratio    = student_short / n_student if n_student else 0.0

    # type-token ratio on all student words combined
    all_student_text = " ".join(content[student_mask])
    all_student_words = _WORD.findall(all_student_text.lower())
    student_ttr = (len(set(all_student_words)) / len(all_student_words)
                   if all_student_words else 0.0)

    student_questions = int(content[student_mask].str.count(r"\?").sum())

    # ---- tutor behaviour ------------------------------------------------
    tutor_questions  = int(content[tutor_mask].str.count(r"\?").sum())
    tutor_praise     = int(content[tutor_mask].map(
                           lambda s: bool(_PRAISE.search(s))).sum())
    tutor_words_per_turn = float(words[tutor_mask].mean()) if tutor_mask.any() else 0.0

    # ---- [unclear] signals ----------------------------------------------
    unclear_in_content = content.map(lambda s: len(_UNCLEAR.findall(s)))
    student_unclear_only = int(
        (student_mask & content.str.fullmatch(r"\s*\[unclear\]\s*", case=False, na=False)).sum()
    )
    unclear_per_student = (
        float(unclear_in_content[student_mask].sum()) / n_student if n_student else 0.0
    )

    # ---- session type / provenance --------------------------------------
    is_tsl     = int(bg_mask.any())
    has_tutor_ph = int(content.map(lambda s: bool(_TUTOR_NAME_PH.search(s))).any())

    # ---- name-call features -------------------------------------------------
    def _count_name_calls(text: str) -> int:
        return sum(
            1 for m in _NAME_CALL_RE.finditer(text)
            if m.group(1).lower() not in _STOP_WORDS
        )
    tutor_name_calls = int(content[tutor_mask].map(_count_name_calls).sum())
    n_tutor_turns = int(tutor_mask.sum())
    tutor_name_calls_per_turn = tutor_name_calls / n_tutor_turns if n_tutor_turns else 0.0
    has_name_call = int(tutor_name_calls > 0)

    # [TUTOR_NAME] token: how often, and who says it
    ph_counts = content.map(lambda s: len(_TUTOR_NAME_PH.findall(s)))
    tutor_ph_count          = int(ph_counts.sum())
    student_says_tutor_name = int(ph_counts[student_mask].sum() > 0)
    tutor_says_tutor_name   = int(ph_counts[tutor_mask].sum() > 0)

    # ---- late-session features (last 20 % of utterances) ---------------
    cutoff = max(1, int(len(df) * 0.8))
    late   = df.iloc[cutoff:]
    late_role    = late["role"].fillna("")
    late_content = late["content"].fillna("").astype(str)
    # The two `.astype(int)` calls are load-bearing, not tidying. `cutoff = max(1, ...)` makes
    # `late` EMPTY for any session of <=1 utterance, and on an empty frame both `.map()` and
    # `.str.count()` return OBJECT-dtype empties whose `.sum()` is `''`, not `0` — so the
    # `int(...)` below raised `invalid literal for int() with base 10: ''`. No delivered TtA
    # transcript is that short (0 of 22,821), which is why this never fired; it surfaced on
    # register-transformed input where merging collapsed 2 sessions to a single turn. It would
    # equally fire at INFERENCE on a 1-utterance hidden-test session, so the fix belongs here
    # rather than in the caller.
    late_words   = late_content.map(lambda s: len(_WORD.findall(s))).astype(int)
    late_qmarks  = late_content.str.count(r"\?").astype(int)
    late_sm      = late_role.eq("student")
    late_tm      = late_role.eq("tutor")
    late_sw      = late_words[late_sm]
    late_student_words     = int(late_sw.sum())
    late_student_questions = int(late_qmarks[late_sm].sum())
    late_tutor_questions   = int(late_qmarks[late_tm].sum())
    late_n_student         = int(late_sm.sum())
    late_student_short_ratio = (
        float((late_sw <= 3).sum()) / late_n_student if late_n_student else 0.0
    )

    # ---- Sandpiper-codebook features ----------------------------------------
    # Use Python sum() with a generator — safe on empty masks (pandas .sum() on
    # an empty object-dtype Series returns '' instead of 0).
    tc = content[tutor_mask]
    sc = content[student_mask]
    tutor_feedback_incorrect    = sum(bool(_FEEDBACK_INCORRECT.search(s))       for s in tc)
    tutor_backchannels          = sum(bool(_BACKCHANNEL.fullmatch(s))           for s in tc)
    tutor_prompts_explanation   = sum(bool(_PROMPT_EXPLANATION.search(s))       for s in tc)
    tutor_prompts_next_step     = sum(bool(_PROMPT_NEXT_STEP.search(s))         for s in tc)
    tutor_prompts_self_correction = sum(bool(_PROMPT_SELF_CORRECTION.search(s)) for s in tc)
    tutor_prompts_alt_repr      = sum(bool(_PROMPT_ALT_REPR.search(s))          for s in tc)
    tutor_tag_questions         = sum(bool(_TAG_QUESTION.search(s))             for s in tc)
    tutor_gives_hint            = sum(bool(_GIVING_HINT.search(s))              for s in tc)
    tutor_gives_example         = sum(bool(_GIVING_EXAMPLE.search(s))           for s in tc)
    tutor_explains_procedural   = sum(bool(_EXPLAINING_PROCEDURAL.search(s))    for s in tc)
    tutor_explains_conceptual   = sum(bool(_EXPLAINING_CONCEPTUAL.search(s))    for s in tc)
    tutor_gives_answer          = sum(bool(_GIVING_ANSWER.search(s) or _GIVING_ANSWER_BARE.fullmatch(s))
                                       for s in tc)
    tutor_prior_knowledge_check = sum(bool(_PRIOR_KNOWLEDGE_CHECK.search(s))    for s in tc)
    tutor_restating             = _count_restating(df)
    student_explains            = sum(bool(_STUDENT_EXPLAINS.search(s))         for s in sc)

    # Derived ratios — compress correlated counts into interpretable signals.
    tutor_prompts_total = (
        tutor_prompts_explanation + tutor_prompts_next_step
        + tutor_prompts_self_correction + tutor_prompts_alt_repr
    )
    tutor_tells_total = tutor_gives_answer + tutor_explains_procedural
    elicit_vs_tell_ratio = tutor_prompts_total / (tutor_tells_total + 1)
    scaffold_vs_tell_ratio = (tutor_gives_hint + tutor_gives_example) / (tutor_gives_answer + 1)
    feedback_positive_ratio = tutor_praise / (tutor_praise + tutor_feedback_incorrect + 1e-9)
    # When the tutor flags an error, do they prompt self-correction rather than give the answer?
    error_response_quality = tutor_prompts_self_correction / (tutor_feedback_incorrect + 1)

    return {
        # original
        "n_utterances":            int(len(df)),
        "n_student":               n_student,
        "n_tutor":                 int(tutor_mask.sum()),
        "n_background":            int(bg_mask.sum()),
        "student_words":           student_words,
        "tutor_words":             tutor_words,
        "student_tutor_word_ratio": student_words / (tutor_words + 1),
        "total_chars":             int(content.map(len).sum()),
        "n_question_marks":        int(content.str.count(r"\?").sum()),
        "n_unclear":               int(unclear_in_content.sum()),
        # pacing
        "session_duration_secs":   session_duration,
        "avg_pause_secs":          avg_pause,
        "n_long_pauses":           n_long_pauses,
        "avg_student_pause_secs":  avg_student_pause,
        # student quality
        "student_words_per_turn":  student_words_per_turn,
        "student_words_var":       student_words_var,
        "student_max_turn_words":  student_max_turn,
        "student_short_turns":     student_short,
        "student_short_turn_ratio": student_short_ratio,
        "student_ttr":             student_ttr,
        "student_questions":       student_questions,
        # tutor behaviour
        "tutor_questions":         tutor_questions,
        "tutor_praise":            tutor_praise,
        "tutor_words_per_turn":    tutor_words_per_turn,
        # unclear
        "student_unclear_only":    student_unclear_only,
        "unclear_per_student_turn": unclear_per_student,
        # provenance
        "is_tsl":                  is_tsl,
        "has_tutor_ph":            has_tutor_ph,
        # name-call
        "tutor_name_calls":        tutor_name_calls,
        "tutor_name_calls_per_turn": tutor_name_calls_per_turn,
        "has_name_call":           has_name_call,
        "tutor_ph_count":          tutor_ph_count,
        "student_says_tutor_name": student_says_tutor_name,
        "tutor_says_tutor_name":   tutor_says_tutor_name,
        # late-session
        "late_student_words":      late_student_words,
        "late_student_questions":  late_student_questions,
        "late_tutor_questions":    late_tutor_questions,
        "late_student_short_ratio": late_student_short_ratio,
        # Sandpiper tutor feedback
        "tutor_feedback_incorrect":  tutor_feedback_incorrect,
        "tutor_backchannels":        tutor_backchannels,
        "feedback_positive_ratio":   feedback_positive_ratio,
        # Sandpiper tutor prompting (eliciting)
        "tutor_prompts_explanation":    tutor_prompts_explanation,
        "tutor_prompts_next_step":      tutor_prompts_next_step,
        "tutor_prompts_self_correction": tutor_prompts_self_correction,
        "tutor_prompts_alt_repr":       tutor_prompts_alt_repr,
        "tutor_tag_questions":          tutor_tag_questions,
        # Sandpiper tutor teaching (telling/scaffolding)
        "tutor_gives_hint":            tutor_gives_hint,
        "tutor_gives_example":         tutor_gives_example,
        "tutor_explains_procedural":   tutor_explains_procedural,
        "tutor_explains_conceptual":   tutor_explains_conceptual,
        "tutor_gives_answer":          tutor_gives_answer,
        "tutor_restating":             tutor_restating,
        # Sandpiper knowledge probing
        "tutor_prior_knowledge_check": tutor_prior_knowledge_check,
        # Sandpiper student
        "student_explains":            student_explains,
        # derived ratios
        "elicit_vs_tell_ratio":      elicit_vs_tell_ratio,
        "scaffold_vs_tell_ratio":    scaffold_vs_tell_ratio,
        "error_response_quality":    error_response_quality,
    }


def build_session_features(session_ids, store=None, show_progress: bool = True) -> pd.DataFrame:
    """For unique session_ids, return a frame indexed by session_id with
    ``transcript_text`` plus structured stat columns.

    ``store`` is a :class:`magnificat.data.TranscriptStore`; defaults to the
    training-zip store. Pass a directory-backed store at inference time.
    """
    store = store or data.default_store()
    uniq = list(dict.fromkeys(session_ids))
    it = store.iter(uniq)
    if show_progress:
        it = tqdm(it, total=len(uniq), desc="transcripts")
    rows = []
    for sid, tdf in it:
        row = {"session_id": sid, "transcript_text": transcript_to_text(tdf)}
        row.update(transcript_stats(tdf))
        rows.append(row)
    return pd.DataFrame(rows).set_index("session_id")


def build_design_matrix(
    meta: pd.DataFrame,
    store=None,
    lo_stats: pd.DataFrame | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build the full design matrix in a single transcript pass.

    Computes per-session transcript stats, per-response LO-transcript alignment
    features, and session task-diversity features.  If ``lo_stats`` is supplied
    (output of :func:`compute_lo_stats` on training labels) the per-LO
    difficulty columns are added; otherwise they are left as zeros.

    Returns a frame indexed by response_id.
    """
    store = store or data.default_store()
    uniq  = list(dict.fromkeys(meta["session_id"]))

    # Build mapping session_id → [(response_id, lo_text), ...]
    session_los: dict = {}
    for _, row in meta[["response_id", "session_id", "learning_objective"]].iterrows():
        session_los.setdefault(row["session_id"], []).append(
            (row["response_id"], row["learning_objective"] or "")
        )

    sess_rows: list[dict] = []
    lo_rows:   list[dict] = []

    it = store.iter(uniq)
    if show_progress:
        it = tqdm(it, total=len(uniq), desc="transcripts")

    for sid, tdf in it:
        sess_row = {"session_id": sid, "transcript_text": transcript_to_text(tdf)}
        sess_row.update(transcript_stats(tdf))
        sess_rows.append(sess_row)

        for rid, lo_text in session_los.get(sid, []):
            lo_row = {"response_id": rid}
            lo_row.update(_lo_transcript_stats(tdf, lo_text))
            lo_rows.append(lo_row)

    sess_df = pd.DataFrame(sess_rows).set_index("session_id")
    lo_df   = pd.DataFrame(lo_rows).set_index("response_id")
    task_df = compute_session_task_features(meta)

    out = meta[["response_id", "session_id", "learning_objective_id",
                "learning_objective"]].copy()
    out = out.merge(sess_df,            on="session_id",  how="left")
    out = out.merge(lo_df,              on="response_id", how="left")
    out = out.merge(task_df, left_on="session_id", right_index=True, how="left")

    out["transcript_text"] = (
        "OBJECTIVE: " + out["learning_objective"].fillna("") + "\n" +
        out["transcript_text"].fillna("")
    )

    _skip = {"transcript_text", "response_id", "session_id",
             "learning_objective_id", "learning_objective"}
    num_cols = [c for c in out.columns if c not in _skip]
    out[num_cols] = out[num_cols].fillna(0)
    out = out.set_index("response_id")

    # LO difficulty (zeros unless lo_stats provided)
    for col in LO_DIFFICULTY_COLS:
        out[col] = 0.0
    if lo_stats is not None:
        out = add_lo_difficulty_features(out, meta, lo_stats)

    return out


def build_meta_matrix(meta: pd.DataFrame) -> pd.DataFrame:
    """Metadata-only design matrix (NO transcripts): keyed by response_id with
    ``learning_objective`` (text) and ``learning_objective_id`` (category).

    Used by the transcript-free baseline; reads no zip, so it's instant.
    """
    out = meta[["response_id", "learning_objective_id", "learning_objective"]].copy()
    out["learning_objective"] = out["learning_objective"].fillna("")
    out["learning_objective_id"] = out["learning_objective_id"].fillna("UNK")
    return out.set_index("response_id")


# ---------------------------------------------------------------------------
# LO curriculum metadata
# ---------------------------------------------------------------------------
import re as _re

def _kw_match(lo_lower: str, keywords: list) -> bool:
    """Check whether any keyword matches lo_lower at a word boundary.

    Using \\b at the *start* of the pattern (only) lets partial stems like
    "multipl" match "multiply" / "multiplication" while preventing "ratio"
    from matching "ope**ratio**ns" or "tally" from matching "men**tally**".
    Multi-word phrases (e.g. "bar chart") are matched verbatim after the
    leading boundary.
    """
    for kw in keywords:
        if _re.search(r"\b" + _re.escape(kw), lo_lower):
            return True
    return False


# Priority order matters: first match wins.  Ratio before Number because
# "proportion" would also match Number-level terms; Geometry before Number
# because "area of" / "perimeter" are unambiguously geometric.
_STRAND_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Ratio",      [
        "ratio", "proportion", "proportional", "scale factor",
        "direct proportion", "inverse proportion", "sharing in ratio",
        "capture.recapture", "capture-recapture",
    ]),
    ("Statistics", [
        "probability", "probabilities", "mean", "median",
        # "mode" omitted — matches "model" as a substring
        "frequency", "scatter", "survey", "histogram", "pictogram",
        "venn", "bar chart", "pie chart", "chart", "cumulative frequency",
        "box plot", "averages", "average speed",
    ]),
    ("Algebra",    [
        "equation", "expression", "formulae", "formula", "substitut",
        "factoris", "factoriz", "quadratic", "simultaneous", "inequalit",
        "completing the square", "like terms", "symbols and letters",
        "function notation", "function machine", "arithmetic sequence",
        "double bracket", "triple bracket", "single bracket",
        # "expand.*bracket" removed — re.escape breaks regex patterns;
        # "double/triple/single bracket" already cover Algebra expand LOs
        "unknown on both sides", "two unknowns",
        "change the subject", "rearranging formula",
        "nth term", "sequence",
        "y=mx", "y=x", "y=-x",     # literals for y=mx+c / y=x+a forms
        "linear equation", "linear graph",
        "index law",
    ]),
    ("Geometry",   [
        "pythagor", "trigonometr", "sohcahtoa", "vector",
        "angle", "triangle", "circle", "quadrilateral", "polygon",
        "perimeter", "volume", "surface area", "coordinat", "symmetr",
        "transform", "reflect", "rotat", "translat",
        "parallel", "perpendicular", "congruent", "similar shape",
        "bearing", "locus", "sector", "arc", "prism", "cylinder",
        "sphere", "cuboid", "net", "tessellat",
        "vertex", "vertices", "edge", "face",
        "2d shape", "3d shape",
        "area of", "area and", "calculating area", "solving area",
        "finding area", "area,",
        "shape",
    ]),
    ("Number",     [
        "fraction", "decimal", "place value", "digit", "factor",
        "multiple", "prime", "count", "add", "subtract", "multipl",
        "divid", "percent", "money", "round", "negative", "integer",
        "times table", "table", "number", "measur", "mass", "weight",
        "capacity", "length", "speed", "calculat", "mental",
        "estimat", "approximat", "power", "square root",
        "cube root", "cube number", "standard form", "surd",
        "scale", "bar model",
        "hundred", "tens and ones", "tens to", "tens",
        "hcf", "lcm", "highest common factor", "lowest common multiple",
        # time
        "telling the time", "units of time", "time to",
        "quarter past", "quarter to", "half past",
        # common primaries
        "difference", "remainder", "inverse", "operation",
        "known fact", "halving", "doubling", "making a whole",
        "recognis", "coin", "sharing", "grouping",
        "upper bound", "lower bound",
        "multi-step",
        "area model",               # multiplication pedagogy, not geometry
    ]),
]

_STRAND_INT = {"Number": 0, "Algebra": 1, "Geometry": 2,
               "Statistics": 3, "Ratio": 4, "Other": 5}

# Bloom's revised taxonomy — \\b at start only (same reason as strands).
# Order: higher cognitive levels first so they win over lower ones.
_BLOOM_VERBS: list[tuple[int, str, list[str]]] = [
    (4, "Analyse",    ["analys", "compar", "distinguish", "examin",
                       "investigat", "differenti", "classify",
                       "categoris", "interpret", "identify pattern"]),
    (3, "Apply",      ["calculat", "solv", "apply", "comput", "construct",
                       "demonstrat", "convert", "determin", "estimat",
                       "find", "measur", "produc", "record", "show",
                       "work out", "work with", "expand", "simplif",
                       "substitut", "factoris", "rearrang", "forming",
                       "generating", "performing", "drawing", "writing",
                       "telling"]),
    (2, "Understand", ["explain", "describ", "recognis", "understand",
                       "illustrat", "paraphras", "summar", "represent",
                       "reason", "justif", "seeing", "developing fluency"]),
    (1, "Remember",   ["recall", "state", "name", "list", "defin",
                       "memoris", "know", "label", "match", "select",
                       "identif"]),
]


def lo_topic_strand_rule(lo_text: str) -> int:
    """Keyword-based topic strand (0=Number…5=Other). No LLM required."""
    lo_lower = lo_text.lower()
    for strand, keywords in _STRAND_KEYWORDS:
        if _kw_match(lo_lower, keywords):
            return _STRAND_INT[strand]
    return _STRAND_INT["Other"]


def lo_bloom_level_rule(lo_text: str) -> int:
    """Action-verb Bloom's level (1=Remember…4=Analyse). No LLM required.
    Returns 3 (Apply) as default — most maths LOs are procedural."""
    lo_lower = lo_text.lower()
    for level, _name, verbs in _BLOOM_VERBS:
        if _kw_match(lo_lower, verbs):
            return level
    return 3  # Apply is the modal level for UK maths LOs


# Features produced by add_lo_curriculum_features().
# When lo_curriculum.json exists (LLM run), all four are populated.
# With the rule-based fallback, lo_year_group and lo_ks_level are 0.
LO_CURRICULUM_COLS = [
    "lo_year_group",    # soft top-5 weighted avg year (float); 0 = unknown
    "lo_ks_level",      # 1–4 (0 = unknown)
    "lo_topic_strand",  # 0=Number 1=Algebra 2=Geometry 3=Stats 4=Ratio 5=Other
    "lo_bloom_level",   # 1=Remember 2=Understand 3=Apply 4=Analyse
    "lo_nc_sim",        # max TF-IDF cosine sim to any NC statement; 0 if no match file
]


def load_lo_curriculum() -> "dict | None":
    """Load artifacts/lo_curriculum.json produced by annotate_lo_curriculum.py.

    Returns None if the file doesn't exist yet.  Callers fall back to the
    rule-based classifiers supplemented by NC matching.
    """
    from . import config as _cfg
    import json
    path = _cfg.ARTIFACTS_DIR / "lo_curriculum.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_nc_curriculum_df() -> "pd.DataFrame | None":
    """Load nc_curriculum.csv — checks assets/ first (submission), then artifacts/.

    This is the only file that needs to be bundled in assets/ for submission.
    Returns None if the file is not found in either location.
    """
    from . import config as _cfg
    from pathlib import Path
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "assets" / "nc_curriculum.csv",
        _cfg.ARTIFACTS_DIR / "nc_curriculum.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


def _nc_match_live(
    lo_ids: list,
    lo_texts: list,
    nc_df: pd.DataFrame,
    top_k: int = 5,
) -> "dict":
    """Compute NC match features live from LO texts. Submission-safe.

    Fits word (1,2)-gram TF-IDF on the NC statements ONLY (a fixed external
    reference table, never on LO text — see rule 3 in the project's rules: "process test
    samples independently, no cross-test pooling"). Each LO is then
    ``.transform()``-ed through that frozen vectorizer independently, so one
    response's feature value can never depend on which other LOs happen to be
    in the same batch/test set.

    Returns {lo_id: (year_soft, ks_level, cosine_sim)}.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos

    _KS = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4}

    nc_stmts = list(nc_df["statement"])
    nc_years  = nc_df["year_group"].to_numpy(float)
    lo_texts_clean = [t or "" for t in lo_texts]

    # Fit ONLY on the fixed NC reference statements — never on LO text — so
    # the vocabulary/IDF weights are fully static and each LO transforms
    # independently of every other test sample.
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                          strip_accents="unicode", min_df=1)
    vec.fit(nc_stmts)
    nc_vecs = vec.transform(nc_stmts)
    lo_vecs = vec.transform(lo_texts_clean)

    sims       = _cos(lo_vecs, nc_vecs)
    best_score = sims.max(axis=1)

    k = min(top_k, sims.shape[1])
    top_idx   = np.argsort(sims, axis=1)[:, -k:]
    top_sims  = sims[np.arange(len(lo_texts))[:, None], top_idx]
    top_years = nc_years[top_idx]
    sim_sum   = top_sims.sum(axis=1, keepdims=True).clip(min=1e-9)
    year_soft = (top_sims * top_years).sum(axis=1) / sim_sum.squeeze()

    result = {}
    for i, lo_id in enumerate(lo_ids):
        yg = float(year_soft[i])
        ks = _KS.get(round(yg), 0)
        result[lo_id] = (yg, ks, float(best_score[i]))
    return result


def add_lo_curriculum_features(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    curriculum: "dict | None",
    nc_df: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Join LO curriculum metadata onto a response-indexed design matrix.

    Tier 1 (best):  ``curriculum`` from lo_curriculum.json (LLM-annotated).
    Tier 2:         NC matching computed live from ``nc_df`` (submission-safe).
    Tier 3 (bare):  rule-based strand/bloom only; year_group = 0, nc_sim = 0.

    ``nc_df`` — output of load_nc_curriculum_df().  Pass None to get tier-3 only.
    The nc_curriculum.csv must be bundled in assets/ for the submission container.
    """
    _SINT = {"Number": 0, "Algebra": 1, "Geometry": 2,
             "Statistics": 3, "Ratio": 4, "Other": 5}

    m = meta.copy()
    if m.index.name != "response_id":
        m = m.set_index("response_id")
    m = m[["learning_objective_id", "learning_objective"]].loc[X.index]

    # Compute NC match once for all unique LOs (not per-row) for efficiency
    nc_lookup: dict = {}
    if nc_df is not None:
        unique_los = m.drop_duplicates("learning_objective_id")
        nc_lookup = _nc_match_live(
            list(unique_los["learning_objective_id"]),
            list(unique_los["learning_objective"]),
            nc_df,
        )

    rows = []
    for lo_id, lo_text in zip(m["learning_objective_id"], m["learning_objective"]):
        lo_text   = lo_text or ""
        nc_entry  = nc_lookup.get(lo_id)          # (year_soft, ks, sim) or None
        nc_sim    = float(nc_entry[2]) if nc_entry else 0.0

        if curriculum is not None and lo_id in curriculum:
            # Tier 1: full LLM annotation
            c = curriculum[lo_id]
            rows.append({
                "lo_year_group":   int(c.get("year_group", 0)),
                "lo_ks_level":     int(c.get("ks_level", 0)),
                "lo_topic_strand": _SINT.get(c.get("topic_strand", "Other"), 5),
                "lo_bloom_level":  int(c.get("bloom_level", 3)),
                "lo_nc_sim":       nc_sim,
            })
        elif nc_entry:
            # Tier 2: live NC match
            yg, ks, _ = nc_entry
            rows.append({
                "lo_year_group":   yg,
                "lo_ks_level":     ks,
                "lo_topic_strand": lo_topic_strand_rule(lo_text),
                "lo_bloom_level":  lo_bloom_level_rule(lo_text),
                "lo_nc_sim":       nc_sim,
            })
        else:
            # Tier 3: pure rule-based
            rows.append({
                "lo_year_group":   0.0,
                "lo_ks_level":     0,
                "lo_topic_strand": lo_topic_strand_rule(lo_text),
                "lo_bloom_level":  lo_bloom_level_rule(lo_text),
                "lo_nc_sim":       0.0,
            })

    cur_df = pd.DataFrame(rows, index=X.index)
    result = X.copy()
    for col in LO_CURRICULUM_COLS:
        result[col] = cur_df[col].values
    return result


SANDPIPER_COLS = [
    "tutor_feedback_incorrect", "tutor_backchannels", "feedback_positive_ratio",
    "tutor_prompts_explanation", "tutor_prompts_next_step",
    "tutor_prompts_self_correction", "tutor_prompts_alt_repr", "tutor_tag_questions",
    "tutor_gives_hint", "tutor_gives_example",
    "tutor_explains_procedural", "tutor_explains_conceptual",
    "tutor_gives_answer", "tutor_restating",
    "tutor_prior_knowledge_check",
    "student_explains",
    "elicit_vs_tell_ratio", "scaffold_vs_tell_ratio", "error_response_quality",
]

# LLM-annotated Sandpiper features (produced by scripts/sandpiper/annotate_sandpiper_train.py
# + scripts/sandpiper/build_sandpiper_features.py).  Present only when the corresponding
# parquet files exist under artifacts/; zeros otherwise.
_LEARNING_SUPPORT_CODES = [
    "PROMPTING_RELATED_CONCEPTS", "PROMPTING_ALTERNATIVE_REPRESENTATION",
    "PROMPTING_SELF_EXPLANATION", "PROMPTING_NEXT_STEP", "PROMPTING_SELF_CORRECTION",
    "FEEDBACK_CORRECT", "FEEDBACK_INCORRECT", "FEEDBACK_NEUTRAL",
    "REVOICING", "RESTATING", "GIVING_HINT", "GIVING_EXAMPLE",
    "EXPLAINING_CONCEPTUAL", "EXPLAINING_PROCEDURAL", "GIVING_ANSWER",
]
_STUDENT_CODES = [
    "DEMONSTRATING_UNDERSTANDING", "CONFUSION_EXPRESSED",
    "SEEKING_CLARIFICATION", "SELF_CORRECTING", "OTHER", "ADMINISTRATIVE",
]

# Whole-session tutor move distribution (21 features)
LLM_SANDPIPER_COLS = [
    *[f"llm_{c.lower()}" for c in _LEARNING_SUPPORT_CODES],
    "llm_high_engagement_ratio",
    "llm_elicit_vs_tell",
    "llm_scaffold_vs_tell",
    "llm_feedback_positive_rate",
    "llm_move_entropy",
    "llm_giving_answer_rate",
]

# Temporal trajectory of tutor moves across session thirds (7 features)
LLM_TEMPORAL_COLS = [
    "llm_he_early", "llm_he_mid", "llm_he_late",
    "llm_scaffold_slope",
    "llm_fallback_count", "llm_fallback_rate",
    "llm_giving_answer_late",
]

# Student knowledge-state annotations: session rates + temporal trajectory (11 features)
LLM_STUDENT_COLS = [
    *[f"llm_stu_{c.lower()}" for c in _STUDENT_CODES],
    "llm_stu_confusion_resolve",
    "llm_stu_late_demonstrating",
    "llm_stu_late_confusion",
    "llm_stu_learning_trajectory",
]

# Final student state: last 10 turns of session (student_end annotation, 4 features)
LLM_STU_END_COLS = [
    "llm_stu_end_demonstrating_understanding",
    "llm_stu_end_confusion_expressed",
    "llm_stu_end_self_correcting",
    "llm_stu_end_seeking_clarification",
]

# Student LO-window knowledge-state rates for one (session, LO) pair (4 features)
LLM_STU_LO_COLS = [
    "llm_lo_stu_demonstrating_understanding",
    "llm_lo_stu_confusion_expressed",
    "llm_lo_stu_seeking_clarification",
    "llm_lo_stu_self_correcting",
]

# Arc + misconception features for one (session, LO) pair (12 features)
# Misconception-conditional features are NaN when misconception_present=False.
LLM_ARC_MISCONCEPTION_COLS = [
    "llm_arc_confusion_count",
    "llm_arc_resolution_rate",
    "llm_arc_final_resolved",
    "llm_arc_student_led",
    "llm_misconception_present",
    "llm_misconception_resolved",
    "llm_misconception_student_self",
    "llm_misconception_tutor_corrected",
    # PCA-reduced embeddings of misconception_description text (NaN if no misconception)
    "llm_misconception_embed_0",
    "llm_misconception_embed_1",
    "llm_misconception_embed_2",
    "llm_misconception_embed_3",
]

# LO cross-feature interactions (2 features)
LLM_LO_INTERACTION_COLS = [
    "llm_lo_tell_confused",
    "llm_lo_prompt_effective",
]

# LO-window tutor move distribution + student rates + interactions (29 features)
LLM_LO_COLS = [
    *[f"llm_lo_{c.lower()}" for c in _LEARNING_SUPPORT_CODES],
    "llm_lo_high_engagement_ratio",
    "llm_lo_elicit_vs_tell",
    "llm_lo_scaffold_vs_tell",
    "llm_lo_giving_answer_rate",
    "llm_lo_move_entropy",
    "llm_lo_recency",
    "llm_lo_window_frac",
    *LLM_STU_LO_COLS,
    *LLM_LO_INTERACTION_COLS,
]


def load_llm_sandpiper_features() -> "pd.DataFrame | None":
    """Load artifacts/sandpiper_llm_features.parquet, or None if not built yet.

    Returns a DataFrame indexed by session_id with
    LLM_SANDPIPER_COLS + LLM_TEMPORAL_COLS + LLM_STUDENT_COLS columns.
    Join onto the design matrix on session_id.
    """
    from . import config as _cfg
    path = _cfg.ARTIFACTS_DIR / "sandpiper_llm_features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_llm_lo_features() -> "pd.DataFrame | None":
    """Load artifacts/sandpiper_lo_features.parquet, or None if not built yet.

    Returns a DataFrame with columns [session_id, lo_id] + LLM_LO_COLS.
    Join onto the design matrix on (session_id, learning_objective_id) after
    casting learning_objective_id to str to match lo_id.
    """
    from . import config as _cfg
    path = _cfg.ARTIFACTS_DIR / "sandpiper_lo_features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_llm_arc_misconception_features() -> "pd.DataFrame | None":
    """Load artifacts/sandpiper_arc_misconception_features.parquet, or None if not built.

    Returns a DataFrame with columns [session_id, lo_id] + LLM_ARC_MISCONCEPTION_COLS.
    Misconception-conditional features (misconception_resolved, student_self,
    tutor_corrected) are NaN when misconception_present=False.
    Join onto the design matrix on (session_id, learning_objective_id).
    """
    from . import config as _cfg
    path = _cfg.ARTIFACTS_DIR / "sandpiper_arc_misconception_features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

STRUCTURED_COLS = [
    # --- transcript stats (session-level) ---
    "n_utterances", "n_student", "n_tutor", "n_background",
    "student_words", "tutor_words", "student_tutor_word_ratio",
    "total_chars", "n_question_marks", "n_unclear",
    "session_duration_secs", "avg_pause_secs", "n_long_pauses", "avg_student_pause_secs",
    "student_words_per_turn", "student_words_var", "student_max_turn_words",
    "student_short_turns", "student_short_turn_ratio", "student_ttr", "student_questions",
    "tutor_questions", "tutor_praise", "tutor_words_per_turn",
    "student_unclear_only", "unclear_per_student_turn",
    "is_tsl", "has_tutor_ph",
    "tutor_name_calls", "tutor_name_calls_per_turn", "has_name_call",
    "tutor_ph_count", "student_says_tutor_name", "tutor_says_tutor_name",
    "late_student_words", "late_student_questions", "late_tutor_questions",
    "late_student_short_ratio",
    # --- LO-transcript alignment (per-response) ---
    *LO_TRANSCRIPT_COLS,
    # --- session task diversity (per-session metadata) ---
    *SESSION_TASK_COLS,
    # --- per-LO difficulty (requires labels; add via add_lo_difficulty_features) ---
    *LO_DIFFICULTY_COLS,
    # --- Sandpiper-codebook features ---
    *SANDPIPER_COLS,
]
