"""Aggregations of a per-turn soft-label sequence into session- and objective-level features.

**One code path, two callers.** `scripts/build_v2_families.py` calls these on the per-turn
probability tables written by `scripts/distil_v2_moves.py`; `magnificat.v2_features` calls the
SAME functions on probabilities predicted live from an unseen transcript. Every previous block
in this repo re-implemented its aggregation on the inference side and two of them drifted —
`nto_features.move_text` records one that reached the leaderboard, `v2_features` records a
second caught at 9.9e-03. Here the aggregation is a pure function of ``(T, S, pos, n)`` and
there is nothing to re-implement.

**Why these forms and not the obvious ones.** Three constraints shape every definition below,
and they are not stylistic:

1. *No cross-sample statistics.* Rule 3 forbids any quantity computed across the batch being
   scored. Every constant here is hardcoded; no quantile, mean or scale is ever read off the
   data. The signature of each function is one session's arrays in, one dict out.
2. *Register transfer.* The shipped corpus is ~100-turn transcribed voice tutoring; the hidden
   test is believed to mix in short typed chat (docs/05). So: **rates and ratios, never counts**;
   every denominator is either a probability mass with an additive guard or a turn count; every
   unbounded ratio is clipped at a hardcoded cap for the reason `distil_v2_moves.ELICIT_TELL_CAP`
   documents — a GBM is invariant to monotone transforms, so `log1p` would be a no-op, but split
   thresholds are learned in TtA's units and a shorter register runs off the end of them.
3. *Width costs score.* Three prior blocks adding 4, 6 and 32 columns each improved in-domain
   and lost on the board, monotone in columns added. So every family here is deliberately
   narrow, and the families are separable so the A/B can price each one.

**The within-session problem.** A session-level scalar is CONSTANT across the objectives that
share a transcript, and the task's discriminator is exactly which objective is being scored.
:func:`lo_window_features` is the only family here that varies within a session; the rest can
only move the session's base level. That asymmetry is why it is built even though the
regex-marker version of the same idea read null (docs/12).
"""
from __future__ import annotations

import numpy as np

# ── code inventories ────────────────────────────────────────────────────────────────────
#: Tutor residuals per docs/39 §5's decision order, steps 2-4 and 6. `TUTOR_OTHER` is NOT
#: here: its definition is "the turn IS pedagogically meaningful, but no code above fits",
#: so it belongs on the instructional side of the denominator.
TUTOR_RESID = ["TUTOR_SOCIAL", "TUTOR_TECHNICAL", "TUTOR_UNINTELLIGIBLE", "TUTOR_NO_MOVE"]
STUDENT_RESID = ["STUDENT_SOCIAL", "STUDENT_TECHNICAL", "STUDENT_UNINTELLIGIBLE",
                 "STUDENT_NO_MOVE"]

#: Elicit / tell, as `distil_v2_moves.RATIO_SPEC["v2"]` defines them — imported by value
#: rather than by reference because `scripts/` is not importable inside the runtime container.
ELICIT = ["PROMPTING_SELF_EXPLANATION", "PROMPTING_NEXT_STEP", "PROMPTING_RELATED_CONCEPTS",
          "PROMPTING_ALTERNATIVE_REPRESENTATION", "PROMPTING_SELF_CORRECTION",
          "RELEASE_HANDOFF"]
TELL = ["EXPLAINING_CONCEPTUAL", "EXPLAINING_PROCEDURAL", "GIVING_ANSWER",
        "WORKED_DEMONSTRATION"]
#: Interrogative moves of any kind — the "is the tutor asking?" axis, which `ASKING_QUESTION`
#: alone under-counts because docs/44 §4.1 measured the two annotators splitting this mass
#: differently (12.5% vs 21.7%) while their union moved far less.
ASK = ELICIT + ["ASKING_QUESTION", "TRANSFER_PROBE", "OFFERING_CHOICE"]
#: Student engagement / struggle axes.
STU_ENGAGE = ["REASONING_SHOWN", "SELF_CORRECTING", "INSIGHT_DELIGHT"]
STU_STRUGGLE = ["CONFUSION_EXPRESSED", "FRUSTRATION", "SEEKING_CLARIFICATION"]

EPS = 1e-3
#: Same value and same reason as `distil_v2_moves.ELICIT_TELL_CAP` / `v2_features`.
RATIO_CAP = 10.0


def _idx(cols: list[str]) -> dict:
    return {c: j for j, c in enumerate(cols)}


def _mass(P: np.ndarray, cols: list[str], names: list[str]) -> np.ndarray:
    """Per-turn summed probability over a named subset. Shape (n,)."""
    ix = _idx(cols)
    js = [ix[c] for c in names if c in ix]
    return P[:, js].sum(axis=1) if js else np.zeros(P.shape[0])


def instructional_weight(P: np.ndarray, cols: list[str], resid: list[str]) -> np.ndarray:
    """Per-turn instructional mass ``w_t = clip(1 - sum(residual codes), 0, 1)``.

    The heads are independent one-vs-rest logistic regressions, so the row does not sum to 1
    and the residual mass can exceed it; clipping is what makes ``w`` a weight rather than a
    signed quantity. This is the user-visible meaning of "swap the denominator": v1 could not
    express it because `ADMINISTRATIVE` fused transcription failure, connection trouble, small
    talk and lesson logistics into one 34.7% bin (docs/39 §5).
    """
    return np.clip(1.0 - _mass(P, cols, resid), 0.0, 1.0)


def _ratio(num: float, den: float) -> float:
    """A bounded ratio. The cap is a CONSTANT — deriving it from the batch would make the
    feature depend on the other samples being scored (rule 3)."""
    return float(min(num / (den + EPS), RATIO_CAP))


# ── family: plain / instructional-denominator / trimmed rates ───────────────────────────
def rates(P: np.ndarray, cols: list[str], weights: np.ndarray | None = None,
          mask: np.ndarray | None = None) -> np.ndarray:
    """Weighted mean of the soft probabilities — the expectation of the oracle's
    ``hits / n_turns``, which is what makes the block length-normalised and transferable.

    ``weights=None`` and ``mask=None`` reproduces the shipped ``P.mean(axis=0)`` exactly.
    """
    if mask is not None:
        if not mask.any():
            return np.full(P.shape[1], np.nan)
        P = P[mask]
        weights = None if weights is None else weights[mask]
    if P.shape[0] == 0:
        return np.full(P.shape[1], np.nan)
    if weights is None:
        return P.mean(axis=0)
    den = float(weights.sum())
    if den <= EPS:                       # a session with no instructional mass at all
        return P.mean(axis=0)            # fall back rather than emit a 0/0 sentinel
    return (P * weights[:, None]).sum(axis=0) / den


def instructional_span(w: np.ndarray, thresh: float = 0.5) -> tuple[int, int]:
    """``(first, last)`` turn index with instructional mass above ``thresh``, inclusive.

    "Let the labels set the boundary" — a fixed first/last-5% trim removes real teaching from
    a short session and leaves warm-up in a long one, whereas the codebook says directly which
    turns were teaching. Falls back to the whole session when fewer than two turns qualify, so
    an 8-turn typed exchange is never trimmed to nothing.
    """
    hit = np.where(w > thresh)[0]
    if hit.size < 2:
        return 0, len(w) - 1
    return int(hit[0]), int(hit[-1])




# ── generalisation: make a rate's NOISE independent of transcript length ────────────────
#
# 🛑 THE MEASURED FAILURE MODE. A rate is `mean_t P[t,c]`, so its sampling sd is `sigma/sqrt(n)`.
# TtA sessions have a median of 102 turns and only **0.5%** below 17; QATD's median is 17 with
# **46.6%** below. At n=8 every column is **3.57x noisier than at n=102 at the same nominal
# scale**, and every split threshold the booster learned was fitted where n~102. Measured
# consequence, on the 46-code block against the shipped reference: QATD `n_utt<17` reads
# -4.11 / -3.41 / -3.72 AUROC with all three CIs excluding zero, while `n_utt>=17` reads
# -3.13 / -2.34 / -2.07 with **none** of the three resolving. The block does not fail off-corpus;
# it fails on SHORT transcripts.
#
# That also explains why the fine vocabulary is the thing that wins in-domain and the thing that
# loses away: 24 of 32 tutor codes carry less than 0.05 probability mass per turn, so finer codes
# buy resolution at n=102 and buy variance at n=8. Collapsing them onto axes gives the resolution
# back without fixing the variance, which is why `narrows` reads null at home AND does not help.
#
# **The fix is shrinkage, not narrowing.** With a prior weight `kappa`,
#
#     r'_c = (n * r_c + kappa * pi_c) / (n + kappa)      sd(r'_c) = sigma * sqrt(n) / (n + kappa)
#
# and that sd is nearly FLAT in n rather than falling as 1/sqrt(n). At kappa=25 the 8-vs-102 sd
# ratio drops from 3.57x to 1.08x. Zero added columns.
#
# Two properties that make this safe to try rather than a gamble:
#   * `pi_c` is a HARDCODED per-code constant, frozen below at build time. It is not a per-LO
#     statistic (rule 1) and is never computed from the batch being scored (rule 3).
#   * In-domain it is close to inert BY CONSTRUCTION: TtA's shrink factor n/(n+kappa) spans
#     0.66 (p5) to 0.87 (p95), so within TtA this is nearly an affine per-code rescaling, which a
#     GBM is invariant to. Off-corpus an 8-turn session shrinks by 0.24. The whole effect is
#     therefore concentrated exactly where the failure was measured.
#
#: Prior "sample size" in turns: how much evidence a session needs before its own rates outweigh
#: the corpus prior. NOT tuned on the evaluation folds -- chosen as the value that flattens
#: sd(r') over the length range the hidden test is believed to span (8..162 turns), where
#: sqrt(n)/(n+kappa) peaks at n=kappa. kappa=25 gives a max/min of 1.47x against 4.50x unshrunk.
SHRINK_KAPPA = 25.0

#: Per-code corpus means over all 2.36M/2.37M scored turns, frozen at build time.
#: Regenerate ONLY when the children change, and record it when you do.
CODE_PRIOR = {
    "ANSWER_ONLY": 0.268468,
    "ASKING_QUESTION": 0.303358,
    "CONFUSION_EXPRESSED": 0.037346,
    "EXPLAINING_CONCEPTUAL": 0.036067,
    "EXPLAINING_PROCEDURAL": 0.058784,
    "EXPLAINING_TOOL": 0.009637,
    "FEEDBACK_CORRECT": 0.088575,
    "FEEDBACK_INCORRECT": 0.018471,
    "FEEDBACK_NEUTRAL": 0.028676,
    "FRUSTRATION": 0.003203,
    "GIVING_ANSWER": 0.014881,
    "GIVING_EXAMPLE": 0.006623,
    "GIVING_HINT": 0.034988,
    "GIVING_PRAISE": 0.076777,
    "GUIDING_SESSION": 0.06555,
    "INSIGHT_DELIGHT": 0.012104,
    "MOTIVATING_RELEVANCE": 0.002032,
    "NORMALIZING_DIFFICULTY": 0.010319,
    "OFFERING_CHOICE": 0.005881,
    "PROMPTING_ALTERNATIVE_REPRESENTATION": 0.005898,
    "PROMPTING_NEXT_STEP": 0.020814,
    "PROMPTING_RELATED_CONCEPTS": 0.010894,
    "PROMPTING_SELF_CORRECTION": 0.015552,
    "PROMPTING_SELF_EXPLANATION": 0.078456,
    "REASONING_SHOWN": 0.169928,
    "RELEASE_HANDOFF": 0.053235,
    "RESTATING": 0.011958,
    "REVOICING": 0.017075,
    "SEEKING_CLARIFICATION": 0.06531,
    "SELF_CORRECTING": 0.018238,
    "SIMPLIFYING_TO_SUBPROBLEM": 0.001225,
    "STUDENT_NO_MOVE": 0.183111,
    "STUDENT_OTHER": 0.036314,
    "STUDENT_SOCIAL": 0.074467,
    "STUDENT_TECHNICAL": 0.04102,
    "STUDENT_UNINTELLIGIBLE": 0.089521,
    "SUMMARIZING_PROGRESS": 0.01146,
    "TRANSFER_PROBE": 0.001369,
    "TUTOR_NO_MOVE": 0.009028,
    "TUTOR_OTHER": 0.019509,
    "TUTOR_SOCIAL": 0.058053,
    "TUTOR_TECHNICAL": 0.048656,
    "TUTOR_UNINTELLIGIBLE": 0.018144,
    "WORKED_DEMONSTRATION": 0.015801,
}




#: PER-CODE empirical-Bayes shrinkage, by method of moments on TRAIN statistics only.
#: `r_s` is a mean of `n_s` per-turn probabilities, so
#:     Var(r_s) = Var_between + E[sigma2_within / n_s]
#: and the variance-optimal weight on a session's own estimate is
#:     w_s = Var_between / (Var_between + sigma2_within/n_s) = n_s / (n_s + k_c),
#:     k_c = sigma2_within,c / Var_between,c.
#: This is DERIVED, not swept — no evaluation corpus enters it, and it is frozen at build time
#: like CODE_PRIOR. Measured range 9.1 .. 91.7, median 34.8: 23 of 32 tutor codes want MORE
#: shrinkage than the global 25, and the ordering is NOT rarity (TRANSFER_PROBE at 0.0014
#: mass/turn wants 16, OFFERING_CHOICE at 0.0058 wants 92), which is exactly why one global
#: constant is the wrong shape.
CODE_KAPPA = {
    "ANSWER_ONLY": 14.838,
    "ASKING_QUESTION": 14.621,
    "CONFUSION_EXPRESSED": 51.87,
    "EXPLAINING_CONCEPTUAL": 25.134,
    "EXPLAINING_PROCEDURAL": 23.857,
    "EXPLAINING_TOOL": 32.758,
    "FEEDBACK_CORRECT": 21.494,
    "FEEDBACK_INCORRECT": 49.819,
    "FEEDBACK_NEUTRAL": 28.583,
    "FRUSTRATION": 12.337,
    "GIVING_ANSWER": 27.285,
    "GIVING_EXAMPLE": 55.568,
    "GIVING_HINT": 32.585,
    "GIVING_PRAISE": 23.369,
    "GUIDING_SESSION": 50.924,
    "INSIGHT_DELIGHT": 67.808,
    "MOTIVATING_RELEVANCE": 60.884,
    "NORMALIZING_DIFFICULTY": 58.953,
    "OFFERING_CHOICE": 91.732,
    "PROMPTING_ALTERNATIVE_REPRESENTATION": 62.618,
    "PROMPTING_NEXT_STEP": 32.08,
    "PROMPTING_RELATED_CONCEPTS": 49.314,
    "PROMPTING_SELF_CORRECTION": 63.444,
    "PROMPTING_SELF_EXPLANATION": 32.473,
    "REASONING_SHOWN": 15.266,
    "RELEASE_HANDOFF": 44.6,
    "RESTATING": 36.878,
    "REVOICING": 43.853,
    "SEEKING_CLARIFICATION": 41.465,
    "SELF_CORRECTING": 50.411,
    "SIMPLIFYING_TO_SUBPROBLEM": 13.34,
    "STUDENT_NO_MOVE": 16.987,
    "STUDENT_OTHER": 21.958,
    "STUDENT_SOCIAL": 72.645,
    "STUDENT_TECHNICAL": 10.922,
    "STUDENT_UNINTELLIGIBLE": 16.786,
    "SUMMARIZING_PROGRESS": 63.047,
    "TRANSFER_PROBE": 16.252,
    "TUTOR_NO_MOVE": 63.382,
    "TUTOR_OTHER": 23.823,
    "TUTOR_SOCIAL": 91.321,
    "TUTOR_TECHNICAL": 10.131,
    "TUTOR_UNINTELLIGIBLE": 9.074,
    "WORKED_DEMONSTRATION": 39.323,
}


def eb_shrunk_rates(P, cols, mask=None):
    """Shrinkage with a PER-CODE kappa. Same shape as `shrunk_rates`, different weights."""
    if mask is not None:
        P = P[mask]
    n = P.shape[0]
    if n == 0:
        return np.full(len(cols), np.nan)
    tot = P.sum(axis=0)
    pri = np.array([CODE_PRIOR.get(c, float(tot[j]/max(n,1))) for j, c in enumerate(cols)])
    kap = np.array([CODE_KAPPA.get(c, SHRINK_KAPPA) for c in cols])
    return (tot + kap*pri) / (n + kap)


def shrunk_rates(P, cols, mask=None, kappa: float = SHRINK_KAPPA):
    """Rates shrunk toward the frozen per-code prior — same columns, length-stable noise."""
    if mask is not None:
        P = P[mask]
    n = P.shape[0]
    if n == 0:
        return np.full(len(cols), np.nan)
    tot = P.sum(axis=0)
    pri = np.array([CODE_PRIOR.get(c, float(tot[j] / max(n, 1)))
                    for j, c in enumerate(cols)])
    return (tot + kappa * pri) / (n + kappa)




# ── generalisation, root fix: make the DENOMINATOR constant instead of correcting for it ──
#
# Shrinkage (above) flattens a rate's sampling sd across lengths. This removes the length
# dependence outright: aggregate over the LAST K turns rather than over all n, so for every
# session with at least K turns the estimator averages **exactly K terms**. Its sampling sd is
# then literally identical on a 100-turn TtA session and a 30-turn typed one, and a split
# threshold the booster learns means the same thing in both. Only sessions genuinely shorter
# than K still vary, and they are the minority the model can no longer confuse with the rest.
#
# The cost is information: 100-turn sessions discard n-K turns. Two measurements say that cost
# is small on this corpus — the `disp` family (temporal spread and centroid) and the `slope`
# family (late-minus-early) were BOTH null, so WHERE in a session a code occurs carries little
# — and CLAUDE.md's own ladder names end-of-session state as where the interpretable signal
# sits. The counter-evidence (docs/13: full 8,192-token context beat last-40 by +2.03 AUROC for
# the LLM) is about INFORMATION for a sequence model, not about VARIANCE for an averaged
# feature, and does not carry over. It is an empirical question, which is why K is swept.
#
#: Primary window. 24 is not tuned on the evaluation — it is the annotation chunk size already
#: used throughout this pipeline (`annotate_v2_moves` / `distil_v2_moves`, and v1 before them),
#: i.e. a constant this project already committed to for an unrelated reason. 12 and 48 are
#: reported as sensitivity so the choice is visible rather than hidden.
WINDOW_K = 24


def window_rates(P: np.ndarray, cols: list[str], mask: np.ndarray | None = None,
                 k: int = WINDOW_K) -> np.ndarray:
    """Mean over the LAST ``k`` turns of this role, or all of them if fewer exist.

    Length-invariant by construction for any session with >= k turns of the role.
    """
    if mask is not None:
        P = P[mask]
    if P.shape[0] == 0:
        return np.full(len(cols), np.nan)
    return P[-k:].mean(axis=0)


# ── family: boundary / shape of the session ─────────────────────────────────────────────
def boundary_features(wT: np.ndarray, wS: np.ndarray, T: np.ndarray, tcols: list[str],
                      pre: str) -> dict:
    """Where the teaching starts and stops, and how densely it fills the session — six
    columns, all fractions of session length so they survive the register change.

    ``turns_to_ask`` is the one the user named: how long before the tutor first asks anything.
    It is reported as a FRACTION of the session, not a turn count, for the same reason.
    """
    n = len(wT)
    if n == 0:
        return {f"{pre}_{k}": np.nan for k in
                ("warmup_frac", "closing_frac", "instr_span_frac", "instr_share",
                 "turns_to_ask", "code_mass")}
    lo, hi = instructional_span(wT)
    ask = _mass(T, tcols, ASK)
    first_ask = np.where(ask > 0.5)[0]
    return {
        f"{pre}_warmup_frac": lo / n,
        f"{pre}_closing_frac": (n - 1 - hi) / n,
        f"{pre}_instr_span_frac": (hi - lo + 1) / n,
        f"{pre}_instr_share": float(wT.mean()),
        # 1.0 = the tutor never asks anything, which is a real and different state from
        # "asks on the last turn"; a NaN here would be read as missing data instead.
        f"{pre}_turns_to_ask": (first_ask[0] / n) if first_ask.size else 1.0,
        # Mean total code mass per turn. >1 means the annotator saw multi-function turns —
        # docs/39 §4 measured `code_2` firing on 8.7% of tutor turns, and this is its soft
        # analogue without needing a second label slot.
        f"{pre}_code_mass": float(T.sum(axis=1).mean()),
    }


# ── family: direction of travel ─────────────────────────────────────────────────────────
def slope_features(P: np.ndarray, cols: list[str], axes: dict, pre: str) -> dict:
    """``late - early`` for a handful of AXES rather than for every code.

    Per-code thirds would cost 3x32 columns to say something the aggregate axes already say,
    and the added-width finding prices that at more than the effect is worth. An axis is a
    named subset summed per turn, so the quantity is still a rate in [0, k].

    Thirds are by TURN INDEX, not by time: `timestamp` is an elapsed offset and turn rhythm
    varies, but a third of the turns is a third of the turns in any register.
    """
    n = P.shape[0]
    out = {}
    if n < 3:
        return {f"{pre}_{k}_slope": np.nan for k in axes}
    c = n // 3
    for name, names in axes.items():
        m = _mass(P, cols, names)
        out[f"{pre}_{name}_slope"] = float(m[-c:].mean() - m[:c].mean())
    return out


# ── family: dispersion in time ──────────────────────────────────────────────────────────
def dispersion_features(P: np.ndarray, cols: list[str], axes: dict, pre: str) -> dict:
    """Is a move spread through the session or bunched into one burst, and where does its
    mass sit?

    ``spread`` is the entropy of the code's temporal distribution ``q_t = p_t / sum(p)``,
    normalised by ``log n`` so it lands in [0, 1] for any session length — 1 is perfectly
    even, 0 is a single burst. ``centroid`` is the mass-weighted mean position in [0, 1].
    Both are invariant to how much of the code there is, which is what separates them from
    the rate they sit beside; a session with the same questioning rate delivered as one
    interrogation versus steadily throughout is a different lesson.
    """
    n = P.shape[0]
    out = {}
    for name, names in axes.items():
        m = _mass(P, cols, names)
        tot = float(m.sum())
        if n < 2 or tot <= EPS:
            out[f"{pre}_{name}_spread"] = np.nan
            out[f"{pre}_{name}_centroid"] = np.nan
            continue
        q = m / tot
        h = -np.sum(np.where(q > 0, q * np.log(np.maximum(q, 1e-12)), 0.0))
        out[f"{pre}_{name}_spread"] = float(h / np.log(n))
        out[f"{pre}_{name}_centroid"] = float((q * (np.arange(n) / (n - 1))).sum())
    return out


# ── family: transitions and contingency ─────────────────────────────────────────────────
#
# The paired-turn layout is what makes these cheap and unambiguous. `turn_text.pkl.gz` stores
# each turn as (tutor_q, student, tutor_reply) and `tutor_reply[t] == tutor_q[t+1]`, so within
# a row the tutor move PRECEDES the student move, and the tutor's response to the student at
# row t is the tutor move at row t+1. Forward pairs therefore read off one row; backward pairs
# read off consecutive rows. No window search, no alignment assumption.
#
# Every one of these is a CONDITIONAL rate: the denominator is how often the antecedent fired,
# so "praise given the student just reasoned" is a different number from "praise", which is the
# distinction the construct rests on.

#: ``name: (antecedent codes, consequent codes, direction)``. ``"t2s"`` reads the student slot
#: of the same row; ``"s2t"`` reads the tutor slot of the next row.
TRANSITIONS = {
    # elicitation that lands vs elicitation that misfires — the elicit-vs-tell axis, measured
    # by what the student DOES next rather than by how often the tutor elicits
    "elicit_to_reason":  (ELICIT, ["REASONING_SHOWN"], "t2s"),
    "elicit_to_confuse": (ELICIT, STU_STRUGGLE, "t2s"),
    "elicit_to_bare":    (ELICIT, ["ANSWER_ONLY"], "t2s"),
    # what the tutor does when the student is stuck: scaffold or take over
    "confuse_to_tell":   (STU_STRUGGLE, TELL, "s2t"),
    "confuse_to_scaffold": (STU_STRUGGLE,
                            ["GIVING_HINT", "PROMPTING_SELF_CORRECTION",
                             "SIMPLIFYING_TO_SUBPROBLEM", "NORMALIZING_DIFFICULTY"], "s2t"),
    # transfer of the work, and whether it is taken up
    "handoff_to_reason": (["RELEASE_HANDOFF"], STU_ENGAGE, "t2s"),
    # repair: does naming an error produce a self-correction?
    "wrong_to_repair":   (["FEEDBACK_INCORRECT"], ["SELF_CORRECTING", "REASONING_SHOWN"],
                          "t2s"),
    # contingent praise: praise BECAUSE the student reasoned, not praise in general
    "reason_to_praise":  (STU_ENGAGE, ["GIVING_PRAISE", "FEEDBACK_CORRECT"], "s2t"),
}


def transition_features(T: np.ndarray, tcols: list[str], S: np.ndarray, scols: list[str],
                        pre: str) -> dict:
    """Soft conditional rates ``sum_t a_t * c_t / (sum_t a_t + eps)``, bounded in [0, k].

    Soft rather than thresholded: a 32-way annotator expresses most of what it knows below
    any threshold, and docs/26 measured the soft answer log carrying more than hard labels.
    """
    n = T.shape[0]
    out = {}
    for name, (ante, cons, direction) in TRANSITIONS.items():
        if n < 2:
            out[f"{pre}_{name}"] = np.nan
            continue
        if direction == "t2s":                      # tutor row t -> student row t
            a, c = _mass(T, tcols, ante), _mass(S, scols, cons)
        else:                                       # "s2t": student row t -> tutor row t+1
            a, c = _mass(S, scols, ante)[:-1], _mass(T, tcols, cons)[1:]
        den = float(a.sum())
        out[f"{pre}_{name}"] = _ratio(float((a * c).sum()), den) if den > EPS else np.nan
    return out


# ── family: bounded ratios ──────────────────────────────────────────────────────────────
def ratio_features(P: np.ndarray, cols: list[str], pre: str) -> dict:
    """Elicit-vs-tell and scaffold-vs-give-away as SHARES, ``a / (a + b)``, not as ``a / b``.

    In-domain this is a free change and looks like one: a GBM is invariant to monotone
    transforms of a single feature, and ``x -> x/(1+x)`` is monotone, so the share and the
    quotient induce identical splits on TtA. **Off-corpus they are not the same feature.**
    The quotient is unbounded, its tail is set by an additive guard, and
    `distil_v2_moves.ELICIT_TELL_CAP` exists because v2 splits the tell side across more codes
    than v1 and drove the max to 28.55. A corpus where tutors explain less shrinks the
    denominator further and the value runs past every threshold learned here — the cap then
    collapses a whole region of the input onto one leaf. The share cannot leave [0, 1], so
    every threshold learned on TtA still partitions the range on a typed corpus.
    """
    el, te = _mass(P, cols, ELICIT).sum(), _mass(P, cols, TELL).sum()
    ask = _mass(P, cols, ASK).sum()
    scaf = _mass(P, cols, ["GIVING_HINT", "SIMPLIFYING_TO_SUBPROBLEM",
                           "PROMPTING_SELF_CORRECTION", "NORMALIZING_DIFFICULTY"]).sum()
    give = _mass(P, cols, ["GIVING_ANSWER", "WORKED_DEMONSTRATION"]).sum()
    return {f"{pre}_elicit_share": float(el / (el + te + EPS)),
            f"{pre}_ask_share": float(ask / (ask + te + EPS)),
            f"{pre}_scaffold_share": float(scaf / (scaf + give + EPS))}


# ── family: how decisively the annotator could code this transcript ─────────────────────
def quality_features(T: np.ndarray, tcols: list[str], S: np.ndarray, scols: list[str],
                     tmask: np.ndarray, pre: str) -> dict:
    """Four register/legibility proxies, all in [0, 1] and all length-normalised.

    Not a measure of tutoring — a measure of how coda-ble the transcript is. `decisive` is the
    mean top-1 probability per turn and `entropy` the mean normalised entropy across codes:
    together they say whether the annotator saw clear moves or a smear. That distinction is
    exactly the axis on which the hidden test is believed to differ (docs/05: transcribed voice
    vs typed chat, and 0/22,821 train sessions are short-and-clean), and none of the rate
    columns can express it — a session that is 30% `TUTOR_UNINTELLIGIBLE` and one the annotator
    simply could not read produce different smears at the same rate.

    ``tutor_turn_share`` is the fraction of paired turns carrying a tutor utterance, which is a
    diarisation and turn-taking property rather than a pedagogical one.
    """
    out = {}
    for name, (P, mask) in (("tutor", (T, tmask)), ("student", (S, None))):
        Q = P[mask] if mask is not None else P
        if Q.shape[0] == 0 or Q.shape[1] < 2:
            out[f"{pre}_{name}_decisive"] = np.nan
            out[f"{pre}_{name}_entropy"] = np.nan
            continue
        out[f"{pre}_{name}_decisive"] = float(Q.max(axis=1).mean())
        tot = Q.sum(axis=1, keepdims=True)
        q = Q / np.maximum(tot, 1e-9)
        h = -(q * np.log(np.maximum(q, 1e-12))).sum(axis=1) / np.log(Q.shape[1])
        # a turn with no mass at all is uninformative, not maximally uncertain
        out[f"{pre}_{name}_entropy"] = float(h[tot[:, 0] > 1e-6].mean()) \
            if (tot[:, 0] > 1e-6).any() else np.nan
    out[f"{pre}_tutor_turn_share"] = float(tmask.mean()) if tmask.size else np.nan
    return out


# ── family: soft LO proximity — ONE column, and the only one that varies by objective ────
#: Decay constant for the proximity kernel. A CONSTANT, not a swept parameter: with one
#: submission window left, a window tuned on the folds it is scored on is fitted to its own
#: evaluation. tau=1 makes the kernel effectively "this turn or the two either side", which is
#: what the hard +/-2 window approximates with a step function.
LO_TAU = 1.0
#: Below this many turns the mean of a decay kernel is dominated by the session being short
#: rather than by where the objective sits, so the column reports NaN rather than a number the
#: booster would read as "the objective is everywhere".
LO_MIN_TURNS = 6


def lo_soft_coverage(hits: np.ndarray, n: int, pre: str, tau: float = LO_TAU) -> dict:
    """``c_LO = (1/n) * sum_t exp(-d_t / tau)``, where ``d_t`` is the distance in turns from
    turn ``t`` to the nearest turn whose words hit the objective's keywords.

    **One column, deliberately.** The hard-window block measured within-session AUROC 0.5353;
    this soft form measures **0.5440** against a 0.5103 null band from matched random windows —
    and one column beats every wider combination of the same idea (soft alone 0.5438, the
    6-column set 0.5430, the 2-column hard window 0.5402). The board's own width law reproduces
    in miniature inside the within-session signal: the extra columns dilute rather than add.

    It is also the only quantity in this module that is a function of the OBJECTIVE as well as
    the transcript, so it is the only one that can separate two responses sharing a session —
    the re-open bar the contingency axis closed 6/6 against, stated verbatim in docs/44 §5.

    NaN, never zero, when the objective is never named or the session is too short: "never
    discussed" and "discussed nowhere near here" are different states, and a zero would be read
    as the second.
    """
    if n < LO_MIN_TURNS or hits is None or not hits.any():
        return {f"{pre}_soft_cov": np.nan}
    idx = np.arange(n)
    hit_at = idx[hits]
    d = np.abs(idx[:, None] - hit_at[None, :]).min(axis=1)
    return {f"{pre}_soft_cov": float(np.exp(-d / tau).mean())}


# ── family: the LO-relevant window (the only one that varies within a session) ───────────
def lo_window_features(T: np.ndarray, tcols: list[str], S: np.ndarray, scols: list[str],
                       win: np.ndarray | None, pre: str) -> dict:
    """Rates on the turns that handled THIS objective, and how they differ from the rest.

    ``win`` is a boolean mask over turns, supplied by the caller so that the keyword rule
    lives in exactly one place (`features._lo_transcript_stats`, via
    `misfit_features._lo_window`) rather than being re-typed here.

    **Reported as inside-minus-outside gaps, not as raw inside rates.** A raw inside rate is
    mostly the session's level, which every other column here already carries; the gap is the
    part that is specific to this objective, and it is what can separate two responses that
    share a transcript. `misfit_features._response_row` uses the same construction for the
    same reason. When the objective is never mentioned, or the window swallows the session,
    the gap columns are NaN — the state the booster already meets for a transcript with no
    turns, never a zero-fill that would read as "no difference".
    """
    keys = ("cover", "pos", "gap_ask", "gap_tell", "gap_reason", "gap_struggle", "gap_instr",
            "end_reason", "end_struggle")
    if win is None or T.shape[0] == 0 or not win.any():
        return {f"{pre}_{k}": np.nan for k in keys}
    n = T.shape[0]
    out = {f"{pre}_cover": float(win.mean()),
           # where in the session this objective was handled — the "distance" feature. An
           # objective taught in the last quarter is a different prediction problem from one
           # taught first and then left alone for 40 turns.
           f"{pre}_pos": float(np.arange(n)[win].mean() / max(1, n - 1))}
    axes = {"ask": (T, tcols, ASK), "tell": (T, tcols, TELL),
            "reason": (S, scols, STU_ENGAGE), "struggle": (S, scols, STU_STRUGGLE),
            "instr": (T, tcols, None)}
    out_mask = ~win
    for name, (P, cols, names) in axes.items():
        m = (instructional_weight(P, cols, TUTOR_RESID) if names is None
             else _mass(P, cols, names))
        if not out_mask.any():
            out[f"{pre}_gap_{name}"] = np.nan       # window is the whole session: no contrast
        else:
            out[f"{pre}_gap_{name}"] = float(m[win].mean() - m[out_mask].mean())

    # How the objective was LEFT, not how it went on average. The modelling ladder names
    # "end-of-session state" as where the interpretable signal lives, and the same argument
    # applies per objective: a student still struggling on the last turn that touched this
    # objective is a different prediction from one who was struggling in the middle and
    # resolved it. Final quarter of the window, at least one turn, so an 8-turn typed exchange
    # still yields a value.
    wi = np.where(win)[0]
    tail = wi[-max(1, len(wi) // 4):]
    for name, (P, cols, names) in (("reason", (S, scols, STU_ENGAGE)),
                                   ("struggle", (S, scols, STU_STRUGGLE))):
        out[f"{pre}_end_{name}"] = float(_mass(P, cols, names)[tail].mean())
    return out
