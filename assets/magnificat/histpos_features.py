"""Programme-position / spaced-re-exposure features (``histpos_*``), computed live.

The inference twin of ``scripts/build_histpos_features.py`` (block (a) only — block (b),
``lorecency_*``, was measured null and is not shipped).

**Why this twin is structurally safe, and why it is still gated.** Every other distilled block
in this package reads a *cached* intermediate on the training side and raw text at inference —
which is exactly how ``recdst_*`` shipped a gate mismatch (47.2 vs 70.3 turns/session) and how a
naive ``losg_*`` twin diverged at 0.285 sd (``turn_text.pkl.gz`` stores ``teacher[-220:]`` /
``student[:220]`` while ``pos``/``n_turns`` index the *unfiltered* list). This block has no
annotator, no classifier and no cached per-turn table: both sides read
``TranscriptStore``-parsed raw transcripts (sorted by ``utterance_id``, ``content`` NaN-filled
to ``""``) plus the row's own learning-objective text. So the two code paths consume the *same*
bytes by construction. That is an argument, not a proof — ``scripts/check_histpos_parity.py``
asserts max|Δ| **== 0** on all four columns over ≥2,000 responses, and asserts its own coverage
before it is allowed to say "pass".

**Four columns, all functions of ONE response's own session and its own objective** (CLAUDE.md
rule 3 — no cross-sample features; in particular this block deliberately carries no rank of the
objective against the session's *other* objectives, which would be reading another test sample):

``histpos_is_first``    the session announces itself as the programme's first, in the opening 40
                        utterances ("this is our first session")
``histpos_prior_head``  ... or refers back to an earlier one there ("last time we looked at…")
``histpos_n_prior``     ``log1p`` of back-references anywhere in the session — a mid-session
                        pivot to the second objective is itself a re-exposure event
``histpos_prior_lo``    a back-reference within ±20 word tokens of **this row's** objective
                        keywords: the spacing × knowledge-component interaction, and the only
                        one of the four that varies between two responses of one session

The keyword rule is ``features._lo_transcript_stats``'s, imported rather than re-typed
(``_WORD``, ``_STOPS``, ``len(w) > 2``), so ``histpos_prior_lo`` and the incumbent
``lo_first_mention_frac`` measure objective presence with one instrument.

🛑 **This module implements the ``v1`` lexicon** — the two regexes below, no forward-reference
veto. The training-side builder has since grown a ``--lexicon`` axis (``wide`` / ``wideprior``,
hand-expanded vocabularies); ``v1`` is the one the −0.00214 clean-row result was measured on and
the one that was approved for adoption. **If a wider lexicon is ever adopted, this file must be
updated in the same commit** — ``scripts/check_histpos_parity.py`` asserts that the builder's
default lexicon is still ``v1`` and fails loudly if it is not.

Processes each session independently — no cross-sample pooling (CLAUDE.md rule 3).
"""
from __future__ import annotations

import bisect
import re

import numpy as np
import pandas as pd

from .features import _STOPS, _WORD

#: "this is our first session/lesson/time", "the very first thing".
FIRST_RE = re.compile(r"first (?:session|lesson|time)|very first", re.IGNORECASE)
#: back-reference to an earlier session in the same programme.
PRIOR_RE = re.compile(r"last (?:week|time|session)|previous session|since our last|"
                      r"we did last|carried on from", re.IGNORECASE)
#: announcements live in the opening exchange; a "last time" 200 turns in is a within-session
#: reference ("last time you tried it you got 7"), not a programme-position claim.
HEAD_N = 40
#: ± word tokens around a prior-reference match searched for this objective's keywords.
LO_WINDOW = 20

#: Column order is PINNED — the booster indexes columns positionally, and this is the order
#: ``artifacts/histpos_features.csv`` emits.
HISTPOS_COLS = [
    "histpos_is_first",
    "histpos_prior_head",
    "histpos_n_prior",
    "histpos_prior_lo",
]


def _lo_words(lo_text) -> set[str]:
    """``features._lo_transcript_stats``'s keyword rule, verbatim.

    ⚠️ ``str(lo_text) if lo_text is not None else ""`` is deliberate and must not be
    "fixed" to a NaN-aware version: the training block applies exactly this, so a missing
    objective yields the token ``{"nan"}`` on both sides. Making the twin smarter here would
    be a divergence, not an improvement.
    """
    return {w.lower() for w in _WORD.findall(str(lo_text) if lo_text is not None else "")
            if w.lower() not in _STOPS and len(w) > 2}


def _session_prep(tdf: pd.DataFrame) -> dict:
    """Everything about a session that does NOT depend on which objective is being scored.

    Mirrors ``build_histpos_features._session_prep``, minus the per-utterance word sets that
    only the (unshipped) ``lorecency_*`` block and its diagnostics consume.
    """
    if tdf is None or not len(tdf):
        return dict(n=0, is_first=np.nan, prior_head=np.nan, n_prior=np.nan,
                    words=[], prior_wi=[])
    content = tdf["content"].fillna("").astype(str).tolist()
    head_text = "\n".join(content[:HEAD_N])
    full_text = "\n".join(content)
    toks = [(m.start(), m.group(0).lower()) for m in _WORD.finditer(full_text)]
    starts = [t[0] for t in toks]
    words = [t[1] for t in toks]
    prior_wi = [bisect.bisect_left(starts, m.start()) for m in PRIOR_RE.finditer(full_text)]
    return dict(
        n=len(content),
        is_first=float(bool(FIRST_RE.search(head_text))),
        prior_head=float(bool(PRIOR_RE.search(head_text))),
        n_prior=float(np.log1p(len(prior_wi))),
        words=words,
        prior_wi=prior_wi,
    )


def _response_row(prep: dict, lo_text) -> dict:
    """One response's four columns, given its session's prep and its own objective text."""
    if prep["n"] == 0:
        # no transcript => "could not measure", not "measured zero" (xgb_pipeline.densify
        # keeps the 0-vs-NaN distinction deliberately)
        return {c: np.nan for c in HISTPOS_COLS}
    prior_lo = 0.0
    lw = _lo_words(lo_text)
    if lw:
        words = prep["words"]
        for wi in prep["prior_wi"]:
            if lw & set(words[max(0, wi - LO_WINDOW): wi + LO_WINDOW + 1]):
                prior_lo = 1.0
                break
    return {"histpos_is_first": prep["is_first"],
            "histpos_prior_head": prep["prior_head"],
            "histpos_n_prior": prep["n_prior"],
            "histpos_prior_lo": prior_lo}


def build_histpos_matrix(feats: pd.DataFrame, store=None,
                         show_progress: bool = False) -> pd.DataFrame:
    """Build :data:`HISTPOS_COLS` live from transcripts, keyed by ``response_id``.

    ``feats`` needs ``response_id``, ``session_id`` and ``learning_objective``. Sessions absent
    from ``store`` come back all-NaN — the same state the booster already meets for a response
    whose transcript has no utterances.

    Unlike the other twins in this package there is no ``clf`` argument: nothing here is
    learned, so there is no train/serve estimator to mismatch.
    """
    from . import data as _data

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc="histpos")
        except ImportError:
            pass

    sess: dict[str, dict] = {}
    for sid, tdf in it:
        try:
            sess[sid] = _session_prep(tdf)
        except Exception:
            sess[sid] = dict(n=0, is_first=np.nan, prior_head=np.nan, n_prior=np.nan,
                             words=[], prior_wi=[])

    blank = dict(n=0, is_first=np.nan, prior_head=np.nan, n_prior=np.nan,
                 words=[], prior_wi=[])
    rows = [_response_row(sess.get(s, blank), lo)
            for s, lo in zip(feats["session_id"], feats["learning_objective"])]
    out = pd.DataFrame(rows, index=pd.Index(feats["response_id"], name="response_id"))
    for c in HISTPOS_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[HISTPOS_COLS].astype(float)
