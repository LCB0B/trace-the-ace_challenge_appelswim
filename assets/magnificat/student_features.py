"""Distilled student-state rates (``studst_*``) computed live from transcripts.

The inference twin of ``scripts/distil_student_states.py``. SWAP-118 ships these 12 columns in
place of E2's 14 student-side regexes (+0.27 AUROC resolved on leakage-free rows, independently
reconfirmed at +0.28), but the block was rejected in July and so never got a twin — which is why
SWAP-118 could be neither packed nor scored off-corpus. Every other distilled block has one:
:mod:`magnificat.evidence_features`, :mod:`magnificat.nto_features`.

⚠️ THE TRUNCATION CHAIN IS THE WHOLE RISK, and it is subtler here than in ``nto_features``.
The head was fitted on ``artifacts/turn_text.pkl.gz``, whose fields were ALREADY truncated when
that table was built (``scripts/build_evidence_features.py:400-401``: ``teacher[-220:]``,
``student[:220]``). ``distil_student_states._text`` then truncates AGAIN
(``tutor_q[-200:]``, ``student[:300]``). So the text the vectoriser actually saw is:

    tutor   = teacher[-220:][-200:]   ==  teacher[-200:]
    student = student[:220][:300]     ==  student[:220]     <-- NOT student[:300]

Feeding ``student[:300]`` at inference would hand the vectoriser up to 80 characters it never
saw in training. That is exactly the bug that cost the NTO block on the leaderboard: 18.8% of
texts differed and **no CV in this repo could detect it**, because every offline experiment
reads the distilled CSV, which is itself built from the truncated text. Both stages are applied
below and the parity check in ``scripts/check_studst_parity.py`` is not optional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence_features import build_turns

#: Codes in the order `distil_student_states.py` writes them (`annotate_student_states.ALL_CODES`).
STUDENT_CODES = ["EXPLAINS_WHY", "SHOWS_METHOD", "ANSWER_ONLY", "CONFUSED", "HEDGED",
                 "INSIGHT", "ASKS_CONCEPTUAL", "ASKS_VERIFICATION", "SELF_CORRECTS",
                 "GUESSES", "MINIMAL", "OFF_TASK"]

#: The 12 shipped columns, in the order the training CSV emits them. Order is PINNED: the
#: booster indexes columns positionally through the `dlg` passthrough transformer.
STUDST_COLS = [f"studst_{c.lower()}_rate" for c in STUDENT_CODES]


def state_text(turns, i: int) -> str:
    """The text the distilled head reads for turn ``i``.

    Mirrors `distil_student_states._text` applied to `turn_text.pkl.gz`'s already-truncated
    fields — see the module docstring. Do not "simplify" the double slice: the outer bound is
    what the head was fitted on, the inner bound is what the table stored, and the effective
    student window is 220 characters, not 300.
    """
    t = turns[i]
    tutor = t["teacher"][-220:][-200:]
    student = t["student"][:220][:300]
    return f"{tutor} [S] {student}"


def _rates(probs: np.ndarray) -> dict:
    """Session rate = MEAN of the soft per-turn probabilities, matching
    `distil_student_states.py`'s `groupby("session_id").mean()` on the training side. Soft, not
    thresholded — the same choice `nto_features._rates` documents."""
    m = probs.mean(axis=0)
    return {col: float(m[j]) for j, col in enumerate(STUDST_COLS)}


def build_studst_matrix(feats: pd.DataFrame, store=None, clf=None,
                        batch: int = 200_000, show_progress: bool = False) -> pd.DataFrame:
    """Build :data:`STUDST_COLS` live from transcripts, keyed by ``response_id``.

    ``clf`` is ``{"vec": ..., "heads": {CODE: LogisticRegression|None}, "base": {CODE: float}}``
    as dumped by `distil_student_states.py:317`. A head is ``None`` when its code was too rare
    to fit; the distillation fell back to the code's base rate and so does this — inventing a
    prediction there would differ from the training-side block.

    ``clf=None`` yields all-NaN, a state the booster already meets whenever a transcript has no
    student turns.
    """
    from . import data as _data

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc="studst")
        except ImportError:
            pass

    texts, owner = [], []
    for sid, tdf in it:
        try:
            turns, _ = build_turns(tdf)
        except Exception:
            turns = []
        for i, t in enumerate(turns):
            # `turn_text.pkl.gz` keeps a row only where the STUDENT block is non-empty
            # (build_evidence_features.py:398-399). Coding a turn with no student utterance
            # would feed the head a class of input it never saw.
            if t["student"].strip():
                texts.append(state_text(turns, i))
                owner.append(sid)

    sess: dict[str, dict] = {}
    if clf is not None and texts:
        arr = np.asarray(texts, dtype=object)
        P = np.zeros((len(arr), len(STUDENT_CODES)), dtype=np.float32)
        for start in range(0, len(arr), batch):
            chunk = arr[start:start + batch]
            Xv = clf["vec"].transform(chunk)
            for j, code in enumerate(STUDENT_CODES):
                head = clf["heads"].get(code)
                P[start:start + len(chunk), j] = (
                    head.predict_proba(Xv)[:, 1] if head is not None
                    else float(clf.get("base", {}).get(code, 0.0)))
        own = np.asarray(owner)
        for sid in dict.fromkeys(owner):
            sess[sid] = _rates(P[own == sid])

    blank = {c: np.nan for c in STUDST_COLS}
    out = pd.DataFrame(
        [sess.get(s, blank) for s in feats["session_id"]],
        index=pd.Index(feats["response_id"], name="response_id"),
    )
    for c in STUDST_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[STUDST_COLS].astype(float)
