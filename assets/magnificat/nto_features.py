"""Distilled NTO tutor-move rates, computed live from transcripts at inference.

The training-side block (`artifacts/nto_dst_features.csv`) is built from the
**session-disjoint OOF** predictions of `scripts/distil_nto_moves.py`; this module is
the inference twin, applying the shipped **all-data** fit
(`artifacts/nto_moves_clf.joblib`, 18.4 MB, pure CPU) to unseen test sessions. That
asymmetry is deliberate and matches the `recdst_*` precedent in
:mod:`magnificat.evidence_features`: at inference the annotator has never seen the test
sessions either, so OOF quality is exactly what the booster was trained to trust.

Measured worth (docs/EXPERIMENTS.md, 2026-07-29): **−0.0029 LL / +0.83 AUROC** on E3 with
XGBoost, replicated on three LO-disjoint partitions, and the distilled block matched or
beat its own Qwen3-32B oracle on all three.

Column order is pinned by :data:`NTODST_COLS` — train and inference must agree, and the
booster consumes positions, not names.

Processes each session independently — no cross-sample pooling (the project's rules rule 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence_features import build_turns

#: move families, in the order `distil_nto_moves.py` wrote them (sorted)
NTO_FAMILIES = ["clarify", "encourage", "explain_conceptual", "explain_procedural",
                "fb_correct", "fb_incorrect", "fb_neutral", "give_answer", "hint",
                "praise", "probe_prior", "probe_understand", "prompt_next_step",
                "prompt_self_correct", "prompt_self_expl", "rapport", "revoicing",
                "strategize", "technical"]
#: the two ratios E2 already builds from regex, rebuilt on the distilled coding
NTODST_COLS = ([f"ntodst_{f}_rate" for f in NTO_FAMILIES]
               + ["ntodst_elicit_to_tell", "ntodst_feedback_positive_ratio"])

_ELICIT = ["prompt_self_expl", "prompt_next_step", "probe_understand", "probe_prior"]
_TELL = ["explain_conceptual", "explain_procedural", "give_answer"]


def move_text(turns, i: int) -> str:
    """The text the distilled head was trained on: the tutor utterance being coded,
    with the preceding student turn as context. Must match `distil_nto_moves.py`
    (`prev_student + " [T] " + tutor_q`) exactly, or the vectoriser sees a different
    distribution than it was fit on."""
    # TRAIN/SERVE SKEW FIX (2026-08-01). The child was fitted on `artifacts/turn_text.pkl.gz`,
    # whose fields were truncated when that table was built
    # (`scripts/build_evidence_features.py:401`): teacher[-220:], student[:220]. Inference fed
    # FULL text, so the vectoriser saw longer strings than anything it was fit on — for the
    # model's largest block (25.7% of SHAP). CV cannot detect this: every local experiment
    # reads `nto_dst_features.csv`, which is itself built from the truncated text, so the skew
    # only ever manifested on the leaderboard. Mirror the truncation exactly.
    prev = (turns[i - 1]["student"] if i > 0 else "")[:220]
    return f"{prev} [T] {turns[i]['teacher'][-220:]}"


def _rates(probs: np.ndarray) -> dict:
    """Session rates = MEAN of soft probabilities, which is the expectation of the
    oracle's hits/n_turns definition. Soft rather than thresholded: docs/26 found the
    soft answer log carried slightly more than hard labels, and a threshold would throw
    away most of what a 19-way multi-label annotator expresses."""
    m = probs.mean(axis=0)
    out = {f"ntodst_{f}_rate": float(m[j]) for j, f in enumerate(NTO_FAMILIES)}
    idx = {f: j for j, f in enumerate(NTO_FAMILIES)}
    el = sum(m[idx[f]] for f in _ELICIT)
    te = sum(m[idx[f]] for f in _TELL)
    out["ntodst_elicit_to_tell"] = float(el / (te + 1e-3))
    fc, fi = m[idx["fb_correct"]], m[idx["fb_incorrect"]]
    out["ntodst_feedback_positive_ratio"] = float(fc / (fc + fi + 1e-3))
    return out


def build_nto_matrix(feats: pd.DataFrame, store=None, clf=None,
                     batch: int = 200_000, show_progress: bool = False) -> pd.DataFrame:
    """Build :data:`NTODST_COLS` live from transcripts, keyed by ``response_id``.

    ``clf`` is ``{"vec": vectoriser, "heads": {family: LogisticRegression}}``. If it is
    None every column comes back NaN — a valid state for the booster, which meets NaN in
    this block whenever a transcript yields no tutor turns.

    Turn texts are vectorised in ONE batched pass over the corpus rather than per
    session: a full test set is ~2.4M turns and 19 heads, so per-session vectorising
    would dominate the 6h budget for no benefit.
    """
    from . import data as _data

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc="nto")
        except ImportError:
            pass

    texts, owner = [], []
    seen = []
    for sid, tdf in it:
        seen.append(sid)
        try:
            turns, _ = build_turns(tdf)
        except Exception:
            turns = []
        for i, t in enumerate(turns):
            if t["teacher"].strip():
                texts.append(move_text(turns, i))
                owner.append(sid)

    sess: dict[str, dict] = {}
    if clf is not None and texts:
        arr = np.asarray(texts, dtype=object)
        P = np.zeros((len(arr), len(NTO_FAMILIES)), dtype=np.float32)
        for start in range(0, len(arr), batch):
            chunk = arr[start:start + batch]
            Xv = clf["vec"].transform(chunk)
            for j, f in enumerate(NTO_FAMILIES):
                head = clf["heads"].get(f)
                # a family too rare to fit in training falls back to its base rate,
                # exactly as the distillation did rather than inventing a prediction
                P[start:start + len(chunk), j] = (
                    head.predict_proba(Xv)[:, 1] if head is not None else 0.0)
        own = np.asarray(owner)
        for sid in dict.fromkeys(owner):
            sess[sid] = _rates(P[own == sid])

    blank = {c: np.nan for c in NTODST_COLS}
    out = pd.DataFrame(
        [sess.get(s, blank) for s in feats["session_id"]],
        index=pd.Index(feats["response_id"], name="response_id"),
    )
    for c in NTODST_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[NTODST_COLS].astype(float)
