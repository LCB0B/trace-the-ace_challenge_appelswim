"""Distilled docs/29 interview-derived rates (``d29dst_*``) computed live from transcripts.

The inference twin of ``scripts/build_doc29_features.py --mode distilled``. Until this module
existed the block was **train-side only** — ``src/magnificat/`` carried the annotation prompts
(:mod:`magnificat.doc29_prompts`) but no feature builder — so it could be neither packed into a
submission nor scored on QATD / MRBench. Every other distilled block has a twin:
:mod:`magnificat.evidence_features`, :mod:`magnificat.nto_features`,
:mod:`magnificat.student_features`.

The 34 columns are 7 tutor moves (rate + late-third share), student reasoning / confidence /
affect, and four derived quantities. All are session-level and get broadcast to every response
row of that session.

⚠️ TWO THINGS THAT MUST MIRROR THE TRAINING SIDE EXACTLY, and one that cannot.

1. **No truncation.** Unlike ``student_features``, whose head was fitted on twice-truncated text
   (the 220/200-char chain that a naive ``[:300]`` would break), ``distil_doc29_labels.build_text``
   applies **no** character cap: full preceding opposite-role turn + full own turn. Do not add
   one "for safety" — that would be the NTO train/serve skew in reverse.

2. **``position_frac`` is WITHIN-ROLE.** ``i / max(len(role_turns) - 1, 1)``, not position in the
   whole transcript. Both the annotator (``sandpiper_live_infer.build_prompts_whole:487``, over
   role-filtered turns) and the all-data scorer (``build_doc29_features.distilled_frame:164``)
   use this, so the two halves of the training CSV agree — verified 2026-08-07. Fourteen of the
   34 columns threshold on it at 2/3, so getting it wrong silently corrupts the whole ``_late``
   family.

3. **Exact parity on the annotated sessions is impossible by construction.** The training CSV is
   a MIXTURE: the 4,000 annotated sessions carry OOF probabilities, the other 18,821 carry the
   all-data fit. At inference there are no annotations, so this module correctly uses the
   all-data head everywhere — which means ``scripts/check_doc29_parity.py`` restricts to the
   unannotated sessions, exactly as ``check_studst_parity.py`` does for ``studst``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Tutor move codes, in the order `distil_doc29_labels.HEADS` declares them. Order is PINNED —
#: it drives the column order below, and the booster indexes columns positionally.
TUTOR_CODES = ["NORMALIZING_DIFFICULTY", "RELEASE_HANDOFF", "TRANSFER_PROBE",
               "SIMPLIFYING_TO_SUBPROBLEM", "MOTIVATING_RELEVANCE", "SUMMARIZING_PROGRESS",
               "OFFERING_CHOICE"]

#: Student head names as `distil_doc29_labels.head_name` emits them.
STU_HEADS = ["stu_answer_only", "stu_reasoning_shown", "stu_hedged", "stu_unmarked",
             "stu_assertive", "stu_insight_delight", "stu_frustration"]

LATE = 2.0 / 3.0          # "final third" boundary on position_frac

#: The 34 shipped columns in the order `artifacts/doc29_features_distilled.csv` emits them.
D29DST_COLS = (
    [f"d29dst_{c}_{s}" for c in TUTOR_CODES for s in ("rate", "late")]
    + ["d29dst_any_move_rate"]
    + [f"d29dst_{n}_{s}"
       for n in ("answer_only", "reasoning_shown", "insight_delight", "frustration",
                 "hedged", "unmarked", "assertive")
       for s in ("rate", "late")]
    + ["d29dst_answering_rate", "d29dst_reasoning_share", "d29dst_confidence_contrast",
       "d29dst_delight_pos_mean", "d29dst_delight_any"]
)


def _prev_other_role(df: pd.DataFrame, other: str) -> list[str]:
    """Most recent STRICTLY PRECEDING utterance by the other role, per row.

    Mirrors `distil_doc29_labels.build_text:134-139`: append first, then update, so a turn never
    sees itself.
    """
    prev, last = [], ""
    for r, c in zip(df["role"].astype(str), df["content"]):
        prev.append(last)
        if r == other:
            last = c
    return prev


def _session_turns(tdf: pd.DataFrame, role: str, marker: str):
    """(texts, position_fracs) for one session's turns of ``role``, in transcript order."""
    df = tdf.sort_values("utterance_id").reset_index(drop=True)
    df = df.assign(content=df["content"].fillna("").astype(str))
    other = "student" if role == "tutor" else "tutor"
    df["_prev"] = _prev_other_role(df, other)
    sub = df[df["role"].astype(str) == role]
    n = max(len(sub) - 1, 1)
    texts = (sub["_prev"] + f" [{marker}] " + sub["content"]).tolist()
    pos = [i / n for i in range(len(sub))]
    return texts, pos


def _agg(P: np.ndarray, names: list[str], denom: np.ndarray, late: np.ndarray,
         prefix: str) -> dict:
    """Rate over ``denom`` plus the late-third share — `build_doc29_features._agg` verbatim."""
    out = {}
    d = float(denom.sum())
    dl = float((denom * late).sum())
    for j, name in enumerate(names):
        v = P[:, j]
        out[f"{prefix}{name}_rate"] = float((v * denom).sum() / d) if d > 0 else np.nan
        out[f"{prefix}{name}_late"] = float((v * denom * late).sum() / dl) if dl > 0 else np.nan
    return out


def _session_row(Pt, Ps, post, poss, prefix: str) -> dict:
    """Compile one session's 34 columns. Mirrors `build_doc29_features.compile_block`."""
    r: dict = {}
    if Pt is not None and len(Pt):
        one = np.ones(len(Pt))
        late = (np.nan_to_num(post, nan=0.0) >= LATE).astype(float)
        r.update(_agg(Pt, TUTOR_CODES, one, late, prefix))
        # coverage: share of tutor turns performing ANY docs/29 move
        r[f"{prefix}any_move_rate"] = float(np.clip(Pt.sum(axis=1), 0, 1).mean())
    if Ps is not None and len(Ps):
        one = np.ones(len(Ps))
        late = (np.nan_to_num(poss, nan=0.0) >= LATE).astype(float)
        ans_only, reas = Ps[:, 0], Ps[:, 1]
        hed, unm, ass = Ps[:, 2], Ps[:, 3], Ps[:, 4]
        del_, fru = Ps[:, 5], Ps[:, 6]
        answering = np.clip(ans_only + reas, 0, 1)      # complement = NOT_ANSWERING
        r.update(_agg(np.stack([ans_only, reas, del_, fru], axis=1),
                      ["answer_only", "reasoning_shown", "insight_delight", "frustration"],
                      one, late, prefix))
        # confidence lives on ANSWERING turns only — see build_doc29_features' docstring
        r.update(_agg(np.stack([hed, unm, ass], axis=1),
                      ["hedged", "unmarked", "assertive"], answering, late, prefix))
        r[f"{prefix}answering_rate"] = float(answering.mean())
        tot = float((ans_only + reas).sum())
        r[f"{prefix}reasoning_share"] = float(reas.sum() / tot) if tot > 0 else np.nan
        hs, as_ = float(hed.sum()), float(ass.sum())
        r[f"{prefix}confidence_contrast"] = ((as_ - hs) / (as_ + hs)
                                             if (as_ + hs) > 0 else np.nan)
        w = float(del_.sum())
        r[f"{prefix}delight_pos_mean"] = (
            float((del_ * np.nan_to_num(poss, nan=0.5)).sum() / w) if w > 0 else np.nan)
        r[f"{prefix}delight_any"] = float(min(w, 1.0))
    return r


def _score(texts, clf, pass_name: str, head_names: list[str], batch: int) -> np.ndarray:
    """Per-turn probabilities, one column per head in ``head_names`` order.

    A head absent from the bundle fell back during distillation (fewer than MIN_POS positives)
    and gets 0.0 — matching `distilled_frame:186-191`'s all-data path, which is the path
    inference takes. ⚠️ The OOF path fills the same head with its BASE RATE instead, so a
    fallen-back head would be discontinuous at the 4,000-session boundary of the training CSV.
    No head fell back in the 2026-08-07 run (all 14 fitted), so this is latent, not active.
    """
    heads = clf[pass_name]["heads"]
    vec = clf[pass_name]["vec"]
    P = np.zeros((len(texts), len(head_names)), dtype=np.float32)
    arr = np.asarray(texts, dtype=object)
    for start in range(0, len(arr), batch):
        chunk = arr[start:start + batch]
        X = vec.transform(chunk)
        for j, h in enumerate(head_names):
            m = heads.get(h)
            if m is not None:
                P[start:start + len(chunk), j] = m.predict_proba(X)[:, 1]
    return P


def build_d29_matrix(feats: pd.DataFrame, store=None, clf=None,
                     batch: int = 100_000, show_progress: bool = False) -> pd.DataFrame:
    """Build :data:`D29DST_COLS` live from transcripts, keyed by ``response_id``.

    ``clf`` is the bundle dumped by `distil_doc29_labels.py`:
    ``{"doc29_tutor": {"vec":…, "heads":{…}}, "doc29_student": {…}}``.
    ``clf=None`` yields all-NaN, the state the booster already meets for a session with no turns.
    """
    from . import data as _data

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc="d29dst")
        except ImportError:
            pass

    t_txt, t_pos, t_own = [], [], []
    s_txt, s_pos, s_own = [], [], []
    for sid, tdf in it:
        try:
            tt, tp = _session_turns(tdf, "tutor", "T")
            st, sp = _session_turns(tdf, "student", "S")
        except Exception:
            continue
        t_txt += tt; t_pos += tp; t_own += [sid] * len(tt)
        s_txt += st; s_pos += sp; s_own += [sid] * len(st)

    sess: dict[str, dict] = {}
    if clf is not None:
        Pt = _score(t_txt, clf, "doc29_tutor", [f"tut_{c.lower()}" for c in TUTOR_CODES], batch) \
            if t_txt else np.zeros((0, len(TUTOR_CODES)), dtype=np.float32)
        Ps = _score(s_txt, clf, "doc29_student", STU_HEADS, batch) \
            if s_txt else np.zeros((0, len(STU_HEADS)), dtype=np.float32)
        to = np.asarray(t_own); so = np.asarray(s_own)
        tpos = np.asarray(t_pos, dtype=float); spos = np.asarray(s_pos, dtype=float)
        for sid in dict.fromkeys(list(t_own) + list(s_own)):
            mt, ms = to == sid, so == sid
            sess[sid] = _session_row(Pt[mt], Ps[ms], tpos[mt], spos[ms], "d29dst_")

    blank = {c: np.nan for c in D29DST_COLS}
    out = pd.DataFrame(
        [{**blank, **sess.get(s, {})} for s in feats["session_id"]],
        index=pd.Index(feats["response_id"], name="response_id"),
    )
    for c in D29DST_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[D29DST_COLS].astype(float)
