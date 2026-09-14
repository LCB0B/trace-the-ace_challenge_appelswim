"""Frozen LO-disjoint CV folds — the partition must not depend on which node you land on.

**The bug this exists to prevent (measured 2026-07-28, DCC hpc).** Rule 1 in the project's rules
says to keep the *same* folds across experiments so comparisons are apples-to-apples.
`GroupKFold(5).split(X, y, learning_objective_id)` does not deliver that on a
heterogeneous cluster:

* sklearn's `GroupKFold` orders groups by size with `np.argsort(...)[::-1]`, i.e. the
  **default unstable quicksort**, then greedily assigns each group to the lightest fold.
* 256 of our 398 learning objectives are **tied** on response count, so the assignment is
  decided entirely by how the sort breaks ties.
* numpy dispatches its sort on CPU features. On an AVX-512 node (Xeon Gold 6142) the tie
  order differs from an AVX2-only node (Xeon E5-2660 v3) — verified: identical `lo` and
  identical count vectors, different `argsort` permutations, different fold vectors
  (`59f0c807dad3ac08` vs `79a0ca503d248ce5`), identical fold *sizes* so nothing looks wrong.

Both partitions are valid and LO-disjoint; they are simply **different experiments**. On
the Model E3 matrix the gap is **0.0046 log loss / 1.7 AUC** — about 10x the booster-seed
noise floor (0.0005) and larger than every effect we are trying to measure (E3 over E2 =
0.0022; XGBoost over LightGBM = 0.0018). It silently turned an XGBoost A/B into a
comparison of two different CV splits until the LightGBM control failed to reproduce its
committed number.

Fitting itself is *not* node-dependent: in the same runs, the arms scored on the Qwen3-8B's
CSV-read partition were bit-identical across the two nodes. Freezing the fold vector is
therefore a complete fix.

**Which partition is frozen, and why not the "correct" one.** `INCUMBENT_SHA` is the
AVX-512 partition, because that is the one every committed number in `docs/EXPERIMENTS.md`
and `docs/26_model_E3.md` was measured on (verified: it reproduces E2's committed seed-42
0.5894 / 61.6618 and E3's 0.5866 exactly). `kind="stable"` would be the more principled
choice for a fresh project — it is node-independent by construction — but adopting it now
would silently invalidate comparisons against the entire logged ladder. Freeze the
incumbent; switch only with a full re-baseline.

Usage — replace `GroupKFold(5).split(...)` with::

    from magnificat.folds import lo_folds
    folds = lo_folds(lo)          # int array, one fold id per row, verified by sha
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import config

#: sha256[:16] of the frozen fold vector — the partition all committed numbers use.
INCUMBENT_SHA = "59f0c807dad3ac08"
PATH = config.ARTIFACTS_DIR / "folds_gkf5_lo.npy"


def fold_sha(folds) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(folds, dtype=np.int64))).hexdigest()[:16]


def _greedy(lo, n_splits: int, kind: str) -> np.ndarray:
    """sklearn's `GroupKFold` assignment, with the sort order made explicit.

    `kind="quicksort"` reproduces sklearn exactly (and inherits its CPU dependence);
    `kind="stable"` is deterministic everywhere. Kept here so the frozen vector can be
    rebuilt and audited without depending on a particular sklearn/numpy build."""
    uniq, inv = np.unique(lo, return_inverse=True)
    counts = np.bincount(inv)
    order = np.argsort(counts, kind=kind)[::-1]
    group_to_fold = np.zeros(len(uniq), dtype=int)
    per_fold = np.zeros(n_splits)
    for gi, w in enumerate(counts[order]):
        lightest = int(np.argmin(per_fold))
        per_fold[lightest] += w
        group_to_fold[order[gi]] = lightest
    return group_to_fold[inv]


def lo_folds(lo, n_splits: int = 5, path=PATH, expect: str | None = INCUMBENT_SHA,
             kind: str = "quicksort") -> np.ndarray:
    """The frozen LO-disjoint fold vector, one entry per row of `lo`.

    Loads `path` if present, otherwise computes and writes it. Either way the result is
    checked against `expect` so a wrong-node rebuild fails loudly instead of quietly
    producing a second, incomparable partition. Pass `expect=None` only when
    deliberately creating a new baseline.
    """
    if n_splits != 5 or path is None:
        return _greedy(lo, n_splits, kind)
    if path.exists():
        folds = np.load(path)
        if len(folds) != len(lo):
            raise SystemExit(f"{path} has {len(folds)} rows, `lo` has {len(lo)} — stale "
                             "fold file; delete it and rebuild on an AVX-512 node.")
        _verify(fold_sha(folds), expect, path)
    else:
        # verify BEFORE writing: a rebuild on the wrong node must fail loudly, not
        # leave a poisoned artifact that every later run silently loads.
        folds = _greedy(lo, n_splits, kind)
        _verify(fold_sha(folds), expect, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, folds)
    return folds


def _verify(got: str, expect: str | None, path) -> None:
    if expect is not None and got != expect:
        raise SystemExit(
            f"fold sha {got} != expected {expect}.\nThis is the CPU-dependent-tie-break "
            "bug (see module docstring): you are on a node whose numpy sort orders tied "
            "group counts differently, so this is a DIFFERENT partition and its numbers "
            "are not comparable to anything in docs/EXPERIMENTS.md. Rebuild "
            f"{path} on an AVX-512 node (e.g. Xeon Gold), or pass expect=None to "
            "deliberately start a new baseline.")


# ── leakage-free row mask ────────────────────────────────────────────────────────────────
#: sha256[:16] of the frozen clean-row mask. Same discipline as INCUMBENT_SHA: the mask is
#: DERIVED from the fold vector, so a different partition silently produces a different mask
#: and every clean-row number becomes incomparable.
CLEAN_SHA = "cc365b58eaea4bdb"
CLEAN_PATH = config.ARTIFACTS_DIR / "clean_rows_gkf5.npy"


def clean_sha(mask) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))).hexdigest()[:16]


def clean_mask(session_id, folds, path=CLEAN_PATH, expect: str | None = CLEAN_SHA):
    """Rows whose transcript is NOT in the training half of their own fold.

    `GroupKFold(learning_objective_id)` controls LO memorisation but abandons
    session-disjointness: 36.7% of sessions carry more than one objective, so one transcript
    gets split across folds and **52.4% of responses (18,385/35,072) are trained on their own
    conversation**. A row is CLEAN iff every response of its session lands in a single fold.

    This is not fixable by better fold construction — connected components of the session↔LO
    graph put **93.6% of the data in one component**, so no 5-way split can be both
    session-disjoint and LO-disjoint. One axis must leak; this mask makes the cost measurable
    instead of silent.

    ⚠️ NOT the same as "multi-objective session". `e3_blend_decide.py:154` computes
    `nlo > 1` (8,364 sessions) as a blend-weight covariate; the leak indicator is
    `fold_span > 1` (7,280 sessions). 1,084 multi-objective sessions do not leak, because both
    their objectives happen to land in the same fold.

    ⚠️ Clean rows are a DIFFERENT POPULATION, not a random subsample: base rate 0.634 vs the
    leaked half's 0.765, 1.14 vs 2.80 responses per session, 321 vs 359 distinct objectives.
    Compare models *within* the clean rows; never compare a clean level against a pooled level
    and call the difference an effect.
    """
    import pandas as pd
    s = pd.Series(np.asarray(session_id))
    n_folds_per_session = s.map(pd.Series(np.asarray(folds)).groupby(s).nunique())
    mask = (n_folds_per_session == 1).to_numpy()
    got = clean_sha(mask)
    if expect not in (None, "PENDING") and got != expect:
        raise SystemExit(
            f"clean-row mask sha {got} != expected {expect}. The mask is derived from the "
            "fold vector, so this means the partition changed — clean-row numbers computed "
            "against it are NOT comparable to anything logged. Check fold_sha() first.")
    if path is not None and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mask)
    return mask
