"""The XGBoost booster for Model E3+Ndst, in the vendored package so it unpickles.

**Why this is not in `scripts/`.** The fitted pipeline contains a `FunctionTransformer`
that holds a reference to :func:`densify`. joblib pickles that reference by
*module path*, so if the function lived in `scripts/e3_xgb_ab.py` the submission
container — which only vendors `magnificat` — would raise `ModuleNotFoundError` on load.
Keeping it here is what makes the model shippable at all.

**Why XGBoost.** `xgboost 3.3.0` entered the official runtime on 2026-07-28
(tutoring-outcomes-runtime@8581996); on the E3 matrix `grow_policy=lossguide` with 31
leaves beats the shipped LightGBM C3 booster by −0.0017 LL / +0.45 AUROC, replicated on
two LO-disjoint partitions it was not selected on (docs/27).

**Why dense.** The preprocessor emits a sparse matrix, and LightGBM reads an unstored CSR
entry as `0` while XGBoost reads it as *missing* — which would collapse E2's deliberate
0-vs-NaN encoding (`confusion_rate = 0` = "never signalled confusion" vs NaN = "could not
measure"). Measured, that costs only 0.0001 (sd 0.0003), so the distinction does not bind;
but the shipped config was tuned dense, so it ships dense rather than re-deriving the
tuning on a different input representation.
"""
from __future__ import annotations

import numpy as np

#: docs/27 winner: C3 translated to XGBoost's vocabulary, then `lossguide` at 31 leaves
#: (`num_leaves=7 -> max_depth=3`, `min_child_samples=600 -> min_child_weight=126`, i.e.
#: 600 x p(1-p) at p~0.7; `min_split_gain -> gamma`).
E3_XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=0, grow_policy="lossguide",
    max_leaves=31, min_child_weight=126.0, reg_lambda=80.0, reg_alpha=10.0,
    colsample_bytree=0.35, subsample=0.6, gamma=0.02, tree_method="hist",
)


def densify(Z):
    """Sparse -> dense float32 so an unstored entry stays 0 instead of becoming
    XGBoost 'missing'. NaN survives as NaN and is still routed as missing.

    Downcast BEFORE materialising: the TF-IDF block is float64, so `toarray()` first
    would peak at 2x the memory of the dense matrix it is about to throw away."""
    if hasattr(Z, "toarray"):
        return Z.astype(np.float32).toarray()
    return np.asarray(Z, dtype=np.float32)


def build_e3_xgb_pipeline(dialogue_cols, n_jobs: int = -1, random_state: int = 42,
                          **overrides):
    """The shipped E3+Ndst pipeline: the SAME preprocessor as
    :func:`magnificat.baseline.build_meta_dialogue_gbm_pipeline` (TF-IDF over the
    objective + one-hot LO id + passthrough numerics), a dense cast, then XGBoost.

    Reusing that preprocessor verbatim is deliberate — it makes the booster swap the only
    difference from the LightGBM model every earlier number was measured against.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from xgboost import XGBClassifier

    from .baseline import build_meta_dialogue_gbm_pipeline

    pre = build_meta_dialogue_gbm_pipeline(dialogue_cols).named_steps["pre"]
    clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                        missing=np.nan, n_jobs=n_jobs, random_state=random_state,
                        verbosity=0, **dict(E3_XGB_PARAMS, **overrides))
    return Pipeline([("pre", pre),
                     ("dense", FunctionTransformer(densify, accept_sparse=True)),
                     ("clf", clf)])
