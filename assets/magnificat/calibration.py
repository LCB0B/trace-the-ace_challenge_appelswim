"""Frozen global recalibration for the submission.

Three monotone, ROC-AUC-preserving recalibrators, all FIT ON TRAINING / OOF DATA
ONLY and then frozen:

- ``GlobalRecenter``   — one frozen bias in logit space; shifts the mean.
- ``ConfidenceShrink`` — shrinks probabilities toward an anchor in logit space;
  tempers over-confidence (compresses spread) without moving the mean.
- ``PlattScaler``      — two-parameter logistic map ``sigmoid(a*logit(p)+b)`` fit
  on out-of-fold predictions (rule 2); the right choice for over-confident boosters.

Each is applied per-sample at inference; no test-set statistic is computed at
inference time, so they comply with the no-cross-test-pooling rule.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class GlobalRecenter:
    """One frozen float ``bias_`` added in logit space."""

    def __init__(self, bias: float = 0.0):
        self.bias_ = float(bias)

    @classmethod
    def from_target_mean(cls, p_base, pi_test: float) -> "GlobalRecenter":
        """Pick the single bias so the MEAN recentred probability over ``p_base``
        equals ``pi_test``. Solved on TRAIN predictions only, then frozen.
        sigmoid is monotone in the bias, so bisection converges."""
        z = _logit(p_base)
        lo, hi = -15.0, 15.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _sigmoid(z + mid).mean() < pi_test:
                lo = mid
            else:
                hi = mid
        return cls(0.5 * (lo + hi))

    def transform(self, p):
        return np.clip(_sigmoid(_logit(p) + self.bias_), EPS, 1 - EPS)


class ConfidenceShrink:
    """Monotone, AUC-PRESERVING shrink of probabilities toward ``anchor`` in logit
    space::

        p -> sigmoid( logit(anchor) + (logit(p) - logit(anchor)) * w )

    with ``0 < w <= 1``. ``w < 1`` reduces OVER-confidence (compresses the spread)
    while keeping the mean ≈ anchor — the right fix when a model discriminates fine
    but is too sure of itself. Strictly monotone in ``p`` for any ``w > 0``, so
    ROC-AUC is EXACTLY preserved. One frozen parameter; applied per-sample; no
    test-set statistic used. Differs from :class:`GlobalRecenter` (which shifts the
    mean) — here we squeeze the variance, not move the centre.
    """

    def __init__(self, w: float = 1.0, anchor: float = 0.7025):
        self.w = float(w)
        self.anchor = float(anchor)

    def transform(self, p):
        a = _logit(self.anchor)
        return np.clip(_sigmoid(a + (_logit(p) - a) * self.w), EPS, 1 - EPS)


class PlattScaler:
    """Frozen TWO-parameter logistic recalibration in logit space::

        p -> sigmoid(a * logit(p) + b),   a > 0

    Fit ``a, b`` once on OUT-OF-FOLD training predictions (rule 2), then frozen.
    Strictly monotone in ``p`` for ``a > 0``, so applied as a SINGLE global map at
    inference it leaves ROC-AUC **exactly** unchanged — it can only fix
    calibration (rescale spread via ``a``, shift mean via ``b``).

    Why Platt and not isotonic here: with only two parameters it cannot overfit
    per-objective noise, which is what sank flexible isotonic on this task. It best
    suits an OVER-confident base (``a < 1`` shrinks the spread) — there must be real
    spread to temper; calibrating an already-calibrated base only overfits. Holds
    two floats, so the pickle is version-proof."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    @classmethod
    def fit(cls, p_oof, y) -> "PlattScaler":
        """Fit ``a, b`` by 1-D logistic regression of ``y`` on ``logit(p_oof)``.
        ``p_oof`` MUST be out-of-fold training predictions; ``y`` their labels."""
        from sklearn.linear_model import LogisticRegression

        z = _logit(p_oof).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(z, np.asarray(y))
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def transform(self, p):
        return np.clip(_sigmoid(self.a * _logit(p) + self.b), EPS, 1 - EPS)


class CalibratedPipeline:
    """Wrap a fitted estimator so ``predict_proba`` applies a frozen 1-D
    probability calibrator. The calibrator is any object exposing
    ``transform(p1d) -> p1d`` (:class:`GlobalRecenter`, :class:`ConfidenceShrink`,
    :class:`PlattScaler`). Exposes the sklearn ``predict_proba`` contract, so
    ``submission_src/main.py`` (which just calls ``model.predict_proba(X)[:, 1]``)
    is UNCHANGED. Pickles cleanly: the base estimator + a tiny calibrator."""

    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator
        self.classes_ = np.asarray(getattr(base, "classes_", np.array([0, 1])))

    @property
    def recenter(self):  # backward-compat alias for older callers/pickles
        return self.calibrator

    def predict_proba(self, X):
        p = self.calibrator.transform(self.base.predict_proba(X)[:, 1])
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SeedEnsembleCalibrated:
    """Like :class:`CalibratedPipeline` but averages K seed-varied base estimators
    BEFORE the frozen calibrator (seed-ensemble variance reduction, docs/18 §; E2
    C3 + K=5 -> −0.0015 Platt LL). Same ``predict_proba(X)[:, 1]`` contract, so
    ``submission_src/main.py`` is UNCHANGED. Pickles cleanly (K small GBM pipes)."""

    def __init__(self, bases, calibrator):
        self.bases = list(bases)
        self.calibrator = calibrator
        self.classes_ = np.asarray(getattr(self.bases[0], "classes_", np.array([0, 1])))

    def predict_proba(self, X):
        p = np.mean([b.predict_proba(X)[:, 1] for b in self.bases], axis=0)
        p = self.calibrator.transform(p)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class IdentityScaler:
    """A calibrator that does nothing — for models scored on AUROC alone.

    Off-corpus evaluation ranks by AUROC because base rates differ from Trace-the-Ace's 0.70
    and a TtA-fitted Platt cannot be honest on a different one (see
    scripts/eval_gbm_transfer.py's guards). Wrapping the seed ensemble in this keeps the
    `predict_proba` contract without pretending the probabilities are calibrated.

    It lives HERE rather than in the script that builds the model because a class defined in
    a script's ``__main__`` pickles as ``__main__.IdentityScaler`` and then fails to unpickle
    inside any *other* script — which is exactly how job 29034131 lost all five transfer
    evaluations to ``AttributeError: Can't get attribute '_Identity'``.
    """

    def transform(self, p):
        return np.asarray(p, float)

    def __repr__(self):
        return "IdentityScaler()"
