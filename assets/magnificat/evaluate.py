""""""
from __future__ import annotations
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

def group_oof(model, X, y, groups, n_splits: int=5, n_jobs: int=5):
    """"""
    return cross_val_predict(model, X, y, cv=GroupKFold(n_splits=n_splits), groups=groups, method='predict_proba', n_jobs=n_jobs)[:, 1]

def ece(y, p, n_bins: int=10) -> float:
    """"""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)

def fold_base_rates(y, groups, n_splits: int=5):
    """"""
    y = np.asarray(y, dtype=float)
    return [float(y[te].mean()) for _, te in GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups)]

def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    return {'logloss': float(log_loss(y, p)), 'auc': float(roc_auc_score(y, p)), 'brier': float(brier_score_loss(y, p)), 'ece': ece(y, p), 'mean_pred': float(np.mean(p)), 'base_rate': float(y.mean()), 'n': int(len(y))}

def metrics_on_subset(y, p, mask) -> dict:
    """"""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.asarray(mask, dtype=bool)
    ys, ps = (y[m], p[m])
    out = {'logloss': float(log_loss(ys, ps, labels=[0.0, 1.0])), 'auc': float('nan'), 'ece': ece(ys, ps), 'mean_pred': float(np.mean(ps)), 'base_rate': float(ys.mean()), 'n': int(m.sum())}
    if len(np.unique(ys)) > 1:
        out['auc'] = float(roc_auc_score(ys, ps))
    return out

def report(y, p, name: str) -> dict:
    m = metrics(y, p)
    print(f"  [{name:>16}] logloss={m['logloss']:.4f}  AUC={m['auc']:.4f}  Brier={m['brier']:.4f}  ECE={m['ece']:.4f}  mean_pred={m['mean_pred']:.4f} (base {m['base_rate']:.4f})  n={m['n']}")
    return m