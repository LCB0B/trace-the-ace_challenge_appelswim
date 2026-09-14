""""""
from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from . import config, data
warnings.filterwarnings('ignore')

def load_Xy():
    """"""
    train = data.load_train()
    train = train.set_index('response_id')
    train['learning_objective'] = train['learning_objective'].fillna('')
    train['learning_objective_id'] = train['learning_objective_id'].fillna('UNK')
    y = train['is_correct'].to_numpy(float)
    groups = train['learning_objective_id'].to_numpy()
    return (train, y, groups)

def grouped_folds(groups, n_splits=5, seed=0):
    """"""
    uniq = np.array(sorted(set(groups)))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    fid = np.array([fold_of[g] for g in groups])
    return [(np.where(fid != k)[0], np.where(fid == k)[0]) for k in range(n_splits)]
_EMB_TAGS = ('lo1024', 'minilm', 'bge_small', 'bge_base', 'gte_base', 'e5_base', 'mpnet')

def _load_lo_emb(tag):
    """"""
    if tag == 'lo1024':
        emb = np.load(config.ARTIFACTS_DIR / 'lo_embeddings.npy')
        idx = pd.read_csv(config.ARTIFACTS_DIR / 'lo_emb_index.csv')['learning_objective_id'].tolist()
    else:
        emb = np.load(config.ARTIFACTS_DIR / f'lo_text_emb_{tag}.npy')
        idx = pd.read_csv(config.ARTIFACTS_DIR / 'lo_text_emb_index.csv')['learning_objective_id'].tolist()
    return (emb, {lo: i for i, lo in enumerate(idx)})

def emb_rows(train, tag):
    """"""
    emb, pos = _load_lo_emb(tag)
    los = train['learning_objective_id'].to_numpy()
    return np.vstack([emb[pos[lo]] if lo in pos else np.zeros(emb.shape[1], np.float32) for lo in los])
_NUM = re.compile('\\d+\\.?\\d*')
_OPS = {'add': '\\badd|sum|plus|total\\b', 'sub': '\\bsubtract|minus|difference|take away\\b', 'mul': '\\bmultipl|times|product|array\\b', 'div': '\\bdivid|quotient|share\\b', 'frac': '\\bfraction|numerator|denominator|equivalent\\b', 'dec': '\\bdecimal\\b', 'pct': '\\bpercent|percentage\\b', 'compare': '\\bcompar|order|greater|less|equal\\b', 'round': '\\bround|estimat|nearest\\b', 'place': '\\bplace value|digit|tens|hundreds|thousand\\b', 'measure': '\\bmeasur|length|mass|volume|capacity|time|money|cm|metre|gram\\b', 'geom': '\\bangle|shape|triangle|rectangle|polygon|perimeter|area|symmetr|coordinate\\b', 'negative': '\\bnegative\\b', 'ratio': '\\bratio|proportion|scale\\b', 'algebra': '\\bequation|expression|variable|formula|substitut\\b', 'word': '\\bproblem|context|word problem|real-life|real life\\b', 'multistep': '\\bmulti-step|multi step|two-step|then\\b'}

def hand_features(train):
    """"""
    raw = train['learning_objective'].fillna('')
    txt = raw.str.lower()
    toks = raw.str.findall('[A-Za-z]+')
    frac_title = toks.map(lambda ts: np.mean([t[0].isupper() for t in ts]) if ts else 0.0)
    rows = {'n_chars': raw.str.len().to_numpy(float), 'n_words': raw.str.split().map(len).to_numpy(float), 'n_numbers': txt.map(lambda s: len(_NUM.findall(s))).to_numpy(float), 'n_and': txt.str.count('\\band\\b').to_numpy(float), 'n_commas': txt.str.count(',').to_numpy(float), 'frac_titlecase': frac_title.to_numpy(float), 'is_titlecase': (frac_title >= 0.5).astype(float).to_numpy(), 'starts_capital': raw.str.match('^[A-Z]').fillna(False).astype(float).to_numpy(), 'ends_period': raw.str.rstrip().str.endswith('.').fillna(False).astype(float).to_numpy(), 'frac_upper_chars': raw.map(lambda s: sum((c.isupper() for c in s)) / max(1, sum((c.isalpha() for c in s)))).to_numpy(float)}
    for name, pat in _OPS.items():
        rows[f'op_{name}'] = txt.str.contains(pat, regex=True).astype(float).to_numpy()
    cols = list(rows)
    return (np.column_stack([rows[c] for c in cols]), cols)

class KNNDifficulty(BaseEstimator, ClassifierMixin):
    """"""
    classes_ = np.array([0.0, 1.0])

    def __init__(self, k=15, m=20.0, normalize=True):
        self.k, self.m, self.normalize = (k, m, normalize)

    def fit(self, E, y):
        E = np.asarray(E, float)
        if self.normalize:
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-09)
        key = np.round(E, 5)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        nlo = inv.max() + 1
        self.base_ = float(np.mean(y))
        cnt = np.bincount(inv, minlength=nlo).astype(float)
        pos = np.bincount(inv, weights=y, minlength=nlo).astype(float)
        rate = (pos + self.m * self.base_) / (cnt + self.m)
        rep = np.zeros((nlo, E.shape[1]), float)
        for g in range(nlo):
            rep[g] = E[inv == g][0]
        self.rep_, self.rate_, self.cnt_ = (rep, rate, cnt)
        return self

    def _proba_pos(self, E):
        E = np.asarray(E, float)
        if self.normalize:
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-09)
        sims = E @ self.rep_.T
        k = min(self.k, self.rep_.shape[0])
        nn = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        snn = np.take_along_axis(sims, nn, axis=1)
        w = np.clip(snn, 1e-06, None)
        rates = self.rate_[nn]
        return (rates * w).sum(1) / w.sum(1)

    def predict_proba(self, E):
        p = np.clip(self._proba_pos(E), 1e-06, 1 - 1e-06)
        return np.column_stack([1 - p, p])

class KNNDiffFeature(BaseEstimator, TransformerMixin):
    """"""

    def __init__(self, k=15, m=20.0):
        self.k, self.m = (k, m)

    def fit(self, E, y=None):
        self.base_ = KNNDifficulty(k=self.k, m=self.m).fit(E, np.asarray(y, float))
        return self

    def transform(self, E):
        p = np.clip(self.base_._proba_pos(E), 1e-06, 1 - 1e-06)
        return np.log(p / (1 - p)).reshape(-1, 1)

def oof_predict(build_fn, X, y, folds):
    """"""
    p = np.zeros(len(y))
    is_df = hasattr(X, 'iloc')
    for tr, te in folds:
        Xtr = X.iloc[tr] if is_df else X[tr]
        Xte = X.iloc[te] if is_df else X[te]
        m = build_fn()
        m.fit(Xtr, y[tr])
        p[te] = m.predict_proba(Xte)[:, 1]
    return p

def metrics(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return {'logloss': float(log_loss(y, p)), 'auc': float(roc_auc_score(y, p)), 'brier': float(brier_score_loss(y, p)), 'ece': _ece(y, p), 'mean_pred': float(p.mean())}

def _ece(y, p, n_bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def multiseed(build_fn, X, y, groups, n_splits=5, seeds=(0,), per_fold=False):
    """"""
    lls, aucs, eces = ([], [], [])
    folds_rec = []
    p0 = None
    for s in seeds:
        folds = grouped_folds(groups, n_splits=n_splits, seed=s)
        p = oof_predict(build_fn, X, y, folds)
        if s == seeds[0]:
            p0 = p
        m = metrics(y, p)
        lls.append(m['logloss'])
        aucs.append(m['auc'])
        eces.append(m['ece'])
        if per_fold:
            for fi, (_, te) in enumerate(folds):
                yt, pt = (y[te], p[te])
                if len(np.unique(yt)) > 1:
                    folds_rec.append((s, fi, _safe_ll(yt, pt), float(_auc(yt, pt))))
    return {'logloss_mean': float(np.mean(lls)), 'logloss_std': float(np.std(lls)), 'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)), 'ece_mean': float(np.mean(eces)), 'oof': p0, 'logloss_per_seed': lls, 'n_folds': len(seeds) * n_splits, 'folds': folds_rec}

def _safe_ll(y, p):
    from sklearn.metrics import log_loss
    return float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0.0, 1.0]))

def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)