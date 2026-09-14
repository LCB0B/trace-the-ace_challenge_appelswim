"""LO-neighbourhood difficulty — the single source of truth for trainer and inference.

Same discipline as `model_e2.py` and `evidence_features.py`: train and inference must build
identical columns or the matrix silently drifts (the project's rules). This module exists because the
feature is a **train-fit statistic**, which is the class of feature this project has been
burned by before (`LO_DIFFICULTY_COLS`, docs/18 §13), so every decision that could make it
behave differently at scoring time is made once, here.

**What the feature is.** For an objective, the empirical-Bayes-shrunk correct-rate of its
`K` nearest *training* objectives in `BAAI/bge-large-en-v1.5` embedding space, weighted by
`max(cos, 0) ** POWER`; plus the max similarity actually used (a confidence the booster can
gate on) and the log effective support behind the estimate.

**Why it is not `LO_DIFFICULTY_COLS`.** That column is an objective's *own* train-label
mean, which is undefined for an objective the training set never saw — and DrivenData have
confirmed some test objectives are unseen. "The difficulty of objectives that mean the same
thing" is defined everywhere. Measured: −0.0019 LL / +0.61 AUC on three LO-disjoint
partitions at 5 seeds, with a shuffled-embedding control at null and a leak control that is
*worse* than the honest version.

**🛑 THE ONE THING THAT MUST NOT DRIFT: exclude-self, at inference too.**
During training every objective's value is computed with itself removed from its own
neighbourhood. A test objective that *also appears in training* (the organizers say only
that "not every" test LO is in train, so some are) must therefore ALSO have itself removed
— otherwise those rows get a self-informed value no training row ever saw, which is exactly
the train/test mismatch the `-self` control measured as actively harmful
(+0.0019 / −0.0000 / −0.0045 vs the honest arm). `exclude_self=True` is the default and
inference must not override it.

**The encoder ships with the runtime, not with us.**
`drivendataorg/tutoring-outcomes-runtime@aee9c8c` adds `BAAI/bge-large-en-v1.5` to
`runtime/huggingface_models.txt`, so the weights are in the offline image and `assets/`
carries only the small fitted table below. Encode with an **empty prefix**: BGE's query
instruction is for asymmetric retrieval, and objective-to-objective similarity is symmetric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: fixed ordered column list — the trainer and `main.py` must agree exactly
LOKNN_COLS = ["loknn_diff", "loknn_sim", "loknn_support"]

#: the Rasch contrast built on top of them (see `irt_columns`)
IRT_COLS = ["irt_margin", "irt_margin_recent", "irt_margin_conf"]

K = 15            # flat optimum; a one-knob-at-a-time sweep over 5/15/40/100 adopted nothing
POWER = 3.0       # cos**3: the median of all 398x397 objective pairs sits at cos 0.582, so a
                  # flat weighting hands an unrelated objective 59% of a near-duplicate's vote
PRIOR_N = 30.0    # empirical-Bayes strength. The MEDIAN objective has only 9 responses, so an
                  # unshrunk rate is mostly noise; ~30 responses to outvote the global mean
ENCODER = "BAAI/bge-large-en-v1.5"


@dataclass
class LoKnnTable:
    """Everything fitted on training data, frozen for inference.

    `emb` are unit-normalised embeddings of the TRAINING objectives only — a test objective
    is scored against these and never enters the neighbour pool, which keeps the pool
    identical to the one every training row saw."""
    lo_ids: np.ndarray        # (n_train_lo,) training objective ids
    emb: np.ndarray           # (n_train_lo, d) unit-normalised, ENCODER
    shrunk: np.ndarray        # (n_train_lo,) empirical-Bayes correct-rate
    count: np.ndarray         # (n_train_lo,) training responses behind each
    prior: float              # training base rate; the fallback for a no-neighbour objective

    def __post_init__(self):
        n = len(self.lo_ids)
        if not (len(self.emb) == len(self.shrunk) == len(self.count) == n):
            raise ValueError(f"LoKnnTable ragged: {n} ids vs {len(self.emb)} emb, "
                             f"{len(self.shrunk)} shrunk, {len(self.count)} count")


def fit(lo, y, emb_by_id: dict, prior_n: float = PRIOR_N) -> LoKnnTable:
    """Fit on training rows. `emb_by_id` maps objective id -> raw embedding vector."""
    lo = np.asarray(lo)
    y = np.asarray(y, float)
    uniq, inv = np.unique(lo, return_inverse=True)
    missing = [l for l in uniq if l not in emb_by_id]
    if missing:
        raise KeyError(f"{len(missing)} training objectives have no embedding, e.g. "
                       f"{missing[:3]}")
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    tot = np.bincount(inv, weights=y, minlength=len(uniq))
    prior = float(y.mean())
    E = np.stack([np.asarray(emb_by_id[l], float) for l in uniq])
    E /= np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)
    return LoKnnTable(lo_ids=uniq, emb=E,
                      shrunk=(tot + prior_n * prior) / (cnt + prior_n),
                      count=cnt, prior=prior)


def transform(lo, emb_by_id: dict, table: LoKnnTable, k: int = K, power: float = POWER,
              exclude_self: bool = True) -> pd.DataFrame:
    """Build `LOKNN_COLS` for each row of `lo`. Never reads labels.

    An objective with no embedding, or none with positive similarity to any training
    objective, falls back to the training base rate with zero similarity and zero support —
    a finite, honest "we could not place this objective" rather than a NaN the booster
    would route arbitrarily."""
    lo = np.asarray(lo)
    uniq, inv = np.unique(lo, return_inverse=True)
    diff = np.full(len(uniq), table.prior)
    simm = np.zeros(len(uniq))
    supp = np.zeros(len(uniq))
    id_pos = {l: i for i, l in enumerate(table.lo_ids)}

    for i, l in enumerate(uniq):
        v = emb_by_id.get(l)
        if v is None:
            continue
        v = np.asarray(v, float)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        s = table.emb @ v
        if exclude_self and l in id_pos:
            # see the module docstring: a test objective that is ALSO a training objective
            # must not vote for itself, or it receives a value no training row ever saw
            s[id_pos[l]] = -np.inf
        kk = min(k, len(s))
        top = np.argpartition(-s, kk - 1)[:kk]
        top = top[np.isfinite(s[top])]
        if not len(top):
            continue
        w = np.clip(s[top], 0, None) ** power
        if w.sum() <= 0:
            continue
        diff[i] = float((w * table.shrunk[top]).sum() / w.sum())
        simm[i] = float(s[top].max())
        supp[i] = float((w * table.count[top]).sum() / w.sum())
    return pd.DataFrame({LOKNN_COLS[0]: diff[inv], LOKNN_COLS[1]: simm[inv],
                         LOKNN_COLS[2]: np.log1p(supp[inv])})


def _logit(p, lo=1e-4, hi=1 - 1e-4):
    p = np.clip(np.asarray(p, float), lo, hi)
    return np.log(p / (1 - p))


def irt_columns(knn: pd.DataFrame, accuracy, recency_accuracy) -> pd.DataFrame:
    """The Rasch contrast: student ABILITY minus objective DIFFICULTY, handed to the tree.

    We now hold both terms explicitly — ability as the distilled answer log
    (`recdst_accuracy`, docs/26's 7th most important feature, rho +0.967) and difficulty as
    `loknn_diff` — and a Rasch model says `P(correct) = sigma(ability - difficulty)`. The
    booster has to reconstruct that difference across splits, and at `colsample_bytree=0.35`
    with 31 leaves the two columns are offered to the same tree only ~12% of the time.
    `evidence_features.contingency_block` already sets the precedent of computing a contrast
    directly "rather than hoping a depth-7 booster reconstructs a difference of two rates".

    NaN in, NaN out: a session with no gradeable turns has no ability estimate, and a margin
    of 0 there would assert "average student on an average objective" — a confident lie of
    exactly the kind `recdst_*` is NaN'd for in the ablated view.

    `irt_margin_conf` scales the margin by the neighbourhood's max similarity, so the tree
    can discount the contrast where the objective sits in a thin part of the space.
    """
    acc = pd.Series(accuracy).to_numpy(float)
    rec = pd.Series(recency_accuracy).to_numpy(float)
    diff = knn[LOKNN_COLS[0]].to_numpy(float)
    sim = knn[LOKNN_COLS[1]].to_numpy(float)
    d_l = _logit(diff)
    m = _logit(acc) - d_l
    mr = _logit(rec) - d_l
    m[~np.isfinite(acc)] = np.nan
    mr[~np.isfinite(rec)] = np.nan
    return pd.DataFrame({IRT_COLS[0]: m, IRT_COLS[1]: mr, IRT_COLS[2]: m * sim},
                        index=knn.index)


#: Where the competition runtime mounts every model listed in the runtime repo's
#: `runtime/huggingface_models.txt` — `<mount>/<repo_id>/`, NOT the Hugging Face cache.
#: Checked in order; the first that exists wins, and a bare repo id is the last resort.
_HF_MOUNTS = ("/code_execution/huggingface_models", "huggingface_models")


def resolve_encoder(model_name: str = ENCODER) -> tuple[str, bool]:
    """(path_or_id, is_local). Resolve the encoder to a mounted local directory if there is one.

    🛑 **Fixed 2026-08-02 after the platform smoke test failed.** The module previously passed the
    bare id `BAAI/bge-large-en-v1.5` to `SentenceTransformer` on the assumption that listing it in
    `huggingface_models.txt` put it in the HF cache. It does not: the runtime **mounts** it at
    `/code_execution/huggingface_models/<repo_id>/`, so the bare id resolves only with internet —
    and there is none. The smoke test died with `NameResolutionError` on `huggingface.co` after
    five retries against `modules.json`, `adapter_config.json` and `config.json`.

    Local runs keep working: the mount is absent, so this returns the bare id and the developer's
    own HF cache serves it. That is also exactly why every local run passed and told us nothing —
    a local pass is not evidence for this code path.
    """
    from pathlib import Path as _P
    for base in _HF_MOUNTS:
        p = _P(base) / model_name
        if p.is_dir():
            return str(p), True
    return model_name, False


def encode(texts, model_name: str = ENCODER, device: str = "cpu", batch_size: int = 64):
    """Embed objective texts. Empty prefix on purpose — see the module docstring.

    398 short texts, so CPU is fine and the 6 h budget is untouched; the caller passes
    `device="cuda"` only if a GPU happens to be free."""
    import os

    from sentence_transformers import SentenceTransformer
    path, is_local = resolve_encoder(model_name)
    kw = {}
    if is_local:
        # Belt and braces: a local dir alone still lets sentence-transformers HEAD the hub for
        # a newer revision, which is a 5-retry stall against a blackholed DNS. Forbid it.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        kw["local_files_only"] = True
    try:
        enc = SentenceTransformer(path, device=device, **kw)
    except TypeError:                      # older sentence-transformers: no local_files_only
        enc = SentenceTransformer(path, device=device)
    return enc.encode(list(texts), normalize_embeddings=True, batch_size=batch_size,
                      convert_to_numpy=True).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# KC difficulty — the same escape from the banned per-LO statistic, but through a
# validated curriculum taxonomy instead of embedding cosine.
# ─────────────────────────────────────────────────────────────────────────────

#: fixed ordered column list, same discipline as LOKNN_COLS
KCDIFF_COLS = ["kcdiff_mean", "kcdiff_min", "kcdiff_max", "kcdiff_n",
               "kcdiff_support", "kcdiff_seen_share"]


def kc_difficulty(lo, y, tr_mask, lo_kcs: dict, prior_n: float = PRIOR_N,
                  exclude_self: bool = True):
    """Empirical-Bayes difficulty of an objective's KNOWLEDGE COMPONENTS, fitted on
    `tr_mask` rows only.

    `lo_kcs` maps objective id -> set of KC ids. A held-out objective has no rate of its
    own, but **78.9% of its KCs do** (measured across the frozen folds; 84.2% have at least
    one), because 52% of the LO-level KC vocabulary is shared by more than one objective.
    That sharing is the whole mechanism.

    **🛑 `exclude_self` is load-bearing and defaults to True.** Under LO-disjoint folds a
    held-out objective contributes nothing to any KC's statistics, but a TRAINING objective's
    own rows do feed its own KCs — so without exclusion train and test rows receive
    systematically different columns and the booster learns to trust a feature that behaves
    differently at scoring time. That is exactly the failure `lo_knn`'s `-self` control
    measured as *actively harmful* (+0.0019 / −0.0000 / −0.0045). Same trap, same guard.

    Objectives with no KCs get NaN on the rate columns and `kcdiff_seen_share = 0`, never a
    rate of 0 — a 0 asserts "every component of this objective is always failed", the
    confident lie docs/26 NaNs the `recdst_*` block for.
    """
    lo = np.asarray(lo)
    y = np.asarray(y, float)
    tr_mask = np.asarray(tr_mask, bool)

    # per-KC totals over TRAINING rows, and per-(KC, objective) totals so a single
    # objective's own contribution can be subtracted back out
    tot: dict = {}
    cnt: dict = {}
    per_lo: dict = {}
    for l, yy in zip(lo[tr_mask], y[tr_mask]):
        ks = lo_kcs.get(l)
        if not ks:
            continue
        for c in ks:
            tot[c] = tot.get(c, 0.0) + yy
            cnt[c] = cnt.get(c, 0.0) + 1.0
            k = (c, l)
            per_lo[k] = (per_lo.get(k, (0.0, 0.0))[0] + yy,
                         per_lo.get(k, (0.0, 0.0))[1] + 1.0)
    prior = float(y[tr_mask].mean())

    out = {c: [] for c in KCDIFF_COLS}
    cache: dict = {}
    for l in lo:
        if l in cache:
            for c in KCDIFF_COLS:
                out[c].append(cache[l][c])
            continue
        ks = lo_kcs.get(l) or set()
        rates, sup = [], 0.0
        for c in ks:
            t, n = tot.get(c, 0.0), cnt.get(c, 0.0)
            if exclude_self:
                st, sn = per_lo.get((c, l), (0.0, 0.0))
                t, n = t - st, n - sn
            if n <= 0:
                continue
            rates.append((t + prior_n * prior) / (n + prior_n))
            sup += n
        d = {c: np.nan for c in KCDIFF_COLS}
        d["kcdiff_n"] = float(len(ks))
        d["kcdiff_seen_share"] = (len(rates) / len(ks)) if ks else 0.0
        if rates:
            d["kcdiff_mean"] = float(np.mean(rates))
            d["kcdiff_min"] = float(np.min(rates))
            d["kcdiff_max"] = float(np.max(rates))
            d["kcdiff_support"] = float(np.log1p(sup))
        cache[l] = d
        for c in KCDIFF_COLS:
            out[c].append(d[c])
    return pd.DataFrame(out)


def shuffled_kc_map(lo_kcs: dict, seed: int = 0, kind: str = "assign") -> dict:
    """Randomise WHICH knowledge components an objective has, holding support fixed.

    **`kind="assign"` is the correct control and the default.** It permutes the
    objective -> KC-set *assignment*: objective A receives objective B's KC set. The
    multiset of KC sets is preserved exactly, so every support statistic — objectives per
    KC, KCs per objective, the singleton share — is bit-identical to the honest map, while
    the objective<->KC correspondence is destroyed. That is what isolates semantics.

    Two earlier versions were both wrong, in opposite directions:

    - ``kind="bijection"`` (default until 2026-08-01) permuted the KC **vocabulary** and
      applied the same renaming to every objective. That is a pure relabeling, and every
      statistic `kc_difficulty` computes is label-invariant — so it is a **mathematical
      no-op**. It was run at domain level (job `28994197`) and returned LL/AUC identical to
      the honest arm **to four decimals on all three partitions**, which is the signature of
      a control that does nothing, not of a null effect.
    - ``kind="resample"`` (the original) drew each objective a fresh KC set by count. It
      preserved each objective's KC *count* but not the KC *usage distribution*, so it
      randomised semantics AND changed support at once (singleton KCs 48.1% -> 19.3%). It
      therefore measured "denser pooling helps", not "the taxonomy carries information".

    Both are kept, reachable only by explicit `kind`, so the two logged results stay
    reproducible.
    """
    rng = np.random.default_rng(seed)
    if kind == "assign":
        keys = sorted(lo_kcs)
        vals = [lo_kcs[k] for k in keys]
        order = rng.permutation(len(vals))
        return {k: set(vals[order[i]]) for i, k in enumerate(keys)}
    vocab = sorted({c for v in lo_kcs.values() for c in v})
    if kind == "bijection":
        perm = dict(zip(vocab, rng.permutation(vocab)))
        return {l: {perm[c] for c in v} for l, v in lo_kcs.items()}
    if kind == "resample":
        return {l: set(rng.choice(vocab, size=min(len(v), len(vocab)), replace=False))
                if v else set() for l, v in lo_kcs.items()}
    raise ValueError(f"unknown kind {kind!r}")


# ─────────────────────────────────────────────────────────────────────────────
# A LEARNED difficulty model — `loknn` is a 15-NN weighted average, i.e. a weak
# learner, on the axis that pays most (13.8% of SHAP from 2.4% of columns).
# A regression from LO embedding -> EB-shrunk difficulty is strictly more expressive.
# ─────────────────────────────────────────────────────────────────────────────

LEARNED_COLS = ["lolrn_diff", "lolrn_resid"]


def learned_difficulty(lo, y, tr_mask, E, kind="ridge", prior_n: float = PRIOR_N,
                       knn_diff=None):
    """Fit LO-embedding -> shrunk difficulty on TRAINING objectives, predict for all.

    Same discipline as `knn_difficulty`: only training rows contribute, and an objective's
    own target is never available to it at prediction time because the model is fitted on
    *other* objectives' targets and a held-out objective is, by construction of the
    LO-disjoint folds, absent from the training set entirely.

    ⚠️ Unlike the kNN there is no `exclude_self` knob, and that asymmetry is deliberate:
    a *training* objective IS in the regression's fit set, so its prediction is partly
    fitted to its own label. That is a genuine train/test asymmetry of exactly the kind the
    `-self` controls measured as harmful, so it is bounded two ways — heavy regularisation,
    and `lolrn_resid` (the disagreement with the kNN) reported so the booster can see where
    the two instruments diverge rather than trusting either blindly.

    `kind`: "ridge" (linear, strongest regularisation) or "gbm" (small XGBoost).
    """
    lo = np.asarray(lo)
    y = np.asarray(y, float)
    tr_mask = np.asarray(tr_mask, bool)
    uniq, inv = np.unique(lo, return_inverse=True)
    cnt = np.bincount(inv[tr_mask], minlength=len(uniq)).astype(float)
    tot = np.bincount(inv[tr_mask], weights=y[tr_mask], minlength=len(uniq))
    prior = float(y[tr_mask].mean())
    shrunk = (tot + prior_n * prior) / (cnt + prior_n)
    seen = cnt > 0
    if seen.sum() < 10:
        return pd.DataFrame({LEARNED_COLS[0]: np.full(len(lo), prior),
                             LEARNED_COLS[1]: np.zeros(len(lo))})

    Etr, ytr, wtr = E[seen], shrunk[seen], cnt[seen]
    if kind == "gbm":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                         reg_lambda=50.0, subsample=0.8, colsample_bytree=0.5,
                         n_jobs=1, verbosity=0)
    else:
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=100.0)
    # weight by support: an objective seen 500 times should shape the fit more than one
    # seen 3 times, the same logic the EB shrink applies within an objective
    m.fit(Etr, ytr, sample_weight=wtr)
    pred = np.clip(m.predict(E), 0.02, 0.98)
    resid = pred - (np.asarray(knn_diff, float)[inv] if knn_diff is not None
                    and len(np.asarray(knn_diff)) == len(uniq) else pred * 0)
    return pd.DataFrame({LEARNED_COLS[0]: pred[inv], LEARNED_COLS[1]: resid[inv]})
