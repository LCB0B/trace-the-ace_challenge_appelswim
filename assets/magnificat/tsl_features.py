"""TSL (Third Space Learning) provenance features.

Third Space Learning publishes its own curriculum objectives across three
public web properties (mathshub.thirdspacelearning.com, the legacy
thirdspacelearning.com/resources/ site, and the maths-tutoring/* programme
pages) -- see external_data/thirdspacelearning/README.md and
docs/05_data_source.md for the scrape that produced ``tsl_corpus.csv``.

These features answer, per response: *does this LO's text match something in
TSL's own published curriculum, and if so, how central/well-documented is it
there?* This is computed **live** by fuzzy-matching the LO text against the
frozen ``tsl_corpus.csv`` reference table -- the same pattern as
``features._nc_match_live`` against ``nc_curriculum.csv``. It is NOT a
per-``learning_objective_id`` lookup fit on train labels, so it generalizes
to novel test LOs by construction: any LO text (train or test) gets matched
against the same static corpus, independent of whether it happened to appear
in train. This keeps it clear of the no-LO-memorization rule (the project's rules) --
unlike LO_DIFFICULTY_COLS, nothing here is a train-label statistic.

Bundling for submission: copy ``external_data/thirdspacelearning/tsl_corpus.csv``
into ``assets/`` (mirrors nc_curriculum.csv's bundling convention).
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd

TSL_COLS = [
    "tsl_found",              # 0/1: LO text matches something on a TSL page
    "tsl_match_score",         # 0-1 float: best fuzzy match ratio (0 if not found)
    "tsl_high_confidence",     # 0/1: best match came from the structured "programmes" pages
    "tsl_n_occurrences",       # log1p(# distinct scraped rows matching) -- how often TSL repeats this LO
    "tsl_n_co_located",        # log1p(# sibling LOs in the matched list/pack)
    "tsl_year_group",          # numeric year group parsed from the matching TSL page/programme; 0 = unknown
    "tsl_is_secondary",        # 0/1: matched content is from the secondary/GCSE programme pages
]


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().rstrip("."))


def load_tsl_corpus() -> "pd.DataFrame | None":
    """Load tsl_corpus.csv -- checks assets/ first (submission), then
    external_data/thirdspacelearning/ (dev). Returns None if not found."""
    from pathlib import Path
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "assets" / "tsl_corpus.csv",
        Path(__file__).resolve().parent.parent.parent / "external_data" / "thirdspacelearning" / "tsl_corpus.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


def _tsl_match_live(lo_texts: list, corpus: pd.DataFrame, cutoff: float = 0.87) -> dict:
    """Match each LO text against the TSL corpus. Returns
    {lo_text: (found, score, high_conf, n_occ, n_co_located, year_group, is_secondary)}.
    """
    norm_texts = corpus["lo_text"].map(_norm)
    # group by normalized text so repeats across pages count as occurrences
    groups = corpus.groupby(norm_texts)
    norm_list = list(groups.groups.keys())

    result = {}
    for lo_text in lo_texts:
        if lo_text in result:
            continue
        n = _norm(lo_text)
        if n in groups.groups:
            score = 1.0
            match_key = n
        else:
            close = difflib.get_close_matches(n, norm_list, n=1, cutoff=cutoff)
            if not close:
                result[lo_text] = (0, 0.0, 0, 0.0, 0.0, 0.0, 0)
                continue
            match_key = close[0]
            score = difflib.SequenceMatcher(None, n, match_key).ratio()

        rows = groups.get_group(match_key)
        high_conf = int(rows["confidence"].eq("high").any())
        n_occ = float(len(rows))
        co_located_lens = rows["co_located_los"].dropna().map(
            lambda s: len([x for x in str(s).split(";") if x])
        )
        n_co_located = float(co_located_lens.max()) if len(co_located_lens) else 0.0
        years = rows["year_group"].dropna()
        year_group = float(years.median()) if len(years) else 0.0
        is_secondary = int(rows["source_site"].eq("programmes_secondary").any())

        result[lo_text] = (1, float(score), high_conf, np.log1p(n_occ),
                            np.log1p(n_co_located), year_group, is_secondary)
    return result


def add_tsl_features(X: pd.DataFrame, meta: pd.DataFrame, corpus: "pd.DataFrame | None") -> pd.DataFrame:
    """Join TSL provenance features onto a response-indexed design matrix.

    ``meta`` must have a ``learning_objective`` column, indexable the same
    way as ``add_lo_curriculum_features``. If ``corpus`` is None (asset
    missing), all TSL_COLS are filled with the "not found" defaults so the
    pipeline still runs (submission-safe degrade, same convention as
    ``load_nc_curriculum_df`` returning None).
    """
    m = meta.copy()
    if m.index.name != "response_id":
        m = m.set_index("response_id")
    m = m[["learning_objective"]].loc[X.index]

    result = X.copy()
    if corpus is None or corpus.empty:
        for col in TSL_COLS:
            result[col] = 0.0
        return result

    unique_texts = m["learning_objective"].dropna().unique().tolist()
    lookup = _tsl_match_live(unique_texts, corpus)

    rows = []
    for lo_text in m["learning_objective"]:
        vals = lookup.get(lo_text, (0, 0.0, 0, 0.0, 0.0, 0.0, 0))
        rows.append(dict(zip(TSL_COLS, vals)))
    tsl_df = pd.DataFrame(rows, index=X.index)
    for col in TSL_COLS:
        result[col] = tsl_df[col].values
    return result
