"""Project paths and tolerant filename resolution.

The competition serves files with hashed suffixes (e.g.
``train_features_TMQTWsB(1).csv``). Rather than force the user to rename them,
we glob on a stable prefix and pick the match.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
SUBMISSIONS_DIR = ROOT / "submissions"


def _resolve(prefix: str, suffix: str) -> Path:
    """Return the single file in DATA_DIR matching ``{prefix}*{suffix}``.

    Raises a clear error if zero or more than one match is found.
    """
    matches = sorted(p for p in DATA_DIR.glob(f"{prefix}*{suffix}") if p.is_file())
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{prefix}*{suffix}' in {DATA_DIR}. "
            "Download the competition data into data/ (see data/README.md)."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous match for '{prefix}*{suffix}' in {DATA_DIR}: "
            f"{[m.name for m in matches]}. Keep only one."
        )
    return matches[0]


def features_path() -> Path:
    return _resolve("train_features", ".csv")


def labels_path() -> Path:
    return _resolve("train_labels", ".csv")


def transcripts_zip() -> Path:
    return _resolve("train_transcripts", ".zip")


def submission_template(full: bool = True) -> Path:
    """Resolve a submission_format template.

    There are two: a ~100-row smoke-test file and the ~10.5k-row full file.
    ``full=True`` picks the larger one by file size.
    """
    matches = sorted(DATA_DIR.glob("submission_format*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No submission_format*.csv in {DATA_DIR} (see data/README.md)."
        )
    matches.sort(key=lambda p: p.stat().st_size, reverse=full)
    return matches[0]
