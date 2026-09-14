"""Data loading: features, labels, and per-session transcripts.

Transcripts come in two physical forms in this competition:
- **training**: a single ``train_transcripts*.zip`` of ``{session_id}.csv`` members;
- **inference** (in the code-execution runtime): a *directory*
  ``data/test_transcripts/{session_id}.csv``.

``TranscriptStore`` reads either form behind one interface so the same feature
code works for training and for the submission's ``main.py``.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from . import config

TRANSCRIPT_COLS = ["session_id", "utterance_id", "role", "content", "timestamp"]


def load_features() -> pd.DataFrame:
    """response_id, session_id, learning_objective_id, learning_objective."""
    return pd.read_csv(config.features_path(), dtype=str)


def load_labels() -> pd.DataFrame:
    """response_id, is_correct (int 0/1)."""
    df = pd.read_csv(config.labels_path())
    df["is_correct"] = df["is_correct"].astype(float).astype(int)
    return df


def load_train() -> pd.DataFrame:
    """Features inner-joined with labels on response_id."""
    return load_features().merge(load_labels(), on="response_id", how="inner")


def _empty_transcript() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSCRIPT_COLS)


def _parse_transcript(raw: bytes | str) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(raw) if isinstance(raw, bytes) else raw, dtype=str)
    df["utterance_id"] = pd.to_numeric(df["utterance_id"], errors="coerce")
    return df.sort_values("utterance_id").reset_index(drop=True)


class TranscriptStore:
    """Read per-session transcripts from a ``.zip`` archive or a directory.

    Parameters
    ----------
    source:
        Path to a zip of ``{session_id}.csv`` members, or a directory containing
        ``{session_id}.csv`` files.
    """

    def __init__(self, source: str | Path):
        self.source = Path(source)
        if self.source.is_dir():
            self.is_zip = False
            self._index = {
                p.stem: p for p in self.source.glob("*.csv")
            }
        elif self.source.is_file() and zipfile.is_zipfile(self.source):
            self.is_zip = True
            with zipfile.ZipFile(self.source) as zf:
                names = [n for n in zf.namelist() if n.endswith(".csv")]
            self._index = {n.rsplit("/", 1)[-1][:-4]: n for n in names}
        else:
            raise FileNotFoundError(
                f"Transcript source not found / unrecognised: {self.source} "
                "(expected a .zip archive or a directory of {session_id}.csv files)."
            )

    def sessions(self) -> set[str]:
        return set(self._index)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._index

    def load(self, session_id: str) -> pd.DataFrame:
        """One session's transcript, or an empty frame if missing."""
        ref = self._index.get(session_id)
        if ref is None:
            return _empty_transcript()
        if self.is_zip:
            with zipfile.ZipFile(self.source) as zf:
                return _parse_transcript(zf.read(ref))
        return _parse_transcript(str(ref))

    def iter(self, session_ids):
        """Yield ``(session_id, transcript_df)`` for the given ids.

        De-dupes while preserving order; opens a zip once for the whole sweep.
        """
        wanted = list(dict.fromkeys(session_ids))
        if not self.is_zip:
            for sid in wanted:
                yield sid, self.load(sid)
            return
        with zipfile.ZipFile(self.source) as zf:
            for sid in wanted:
                ref = self._index.get(sid)
                yield sid, (_parse_transcript(zf.read(ref)) if ref else _empty_transcript())


# ---- convenience wrappers over the default (training) store ----

_default_store: TranscriptStore | None = None


def default_store() -> TranscriptStore:
    """Lazily-built store over the train transcripts zip in ``data/``."""
    global _default_store
    if _default_store is None:
        _default_store = TranscriptStore(config.transcripts_zip())
    return _default_store


def available_sessions() -> set[str]:
    return default_store().sessions()


def load_transcript(session_id: str) -> pd.DataFrame:
    return default_store().load(session_id)


def iter_transcripts(session_ids):
    yield from default_store().iter(session_ids)
