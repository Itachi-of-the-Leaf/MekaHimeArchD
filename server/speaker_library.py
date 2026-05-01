"""
server/speaker_library.py — Phase 3: NumPy Flat-File Speaker Identity Store

Hot storage : in-memory [N, EMBEDDING_DIM] float32 matrix, pre-L2-normalised.
              Cosine similarity = matrix @ query_norm  →  single BLAS call, µs range.
Cold storage: speakers.npy  (matrix)
              metadata.json (names, enrolled_at, last_seen)  — atomic rename on write.
Thread-safe : threading.RLock on all public methods.

⚠️  DIMENSION NOTE:
    EMBEDDING_DIM is set to 256 because wespeaker.load_model('english') downloads
    ResNet34, which outputs 256-dimensional embeddings. If you supply a custom
    512D checkpoint (e.g. a fine-tuned ResNet293 or CAM++ variant), change the
    constant below — the rest of the class is dimension-agnostic.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

log = logging.getLogger("amika.library")

# ── Tunables ──────────────────────────────────────────────────────────────────
EMBEDDING_DIM        = 256    # wespeaker ResNet34 default; change to 512 for CAM++
SIMILARITY_THRESHOLD = 0.65   # cosine similarity floor to claim a match


class IdentifyResult(NamedTuple):
    name:  str
    score: float


class SpeakerLibrary:
    """
    Flat-file speaker identity store backed by a NumPy matrix and JSON metadata.

    Cosine similarity is O(N × D) pure NumPy BLAS — no external DB, no Qdrant.
    """

    _NPY_FILE  = "speakers.npy"
    _META_FILE = "metadata.json"

    def __init__(self, data_dir: str | Path = "server") -> None:
        self._dir  = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._lock: threading.RLock = threading.RLock()

        # Hot matrix — rows are L2-normalised embeddings, shape [N, EMBEDDING_DIM]
        self._matrix:      np.ndarray       = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self._names:       list[str]        = []
        self._last_seen:   dict[str, float] = {}
        self._enrolled_at: dict[str, float] = {}

        self._load_from_disk()
        log.info("[SpeakerLibrary] Ready — %d speaker(s) in %s",
                 len(self._names), self._dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def enroll_speaker(self, name: str, embedding: np.ndarray) -> None:
        """
        Append a new voice print or soft-update an existing one.
        Blend (70 % existing / 30 % new) prevents drift from noisy frames.
        """
        norm = self._normalise(embedding)
        now  = time.time()

        with self._lock:
            if name in self._names:
                idx = self._names.index(name)
                self._matrix[idx] = self._normalise(
                    0.7 * self._matrix[idx] + 0.3 * norm
                )
                log.info("[SpeakerLibrary] Updated: '%s'", name)
            else:
                row = norm[np.newaxis, :]                              # [1, D]
                self._matrix = (
                    np.vstack([self._matrix, row])
                    if len(self._names) > 0 else row.copy()
                )
                self._names.append(name)
                self._enrolled_at[name] = now
                log.info("[SpeakerLibrary] Enrolled: '%s' (total=%d)",
                         name, len(self._names))

            self._last_seen[name] = now
            self._persist()

    def identify_speaker(self, embedding: np.ndarray) -> IdentifyResult:
        """
        Vectorised cosine similarity search.
        Returns (name, score) if score >= SIMILARITY_THRESHOLD, else ("Unknown", score).
        """
        with self._lock:
            if self._matrix.shape[0] == 0:
                return IdentifyResult("Unknown", 0.0)

            query  = self._normalise(embedding)      # [D,]
            scores = self._matrix @ query            # [N,] — cosine similarities

            best_idx   = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= SIMILARITY_THRESHOLD:
                matched = self._names[best_idx]
                self._last_seen[matched] = time.time()
                self._persist_metadata()             # update last_seen only
                return IdentifyResult(matched, best_score)

            return IdentifyResult("Unknown", best_score)

    def get_active_batch(self, names: list[str]) -> np.ndarray:
        """
        Returns a stacked [K, D] matrix of embeddings for the requested speakers.
        Phase 4 hook: used to build the conditioning tensor for pBSRNN.
        """
        with self._lock:
            indices = [i for i, n in enumerate(self._names) if n in names]
            if not indices:
                return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            return self._matrix[indices].copy()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """Write hot matrix + metadata. Call with self._lock held."""
        np.save(str(self._dir / self._NPY_FILE), self._matrix)
        self._persist_metadata()

    def _persist_metadata(self) -> None:
        """Atomic JSON write (tmp file + rename). Call with self._lock held."""
        meta = {
            "version":       1,
            "embedding_dim": EMBEDDING_DIM,
            "count":         len(self._names),
            "speakers": [
                {
                    "index":       i,
                    "name":        name,
                    "enrolled_at": self._enrolled_at.get(name, 0.0),
                    "last_seen":   self._last_seen.get(name,   0.0),
                }
                for i, name in enumerate(self._names)
            ],
        }
        tmp = self._dir / (self._META_FILE + ".tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        tmp.replace(self._dir / self._META_FILE)     # atomic on POSIX + Windows

    def _load_from_disk(self) -> None:
        npy  = self._dir / self._NPY_FILE
        meta = self._dir / self._META_FILE

        if npy.exists():
            loaded = np.load(str(npy)).astype(np.float32)
            self._matrix = loaded
            log.debug("[SpeakerLibrary] Loaded matrix %s", self._matrix.shape)

        if meta.exists():
            with open(meta) as f:
                data = json.load(f)
            speakers           = data.get("speakers", [])
            self._names        = [s["name"]                      for s in speakers]
            self._enrolled_at  = {s["name"]: s.get("enrolled_at", 0.0) for s in speakers}
            self._last_seen    = {s["name"]: s.get("last_seen",   0.0) for s in speakers}

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)
