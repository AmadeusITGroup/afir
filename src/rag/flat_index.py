"""Exact inner-product search in numpy; drop-in for ``faiss.IndexFlatIP`` with no second OpenMP runtime."""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Sentinel values mirroring faiss: callers skip idx < 0; padded slots cannot beat real ones.
_MISSING_INDEX = -1
_MISSING_SCORE = -np.inf


class FlatIP:
    """Brute-force inner-product index. Expects unit-norm vectors; does not re-normalise (that would hide a provider that stopped)."""

    def __init__(self, dim: int) -> None:
        self.d = int(dim)
        self._vectors = np.zeros((0, self.d), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        """Number of indexed vectors (faiss API name)."""
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        """Append vectors; raises on width mismatch with informative message (faiss asserts silently)."""
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-D array of vectors, got shape {arr.shape}")
        if arr.shape[1] != self.d:
            raise ValueError(
                f"cannot add {arr.shape[1]}-dimensional vectors to a "
                f"{self.d}-dimensional index"
            )
        self._vectors = (
            arr.copy() if self.ntotal == 0 else np.vstack([self._vectors, arr])
        )

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Top-``k`` by inner product; ``(distances, indices)`` both ``(nq, k)``; stable descending order."""
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.d:
            raise ValueError(
                f"cannot search a {self.d}-dimensional index with "
                f"{q.shape[1]}-dimensional queries"
            )
        k = int(k)
        if k <= 0:
            return (
                np.zeros((q.shape[0], 0), dtype=np.float32),
                np.zeros((q.shape[0], 0), dtype=np.int64),
            )

        nq = q.shape[0]
        distances = np.full((nq, k), _MISSING_SCORE, dtype=np.float32)
        indices = np.full((nq, k), _MISSING_INDEX, dtype=np.int64)
        if self.ntotal == 0:
            # Normal on first boot; sentinels let the caller's idx < 0 check handle it.
            return distances, indices

        scores = q @ self._vectors.T  # (nq, ntotal), exact
        take = min(k, self.ntotal)
        # Partition first O(n), then sort the slice O(k log k).
        part = np.argpartition(-scores, take - 1, axis=1)[:, :take]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        top = np.take_along_axis(part, order, axis=1)
        distances[:, :take] = np.take_along_axis(scores, top, axis=1).astype(np.float32)
        indices[:, :take] = top
        return distances, indices

    def reconstruct_n(self, start: int = 0, num: int | None = None) -> np.ndarray:
        """The stored vectors, faiss's accessor name. Used when rebuilding from a pickle."""
        if num is None:
            num = self.ntotal - start
        return self._vectors[start : start + num].copy()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"FlatIP(d={self.d}, ntotal={self.ntotal})"
