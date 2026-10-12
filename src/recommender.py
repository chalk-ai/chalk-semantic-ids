"""Next-video scoring over semantic IDs, shared by `build_recommender.py` and the demo.

Numpy only, so the demo's image needs no pandas or torch. Sparse transition tables are
CSR-style: `keys` sorted, row `i` spans `targets[pointers[i]:pointers[i + 1]]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CODEBOOK = 256


@dataclass(frozen=True)
class Sparse:
    keys: np.ndarray
    pointers: np.ndarray
    targets: np.ndarray
    counts: np.ndarray

    @classmethod
    def from_dict(cls, table: dict[int, dict[int, float]]) -> Sparse:
        keys = np.array(sorted(table), dtype=np.int64)
        pointers = np.zeros(len(keys) + 1, dtype=np.int64)
        targets: list[int] = []
        counts: list[float] = []
        for i, key in enumerate(keys):
            row = table[int(key)]
            targets.extend(row.keys())
            counts.extend(row.values())
            pointers[i + 1] = len(targets)
        return cls(keys, pointers, np.array(targets, dtype=np.int64), np.array(counts, dtype=np.float32))

    def row(self, key: int) -> tuple[np.ndarray, np.ndarray]:
        i = int(np.searchsorted(self.keys, key))
        if i == len(self.keys) or self.keys[i] != key:
            return self.targets[:0], self.counts[:0]
        return self.targets[self.pointers[i] : self.pointers[i + 1]], self.counts[
            self.pointers[i] : self.pointers[i + 1]
        ]


class Recommender:
    """score(v) = log P(prefix0 | history) + w01 * log P(prefix01 | history)
    + w_content * cos(history, v) + w_pop * log pop(v) + w_item * log(1 + item transitions).

    The prefix terms come from transitions between *semantic IDs* in watch sequences,
    so they apply to a video with no views, or a history entry the catalog has never
    seen. The item term only helps videos that already have watch history.
    """

    def __init__(
        self,
        codes: np.ndarray,
        embeddings: np.ndarray,
        popularity: np.ndarray,
        level0: np.ndarray,
        level01: Sparse,
        items: Sparse,
        weights: dict[str, float],
    ) -> None:
        self.codes = codes
        self.embeddings = embeddings
        self.log_pop = np.log1p(popularity)
        self.level0 = level0
        self.level01 = level01
        self.items = items
        self.weights = weights
        self.prefix01 = codes[:, 0].astype(np.int64) * CODEBOOK + codes[:, 1]

    def parts(
        self, history_codes: np.ndarray, history_vectors: np.ndarray, history_items: list[int | None]
    ) -> dict[str, np.ndarray]:
        # Later history entries predict the next video better; weight them up.
        recency = np.array([0.5 ** (len(history_codes) - 1 - i) for i in range(len(history_codes))])
        recency = recency / recency.sum()

        p0 = np.zeros(CODEBOOK)
        p01 = np.zeros(CODEBOOK * CODEBOOK)
        for w, code in zip(recency, history_codes, strict=True):
            row0 = self.level0[code[0]] + self.weights["alpha0"]
            p0 += w * row0 / row0.sum()
            targets, counts = self.level01.row(int(code[0]) * CODEBOOK + int(code[1]))
            if len(targets):
                np.add.at(p01, targets, w * counts / counts.sum())

        item_counts = np.zeros(len(self.codes))
        for item in history_items:
            if item is not None:
                targets, counts = self.items.row(item)
                np.add.at(item_counts, targets, counts)

        query = history_vectors.mean(axis=0)
        query = query / np.linalg.norm(query)
        return {
            "p0": p0,
            "level0": np.log(p0[self.codes[:, 0]]),
            "level01": np.log(p01[self.prefix01] + self.weights["alpha01"]),
            "cosine": self.embeddings @ query,
            "popularity": self.log_pop,
            "item_counts": item_counts,
        }

    def semantic_score(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        w = self.weights
        return (
            parts["level0"]
            + w["level01"] * parts["level01"]
            + w["content"] * parts["cosine"]
            + w["popularity"] * parts["popularity"]
        )

    def hybrid_score(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        return self.semantic_score(parts) + self.weights["item"] * np.log1p(parts["item_counts"])
