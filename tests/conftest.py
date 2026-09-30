# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers: a deterministic stand-in for the generator's text encoder."""
from __future__ import annotations

import numpy as np
import pytest
import torch


class TableEncoder:
    """A text encoder backed by a lookup table of vectors.

    Keys are matched after collapsing whitespace, lower-casing and dropping a
    trailing period, so texts normalised by Kimodo's sanitiser still resolve.
    Unknown texts map to a small constant vector.

    It serves both interfaces CSF uses: ``encoder(text) -> 1-D array`` (the
    gate and reference selection) and ``encoder.batch(texts) -> (feat, lengths)``
    with ``feat`` of shape [B, 1, d] (the handles).
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = {self.key(k): np.asarray(v, dtype=np.float64) for k, v in table.items()}
        dim = len(next(iter(self.table.values())))
        self.unknown = np.full(dim, 1e-3)
        self.batch_calls: list[list[str]] = []

    @staticmethod
    def key(text: str) -> str:
        return " ".join(text.split()).rstrip(".").lower()

    def __call__(self, text: str) -> np.ndarray:
        return self.table.get(self.key(text), self.unknown)

    def batch(self, texts):
        self.batch_calls.append(list(texts))
        feat = torch.tensor(np.stack([self(t) for t in texts]), dtype=torch.float32)
        return feat[:, None, :], [1] * len(texts)


@pytest.fixture()
def make_encoder():
    """Factory for `TableEncoder` stubs."""
    return TableEncoder
