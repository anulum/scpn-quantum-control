# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the QRNG streaming harness
"""Tests for the buffered quantum random-number streaming harness.

Covers the empty-input entropy estimate, injection of a custom entropy backend,
the debiasing refill loop's skip of empty draws, the sample/stream/health-check
input guards, and buffer release — all against a deterministic fake backend so
no simulator run is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.entropy.qrng_stream import QRNGStream, _entropy_per_bit


class _ConstantBackend:
    """Entropy backend returning a fixed bit value (debiases to nothing)."""

    def sample_bits(self, n_bits: int) -> NDArray[np.int8]:
        """Return ``n_bits`` identical zero bits."""
        return np.zeros(n_bits, dtype=np.int8)


class _EmptyThenAlternatingBackend:
    """Backend whose first draw debiases to empty, then yields usable entropy."""

    def __init__(self) -> None:
        self.calls = 0

    def sample_bits(self, n_bits: int) -> NDArray[np.int8]:
        """Return constant bits first (debias → empty), then alternating bits."""
        self.calls += 1
        if self.calls == 1:
            return np.zeros(n_bits, dtype=np.int8)
        return np.tile(np.array([0, 1], dtype=np.int8), (n_bits // 2) + 1)[:n_bits]


def test_entropy_per_bit_empty_is_zero() -> None:
    """An empty bit array has zero Shannon and min-entropy."""
    assert _entropy_per_bit(np.empty(0, dtype=np.int8)) == (0.0, 0.0)


def test_accepts_injected_backend_without_debias() -> None:
    """A custom entropy backend is used directly; raw bits stream through."""
    stream = QRNGStream(source=_ConstantBackend(), debias=False)
    bits = stream.sample(8)
    assert bits.dtype == np.uint8
    assert bits.tolist() == [0] * 8


def test_refill_skips_empty_debiased_draws() -> None:
    """A draw that debiases to empty is skipped and the refill loop continues."""
    backend = _EmptyThenAlternatingBackend()
    stream = QRNGStream(source=backend, debias=True)
    bits = stream.sample(8)
    assert bits.size == 8
    assert backend.calls >= 2


def test_sample_rejects_negative_count() -> None:
    """A negative sample request is rejected."""
    stream = QRNGStream(source=_ConstantBackend(), debias=False)
    with pytest.raises(ValueError, match="n_bits must be a non-negative integer"):
        stream.sample(-1)


def test_sample_zero_returns_empty_uint8() -> None:
    """A zero-bit request returns an empty uint8 array."""
    stream = QRNGStream(source=_ConstantBackend(), debias=False)
    result = stream.sample(0)
    assert result.dtype == np.uint8
    assert result.size == 0


def test_stream_rejects_non_positive_chunk() -> None:
    """The streaming iterator requires a positive chunk size."""
    stream = QRNGStream(source=_ConstantBackend(), debias=False)
    with pytest.raises(ValueError, match="chunk_bits must be a positive integer"):
        next(stream.stream(0))


def test_health_check_requires_minimum_bits() -> None:
    """health_check refuses a sample smaller than the FIPS power-up block."""
    stream = QRNGStream(source=_ConstantBackend(), debias=False)
    with pytest.raises(ValueError, match="needs at least"):
        stream.health_check(1)


def test_close_releases_buffer() -> None:
    """Closing the stream empties the internal buffer."""
    stream = QRNGStream(source=_ConstantBackend(), debias=False)
    stream.sample(4)
    stream.close()
    assert stream.sample(0).size == 0
