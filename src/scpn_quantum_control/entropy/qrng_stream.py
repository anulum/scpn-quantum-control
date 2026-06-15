# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum random-number streaming harness
"""Production streaming quantum random-number generator.

:class:`QRNGStream` wraps a quantum measurement entropy source
(:mod:`~scpn_quantum_control.entropy.quantum_source`), optionally applies Von
Neumann debiasing, buffers bits, and serves exact-length samples. Periodic
health checks run the FIPS 140-2 power-up suite and a subset of the NIST
SP 800-22 tests and estimate the Shannon and min-entropy per bit.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .fips_140_2 import FIPS_SAMPLE_BITS, FipsHealthReport, fips_140_2_tests
from .nist_sp800_22 import (
    block_frequency_test,
    cumulative_sums_test,
    frequency_test,
    runs_test,
)
from .quantum_source import (
    AerQuantumEntropySource,
    EntropyBackend,
    QuantumSourceKind,
    von_neumann_debias,
)


@dataclass(frozen=True)
class EntropyHealthReport:
    """Health verdict for a block of generated quantum random bits."""

    fips: FipsHealthReport
    nist_p_values: Mapping[str, float]
    shannon_entropy_per_bit: float
    min_entropy_per_bit: float
    bit_rate_kbit_per_s: float
    n_bits: int
    timestamp_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "nist_p_values", MappingProxyType(dict(self.nist_p_values)))

    @property
    def healthy(self) -> bool:
        """Return whether FIPS passed and every recorded NIST P-value cleared 0.01."""
        return self.fips.passed and all(p >= 0.01 for p in self.nist_p_values.values())


def _entropy_per_bit(bits: NDArray[np.int8]) -> tuple[float, float]:
    n = bits.size
    if n == 0:
        return 0.0, 0.0
    ones = float(np.sum(bits))
    p1 = ones / n
    p0 = 1.0 - p1
    shannon = 0.0
    for p in (p0, p1):
        if p > 0.0:
            shannon -= p * np.log2(p)
    min_entropy = -np.log2(max(p0, p1)) if max(p0, p1) > 0.0 else 0.0
    return float(shannon), float(min_entropy)


class QRNGStream:
    """Buffered streaming quantum random-number generator."""

    def __init__(
        self,
        source: QuantumSourceKind | EntropyBackend = "xy_measurement",
        *,
        register_qubits: int = 64,
        debias: bool = True,
        seed: int | None = None,
    ) -> None:
        if isinstance(source, str):
            self._source: EntropyBackend = AerQuantumEntropySource(
                source, register_qubits=register_qubits, seed=seed
            )
        else:
            self._source = source
        self._debias = bool(debias)
        self._buffer: NDArray[np.int8] = np.empty(0, dtype=np.int8)

    def _draw(self, n_raw: int) -> NDArray[np.int8]:
        raw = self._source.sample_bits(n_raw)
        return von_neumann_debias(raw) if self._debias else raw

    def _fill_to(self, n_bits: int) -> None:
        while self._buffer.size < n_bits:
            # Debiasing yields ~1 bit per 4 raw bits; oversample to converge fast.
            deficit = n_bits - self._buffer.size
            n_raw = max(4096, (deficit * 4 + 1024) if self._debias else deficit)
            fresh = self._draw(int(n_raw))
            if fresh.size == 0:
                continue
            self._buffer = np.concatenate([self._buffer, fresh])

    def sample(self, n_bits: int) -> NDArray[np.uint8]:
        """Return exactly ``n_bits`` random bits as a ``uint8`` array of 0/1."""
        if not isinstance(n_bits, int) or n_bits < 0:
            raise ValueError("n_bits must be a non-negative integer")
        if n_bits == 0:
            return np.empty(0, dtype=np.uint8)
        self._fill_to(n_bits)
        out = self._buffer[:n_bits]
        self._buffer = np.ascontiguousarray(self._buffer[n_bits:], dtype=np.int8)
        return out.astype(np.uint8)

    def stream(self, chunk_bits: int) -> Iterator[NDArray[np.uint8]]:
        """Yield fixed-size chunks indefinitely."""
        if not isinstance(chunk_bits, int) or chunk_bits < 1:
            raise ValueError("chunk_bits must be a positive integer")
        while True:
            yield self.sample(chunk_bits)

    def health_check(self, n_bits: int = FIPS_SAMPLE_BITS) -> EntropyHealthReport:
        """Sample ``n_bits`` and run FIPS + NIST + entropy diagnostics on them."""
        if n_bits < FIPS_SAMPLE_BITS:
            raise ValueError(f"health_check needs at least {FIPS_SAMPLE_BITS} bits")
        start = time.perf_counter()
        bits = self.sample(n_bits).astype(np.int8)
        elapsed = time.perf_counter() - start
        fips = fips_140_2_tests(bits[:FIPS_SAMPLE_BITS])
        nist = {
            "frequency": frequency_test(bits).p_value,
            "block_frequency": block_frequency_test(bits, block_size=128).p_value,
            "runs": runs_test(bits).p_value,
            "cumulative_sums": cumulative_sums_test(bits).p_value,
        }
        shannon, min_entropy = _entropy_per_bit(bits)
        rate = bits.size / elapsed / 1_000.0 if elapsed > 0 else 0.0
        return EntropyHealthReport(
            fips=fips,
            nist_p_values=nist,
            shannon_entropy_per_bit=shannon,
            min_entropy_per_bit=min_entropy,
            bit_rate_kbit_per_s=float(rate),
            n_bits=int(bits.size),
            timestamp_ns=time.time_ns(),
        )

    def close(self) -> None:
        """Release the internal buffer."""
        self._buffer = np.empty(0, dtype=np.int8)


__all__ = [
    "EntropyHealthReport",
    "QRNGStream",
]
