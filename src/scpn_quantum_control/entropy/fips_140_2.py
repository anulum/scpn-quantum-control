# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FIPS 140-2 Annex C startup tests
"""FIPS 140-2 Annex C power-up randomness tests on a 20 000-bit sample.

Reference: FIPS PUB 140-2, Annex C (Approved Random Number Generators), the
continuous and power-up statistical tests — monobit, poker, runs, and long-run.
A single fixed 20 000-bit window is consumed; each sub-test has the published
acceptance interval and the suite fails closed on the first violation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FIPS_SAMPLE_BITS = 20_000

# Published FIPS 140-2 Annex C run-length acceptance intervals for 20 000 bits.
_RUN_INTERVALS: tuple[tuple[int, int], ...] = (
    (2315, 2685),  # length 1
    (1114, 1386),  # length 2
    (527, 723),  # length 3
    (240, 384),  # length 4
    (103, 209),  # length 5
    (103, 209),  # length 6 or more
)
_LONG_RUN_LIMIT = 26


@dataclass(frozen=True)
class FipsHealthReport:
    """Outcome of the FIPS 140-2 Annex C power-up tests."""

    monobit_pass: bool
    poker_pass: bool
    runs_pass: bool
    long_run_pass: bool
    ones: int
    poker_statistic: float
    longest_run: int

    @property
    def passed(self) -> bool:
        """Return whether every FIPS sub-test passed."""
        return self.monobit_pass and self.poker_pass and self.runs_pass and self.long_run_pass


def _run_lengths(bits: NDArray[np.int8]) -> dict[int, list[int]]:
    """Return per-symbol run-length lists keyed by the symbol (0 or 1)."""
    runs: dict[int, list[int]] = {0: [], 1: []}
    boundaries = np.flatnonzero(np.diff(bits)) + 1
    segments = np.split(bits, boundaries)
    for segment in segments:
        runs[int(segment[0])].append(segment.size)
    return runs


def fips_140_2_tests(sequence: Sequence[int] | NDArray[np.integer]) -> FipsHealthReport:
    """Run the FIPS 140-2 Annex C power-up tests on exactly 20 000 bits."""
    bits = np.ascontiguousarray(sequence, dtype=np.int8)
    if bits.ndim != 1 or bits.size != FIPS_SAMPLE_BITS:
        raise ValueError(f"FIPS 140-2 requires exactly {FIPS_SAMPLE_BITS} bits")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bit sequence must contain only 0 and 1")

    ones = int(np.sum(bits))
    monobit_pass = 9_725 < ones < 10_275

    nibbles = bits.reshape(-1, 4)
    codes = nibbles.astype(np.int64) @ np.array([8, 4, 2, 1], dtype=np.int64)
    freqs = np.bincount(codes, minlength=16).astype(np.float64)
    poker = (16.0 / 5_000.0) * float(np.sum(freqs**2)) - 5_000.0
    poker_pass = 2.16 < poker < 46.17

    runs = _run_lengths(bits)
    runs_pass = True
    for length_idx, (low, high) in enumerate(_RUN_INTERVALS, start=1):
        for symbol in (0, 1):
            if length_idx < 6:
                count = sum(1 for r in runs[symbol] if r == length_idx)
            else:
                count = sum(1 for r in runs[symbol] if r >= 6)
            if not (low <= count <= high):
                runs_pass = False
    longest_run = max((max(runs[0], default=0), max(runs[1], default=0)))
    long_run_pass = longest_run < _LONG_RUN_LIMIT

    return FipsHealthReport(
        monobit_pass=monobit_pass,
        poker_pass=poker_pass,
        runs_pass=runs_pass,
        long_run_pass=long_run_pass,
        ones=ones,
        poker_statistic=poker,
        longest_run=longest_run,
    )


def enforce_fips_140_2(sequence: Sequence[int] | NDArray[np.integer]) -> FipsHealthReport:
    """Run the FIPS 140-2 power-up tests and raise on the first failure."""
    report = fips_140_2_tests(sequence)
    if not report.passed:
        failed = [
            name
            for name, ok in (
                ("monobit", report.monobit_pass),
                ("poker", report.poker_pass),
                ("runs", report.runs_pass),
                ("long_run", report.long_run_pass),
            )
            if not ok
        ]
        raise RuntimeError(f"FIPS 140-2 power-up test failed: {', '.join(failed)}")
    return report


__all__ = [
    "FIPS_SAMPLE_BITS",
    "FipsHealthReport",
    "enforce_fips_140_2",
    "fips_140_2_tests",
]
