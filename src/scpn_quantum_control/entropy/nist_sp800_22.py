# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — NIST SP 800-22 Revision 1a statistical test suite
"""NIST SP 800-22 Revision 1a statistical tests for (pseudo)random bit streams.

Reference: A. Rukhin et al., *A Statistical Test Suite for Random and
Pseudorandom Number Generators for Cryptographic Applications*, NIST Special
Publication 800-22 Revision 1a (April 2010).

Each test returns a :class:`NistTestResult` carrying the test name, the
P-value(s), the test statistic, the pass/fail decision at significance level
``alpha`` (default 0.01, the NIST default), and a details mapping. The
mathematical formulation follows the publication exactly; the incomplete gamma
function ``igamc(a, x) = Q(a, x)`` and the complementary error function are
taken from :mod:`scipy.special` (``gammaincc`` and ``erfc``).

This module implements the worked-example-validated subset of the suite. The
remaining large-sample tests live alongside it and share the same result type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, gammaincc

_DEFAULT_ALPHA = 0.01


@dataclass(frozen=True)
class NistTestResult:
    """Outcome of a single NIST SP 800-22 test (or test family)."""

    name: str
    p_value: float
    passed: bool
    statistic: float
    alpha: float = _DEFAULT_ALPHA
    p_values: tuple[float, ...] = ()
    details: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        object.__setattr__(self, "p_values", tuple(float(p) for p in self.p_values))


def as_bits(
    sequence: Sequence[int] | NDArray[np.integer], *, minimum: int = 1
) -> NDArray[np.int8]:
    """Validate and coerce a 0/1 sequence into a contiguous ``int8`` bit array."""
    bits = np.ascontiguousarray(sequence, dtype=np.int8)
    if bits.ndim != 1:
        raise ValueError("bit sequence must be one-dimensional")
    if bits.size < minimum:
        raise ValueError(f"bit sequence must contain at least {minimum} bits, got {bits.size}")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bit sequence must contain only 0 and 1")
    return bits


def _passed(p_value: float, alpha: float) -> bool:
    return p_value >= alpha


# --------------------------------------------------------------------------- #
# §2.1 Frequency (Monobit) Test
# --------------------------------------------------------------------------- #
def frequency_test(
    sequence: Sequence[int] | NDArray[np.integer], *, alpha: float = _DEFAULT_ALPHA
) -> NistTestResult:
    """NIST SP 800-22 §2.1 Frequency (Monobit) Test."""
    bits = as_bits(sequence)
    n = bits.size
    s_n = int(np.sum(2 * bits.astype(np.int64) - 1))
    s_obs = abs(s_n) / np.sqrt(n)
    p_value = float(erfc(s_obs / np.sqrt(2.0)))
    return NistTestResult(
        name="frequency_monobit",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=float(s_obs),
        alpha=alpha,
        details={"n": float(n), "s_n": float(s_n)},
    )


# --------------------------------------------------------------------------- #
# §2.2 Frequency Test within a Block
# --------------------------------------------------------------------------- #
def block_frequency_test(
    sequence: Sequence[int] | NDArray[np.integer],
    *,
    block_size: int = 128,
    alpha: float = _DEFAULT_ALPHA,
) -> NistTestResult:
    """NIST SP 800-22 §2.2 Frequency Test within a Block."""
    bits = as_bits(sequence)
    n = bits.size
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    n_blocks = n // block_size
    if n_blocks < 1:
        raise ValueError("block_size larger than the sequence length")
    trimmed = bits[: n_blocks * block_size].astype(np.float64).reshape(n_blocks, block_size)
    pi = trimmed.mean(axis=1)
    chi_sq = 4.0 * block_size * float(np.sum((pi - 0.5) ** 2))
    p_value = float(gammaincc(n_blocks / 2.0, chi_sq / 2.0))
    return NistTestResult(
        name="block_frequency",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=chi_sq,
        alpha=alpha,
        details={"n": float(n), "block_size": float(block_size), "n_blocks": float(n_blocks)},
    )


# --------------------------------------------------------------------------- #
# §2.3 Runs Test
# --------------------------------------------------------------------------- #
def runs_test(
    sequence: Sequence[int] | NDArray[np.integer], *, alpha: float = _DEFAULT_ALPHA
) -> NistTestResult:
    """NIST SP 800-22 §2.3 Runs Test."""
    bits = as_bits(sequence)
    n = bits.size
    pi = float(np.mean(bits))
    tau = 2.0 / np.sqrt(n)
    if abs(pi - 0.5) >= tau:
        # Frequency precondition fails; the runs test is not applicable.
        return NistTestResult(
            name="runs",
            p_value=0.0,
            passed=False,
            statistic=float("nan"),
            alpha=alpha,
            details={"n": float(n), "pi": pi, "tau": float(tau), "precondition_failed": 1.0},
        )
    v_obs = int(np.sum(bits[:-1] != bits[1:])) + 1
    numerator = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
    denominator = 2.0 * np.sqrt(2.0 * n) * pi * (1.0 - pi)
    p_value = float(erfc(numerator / denominator))
    return NistTestResult(
        name="runs",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=float(v_obs),
        alpha=alpha,
        details={"n": float(n), "pi": pi, "v_obs": float(v_obs)},
    )


# --------------------------------------------------------------------------- #
# §2.4 Test for the Longest Run of Ones in a Block
# --------------------------------------------------------------------------- #
# Block size M and category probabilities per NIST SP 800-22 §2.4.
_LONGEST_RUN_PARAMS: tuple[tuple[int, int, int, int, tuple[float, ...]], ...] = (
    # (n_min, M, K, first_category_upper, probabilities)
    (128, 8, 3, 1, (0.2148, 0.3672, 0.2305, 0.1875)),
    (6272, 128, 5, 4, (0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124)),
    (
        750000,
        10000,
        6,
        10,
        (0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727),
    ),
)


def _longest_run_in_block(block: NDArray[np.int8]) -> int:
    longest = 0
    current = 0
    for value in block:
        if value == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def longest_run_of_ones_test(
    sequence: Sequence[int] | NDArray[np.integer], *, alpha: float = _DEFAULT_ALPHA
) -> NistTestResult:
    """NIST SP 800-22 §2.4 Test for the Longest Run of Ones in a Block."""
    bits = as_bits(sequence, minimum=128)
    n = bits.size
    params = None
    for n_min, m, k, first_upper, probs in _LONGEST_RUN_PARAMS:
        if n >= n_min:
            params = (m, k, first_upper, probs)
    if params is None:
        raise ValueError("sequence too short for the longest-run test (need >= 128 bits)")
    m, k, first_upper, probs = params
    n_blocks = n // m
    trimmed = bits[: n_blocks * m].reshape(n_blocks, m)
    # Bin the per-block longest run into K+1 categories clamped to [first_upper, first_upper+K].
    nu = np.zeros(k + 1, dtype=np.int64)
    for block in trimmed:
        longest = _longest_run_in_block(block)
        index = min(max(longest, first_upper), first_upper + k) - first_upper
        nu[index] += 1
    expected = n_blocks * np.asarray(probs, dtype=np.float64)
    chi_sq = float(np.sum((nu - expected) ** 2 / expected))
    p_value = float(gammaincc(k / 2.0, chi_sq / 2.0))
    return NistTestResult(
        name="longest_run_of_ones",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=chi_sq,
        alpha=alpha,
        details={
            "n": float(n),
            "block_size": float(m),
            "n_blocks": float(n_blocks),
            "k": float(k),
        },
    )


# --------------------------------------------------------------------------- #
# §2.6 Discrete Fourier Transform (Spectral) Test
# --------------------------------------------------------------------------- #
def dft_spectral_test(
    sequence: Sequence[int] | NDArray[np.integer], *, alpha: float = _DEFAULT_ALPHA
) -> NistTestResult:
    """NIST SP 800-22 §2.6 Discrete Fourier Transform (Spectral) Test."""
    bits = as_bits(sequence)
    n = bits.size
    x = 2.0 * bits.astype(np.float64) - 1.0
    magnitudes = np.abs(np.fft.fft(x))[: n // 2]
    threshold = np.sqrt(np.log(1.0 / 0.05) * n)
    n0 = 0.95 * n / 2.0
    n1 = float(np.sum(magnitudes < threshold))
    d = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4.0)
    p_value = float(erfc(abs(d) / np.sqrt(2.0)))
    return NistTestResult(
        name="dft_spectral",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=float(d),
        alpha=alpha,
        details={"n": float(n), "n0": float(n0), "n1": n1, "threshold": float(threshold)},
    )


# --------------------------------------------------------------------------- #
# §2.11 Serial Test
# --------------------------------------------------------------------------- #
def _psi_sq_m(bits: NDArray[np.int8], m: int) -> float:
    n = bits.size
    if m <= 0:
        return 0.0
    # Augment with the first m-1 bits to make the sequence circular.
    augmented = np.concatenate([bits, bits[: m - 1]]) if m > 1 else bits
    # Encode each overlapping m-bit window as an integer in [0, 2^m).
    windows = np.lib.stride_tricks.sliding_window_view(augmented, m)[:n]
    weights = (1 << np.arange(m - 1, -1, -1)).astype(np.int64)
    codes = windows.astype(np.int64) @ weights
    counts = np.bincount(codes, minlength=1 << m)
    return float((1 << m) / n * np.sum(counts.astype(np.float64) ** 2) - n)


def serial_test(
    sequence: Sequence[int] | NDArray[np.integer],
    *,
    block_size: int = 16,
    alpha: float = _DEFAULT_ALPHA,
) -> NistTestResult:
    """NIST SP 800-22 §2.11 Serial Test (returns the two P-values)."""
    bits = as_bits(sequence)
    m = block_size
    if not isinstance(m, int) or m < 2:
        raise ValueError("block_size must be an integer >= 2")
    psi_m = _psi_sq_m(bits, m)
    psi_m1 = _psi_sq_m(bits, m - 1)
    psi_m2 = _psi_sq_m(bits, m - 2)
    del_psi = psi_m - psi_m1
    del2_psi = psi_m - 2.0 * psi_m1 + psi_m2
    p_value1 = float(gammaincc(2.0 ** (m - 2), del_psi / 2.0))
    p_value2 = float(gammaincc(2.0 ** (m - 3), del2_psi / 2.0))
    return NistTestResult(
        name="serial",
        p_value=min(p_value1, p_value2),
        passed=_passed(p_value1, alpha) and _passed(p_value2, alpha),
        statistic=del_psi,
        alpha=alpha,
        p_values=(p_value1, p_value2),
        details={"n": float(bits.size), "block_size": float(m), "del_psi": del_psi},
    )


# --------------------------------------------------------------------------- #
# §2.12 Approximate Entropy Test
# --------------------------------------------------------------------------- #
def _phi_m(bits: NDArray[np.int8], m: int) -> float:
    n = bits.size
    if m == 0:
        return 0.0
    augmented = np.concatenate([bits, bits[: m - 1]]) if m > 1 else bits
    windows = np.lib.stride_tricks.sliding_window_view(augmented, m)[:n]
    weights = (1 << np.arange(m - 1, -1, -1)).astype(np.int64)
    codes = windows.astype(np.int64) @ weights
    counts = np.bincount(codes, minlength=1 << m).astype(np.float64)
    probs = counts[counts > 0] / n
    return float(np.sum(probs * np.log(probs)))


def approximate_entropy_test(
    sequence: Sequence[int] | NDArray[np.integer],
    *,
    block_size: int = 10,
    alpha: float = _DEFAULT_ALPHA,
) -> NistTestResult:
    """NIST SP 800-22 §2.12 Approximate Entropy Test."""
    bits = as_bits(sequence)
    n = bits.size
    m = block_size
    if not isinstance(m, int) or m < 1:
        raise ValueError("block_size must be a positive integer")
    ap_en = _phi_m(bits, m) - _phi_m(bits, m + 1)
    chi_sq = 2.0 * n * (np.log(2.0) - ap_en)
    p_value = float(gammaincc(2.0 ** (m - 1), chi_sq / 2.0))
    return NistTestResult(
        name="approximate_entropy",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=float(chi_sq),
        alpha=alpha,
        details={"n": float(n), "block_size": float(m), "ap_en": float(ap_en)},
    )


# --------------------------------------------------------------------------- #
# §2.13 Cumulative Sums (Cusum) Test
# --------------------------------------------------------------------------- #
def _normal_cdf(z: float) -> float:
    from scipy.special import ndtr

    return float(ndtr(z))


def _cusum_p_value(z: int, n: int) -> float:
    if z == 0:
        return 1.0
    sqrt_n = np.sqrt(n)
    lower1 = int((-n / z + 1) / 4.0)
    upper1 = int((n / z - 1) / 4.0)
    term1 = 0.0
    for k in range(lower1, upper1 + 1):
        term1 += _normal_cdf((4 * k + 1) * z / sqrt_n) - _normal_cdf((4 * k - 1) * z / sqrt_n)
    lower2 = int((-n / z - 3) / 4.0)
    upper2 = int((n / z - 1) / 4.0)
    term2 = 0.0
    for k in range(lower2, upper2 + 1):
        term2 += _normal_cdf((4 * k + 3) * z / sqrt_n) - _normal_cdf((4 * k + 1) * z / sqrt_n)
    return float(1.0 - term1 + term2)


def cumulative_sums_test(
    sequence: Sequence[int] | NDArray[np.integer],
    *,
    mode: str = "forward",
    alpha: float = _DEFAULT_ALPHA,
) -> NistTestResult:
    """NIST SP 800-22 §2.13 Cumulative Sums (Cusum) Test."""
    bits = as_bits(sequence)
    n = bits.size
    if mode not in ("forward", "backward"):
        raise ValueError("mode must be 'forward' or 'backward'")
    x = 2 * bits.astype(np.int64) - 1
    walk = np.cumsum(x if mode == "forward" else x[::-1])
    z = int(np.max(np.abs(walk)))
    p_value = _cusum_p_value(z, n)
    return NistTestResult(
        name=f"cumulative_sums_{mode}",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=float(z),
        alpha=alpha,
        details={"n": float(n), "z": float(z)},
    )


# --------------------------------------------------------------------------- #
# §2.5 Binary Matrix Rank Test
# --------------------------------------------------------------------------- #
def _gf2_rank(matrix: NDArray[np.int8]) -> int:
    """Rank of a binary matrix over GF(2) via forward elimination."""
    m = matrix.copy().astype(np.int8)
    rows, cols = m.shape
    rank = 0
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        pivot = pivot_row + int(np.argmax(m[pivot_row:, col]))
        if m[pivot, col] == 0:
            continue
        if pivot != pivot_row:
            m[[pivot_row, pivot]] = m[[pivot, pivot_row]]
        mask = m[:, col].astype(bool).copy()
        mask[pivot_row] = False
        m[mask] ^= m[pivot_row]
        rank += 1
        pivot_row += 1
    return rank


# Probabilities for a 32x32 GF(2) matrix having rank 32, 31, and <= 30
# (NIST SP 800-22 §2.5; M = Q = 32).
_MATRIX_RANK_PROBS = (0.2888, 0.5776, 0.1336)


def binary_matrix_rank_test(
    sequence: Sequence[int] | NDArray[np.integer],
    *,
    rows: int = 32,
    cols: int = 32,
    alpha: float = _DEFAULT_ALPHA,
) -> NistTestResult:
    """NIST SP 800-22 §2.5 Binary Matrix Rank Test."""
    bits = as_bits(sequence, minimum=rows * cols)
    block = rows * cols
    n_matrices = bits.size // block
    if n_matrices < 1:
        raise ValueError("sequence too short for one matrix")
    full = rows  # full rank
    counts = [0, 0, 0]  # rank == full, rank == full-1, rank <= full-2
    for i in range(n_matrices):
        chunk = bits[i * block : (i + 1) * block].reshape(rows, cols)
        rank = _gf2_rank(chunk)
        if rank == full:
            counts[0] += 1
        elif rank == full - 1:
            counts[1] += 1
        else:
            counts[2] += 1
    expected = [p * n_matrices for p in _MATRIX_RANK_PROBS]
    chi_sq = float(sum((c - e) ** 2 / e for c, e in zip(counts, expected, strict=True)))
    p_value = float(np.exp(-chi_sq / 2.0))
    return NistTestResult(
        name="binary_matrix_rank",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=chi_sq,
        alpha=alpha,
        details={"n_matrices": float(n_matrices), "f_full": float(counts[0])},
    )


# --------------------------------------------------------------------------- #
# §2.7 Non-overlapping Template Matching Test
# --------------------------------------------------------------------------- #
def non_overlapping_template_test(
    sequence: Sequence[int] | NDArray[np.integer],
    *,
    template: Sequence[int] = (0, 0, 0, 0, 0, 0, 0, 0, 1),
    n_blocks: int = 8,
    alpha: float = _DEFAULT_ALPHA,
) -> NistTestResult:
    """NIST SP 800-22 §2.7 Non-overlapping Template Matching Test (single template)."""
    bits = as_bits(sequence)
    pattern = np.asarray(template, dtype=np.int8)
    m = pattern.size
    if m < 1:
        raise ValueError("template must be non-empty")
    n = bits.size
    block_len = n // n_blocks
    if block_len <= m:
        raise ValueError("block length must exceed the template length")
    mu = (block_len - m + 1) / 2.0**m
    var = block_len * (1.0 / 2.0**m - (2.0 * m - 1.0) / 2.0 ** (2 * m))
    counts = np.empty(n_blocks, dtype=np.float64)
    for b in range(n_blocks):
        block = bits[b * block_len : (b + 1) * block_len]
        hits = 0
        pos = 0
        limit = block_len - m
        while pos <= limit:
            if np.array_equal(block[pos : pos + m], pattern):
                hits += 1
                pos += m
            else:
                pos += 1
        counts[b] = hits
    chi_sq = float(np.sum((counts - mu) ** 2) / var)
    p_value = float(gammaincc(n_blocks / 2.0, chi_sq / 2.0))
    return NistTestResult(
        name="non_overlapping_template",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=chi_sq,
        alpha=alpha,
        details={"n": float(n), "m": float(m), "n_blocks": float(n_blocks), "mu": mu},
    )


# --------------------------------------------------------------------------- #
# §2.8 Overlapping Template Matching Test
# --------------------------------------------------------------------------- #
# Theoretical category probabilities for the default m=9 all-ones template
# (NIST SP 800-22 §2.8; K = 5, M = 1032).
_OVERLAPPING_TEMPLATE_PROBS = (
    0.364091,
    0.185659,
    0.139381,
    0.100571,
    0.070432,
    0.139865,
)


def overlapping_template_test(
    sequence: Sequence[int] | NDArray[np.integer],
    *,
    template_length: int = 9,
    block_len: int = 1032,
    alpha: float = _DEFAULT_ALPHA,
) -> NistTestResult:
    """NIST SP 800-22 §2.8 Overlapping Template Matching Test (all-ones template)."""
    bits = as_bits(sequence, minimum=block_len)
    m = template_length
    n_blocks = bits.size // block_len
    if n_blocks < 1:
        raise ValueError("sequence too short for one block")
    nu = np.zeros(6, dtype=np.int64)
    for b in range(n_blocks):
        block = bits[b * block_len : (b + 1) * block_len]
        windows = np.lib.stride_tricks.sliding_window_view(block, m)
        hits = int(np.sum(np.all(windows == 1, axis=1)))
        nu[min(hits, 5)] += 1
    expected = n_blocks * np.asarray(_OVERLAPPING_TEMPLATE_PROBS, dtype=np.float64)
    chi_sq = float(np.sum((nu - expected) ** 2 / expected))
    p_value = float(gammaincc(5 / 2.0, chi_sq / 2.0))
    return NistTestResult(
        name="overlapping_template",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=chi_sq,
        alpha=alpha,
        details={"n_blocks": float(n_blocks), "m": float(m)},
    )


# --------------------------------------------------------------------------- #
# §2.9 Maurer's Universal Statistical Test
# --------------------------------------------------------------------------- #
# Expected value and variance of the test statistic per block length L
# (NIST SP 800-22 §2.9, Table from Appendix).
_MAURER_TABLE: dict[int, tuple[float, float]] = {
    6: (5.2177052, 2.954),
    7: (6.1962507, 3.125),
    8: (7.1836656, 3.238),
    9: (8.1764248, 3.311),
    10: (9.1723243, 3.356),
    11: (10.170032, 3.384),
    12: (11.168765, 3.401),
    13: (12.168070, 3.410),
    14: (13.167693, 3.416),
    15: (14.167488, 3.419),
    16: (15.167379, 3.421),
}


def maurers_universal_test(
    sequence: Sequence[int] | NDArray[np.integer], *, alpha: float = _DEFAULT_ALPHA
) -> NistTestResult:
    """NIST SP 800-22 §2.9 Maurer's Universal Statistical Test."""
    bits = as_bits(sequence, minimum=1010 * 6)
    n = bits.size
    # Choose L per the NIST recommendation; require n >= (Q + K) * L with Q = 10*2^L.
    block_length = 6
    for candidate in range(16, 5, -1):
        q_blocks = 10 * 2**candidate
        if n >= (q_blocks + 1000) * candidate:
            block_length = candidate
            break
    l_len = block_length
    q_blocks = 10 * 2**l_len
    total_blocks = n // l_len
    k_blocks = total_blocks - q_blocks
    if k_blocks <= 0:
        raise ValueError("sequence too short for Maurer's universal test")
    weights = (1 << np.arange(l_len - 1, -1, -1)).astype(np.int64)
    codes = (bits[: total_blocks * l_len].reshape(total_blocks, l_len).astype(np.int64)) @ weights
    last_seen = np.zeros(1 << l_len, dtype=np.int64)
    for i in range(q_blocks):
        last_seen[codes[i]] = i + 1
    total = 0.0
    for i in range(q_blocks, total_blocks):
        total += np.log2((i + 1) - last_seen[codes[i]])
        last_seen[codes[i]] = i + 1
    fn = total / k_blocks
    expected, variance = _MAURER_TABLE[l_len]
    c = 0.7 - 0.8 / l_len + (4.0 + 32.0 / l_len) * k_blocks ** (-3.0 / l_len) / 15.0
    sigma = c * np.sqrt(variance / k_blocks)
    p_value = float(erfc(abs((fn - expected) / (np.sqrt(2.0) * sigma))))
    return NistTestResult(
        name="maurers_universal",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=float(fn),
        alpha=alpha,
        details={"n": float(n), "L": float(l_len), "K": float(k_blocks)},
    )


# --------------------------------------------------------------------------- #
# §2.10 Linear Complexity Test
# --------------------------------------------------------------------------- #
def berlekamp_massey(bits: NDArray[np.int8]) -> int:
    """Length of the shortest LFSR generating ``bits`` (Berlekamp-Massey over GF(2)).

    Dispatches to the Rust kernel when the acceleration engine is installed; the
    pure-Python implementation is the bit-identical fallback.
    """
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "nist_berlekamp_massey"):
            return int(_engine.nist_berlekamp_massey(np.ascontiguousarray(bits, dtype=np.int8)))
    except (ImportError, AttributeError, ValueError):
        pass
    return _berlekamp_massey_python(bits)


def _berlekamp_massey_python(bits: NDArray[np.int8]) -> int:
    n = bits.size
    c = np.zeros(n, dtype=np.int8)
    b = np.zeros(n, dtype=np.int8)
    c[0] = 1
    b[0] = 1
    length = 0
    m = -1
    for i in range(n):
        discrepancy = bits[i]
        for j in range(1, length + 1):
            discrepancy ^= c[j] & bits[i - j]
        if discrepancy:
            t = c.copy()
            shift = i - m
            if shift < n:
                c[shift:] ^= b[: n - shift]
            if 2 * length <= i:
                length = i + 1 - length
                m = i
                b = t
    return length


# Category probabilities for the linear-complexity test (NIST SP 800-22 §2.10, K = 6).
_LINEAR_COMPLEXITY_PROBS = (
    0.010417,
    0.03125,
    0.125,
    0.5,
    0.25,
    0.0625,
    0.020833,
)


def linear_complexity_test(
    sequence: Sequence[int] | NDArray[np.integer],
    *,
    block_size: int = 500,
    alpha: float = _DEFAULT_ALPHA,
) -> NistTestResult:
    """NIST SP 800-22 §2.10 Linear Complexity Test."""
    bits = as_bits(sequence, minimum=block_size)
    m = block_size
    n_blocks = bits.size // m
    if n_blocks < 1:
        raise ValueError("sequence too short for one block")
    sign = (-1.0) ** m
    mu = m / 2.0 + (9.0 + (-1.0) ** (m + 1)) / 36.0 - (m / 3.0 + 2.0 / 9.0) / 2.0**m
    nu = np.zeros(7, dtype=np.int64)
    for b in range(n_blocks):
        block = bits[b * m : (b + 1) * m]
        complexity = berlekamp_massey(block)
        t = sign * (complexity - mu) + 2.0 / 9.0
        if t <= -2.5:
            nu[0] += 1
        elif t <= -1.5:
            nu[1] += 1
        elif t <= -0.5:
            nu[2] += 1
        elif t <= 0.5:
            nu[3] += 1
        elif t <= 1.5:
            nu[4] += 1
        elif t <= 2.5:
            nu[5] += 1
        else:
            nu[6] += 1
    expected = n_blocks * np.asarray(_LINEAR_COMPLEXITY_PROBS, dtype=np.float64)
    chi_sq = float(np.sum((nu - expected) ** 2 / expected))
    p_value = float(gammaincc(6 / 2.0, chi_sq / 2.0))
    return NistTestResult(
        name="linear_complexity",
        p_value=p_value,
        passed=_passed(p_value, alpha),
        statistic=chi_sq,
        alpha=alpha,
        details={"n_blocks": float(n_blocks), "block_size": float(m), "mu": mu},
    )


# --------------------------------------------------------------------------- #
# §2.14 / §2.15 Random Excursions (and Variant) Tests
# --------------------------------------------------------------------------- #
def _excursion_cycles(bits: NDArray[np.int8]) -> tuple[NDArray[np.int64], int]:
    x = 2 * bits.astype(np.int64) - 1
    walk = np.concatenate([[0], np.cumsum(x), [0]])
    zero_positions = np.flatnonzero(walk == 0)
    n_cycles = zero_positions.size - 1
    return walk, n_cycles


def random_excursions_test(
    sequence: Sequence[int] | NDArray[np.integer], *, alpha: float = _DEFAULT_ALPHA
) -> NistTestResult:
    """NIST SP 800-22 §2.14 Random Excursions Test (eight P-values)."""
    bits = as_bits(sequence, minimum=1000)
    walk, n_cycles = _excursion_cycles(bits)
    states = (-4, -3, -2, -1, 1, 2, 3, 4)
    if n_cycles == 0:
        raise ValueError("no cycles found; sequence unsuitable for random-excursions test")
    zero_positions = np.flatnonzero(walk == 0)
    # Per-cycle visit counts to each state.
    visit_counts = {state: np.zeros(n_cycles, dtype=np.int64) for state in states}
    for c in range(n_cycles):
        segment = walk[zero_positions[c] : zero_positions[c + 1] + 1]
        for state in states:
            visit_counts[state][c] = int(np.sum(segment == state))
    p_values: list[float] = []
    min_p = 1.0
    for state in states:
        x = abs(state)
        # Theoretical probabilities pi_k(x) for k = 0..>=5 visits in a cycle.
        pi0 = 1.0 - 1.0 / (2.0 * x)
        pis = [pi0]
        for k in range(1, 5):
            pis.append(1.0 / (4.0 * x**2) * (1.0 - 1.0 / (2.0 * x)) ** (k - 1))
        pis.append(1.0 / (2.0 * x) * (1.0 - 1.0 / (2.0 * x)) ** 4)
        nu = np.zeros(6, dtype=np.int64)
        for count in visit_counts[state]:
            nu[min(int(count), 5)] += 1
        expected = n_cycles * np.asarray(pis, dtype=np.float64)
        chi_sq = float(np.sum((nu - expected) ** 2 / expected))
        p = float(gammaincc(5 / 2.0, chi_sq / 2.0))
        p_values.append(p)
        min_p = min(min_p, p)
    return NistTestResult(
        name="random_excursions",
        p_value=min_p,
        passed=all(_passed(p, alpha) for p in p_values),
        statistic=float(n_cycles),
        alpha=alpha,
        p_values=tuple(p_values),
        details={"n_cycles": float(n_cycles)},
    )


def random_excursions_variant_test(
    sequence: Sequence[int] | NDArray[np.integer], *, alpha: float = _DEFAULT_ALPHA
) -> NistTestResult:
    """NIST SP 800-22 §2.15 Random Excursions Variant Test (eighteen P-values)."""
    bits = as_bits(sequence, minimum=1000)
    walk, n_cycles = _excursion_cycles(bits)
    if n_cycles == 0:
        raise ValueError("no cycles found; sequence unsuitable for the variant test")
    states = [s for s in range(-9, 10) if s != 0]
    p_values: list[float] = []
    min_p = 1.0
    for state in states:
        xi = int(np.sum(walk == state))
        p = float(erfc(abs(xi - n_cycles) / np.sqrt(2.0 * n_cycles * (4.0 * abs(state) - 2.0))))
        p_values.append(p)
        min_p = min(min_p, p)
    return NistTestResult(
        name="random_excursions_variant",
        p_value=min_p,
        passed=all(_passed(p, alpha) for p in p_values),
        statistic=float(n_cycles),
        alpha=alpha,
        p_values=tuple(p_values),
        details={"n_cycles": float(n_cycles)},
    )


__all__ = [
    "NistTestResult",
    "as_bits",
    "approximate_entropy_test",
    "berlekamp_massey",
    "binary_matrix_rank_test",
    "block_frequency_test",
    "cumulative_sums_test",
    "dft_spectral_test",
    "frequency_test",
    "linear_complexity_test",
    "longest_run_of_ones_test",
    "maurers_universal_test",
    "non_overlapping_template_test",
    "overlapping_template_test",
    "random_excursions_test",
    "random_excursions_variant_test",
    "runs_test",
    "serial_test",
]
