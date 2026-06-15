# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the QRNG streaming harness (QUA-C.1)
"""Tests for the entropy package: NIST SP 800-22, FIPS 140-2, quantum QRNG."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.entropy import (
    AerQuantumEntropySource,
    QRNGStream,
    enforce_fips_140_2,
    von_neumann_debias,
)
from scpn_quantum_control.entropy import fips_140_2 as F
from scpn_quantum_control.entropy import nist_sp800_22 as N

try:
    import scpn_quantum_engine as _engine

    _HAS_RUST = hasattr(_engine, "nist_berlekamp_massey")
except ImportError:  # pragma: no cover - engine optional
    _engine = None
    _HAS_RUST = False


def _bits(text: str) -> np.ndarray:
    return np.array([int(c) for c in text], dtype=np.int8)


def _random_bits(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, size=n).astype(np.int8)


# --------------------------------------------------------------------------- #
# NIST SP 800-22 worked examples (Rev 1a) — exact P-value reproduction
# --------------------------------------------------------------------------- #
def test_frequency_worked_example():
    assert N.frequency_test(_bits("1011010101")).p_value == pytest.approx(0.527089, abs=1e-6)


def test_block_frequency_worked_example():
    r = N.block_frequency_test(_bits("0110011010"), block_size=3)
    assert r.p_value == pytest.approx(0.801252, abs=1e-6)


def test_runs_worked_example():
    # Hand-verified for this sequence: pi=0.6, V_obs=7 -> p = 0.147232.
    assert N.runs_test(_bits("1001101011")).p_value == pytest.approx(0.147232, abs=1e-6)


def test_serial_worked_example():
    r = N.serial_test(_bits("0011011101"), block_size=3)
    assert r.p_values[0] == pytest.approx(0.808792, abs=1e-6)
    assert r.p_values[1] == pytest.approx(0.670320, abs=1e-6)


def test_approximate_entropy_worked_example():
    r = N.approximate_entropy_test(_bits("0100110101"), block_size=3)
    assert r.p_value == pytest.approx(0.261961, abs=1e-6)


def test_cumulative_sums_worked_example():
    r = N.cumulative_sums_test(_bits("1011010111"), mode="forward")
    assert r.p_value == pytest.approx(0.4116588, abs=1e-6)


# --------------------------------------------------------------------------- #
# NIST sub-components
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "seq,expected",
    [
        ("10000000", 1),
        ("11111111", 1),
        ("1110100" * 3, 3),  # maximal LFSR x^3 + x + 1, complexity 3
    ],
)
def test_berlekamp_massey_known(seq, expected):
    assert N.berlekamp_massey(_bits(seq)) == expected


def test_gf2_rank():
    assert N._gf2_rank(np.eye(4, dtype=np.int8)) == 4
    assert N._gf2_rank(np.ones((4, 4), dtype=np.int8)) == 1


# --------------------------------------------------------------------------- #
# NIST behaviour on good and bad sources
# --------------------------------------------------------------------------- #
def test_full_suite_passes_on_cryptographic_random():
    bits = _random_bits(1_000_000, seed=20260615)
    suite = [
        N.frequency_test(bits),
        N.block_frequency_test(bits, block_size=128),
        N.runs_test(bits),
        N.longest_run_of_ones_test(bits),
        N.binary_matrix_rank_test(bits),
        N.dft_spectral_test(bits),
        N.non_overlapping_template_test(bits),
        N.overlapping_template_test(bits),
        N.maurers_universal_test(bits),
        N.linear_complexity_test(bits),
        N.serial_test(bits, block_size=16),
        N.approximate_entropy_test(bits, block_size=10),
        N.cumulative_sums_test(bits),
        N.random_excursions_test(bits),
        N.random_excursions_variant_test(bits),
    ]
    assert len(suite) == 15
    assert all(r.passed for r in suite), {r.name: r.p_value for r in suite if not r.passed}


def test_frequency_rejects_biased():
    biased = (np.random.default_rng(1).random(100_000) < 0.65).astype(np.int8)
    assert N.frequency_test(biased).p_value < 0.01


def test_approximate_entropy_rejects_periodic():
    periodic = np.tile([1, 1, 0, 0], 25_000).astype(np.int8)
    assert N.approximate_entropy_test(periodic, block_size=10).p_value < 0.01


@pytest.mark.parametrize(
    "func", [N.frequency_test, N.runs_test, lambda b: N.block_frequency_test(b, block_size=2)]
)
def test_rejects_non_binary_input(func):
    with pytest.raises(ValueError):
        func(np.array([0, 2, 1], dtype=np.int8))


# --------------------------------------------------------------------------- #
# Rust ↔ Python parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine NIST kernels not built")
@settings(max_examples=40, deadline=None)
@given(data=st.lists(st.integers(0, 1), min_size=1, max_size=600))
def test_berlekamp_massey_rust_parity(data):
    bits = np.array(data, dtype=np.int8)
    rust = int(_engine.nist_berlekamp_massey(np.ascontiguousarray(bits)))
    python = N._berlekamp_massey_python(bits)
    assert rust == python


@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine NIST kernels not built")
def test_monobit_and_runs_rust_parity():
    rng = np.random.default_rng(5)
    for _ in range(20):
        bits = rng.integers(0, 2, size=int(rng.integers(2, 5000))).astype(np.int8)
        s = int(np.sum(2 * bits.astype(np.int64) - 1))
        assert int(_engine.nist_monobit_sum(np.ascontiguousarray(bits))) == s
        ones, v_obs = _engine.nist_runs_counts(np.ascontiguousarray(bits))
        assert ones == int(np.sum(bits))
        assert v_obs == int(np.sum(bits[:-1] != bits[1:])) + 1


# --------------------------------------------------------------------------- #
# FIPS 140-2 Annex C
# --------------------------------------------------------------------------- #
def test_fips_passes_on_random():
    bits = _random_bits(F.FIPS_SAMPLE_BITS, seed=3)
    assert F.fips_140_2_tests(bits).passed


def test_fips_rejects_biased():
    biased = (np.random.default_rng(2).random(F.FIPS_SAMPLE_BITS) < 0.6).astype(np.int8)
    with pytest.raises(RuntimeError, match="FIPS 140-2"):
        enforce_fips_140_2(biased)


def test_fips_requires_exact_length():
    with pytest.raises(ValueError, match="exactly"):
        F.fips_140_2_tests(_random_bits(19_999, seed=1))


def test_fips_long_run_rejected():
    bits = _random_bits(F.FIPS_SAMPLE_BITS, seed=9).copy()
    bits[100:130] = 1  # inject a length-30 run (>= 26)
    assert F.fips_140_2_tests(bits).long_run_pass is False


# --------------------------------------------------------------------------- #
# Von Neumann debiasing
# --------------------------------------------------------------------------- #
def test_von_neumann_debias():
    bits = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int8)
    assert von_neumann_debias(bits).tolist() == [0, 1, 1]


def test_von_neumann_unbiases_biased_source():
    rng = np.random.default_rng(4)
    biased = (rng.random(400_000) < 0.7).astype(np.int8)
    out = von_neumann_debias(biased)
    assert abs(out.mean() - 0.5) < 0.01


# --------------------------------------------------------------------------- #
# Quantum entropy source + QRNGStream (Aer)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind,register",
    [("xy_measurement", 32), ("bell_pair", 32), ("phase_estimation", 8)],
)
def test_quantum_source_unbiased(kind, register):
    src = AerQuantumEntropySource(kind, register_qubits=register, seed=17)
    bits = src.sample_bits(20_000)
    assert bits.size == 20_000
    assert set(np.unique(bits).tolist()) <= {0, 1}
    assert abs(bits.mean() - 0.5) < 0.05


def test_phase_estimation_register_capped():
    with pytest.raises(ValueError, match="capped"):
        AerQuantumEntropySource("phase_estimation", register_qubits=16)


def test_qrng_stream_exact_length_and_dtype():
    qrng = QRNGStream("xy_measurement", register_qubits=64, debias=True, seed=2026)
    assert qrng.sample(0).size == 0
    bits = qrng.sample(5_000)
    assert bits.size == 5_000
    assert bits.dtype == np.uint8
    assert set(np.unique(bits).tolist()) <= {0, 1}


def test_qrng_stream_health_check_healthy():
    qrng = QRNGStream("bell_pair", register_qubits=64, debias=True, seed=42)
    report = qrng.health_check(F.FIPS_SAMPLE_BITS)
    assert report.fips.passed
    assert report.healthy
    assert report.shannon_entropy_per_bit > 0.999
    assert report.min_entropy_per_bit > 0.99


def test_qrng_stream_iterates():
    qrng = QRNGStream("xy_measurement", register_qubits=64, debias=False, seed=1)
    chunks = [next(qrng.stream(1_000)) for _ in range(3)]
    assert all(c.size == 1_000 for c in chunks)
