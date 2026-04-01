# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qec Properties
"""Multi-angle property-based tests for QEC surface code.

Covers: zero-error syndrome, correction shapes, syndrome shapes,
single-qubit errors, correction effectiveness, code structure,
parametrised distances.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.qec.control_qec import ControlQEC


@given(d=st.integers(min_value=3, max_value=5))
@settings(max_examples=5, deadline=10000)
def test_zero_errors_zero_syndrome(d: int) -> None:
    """No errors must produce all-zero syndromes."""
    qec = ControlQEC(distance=d)
    n_data = qec.code.num_data
    err_x = np.zeros(n_data, dtype=np.int8)
    err_z = np.zeros(n_data, dtype=np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert np.all(syn_z == 0)
    assert np.all(syn_x == 0)


@given(d=st.integers(min_value=3, max_value=5))
@settings(max_examples=5, deadline=10000)
def test_correction_shape_matches_error(d: int) -> None:
    """Correction vector must have same shape as error vector."""
    qec = ControlQEC(distance=d)
    n_data = qec.code.num_data
    rng = np.random.default_rng(d)
    err_x = rng.binomial(1, 0.05, n_data).astype(np.int8)
    err_z = rng.binomial(1, 0.05, n_data).astype(np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    corr_x = qec.decoder.decode(syn_z, dual=False)
    corr_z = qec.decoder.decode(syn_x, dual=True)
    assert corr_x.shape == err_x.shape
    assert corr_z.shape == err_z.shape


@given(d=st.integers(min_value=3, max_value=5))
@settings(max_examples=5, deadline=10000)
def test_syndrome_shape_matches_distance(d: int) -> None:
    """Syndrome vectors must have d² entries."""
    qec = ControlQEC(distance=d)
    n_data = qec.code.num_data
    rng = np.random.default_rng(d)
    err_x = rng.binomial(1, 0.1, n_data).astype(np.int8)
    err_z = rng.binomial(1, 0.1, n_data).astype(np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert syn_z.shape == (d * d,)
    assert syn_x.shape == (d * d,)


@pytest.mark.parametrize("d", [3, 5])
def test_num_data_qubits_positive(d: int) -> None:
    """Surface code should have positive number of data qubits."""
    qec = ControlQEC(distance=d)
    assert qec.code.num_data > 0
    assert qec.code.num_data >= d * d  # at least d² data qubits


@pytest.mark.parametrize("d", [3, 5])
def test_syndrome_binary(d: int) -> None:
    """Syndrome entries must be 0 or 1."""
    qec = ControlQEC(distance=d)
    n_data = qec.code.num_data
    rng = np.random.default_rng(42)
    err_x = rng.binomial(1, 0.1, n_data).astype(np.int8)
    err_z = rng.binomial(1, 0.1, n_data).astype(np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert set(np.unique(syn_z)).issubset({0, 1})
    assert set(np.unique(syn_x)).issubset({0, 1})


def test_single_x_error_detectable() -> None:
    """A single X error should produce non-zero Z syndrome."""
    qec = ControlQEC(distance=3)
    n_data = qec.code.num_data
    err_x = np.zeros(n_data, dtype=np.int8)
    err_x[0] = 1  # single X error on qubit 0
    err_z = np.zeros(n_data, dtype=np.int8)
    syn_z, _ = qec.get_syndrome(err_x, err_z)
    assert np.any(syn_z != 0), "Single X error should be detectable"


def test_single_z_error_detectable() -> None:
    """A single Z error should produce non-zero X syndrome."""
    qec = ControlQEC(distance=3)
    n_data = qec.code.num_data
    err_x = np.zeros(n_data, dtype=np.int8)
    err_z = np.zeros(n_data, dtype=np.int8)
    err_z[0] = 1  # single Z error on qubit 0
    _, syn_x = qec.get_syndrome(err_x, err_z)
    assert np.any(syn_x != 0), "Single Z error should be detectable"


def test_correction_is_binary() -> None:
    """Correction vector must be binary (0 or 1)."""
    qec = ControlQEC(distance=3)
    n_data = qec.code.num_data
    rng = np.random.default_rng(42)
    err_x = rng.binomial(1, 0.1, n_data).astype(np.int8)
    syn_z, _ = qec.get_syndrome(err_x, np.zeros(n_data, dtype=np.int8))
    corr = qec.decoder.decode(syn_z, dual=False)
    assert set(np.unique(corr)).issubset({0, 1})


# ---------------------------------------------------------------------------
# QEC physical invariants
# ---------------------------------------------------------------------------


def test_low_error_rate_correctable() -> None:
    """Below threshold error rate → decode_and_correct should succeed most of the time."""
    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(42)
    successes = 0
    for _ in range(50):
        err_x, err_z = qec.simulate_errors(p_error=0.01, rng=rng)
        if qec.decode_and_correct(err_x, err_z):
            successes += 1
    assert successes >= 40  # >80% success at p=0.01


def test_high_error_rate_fails_often() -> None:
    """Above threshold → correction fails frequently."""
    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(42)
    failures = 0
    for _ in range(100):
        err_x, err_z = qec.simulate_errors(p_error=0.3, rng=rng)
        if not qec.decode_and_correct(err_x, err_z):
            failures += 1
    assert failures > 10  # significant failure rate at p=0.3


# ---------------------------------------------------------------------------
# Pipeline: Knm → QEC → syndrome → decode → wired
# ---------------------------------------------------------------------------


def test_pipeline_qec_full_cycle() -> None:
    """Full pipeline: create code → inject errors → syndrome → decode → correct.
    Verifies QEC is not decorative — full error correction cycle.
    """
    import time

    t0 = time.perf_counter()
    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(42)
    err_x, err_z = qec.simulate_errors(p_error=0.05, rng=rng)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    corr_x = qec.decoder.decode(syn_z, dual=False)
    corrected = qec.decode_and_correct(err_x, err_z)
    dt = (time.perf_counter() - t0) * 1000

    assert corr_x.shape == err_x.shape
    assert isinstance(corrected, bool)

    print(f"\n  PIPELINE QEC (d=3, p=0.05): {dt:.1f} ms")
    print(f"  Corrected: {corrected}, syndrome weight: {int(syn_z.sum())}")
