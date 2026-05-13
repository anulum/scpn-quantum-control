# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Rust FFI Boundary Validation Tests

from __future__ import annotations

import numpy as np
import pytest

engine = pytest.importorskip("scpn_quantum_engine")


def test_feedback_policy_batch_rejects_non_contiguous_r_values() -> None:
    r_values = np.linspace(0.1, 0.9, 8, dtype=np.float64)[::2]

    with pytest.raises(ValueError, match=r_values_error_pattern()):
        engine.feedback_policy_batch(
            r_values,
            target_r=0.8,
            deadband=0.05,
            base_gain=0.5,
            max_gain=2.0,
        )


def test_feedback_policy_batch_accepts_contiguous_r_values() -> None:
    r_values = np.ascontiguousarray([0.2, 0.8, 0.95], dtype=np.float64)

    actions, gains, errors = engine.feedback_policy_batch(
        r_values,
        target_r=0.8,
        deadband=0.05,
        base_gain=0.5,
        max_gain=2.0,
    )

    np.testing.assert_array_equal(np.asarray(actions), np.array([1, 0, -1]))
    np.testing.assert_allclose(np.asarray(errors), np.array([0.6, 0.0, -0.15]))
    assert np.asarray(gains)[0] > 1.0
    assert np.asarray(gains)[1] == pytest.approx(1.0)
    assert 0.0 < np.asarray(gains)[2] < 1.0


def test_guess_extrapolate_batch_rejects_non_contiguous_symmetry_values() -> None:
    target = np.ascontiguousarray([0.5, 0.6, 0.7], dtype=np.float64)
    symmetry = np.linspace(4.0, 2.0, 6, dtype=np.float64)[::2]

    with pytest.raises(ValueError, match="symmetry_noisy must be a C-contiguous NumPy array"):
        engine.guess_extrapolate_batch(target, symmetry, 4.0, 0.2)


def test_guess_extrapolate_batch_rejects_length_mismatch() -> None:
    target = np.ascontiguousarray([0.5, 0.6], dtype=np.float64)
    symmetry = np.ascontiguousarray([4.0, 3.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="target_noisy length 2 != symmetry_noisy length 3"):
        engine.guess_extrapolate_batch(target, symmetry, 4.0, 0.2)


def test_guess_extrapolate_batch_rejects_non_finite_alpha() -> None:
    target = np.ascontiguousarray([0.5], dtype=np.float64)
    symmetry = np.ascontiguousarray([4.0], dtype=np.float64)

    with pytest.raises(ValueError, match="alpha must be finite"):
        engine.guess_extrapolate_batch(target, symmetry, 4.0, np.nan)


def test_guess_extrapolate_batch_preserves_zero_symmetry_fallback() -> None:
    target = np.ascontiguousarray([0.5, 0.6], dtype=np.float64)
    symmetry = np.ascontiguousarray([4.0, 0.0], dtype=np.float64)

    mitigated = engine.guess_extrapolate_batch(target, symmetry, 4.0, 0.5)

    np.testing.assert_allclose(np.asarray(mitigated), np.array([0.5, 0.6]))


def test_fit_symmetry_decay_rejects_non_contiguous_noise_scales() -> None:
    noisy = np.ascontiguousarray([4.0, 3.2, 2.56], dtype=np.float64)
    scales = np.linspace(1.0, 5.0, 6, dtype=np.float64)[::2]

    with pytest.raises(ValueError, match="noise_scales must be a C-contiguous NumPy array"):
        engine.fit_symmetry_decay(4.0, noisy, scales)


def test_fit_symmetry_decay_rejects_too_few_points() -> None:
    noisy = np.ascontiguousarray([4.0], dtype=np.float64)
    scales = np.ascontiguousarray([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="at least 2 points"):
        engine.fit_symmetry_decay(4.0, noisy, scales)


def test_fit_symmetry_decay_recovers_exact_exponential() -> None:
    s_ideal = 4.0
    alpha_true = 0.15
    scales = np.ascontiguousarray([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    noisy = np.ascontiguousarray(
        s_ideal * np.exp(-alpha_true * (scales - 1.0)),
        dtype=np.float64,
    )

    alpha, residual = engine.fit_symmetry_decay(s_ideal, noisy, scales)

    assert alpha == pytest.approx(alpha_true, abs=1e-10)
    assert residual < 1e-10


def test_concatenated_logical_rate_rejects_non_contiguous_distances() -> None:
    distances = np.arange(1, 8, dtype=np.int64)[::2]

    with pytest.raises(ValueError, match="distances must be a C-contiguous NumPy array"):
        engine.concatenated_logical_rate_rust(0.003, distances, 0.01, 0.1)


def test_concatenated_logical_rate_rejects_non_positive_distances() -> None:
    distances = np.ascontiguousarray([3, 0, 5], dtype=np.int64)

    with pytest.raises(ValueError, match=r"distances\[1\] must be at least 1"):
        engine.concatenated_logical_rate_rust(0.003, distances, 0.01, 0.1)


def test_concatenated_logical_rate_accepts_d1_contract() -> None:
    distances = np.ascontiguousarray([1], dtype=np.int64)

    rates = engine.concatenated_logical_rate_rust(0.003, distances, 0.01, 0.1)

    np.testing.assert_allclose(np.asarray(rates), np.array([0.03]))


def test_concatenated_logical_rate_matches_two_level_reference() -> None:
    distances = np.ascontiguousarray([3, 5], dtype=np.int64)

    rates = np.asarray(engine.concatenated_logical_rate_rust(0.003, distances, 0.01, 0.1))

    first = 0.1 * (0.003 / 0.01) ** 2
    second = 0.1 * (first / 0.01) ** 3
    np.testing.assert_allclose(rates, np.array([first, second]))


def test_score_regions_batch_rejects_non_contiguous_gate_errors() -> None:
    gate_errors = np.linspace(0.0, 0.03, 32, dtype=np.float64)[::2]
    offsets = np.ascontiguousarray([0, 4], dtype=np.int64)
    qubits = np.ascontiguousarray([0, 1, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError, match="gate_errors_flat must be a C-contiguous NumPy array"):
        engine.score_regions_batch(gate_errors, 4, offsets, qubits)


def test_score_regions_batch_rejects_wrong_gate_error_shape() -> None:
    gate_errors = np.ascontiguousarray(np.zeros(15, dtype=np.float64))
    offsets = np.ascontiguousarray([0, 4], dtype=np.int64)
    qubits = np.ascontiguousarray([0, 1, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError, match="gate_errors_flat length 15 != n_qubits² = 16"):
        engine.score_regions_batch(gate_errors, 4, offsets, qubits)


def test_score_regions_batch_rejects_invalid_gate_error_value() -> None:
    gate_errors = np.ascontiguousarray(np.zeros((4, 4), dtype=np.float64).ravel())
    gate_errors[1] = 1.2
    offsets = np.ascontiguousarray([0, 4], dtype=np.int64)
    qubits = np.ascontiguousarray([0, 1, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError, match=r"gate_errors_flat\[1\] must be in \[0, 1\]"):
        engine.score_regions_batch(gate_errors, 4, offsets, qubits)


def test_score_regions_batch_rejects_odd_region_offsets() -> None:
    gate_errors = np.ascontiguousarray(np.zeros((4, 4), dtype=np.float64).ravel())
    offsets = np.ascontiguousarray([0, 4, 4], dtype=np.int64)
    qubits = np.ascontiguousarray([0, 1, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError, match="region_offsets length 3 must be even"):
        engine.score_regions_batch(gate_errors, 4, offsets, qubits)


def test_score_regions_batch_rejects_offset_beyond_qubit_buffer() -> None:
    gate_errors = np.ascontiguousarray(np.zeros((4, 4), dtype=np.float64).ravel())
    offsets = np.ascontiguousarray([0, 5], dtype=np.int64)
    qubits = np.ascontiguousarray([0, 1, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError, match="end 5 exceeds region_qubits length 4"):
        engine.score_regions_batch(gate_errors, 4, offsets, qubits)


def test_score_regions_batch_rejects_out_of_range_qubit() -> None:
    gate_errors = np.ascontiguousarray(np.zeros((4, 4), dtype=np.float64).ravel())
    offsets = np.ascontiguousarray([0, 4], dtype=np.int64)
    qubits = np.ascontiguousarray([0, 1, 2, 4], dtype=np.int64)

    with pytest.raises(ValueError, match="region_qubits\\[3\\]=4 exceeds n_qubits=4"):
        engine.score_regions_batch(gate_errors, 4, offsets, qubits)


def test_score_regions_batch_accepts_complete_graph_scores() -> None:
    n_qubits = 4
    gate_errors = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    gate_errors[~np.eye(n_qubits, dtype=bool)] = 0.01
    offsets = np.ascontiguousarray([0, 4], dtype=np.int64)
    qubits = np.ascontiguousarray([0, 1, 2, 3], dtype=np.int64)

    connectivity, fidelity, composite = engine.score_regions_batch(
        np.ascontiguousarray(gate_errors.ravel()),
        n_qubits,
        offsets,
        qubits,
    )

    np.testing.assert_allclose(np.asarray(connectivity), np.array([1.0]))
    np.testing.assert_allclose(np.asarray(fidelity), np.array([0.99]))
    np.testing.assert_allclose(np.asarray(composite), np.array([0.99]))


def r_values_error_pattern() -> str:
    return r"r_values must be a C-contiguous NumPy array"
