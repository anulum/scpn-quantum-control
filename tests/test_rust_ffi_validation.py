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


def test_mc_xy_simulate_rejects_non_contiguous_couplings() -> None:
    k_flat = np.linspace(0.0, 1.0, 32, dtype=np.float64)[::2]

    with pytest.raises(ValueError, match="k_flat must be a C-contiguous NumPy array"):
        engine.mc_xy_simulate(k_flat, 4, 0.1, 2, 2, 42)


def test_mc_xy_simulate_rejects_wrong_coupling_shape() -> None:
    k_flat = np.ascontiguousarray(np.zeros(15, dtype=np.float64))

    with pytest.raises(ValueError, match="k_flat length 15 != 4² = 16"):
        engine.mc_xy_simulate(k_flat, 4, 0.1, 2, 2, 42)


def test_mc_xy_simulate_rejects_non_finite_coupling() -> None:
    k_flat = np.ascontiguousarray(np.zeros((4, 4), dtype=np.float64).ravel())
    k_flat[3] = np.inf

    with pytest.raises(ValueError, match=r"k_flat\[3\] is not finite"):
        engine.mc_xy_simulate(k_flat, 4, 0.1, 2, 2, 42)


def test_mc_xy_simulate_rejects_negative_coupling() -> None:
    k_flat = np.ascontiguousarray(np.zeros((4, 4), dtype=np.float64).ravel())
    k_flat[1] = -0.2

    with pytest.raises(ValueError, match=r"k_flat\[1\] must be non-negative"):
        engine.mc_xy_simulate(k_flat, 4, 0.1, 2, 2, 42)


def test_mc_xy_simulate_rejects_zero_measurements() -> None:
    k_flat = np.ascontiguousarray(np.zeros((4, 4), dtype=np.float64).ravel())

    with pytest.raises(ValueError, match="n_measure must be > 0"):
        engine.mc_xy_simulate(k_flat, 4, 0.1, 2, 0, 42)


def test_mc_xy_simulate_accepts_zero_coupling_reference() -> None:
    k_flat = np.ascontiguousarray(np.zeros((4, 4), dtype=np.float64).ravel())

    energy, order, helicity = engine.mc_xy_simulate(k_flat, 4, 0.1, 2, 3, 42)

    assert energy == pytest.approx(0.0)
    assert 0.0 <= order <= 1.0
    assert helicity == pytest.approx(0.0)


def test_hierarchical_prediction_error_rejects_non_contiguous_observations() -> None:
    observations = np.linspace(0.0, 1.0, 8, dtype=np.float64)[::2]
    beliefs = np.ascontiguousarray(np.zeros(4, dtype=np.float64))
    k = np.ascontiguousarray(np.eye(4, dtype=np.float64))

    with pytest.raises(ValueError, match="observations must be a C-contiguous NumPy array"):
        engine.hierarchical_prediction_error_rust(observations, beliefs, k)


def test_hierarchical_prediction_error_rejects_length_mismatch() -> None:
    observations = np.ascontiguousarray(np.zeros(4, dtype=np.float64))
    beliefs = np.ascontiguousarray(np.zeros(3, dtype=np.float64))
    k = np.ascontiguousarray(np.eye(4, dtype=np.float64))

    with pytest.raises(ValueError, match="beliefs length 3 != observations length 4"):
        engine.hierarchical_prediction_error_rust(observations, beliefs, k)


def test_hierarchical_prediction_error_rejects_wrong_k_shape() -> None:
    observations = np.ascontiguousarray(np.zeros(4, dtype=np.float64))
    beliefs = np.ascontiguousarray(np.zeros(4, dtype=np.float64))
    k = np.ascontiguousarray(np.zeros((4, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="k shape 4x3 != observations length 4 squared"):
        engine.hierarchical_prediction_error_rust(observations, beliefs, k)


def test_hierarchical_prediction_error_rejects_non_finite_belief() -> None:
    observations = np.ascontiguousarray(np.zeros(4, dtype=np.float64))
    beliefs = np.ascontiguousarray(np.zeros(4, dtype=np.float64))
    beliefs[2] = np.nan
    k = np.ascontiguousarray(np.eye(4, dtype=np.float64))

    with pytest.raises(ValueError, match=r"beliefs\[2\] is not finite"):
        engine.hierarchical_prediction_error_rust(observations, beliefs, k)


def test_hierarchical_prediction_error_matches_isolated_and_coupled_rows() -> None:
    observations = np.ascontiguousarray([1.0, 2.0], dtype=np.float64)
    beliefs = np.ascontiguousarray([0.5, 1.0], dtype=np.float64)
    k = np.ascontiguousarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)

    errors = engine.hierarchical_prediction_error_rust(observations, beliefs, k)

    np.testing.assert_allclose(np.asarray(errors), np.array([0.5, 3.0]))


def test_brute_mpc_rejects_non_contiguous_b_matrix() -> None:
    b_flat = np.linspace(0.0, 1.0, 8, dtype=np.float64)[::2]
    target = np.ascontiguousarray([1.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="b_flat must be a C-contiguous NumPy array"):
        engine.brute_mpc(b_flat, target, 2, 2)


def test_brute_mpc_rejects_wrong_b_matrix_shape() -> None:
    b_flat = np.ascontiguousarray(np.zeros(5, dtype=np.float64))
    target = np.ascontiguousarray([1.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="b_flat length 5 != 2² = 4"):
        engine.brute_mpc(b_flat, target, 2, 2)


def test_brute_mpc_rejects_wrong_target_length() -> None:
    b_flat = np.ascontiguousarray(np.eye(2, dtype=np.float64).ravel())
    target = np.ascontiguousarray([1.0, 0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="target length 3 != dim 2"):
        engine.brute_mpc(b_flat, target, 2, 2)


def test_brute_mpc_rejects_non_finite_target() -> None:
    b_flat = np.ascontiguousarray(np.eye(2, dtype=np.float64).ravel())
    target = np.ascontiguousarray([1.0, np.nan], dtype=np.float64)

    with pytest.raises(ValueError, match=r"target\[1\] is not finite"):
        engine.brute_mpc(b_flat, target, 2, 2)


def test_brute_mpc_matches_two_dimensional_reference() -> None:
    b_flat = np.ascontiguousarray(np.eye(2, dtype=np.float64).ravel())
    target = np.ascontiguousarray([0.8, 0.6], dtype=np.float64)

    actions, cost, costs, n_evaluated = engine.brute_mpc(b_flat, target, 2, 3)

    assert n_evaluated == 8
    np.testing.assert_array_equal(np.asarray(actions), np.zeros(3, dtype=np.int64))
    assert cost == pytest.approx(1.0 / 3.0)
    assert np.asarray(costs).shape == (8,)


def test_analog_coupling_terms_rejects_non_contiguous_couplings() -> None:
    k_flat = np.linspace(0.0, 1.0, 32, dtype=np.float64)[::2]

    with pytest.raises(ValueError, match="k_flat must be a C-contiguous NumPy array"):
        engine.analog_coupling_terms(k_flat, 4, 0, 1.0, 64.0, 1e-12)


def test_analog_coupling_terms_rejects_wrong_coupling_shape() -> None:
    k_flat = np.ascontiguousarray(np.zeros(15, dtype=np.float64))

    with pytest.raises(ValueError, match="k_flat length 15 != 4² = 16"):
        engine.analog_coupling_terms(k_flat, 4, 0, 1.0, 64.0, 1e-12)


def test_analog_coupling_terms_preserves_signed_neutral_atom_edges() -> None:
    k_matrix = np.ascontiguousarray(
        [
            [0.0, 0.5, -0.25],
            [0.5, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    rows, cols, strengths, phases, radii = engine.analog_coupling_terms(
        np.ascontiguousarray(k_matrix.ravel()),
        3,
        0,
        1.0,
        64.0,
        1e-12,
    )

    np.testing.assert_array_equal(np.asarray(rows), np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(cols), np.array([1, 2], dtype=np.int64))
    np.testing.assert_allclose(np.asarray(strengths), np.array([0.5, 0.25]))
    np.testing.assert_allclose(np.asarray(phases), np.array([0.0, np.pi]))
    np.testing.assert_allclose(np.asarray(radii), (64.0 / np.array([0.5, 0.25])) ** (1.0 / 6.0))


def test_hybrid_coupling_partition_rejects_non_contiguous_couplings() -> None:
    k_flat = np.linspace(0.0, 1.0, 32, dtype=np.float64)[::2]

    with pytest.raises(ValueError, match="k_flat must be a C-contiguous NumPy array"):
        engine.hybrid_coupling_partition(k_flat, 4, 1, 0.0, 1e-12)


def test_hybrid_coupling_partition_selects_budgeted_tie_break_edge() -> None:
    k_matrix = np.ascontiguousarray(
        [
            [0.0, 0.75, 0.75],
            [0.75, 0.0, 0.5],
            [0.75, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    analog, digital, rows, cols, route_codes = engine.hybrid_coupling_partition(
        np.ascontiguousarray(k_matrix.ravel()),
        3,
        1,
        0.0,
        1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(analog).reshape(3, 3),
        np.array([[0.0, 0.75, 0.0], [0.75, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(digital).reshape(3, 3),
        np.array([[0.0, 0.0, 0.75], [0.0, 0.0, 0.5], [0.75, 0.5, 0.0]]),
    )
    np.testing.assert_array_equal(np.asarray(rows), np.array([0, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(cols), np.array([1, 2, 2], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(route_codes), np.array([1, 0, 0], dtype=np.int64))


def r_values_error_pattern() -> str:
    return r"r_values must be a C-contiguous NumPy array"
