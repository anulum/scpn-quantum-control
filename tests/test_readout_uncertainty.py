# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Readout uncertainty qualification
"""Qualify signed readout covariance through the public production API."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import binomtest

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.mitigation import (
    ReadoutConfusionMatrix,
    ReadoutMitigationEstimate,
    ReadoutZNEResult,
    build_readout_confusion_matrix,
    mitigate_counts_with_uncertainty,
    zne_extrapolate_with_readout_uncertainty,
)


@pytest.fixture
def calibration() -> ReadoutConfusionMatrix:
    """Build an actual two-state finite-shot calibration."""
    return build_readout_confusion_matrix(
        {"0": {"0": 900, "1": 100}, "1": {"0": 200, "1": 800}}, 1
    )


def test_signed_covariance_matches_closed_bernoulli_derivatives(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """Compare both covariance components with independent scalar derivatives."""
    result = mitigate_counts_with_uncertainty({"0": 410, "1": 590}, calibration)
    q0 = (0.41 - 0.2) / 0.7
    sampling = 0.41 * 0.59 / (1000 * 0.7**2)
    calibration_variance = (q0**2 * 0.9 * 0.1 + (1 - q0) ** 2 * 0.2 * 0.8) / (1000 * 0.7**2)
    sign_covariance = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert isinstance(result, ReadoutMitigationEstimate)
    np.testing.assert_allclose(result.quasiprobabilities, [q0, 1 - q0], atol=1e-14)
    np.testing.assert_allclose(result.sampling_covariance, sign_covariance * sampling, atol=1e-15)
    np.testing.assert_allclose(
        result.calibration_covariance, sign_covariance * calibration_variance, atol=1e-15
    )
    assert result.shots == 1000
    assert result.condition_number > 1
    calibration.matrix[:] = 0
    assert result.quasiprobabilities == pytest.approx((0.3, 0.7))
    assert isinstance(result.sampling_covariance, tuple)


def test_identity_preserves_observed_expectation_and_sampling_variance() -> None:
    """Identity calibration leaves counts unchanged and has zero empirical fit variance."""
    identity = build_readout_confusion_matrix({"0": {"0": 20}, "1": {"1": 30}}, 1)
    counts = cast(dict[str, int], {"0": np.int64(6), "1": np.int64(4)})
    result = mitigate_counts_with_uncertainty(counts, identity)
    np.testing.assert_array_equal(result.quasiprobabilities, [0.6, 0.4])
    np.testing.assert_array_equal(result.calibration_covariance, np.zeros((2, 2)))
    np.testing.assert_allclose(result.sampling_covariance, [[0.024, -0.024], [-0.024, 0.024]])


def test_more_independent_calibration_shots_reduce_only_calibration_error(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """Double preparation shots at fixed frequencies halves their covariance alone."""
    original = mitigate_counts_with_uncertainty({"0": 410, "1": 590}, calibration)
    doubled = replace(calibration, shots_by_prepared_state={"0": 2000, "1": 2000})
    result = mitigate_counts_with_uncertainty({"0": 410, "1": 590}, doubled)
    np.testing.assert_array_equal(result.quasiprobabilities, original.quasiprobabilities)
    np.testing.assert_array_equal(result.sampling_covariance, original.sampling_covariance)
    np.testing.assert_allclose(
        result.calibration_covariance, np.asarray(original.calibration_covariance) / 2
    )


def test_complex_calibration_refuses_imaginary_data_loss(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """Refuse complex calibration rather than silently discarding its imaginary part."""
    matrix = calibration.matrix.astype(np.complex128) + 0.1j
    with pytest.raises(ValueError, match="real"):
        mitigate_counts_with_uncertainty(
            {"0": 1, "1": 1}, replace(calibration, matrix=cast(Any, matrix))
        )


def test_complex_observable_refuses_imaginary_data_loss(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """A scalar real-observable model cannot discard a complex component."""
    observable = np.array([1 + 0.1j, -1], dtype=np.complex128)
    with pytest.raises(ValueError, match="real"):
        zne_extrapolate_with_readout_uncertainty(
            [1, 3], [{"0": 410, "1": 590}] * 2, calibration, cast(Any, observable)
        )


@pytest.mark.parametrize(
    "matrix",
    [
        np.ones((2, 3)),
        np.array([[np.nan, 0.2], [0.1, 0.8]]),
        np.array([[-0.1, 0.2], [1.1, 0.8]]),
        np.array([[0.8, 0.2], [0.1, 0.8]]),
    ],
)
def test_malformed_calibration_matrix_refuses_without_mutation(
    calibration: ReadoutConfusionMatrix, matrix: NDArray[np.float64]
) -> None:
    """Reject invalid matrices instead of normalising or replacing caller evidence."""
    broken = replace(calibration, matrix=matrix)
    before = matrix.copy()
    with pytest.raises(ValueError, match="column-stochastic"):
        mitigate_counts_with_uncertainty({"0": 1, "1": 1}, broken)
    np.testing.assert_array_equal(matrix, before)


def test_partial_calibration_and_duplicate_labels_refuse(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """A full measured basis and distinct original labels are mandatory."""
    with pytest.raises(ValueError, match="full measured-basis"):
        mitigate_counts_with_uncertainty({"0": 1}, replace(calibration, n_qubits=2))
    with pytest.raises(ValueError, match="not repeat"):
        mitigate_counts_with_uncertainty({"0": 1}, replace(calibration, labels=("0", "0")))


def test_label_width_cannot_relabel_a_subset_as_full_basis(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """Two selected two-bit states cannot impersonate a full one-qubit calibration."""
    subset = replace(
        calibration, labels=("00", "11"), shots_by_prepared_state={"00": 1000, "11": 1000}
    )
    with pytest.raises(ValueError, match="full measured-basis"):
        mitigate_counts_with_uncertainty({"00": 50, "11": 50}, subset)


@pytest.mark.parametrize("ceiling", [True, np.bool_(True), "2", np.inf, np.nan, 0.9])
def test_invalid_condition_ceiling_refuses(
    calibration: ReadoutConfusionMatrix, ceiling: object
) -> None:
    """Reject coercion, nonfinite values and condition ceilings below one."""
    with pytest.raises(ValueError, match="max_condition_number"):
        mitigate_counts_with_uncertainty(
            {"0": 1}, calibration, max_condition_number=cast(Any, ceiling)
        )


def test_cached_condition_is_not_an_admission_certificate(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """Recompute conditioning rather than trusting a caller's favourable cache."""
    with pytest.raises(ValueError, match="condition"):
        mitigate_counts_with_uncertainty(
            {"0": 1}, replace(calibration, condition_number=1.0), max_condition_number=1.1
        )


@pytest.mark.parametrize("shots", [False, np.bool_(True), 0, -1, 1.5, 10**400])
def test_invalid_or_unrepresentable_calibration_shots_refuse(
    calibration: ReadoutConfusionMatrix, shots: object
) -> None:
    """The finite-shot model requires exact positive representable sample sizes."""
    broken = replace(calibration, shots_by_prepared_state={"0": cast(Any, shots), "1": 1000})
    with pytest.raises(ValueError, match="shots|shot-count"):
        mitigate_counts_with_uncertainty({"0": 1}, broken)


def test_missing_prepared_shot_metadata_refuses(calibration: ReadoutConfusionMatrix) -> None:
    """Every full-basis column needs its own independent calibration sample size."""
    with pytest.raises(ValueError, match="every prepared label"):
        mitigate_counts_with_uncertainty(
            {"0": 1}, replace(calibration, shots_by_prepared_state={"0": 1000})
        )


@pytest.mark.parametrize("counts", [{}, {"0": -1}, {"2": 1}, {"0": True}, {"0": 1.5}])
def test_invalid_measurement_counts_refuse(
    calibration: ReadoutConfusionMatrix, counts: dict[str, Any]
) -> None:
    """Malformed observations never produce a corrected result or alter calibration."""
    original = calibration.matrix.copy()
    with pytest.raises(ValueError):
        mitigate_counts_with_uncertainty(counts, calibration)
    np.testing.assert_array_equal(calibration.matrix, original)


def test_measurement_reciprocal_underflow_refuses(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """Do not present an unrepresentable finite-shot variance as exact certainty."""
    with pytest.raises(ValueError, match="not representable"):
        mitigate_counts_with_uncertainty({"0": 10**400}, calibration)


def test_unrepresentable_propagation_refuses(calibration: ReadoutConfusionMatrix) -> None:
    """An explicitly admitted extreme matrix still cannot return nonfinite covariance."""
    extreme = replace(calibration, matrix=np.array([[1.0, 1.0], [0.0, 1e-155]]))
    with pytest.raises(ValueError, match="not finite"):
        mitigate_counts_with_uncertainty({"1": 100}, extreme, max_condition_number=1e200)


def test_dense_admission_precedes_exponential_calibration_allocation(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """Refuse an oversized declared measured basis before enumerating or solving it."""
    with pytest.raises(DenseAllocationError):
        mitigate_counts_with_uncertainty(
            {"0": 1}, replace(calibration, n_qubits=40), max_dense_gib=0.001
        )


@pytest.mark.parametrize("scales,counts", [([], []), ([1, 3], [{"0": 1}])])
def test_extrapolation_requires_one_measurement_batch_per_scale(
    calibration: ReadoutConfusionMatrix,
    scales: list[float],
    counts: list[dict[str, int]],
) -> None:
    """Reject missing scale batches before extrapolating."""
    with pytest.raises(ValueError, match="counts_by_scale"):
        zne_extrapolate_with_readout_uncertainty(scales, counts, calibration, [1.0, -1.0])


@pytest.mark.parametrize("observable", [[1.0], [1.0, np.inf]])
def test_observable_requires_finite_label_alignment(
    calibration: ReadoutConfusionMatrix, observable: list[float]
) -> None:
    """A malformed observable cannot silently reorder or discard measured states."""
    with pytest.raises(ValueError, match="observable_weights"):
        zne_extrapolate_with_readout_uncertainty(
            [1, 3], [{"0": 1, "1": 1}] * 2, calibration, observable
        )


def test_extrapolation_retained_buffers_are_admitted(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """A calibration fitting the allowance does not admit arbitrarily many scale buffers."""
    with pytest.raises(DenseAllocationError, match="extrapolation buffers"):
        zne_extrapolate_with_readout_uncertainty(
            [1, 3],
            [{"0": 1, "1": 1}] * 2,
            calibration,
            [1.0, -1.0],
            max_dense_gib=1300 / 2**30,
        )


def test_singular_empirical_observable_covariance_refuses() -> None:
    """Constant empirical observations provide no positive-definite delta covariance."""
    identity = build_readout_confusion_matrix({"0": {"0": 10}, "1": {"1": 10}}, 1)
    with pytest.raises(ValueError, match="positive definite"):
        zne_extrapolate_with_readout_uncertainty(
            [1, 3], [{"0": 100}, {"0": 100}], identity, [1.0, -1.0]
        )


def test_shared_calibration_result_retains_both_error_components(
    calibration: ReadoutConfusionMatrix,
) -> None:
    """The actual pipeline returns its signed estimates and full covariance."""
    result = zne_extrapolate_with_readout_uncertainty(
        [1, 3], [{"0": 410, "1": 590}, {"0": 550, "1": 450}], calibration, [1.0, -1.0]
    )
    assert isinstance(result, ReadoutZNEResult)
    assert len(result.readout_estimates) == 2
    assert result.observation_covariance[0][1] > 0
    for index, estimate in enumerate(result.readout_estimates):
        combined = np.asarray(estimate.sampling_covariance) + estimate.calibration_covariance
        expected = float(np.array([1.0, -1.0]) @ combined @ np.array([1.0, -1.0]))
        assert result.observation_covariance[index][index] == pytest.approx(expected)


def test_binomial_interval_coverage_retains_every_frozen_replicate(tmp_path: Path) -> None:
    """Report finite-shot coverage over all 200 predetermined independent seeds."""
    replicates = []
    covered = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        prepared_zero = int(rng.binomial(1000, 0.9))
        prepared_one = int(rng.binomial(1000, 0.2))
        calibration_counts = {
            "0": {"0": prepared_zero, "1": 1000 - prepared_zero},
            "1": {"0": prepared_one, "1": 1000 - prepared_one},
        }
        observations = []
        for probability in (0.62, 0.48):
            zeros = int(rng.binomial(1000, probability))
            observations.append({"0": zeros, "1": 1000 - zeros})
        calibration = build_readout_confusion_matrix(calibration_counts, 1)
        result = zne_extrapolate_with_readout_uncertainty(
            [1, 3], observations, calibration, [1.0, -1.0]
        ).extrapolation
        contains_truth = result.low <= 0.4 <= result.high
        covered += int(contains_truth)
        replicates.append(
            {
                "seed": seed,
                "calibration_counts": calibration_counts,
                "measurement_counts": observations,
                "estimate": result.zero_noise_estimate,
                "standard_error": result.standard_error,
                "interval": [result.low, result.high],
                "covered": contains_truth,
            }
        )
    interval = binomtest(covered, 200).proportion_ci(confidence_level=0.95, method="exact")
    report = {
        "model": "stationary independent multinomial; shared finite calibration",
        "seeds": "0..199 inclusive, all retained",
        "samples_per_independent_batch": 1000,
        "truth": 0.4,
        "nominal_coverage": 0.95,
        "observed_coverage": covered / 200,
        "observed_coverage_exact_binomial_interval": [float(interval.low), float(interval.high)],
        "diagnostic_minimum_covered": 180,
        "boundary": "Finite fixed-model diagnostic, not a universal coverage guarantee.",
        "replicates": replicates,
    }
    (tmp_path / "readout_interval_coverage.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    assert covered >= 180
