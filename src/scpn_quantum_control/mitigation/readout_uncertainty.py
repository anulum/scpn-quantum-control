# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Readout calibration uncertainty
"""Preserve signed readout estimates and propagate finite calibration uncertainty.

The model has independent multinomial preparations and measurement batches,
with one shared calibration across extrapolation scales. Covariances use the
first-order delta method at empirical frequencies. They exclude calibration
drift, preparation bias and model error; zero empirical variance is not proof
of certainty. Legacy clipped probability APIs retain their separate meaning.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..dense_budget import DenseAllocationError, dense_budget_bytes, require_dense_allocation
from ._readout_svd import admit_readout_svd, readout_svd_condition
from .readout_matrix import ReadoutConfusionMatrix, counts_to_probabilities, label_index_map
from .zne_uncertainty import ZNEUncertaintyResult, zne_extrapolate_with_uncertainty


@dataclass(frozen=True)
class ReadoutMitigationEstimate:
    """Signed inverse estimate with separate sampling and calibration covariance.

    Parameters
    ----------
    labels
        Original measured-basis label order, including a declared permutation.
    quasiprobabilities
        Dimensionless inverse estimate in that order, without clipping.
    sampling_covariance
        Dimensionless squared covariance from the measurement multinomial.
    calibration_covariance
        Dimensionless squared delta covariance from independent calibration columns.
    shots
        Exact total measurement count.
    condition_number
        Measured spectral condition of the admitted calibration matrix.

    Notes
    -----
    Tuple snapshots preserve values without aliasing caller arrays. Covariances
    are empirical approximations, not finite-sample confidence guarantees.

    """

    labels: tuple[str, ...]
    quasiprobabilities: tuple[float, ...]
    sampling_covariance: tuple[tuple[float, ...], ...]
    calibration_covariance: tuple[tuple[float, ...], ...]
    shots: int
    condition_number: float


@dataclass(frozen=True)
class ReadoutZNEResult:
    """Readout estimates and covariance consumed by the actual extrapolator.

    Parameters
    ----------
    readout_estimates
        Signed per-scale estimates in noise-scale input order.
    observation_covariance
        Full scalar-observable covariance, retaining shared-calibration correlations.
    extrapolation
        Existing normal/delta GLS result for those observations and covariance.

    """

    readout_estimates: tuple[ReadoutMitigationEstimate, ...]
    observation_covariance: tuple[tuple[float, ...], ...]
    extrapolation: ZNEUncertaintyResult


def _freeze_matrix(matrix: NDArray[np.float64]) -> tuple[tuple[float, ...], ...]:
    """Copy a numerical matrix into immutable scalar tuples.

    Parameters
    ----------
    matrix
        Two-dimensional finite covariance.

    Returns
    -------
    tuple of tuple of float
        Independent row-major values.

    """
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _reciprocal_shots(value: object) -> float:
    """Admit an exact positive shot count with representable inverse.

    Parameters
    ----------
    value
        Python or NumPy integer count.

    Returns
    -------
    float
        Positive reciprocal used in multinomial covariance.

    Raises
    ------
    ValueError
        If counts are invalid or their reciprocal underflows float64.

    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("calibration and measurement shots must be positive integers")
    if value <= 0:
        raise ValueError("calibration and measurement shots must be positive")
    reciprocal = 1 / int(value)
    if reciprocal == 0:
        raise ValueError("shot-count uncertainty is not representable in float64")
    return reciprocal


def _prepare_calibration(
    calibration: ReadoutConfusionMatrix,
    max_condition_number: float,
    max_dense_gib: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Admit the existing measured-basis matrix and its calibration sample sizes.

    Parameters
    ----------
    calibration
        Existing full measured-basis calibration and column counts.
    max_condition_number
        Finite accepted spectral condition ceiling.
    max_dense_gib
        Declared numerical-buffer allowance in GiB.

    Returns
    -------
    tuple
        Matrix, inverse, reciprocal column counts and recomputed condition.

    Raises
    ------
    ValueError
        If the real matrix, labels, shot counts or conditioning are inadmissible.
    DenseAllocationError
        If declared matrix buffers exceed the allowance.
    numpy.linalg.LinAlgError
        If the admitted numerical decomposition fails.

    """
    admitted = require_dense_allocation(
        calibration.n_qubits,
        dtype=np.float64,
        object_count=12,
        max_gib=max_dense_gib,
        label="readout uncertainty buffers",
    )
    labels = tuple(label_index_map(calibration.labels))
    if len(labels) != admitted.dimension or any(
        len(label) != calibration.n_qubits for label in labels
    ):
        raise ValueError("uncertainty requires a full measured-basis calibration")
    if np.iscomplexobj(calibration.matrix):
        raise ValueError("calibration must be a real column-stochastic matrix")
    matrix = np.asarray(calibration.matrix, dtype=np.float64)
    if (
        matrix.shape != (admitted.dimension, admitted.dimension)
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
        or not np.allclose(matrix.sum(axis=0), 1.0, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("calibration must be a finite column-stochastic square matrix")
    if (
        isinstance(max_condition_number, (bool, np.bool_))
        or not isinstance(max_condition_number, (int, float, np.integer, np.floating))
        or not math.isfinite(max_condition_number)
        or max_condition_number < 1
    ):
        raise ValueError("max_condition_number must be finite and at least one")
    if set(calibration.shots_by_prepared_state) != set(labels):
        raise ValueError("calibration shots must identify every prepared label exactly")
    reciprocal = np.array(
        [_reciprocal_shots(calibration.shots_by_prepared_state[label]) for label in labels]
    )
    work = admit_readout_svd(calibration.n_qubits, max_dense_gib)
    condition = readout_svd_condition(matrix, work)
    if not math.isfinite(condition) or condition > max_condition_number:
        raise ValueError("calibration condition exceeds the admitted ceiling")
    inverse = np.asarray(np.linalg.solve(matrix, np.eye(admitted.dimension)), dtype=np.float64)
    return matrix, inverse, reciprocal, condition


def _estimate_counts(
    counts: Mapping[str, int],
    calibration: ReadoutConfusionMatrix,
    matrix: NDArray[np.float64],
    inverse: NDArray[np.float64],
    reciprocal: NDArray[np.float64],
    condition: float,
) -> ReadoutMitigationEstimate:
    """Propagate one actual count vector through the admitted inverse.

    Parameters
    ----------
    counts
        Original unmodified observations.
    calibration
        Existing matrix carrying the authoritative label order.
    matrix
        Admitted column-stochastic matrix.
    inverse
        Actual matrix inverse.
    reciprocal
        Reciprocal independent calibration-column counts.
    condition
        Recomputed spectral condition.

    Returns
    -------
    ReadoutMitigationEstimate
        Immutable signed estimate and covariance components.

    Raises
    ------
    ValueError
        If observations are invalid or propagated values are not finite.

    """
    observed = counts_to_probabilities(counts, calibration.labels)
    shots = sum(int(value) for value in counts.values())
    inverse_shots = _reciprocal_shots(shots)
    with np.errstate(over="ignore", invalid="ignore"):
        quasi = inverse @ observed
        sampling = inverse @ (np.diag(observed) - np.outer(observed, observed)) @ inverse.T
        sampling *= inverse_shots
        weights = quasi**2 * reciprocal
        calibration_noise = np.diag(matrix @ weights) - (matrix * weights) @ matrix.T
        calibration_covariance = inverse @ calibration_noise @ inverse.T
    if not all(np.all(np.isfinite(value)) for value in (quasi, sampling, calibration_covariance)):
        raise ValueError("propagated readout uncertainty is not finite")
    return ReadoutMitigationEstimate(
        labels=tuple(calibration.labels),
        quasiprobabilities=tuple(float(value) for value in quasi),
        sampling_covariance=_freeze_matrix(sampling),
        calibration_covariance=_freeze_matrix(calibration_covariance),
        shots=shots,
        condition_number=condition,
    )


def mitigate_counts_with_uncertainty(
    counts: Mapping[str, int],
    confusion_matrix: ReadoutConfusionMatrix,
    *,
    max_condition_number: float = 1e8,
    max_dense_gib: float | None = None,
) -> ReadoutMitigationEstimate:
    """Correct readout without clipping and retain both empirical error components.

    Parameters
    ----------
    counts
        Nonnegative integer observations in the measured register's label vocabulary.
    confusion_matrix
        Existing full measured-basis calibration with exact per-preparation counts.
        Its columns and rows use its own declared label order.
    max_condition_number
        Accepted spectral condition ceiling; the cached condition is not trusted.
    max_dense_gib
        Numerical-buffer allowance in GiB, or the current process budget.

    Returns
    -------
    ReadoutMitigationEstimate
        Signed inverse estimate and separate multinomial sampling/calibration
        delta covariances. Raw observations and calibration are never modified.

    Raises
    ------
    ValueError
        If inputs, conditioning or float64 propagation are inadmissible.
    DenseAllocationError
        If declared numerical buffers exceed the allowance.
    numpy.linalg.LinAlgError
        If an admitted numerical decomposition fails.

    Notes
    -----
    Assumes independent calibration preparations and an independent measurement
    batch under the same stationary response. It does not model calibration drift,
    preparation bias, arbitrary correlated shots or a state-specific parity fit.
    Negative inverse components remain visible and need not be probabilities.

    """
    matrix, inverse, reciprocal, condition = _prepare_calibration(
        confusion_matrix, max_condition_number, max_dense_gib
    )
    return _estimate_counts(counts, confusion_matrix, matrix, inverse, reciprocal, condition)


def zne_extrapolate_with_readout_uncertainty(
    noise_scales: Sequence[float],
    counts_by_scale: Sequence[Mapping[str, int]],
    confusion_matrix: ReadoutConfusionMatrix,
    observable_weights: Sequence[float],
    *,
    order: int = 1,
    coverage: float = 0.95,
    max_condition_number: float = 1e8,
    max_dense_gib: float | None = None,
) -> ReadoutZNEResult:
    """Extrapolate corrected observables with their shared-calibration covariance.

    Parameters
    ----------
    noise_scales
        Distinct amplification factors in the existing extrapolator's domain.
    counts_by_scale
        Independent measurement batches in noise-scale order.
    confusion_matrix
        One shared stationary calibration with independent prepared-state batches.
    observable_weights
        Finite real scalar observable values aligned with the calibration labels.
    order
        Polynomial degree of the existing extrapolator.
    coverage
        Existing normal/delta interval's nominal probability in (0, 1).
    max_condition_number
        Accepted calibration condition ceiling.
    max_dense_gib
        Numerical-buffer allowance in GiB, or the current process budget.

    Returns
    -------
    ReadoutZNEResult
        Signed per-scale estimates, complete observation covariance and actual GLS
        output. Correlated calibration error is not treated as independent shot noise.

    Raises
    ------
    ValueError
        If inputs, conditioning, propagation or positive-definite covariance fail.
    DenseAllocationError
        If declared retained and extrapolation buffers exceed the allowance.
    numpy.linalg.LinAlgError
        If an admitted numerical decomposition fails.

    Notes
    -----
    Uses first-order multinomial delta covariance at empirical frequencies and
    the original normal interval definition. It excludes model mismatch and
    calibration drift; extrapolation can worsen a result. Singular empirical
    covariance refuses rather than manufacturing a confidence interval.

    """
    if not counts_by_scale or len(counts_by_scale) != len(noise_scales):
        raise ValueError("counts_by_scale must match nonempty noise_scales")
    matrix, inverse, reciprocal, condition = _prepare_calibration(
        confusion_matrix, max_condition_number, max_dense_gib
    )
    points, dimension = len(counts_by_scale), len(confusion_matrix.labels)
    required_bytes = 8 * (
        12 * points**2 + (12 + 6 * points) * dimension**2 + 8 * points * dimension
    )
    if required_bytes > dense_budget_bytes(max_dense_gib):
        raise DenseAllocationError("readout extrapolation buffers exceed the active dense budget")
    if np.iscomplexobj(observable_weights):
        raise ValueError("observable_weights must be real")
    observable = np.asarray(observable_weights, dtype=np.float64)
    if observable.shape != (dimension,) or not np.all(np.isfinite(observable)):
        raise ValueError("observable_weights must be finite and aligned with calibration labels")
    estimates = tuple(
        _estimate_counts(counts, confusion_matrix, matrix, inverse, reciprocal, condition)
        for counts in counts_by_scale
    )
    quasi = np.array([estimate.quasiprobabilities for estimate in estimates])
    projected_inverse = inverse.T @ observable
    column_variance = (
        projected_inverse**2 @ matrix - (projected_inverse @ matrix) ** 2
    ) * reciprocal
    covariance = (quasi * column_variance) @ quasi.T
    sampling_variance = np.array(
        [
            float(observable @ np.asarray(estimate.sampling_covariance) @ observable)
            for estimate in estimates
        ]
    )
    covariance += np.diag(sampling_variance)
    # The analytical covariance is symmetric; remove only multiplication-order roundoff.
    covariance = (covariance + covariance.T) / 2
    means = quasi @ observable
    extrapolation = zne_extrapolate_with_uncertainty(
        noise_scales,
        [float(value) for value in means],
        covariance=covariance,
        order=order,
        coverage=coverage,
        max_dense_gib=max_dense_gib,
    )
    return ReadoutZNEResult(
        readout_estimates=estimates,
        observation_covariance=_freeze_matrix(covariance),
        extrapolation=extrapolation,
    )
