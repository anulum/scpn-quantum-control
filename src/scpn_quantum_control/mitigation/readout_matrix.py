# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — readout matrix module
# scpn-quantum-control -- full-basis readout matrix mitigation
"""Full-basis readout confusion-matrix mitigation utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._readout_svd import admit_readout_svd, readout_svd_condition


@dataclass(frozen=True)
class ReadoutConfusionMatrix:
    """Column-stochastic readout matrix in computational-bitstring order.

    The matrix convention is ``observed_probabilities = matrix @ true_probabilities``.
    Columns are prepared basis states and rows are observed basis states.
    """

    n_qubits: int
    labels: tuple[str, ...]
    matrix: NDArray[np.float64]
    condition_number: float
    shots_by_prepared_state: dict[str, int]


def computational_basis_labels(n_qubits: int) -> tuple[str, ...]:
    """Return big-endian computational basis labels for ``n_qubits``."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    return tuple(format(index, f"0{n_qubits}b") for index in range(2**n_qubits))


def label_index_map(labels: Sequence[str]) -> dict[str, int]:
    """Return a unique label-to-position mapping for a readout label order.

    The position a label occupies in ``labels`` is authoritative. It coincides
    with the label's numeric value only for the canonical big-endian order, so a
    permuted or partial order must be resolved through this mapping rather than
    by parsing the bitstring.

    Parameters
    ----------
    labels
        Readout labels in the order the probability vector uses. Surrounding
        and embedded ASCII spaces are ignored, as they are in count keys.
        Labels must be nonempty binary strings of one common register width.
        Partial and permuted registers are supported.

    Returns
    -------
    dict of str to int
        Cleaned label to its position in ``labels``.

    Raises
    ------
    ValueError
        If labels are empty, nonbinary, mixed-width or duplicated after space
        removal. A repeated label cannot resolve to one position.

    """
    if not labels:
        raise ValueError("labels must not be empty")
    mapping: dict[str, int] = {}
    width: int | None = None
    for index, label in enumerate(labels):
        if not isinstance(label, str):
            raise ValueError("labels must be binary strings")
        clean = label.replace(" ", "")
        if not clean or any(bit not in "01" for bit in clean):
            raise ValueError("labels must be non-empty binary strings")
        if width is None:
            width = len(clean)
        elif len(clean) != width:
            raise ValueError("labels must have one common bitstring width")
        if clean in mapping:
            raise ValueError(f"labels must not repeat a bitstring: {clean!r}")
        mapping[clean] = index
    return mapping


def bitstring_index(bitstring: str, labels: Sequence[str] | None = None) -> int:
    """Return the position of ``bitstring`` in a readout label order.

    Parameters
    ----------
    bitstring
        Computational-basis outcome; embedded spaces are ignored.
    labels
        Label order to resolve against. When ``None`` the canonical big-endian
        numeric value is returned instead.

    Returns
    -------
    int
        Position within ``labels``, or the numeric value when ``labels`` is
        ``None``.

    Raises
    ------
    ValueError
        If ``labels`` repeats a bitstring, or does not contain ``bitstring``.

    """
    clean = bitstring.replace(" ", "")
    if labels is None:
        label_index_map((bitstring,))
        return int(clean, 2)
    mapping = label_index_map(labels)
    if clean not in mapping:
        raise ValueError(f"unknown bitstring label: {bitstring!r}")
    return mapping[clean]


def counts_to_probabilities(
    counts: Mapping[str, int],
    labels: Sequence[str],
) -> NDArray[np.float64]:
    """Convert a count dictionary into a probability vector over ``labels``.

    Each count is placed at the position its label occupies in ``labels``.
    Outcomes absent from ``counts`` stay zero; they are not shifted, and the
    label order is never re-derived from the bitstring value.

    Parameters
    ----------
    counts
        Non-negative integer observations keyed by computational-basis outcome.
        Booleans, numeric strings and floating-point values are rejected.
    labels
        Readout label order defining the returned vector's positions.

    Returns
    -------
    numpy.ndarray
        Probabilities over ``labels``, summing to one.

    Raises
    ------
    ValueError
        If ``labels`` is empty or repeats a bitstring, a count is negative, the
        counts are empty or total zero, or a count key is absent from ``labels``.

    """
    mapping = label_index_map(labels)
    total_count = 0
    for value in counts.values():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be non-negative integers")
        count = int(value)
        if count < 0:
            raise ValueError("counts must be non-negative")
        total_count += count
    if total_count <= 0:
        raise ValueError("empty count dictionary")
    probabilities = np.zeros(len(labels), dtype=np.float64)
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        if clean not in mapping:
            raise ValueError(f"count dictionary contains unknown bitstring {bitstring!r}")
        probabilities[mapping[clean]] += int(count) / total_count
    return probabilities


def build_readout_confusion_matrix(
    calibration_counts: Mapping[str, Mapping[str, int]],
    n_qubits: int,
    *,
    max_dense_gib: float | None = None,
) -> ReadoutConfusionMatrix:
    """Build a full-basis readout confusion matrix from calibration counts.

    ``calibration_counts`` must map prepared computational-basis labels to
    observed count dictionaries. Missing prepared states are rejected; this keeps
    the mitigation claim distinct from partial exact-state corrections.

    The matrix is full-basis, so it is ``2**n_qubits`` square. Admission is
    checked before the basis labels are enumerated, because that tuple is itself
    ``2**n_qubits`` entries long and would be the first exponential allocation
    to run.
    The condition-number SVD also admits its input copy, queried LAPACK work
    arrays and singular values before enumeration. This covers declared numeric
    buffers, not Python metadata or total process memory.

    Parameters
    ----------
    calibration_counts
        Prepared-state label to observed counts.
    n_qubits
        Number of measured qubits.
    max_dense_gib
        Optional dense-allocation budget in GiB. ``None`` uses the active
        process budget.

    Returns
    -------
    ReadoutConfusionMatrix
        The calibrated matrix with its label order and condition number.

    Raises
    ------
    DenseAllocationError
        If the matrix and declared SVD buffers exceed the budget or native limits.
    numpy.linalg.LinAlgError
        If the SVD workspace query fails or the decomposition does not converge.
    ValueError
        If a prepared computational-basis state has no calibration counts.

    """
    lwork = admit_readout_svd(n_qubits, max_dense_gib)
    labels = computational_basis_labels(n_qubits)
    missing = [label for label in labels if label not in calibration_counts]
    if missing:
        raise ValueError(f"missing calibration states: {', '.join(missing)}")

    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    shots_by_prepared_state: dict[str, int] = {}
    for column, prepared in enumerate(labels):
        counts = calibration_counts[prepared]
        probabilities = counts_to_probabilities(counts, labels)
        matrix[:, column] = probabilities
        shots_by_prepared_state[prepared] = int(sum(int(value) for value in counts.values()))

    return ReadoutConfusionMatrix(
        n_qubits=n_qubits,
        labels=labels,
        matrix=matrix,
        condition_number=readout_svd_condition(matrix, lwork),
        shots_by_prepared_state=shots_by_prepared_state,
    )


def mitigate_probabilities(
    observed_probabilities: NDArray[np.float64],
    confusion_matrix: ReadoutConfusionMatrix,
    *,
    rcond: float = 1e-10,
) -> NDArray[np.float64]:
    """Invert a readout matrix with clipping and renormalisation.

    Parameters
    ----------
    observed_probabilities
        Normalised non-negative vector in the calibration's label order.
    confusion_matrix
        Calibration whose rows and columns share a unique binary label order.
    rcond
        Relative singular-value cutoff passed to the pseudoinverse.

    Returns
    -------
    numpy.ndarray
        Clipped and normalised estimate in the same label order.

    Raises
    ------
    ValueError
        If labels or observed probabilities are invalid, or inversion leaves no mass.
    """
    label_index_map(confusion_matrix.labels)
    if observed_probabilities.shape != (len(confusion_matrix.labels),):
        raise ValueError("observed probability vector has incompatible shape")
    observed_total = float(np.sum(observed_probabilities))
    if (
        not np.all(np.isfinite(observed_probabilities))
        or np.any(observed_probabilities < 0.0)
        or observed_total <= 0.0
        or not np.isclose(observed_total, 1.0, rtol=1e-9, atol=1e-12)
    ):
        raise ValueError("observed probabilities must be finite, non-negative, and sum to one")
    raw = np.linalg.pinv(confusion_matrix.matrix, rcond=rcond) @ observed_probabilities
    clipped = np.clip(raw, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError("readout mitigation produced a zero vector")
    return clipped / total


def mitigate_counts(
    counts: Mapping[str, int],
    confusion_matrix: ReadoutConfusionMatrix,
    *,
    rcond: float = 1e-10,
) -> NDArray[np.float64]:
    """Apply full-basis readout mitigation to a count dictionary."""
    observed = counts_to_probabilities(counts, confusion_matrix.labels)
    return mitigate_probabilities(observed, confusion_matrix, rcond=rcond)


def probability_state_retention(
    probabilities: NDArray[np.float64],
    labels: Sequence[str],
    target_bitstring: str,
) -> float:
    """Return probability mass on ``target_bitstring``."""
    return float(probabilities[bitstring_index(target_bitstring, labels)])


def probability_parity_leakage(
    probabilities: NDArray[np.float64],
    labels: Sequence[str],
    target_bitstring: str,
) -> float:
    """Return mass outside the parity sector of a target in the unique label order.

    Duplicate labels and unknown targets raise ValueError, as for retention.
    Labels may be partial or permuted; probabilities follow their supplied order.

    Parameters
    ----------
    probabilities
        Probability vector aligned with labels.
    labels
        Unique same-width binary labels; ASCII spaces are ignored.
    target_bitstring
        Known label defining the target parity.

    Returns
    -------
    float
        Probability mass in the opposite parity sector.

    Raises
    ------
    ValueError
        If labels are invalid, the target is unknown or vector lengths differ.
    """
    bitstring_index(target_bitstring, labels)
    target = target_bitstring.count("1") % 2
    return float(
        sum(
            probability
            for label, probability in zip(labels, probabilities, strict=True)
            if label.count("1") % 2 != target
        )
    )


def probability_magnetisation_leakage(
    probabilities: NDArray[np.float64],
    labels: Sequence[str],
    target_bitstring: str,
) -> float:
    """Return mass outside the magnetisation sector of a known target label.

    Duplicate labels and unknown targets raise ValueError, as for retention.
    Labels may be partial or permuted; probabilities follow their supplied order.

    Parameters
    ----------
    probabilities
        Probability vector aligned with labels.
    labels
        Unique same-width binary labels; ASCII spaces are ignored.
    target_bitstring
        Known label defining the target magnetisation.

    Returns
    -------
    float
        Probability mass outside the target magnetisation sector.

    Raises
    ------
    ValueError
        If labels are invalid, the target is unknown or vector lengths differ.
    """
    bitstring_index(target_bitstring, labels)
    target = _magnetisation(target_bitstring)
    return float(
        sum(
            probability
            for label, probability in zip(labels, probabilities, strict=True)
            if _magnetisation(label) != target
        )
    )


def probability_mean_magnetisation(
    probabilities: NDArray[np.float64],
    labels: Sequence[str],
) -> float:
    """Return probability-weighted magnetisation over a unique binary label order.

    Partial and permuted orders are supported; invalid or duplicate labels raise
    ValueError. Probabilities follow supplied label positions.

    Parameters
    ----------
    probabilities
        Probability vector aligned with labels.
    labels
        Unique same-width binary labels; ASCII spaces are ignored.

    Returns
    -------
    float
        Mean of number-of-zeros minus number-of-ones across outcomes.

    Raises
    ------
    ValueError
        If labels are invalid or vector lengths differ.
    """
    label_index_map(labels)
    return float(
        sum(
            _magnetisation(label) * probability
            for label, probability in zip(labels, probabilities, strict=True)
        )
    )


def _magnetisation(bitstring: str) -> int:
    clean = bitstring.replace(" ", "")
    return len(clean) - 2 * clean.count("1")
