# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- full-basis readout matrix mitigation
"""Full-basis readout confusion-matrix mitigation utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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


def bitstring_index(bitstring: str, labels: Sequence[str] | None = None) -> int:
    """Return the computational-basis index for ``bitstring``."""

    clean = bitstring.replace(" ", "")
    if labels is not None and clean not in labels:
        raise ValueError(f"unknown bitstring label: {bitstring!r}")
    return int(clean, 2)


def counts_to_probabilities(
    counts: Mapping[str, int],
    labels: Sequence[str],
) -> NDArray[np.float64]:
    """Convert a count dictionary into a probability vector over ``labels``."""

    total_count = 0
    for value in counts.values():
        count = int(value)
        if count < 0:
            raise ValueError("counts must be non-negative")
        total_count += count
    total = float(total_count)
    if total <= 0.0:
        raise ValueError("empty count dictionary")
    probabilities = np.zeros(len(labels), dtype=np.float64)
    label_set = set(labels)
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        if clean not in label_set:
            raise ValueError(f"count dictionary contains unknown bitstring {bitstring!r}")
        probabilities[bitstring_index(clean, labels)] += int(count) / total
    return probabilities


def build_readout_confusion_matrix(
    calibration_counts: Mapping[str, Mapping[str, int]],
    n_qubits: int,
) -> ReadoutConfusionMatrix:
    """Build a full-basis readout confusion matrix from calibration counts.

    ``calibration_counts`` must map prepared computational-basis labels to
    observed count dictionaries. Missing prepared states are rejected; this keeps
    the mitigation claim distinct from partial exact-state corrections.
    """

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
        condition_number=float(np.linalg.cond(matrix)),
        shots_by_prepared_state=shots_by_prepared_state,
    )


def mitigate_probabilities(
    observed_probabilities: NDArray[np.float64],
    confusion_matrix: ReadoutConfusionMatrix,
    *,
    rcond: float = 1e-10,
) -> NDArray[np.float64]:
    """Invert a readout matrix with clipping and renormalisation."""

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
    """Return probability mass outside the target parity sector."""

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
    """Return probability mass outside the target magnetisation sector."""

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
    """Return the probability-weighted computational-basis magnetisation."""

    return float(
        sum(
            _magnetisation(label) * probability
            for label, probability in zip(labels, probabilities, strict=True)
        )
    )


def _magnetisation(bitstring: str) -> int:
    clean = bitstring.replace(" ", "")
    return len(clean) - 2 * clean.count("1")
