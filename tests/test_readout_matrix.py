# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- readout matrix mitigation tests
"""Tests for full-basis readout matrix mitigation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.mitigation.readout_matrix import (
    bitstring_index,
    build_readout_confusion_matrix,
    computational_basis_labels,
    counts_to_probabilities,
    mitigate_counts,
    mitigate_probabilities,
    probability_magnetisation_leakage,
    probability_mean_magnetisation,
    probability_parity_leakage,
    probability_state_retention,
)


def test_computational_basis_labels_are_big_endian() -> None:
    assert computational_basis_labels(2) == ("00", "01", "10", "11")


def test_identity_confusion_matrix_preserves_probabilities() -> None:
    calibrations = {
        "00": {"00": 100},
        "01": {"01": 100},
        "10": {"10": 100},
        "11": {"11": 100},
    }
    matrix = build_readout_confusion_matrix(calibrations, 2)
    observed = counts_to_probabilities({"01": 25, "10": 75}, matrix.labels)

    mitigated = mitigate_probabilities(observed, matrix)

    np.testing.assert_allclose(matrix.matrix.sum(axis=0), np.ones(4))
    assert matrix.condition_number == pytest.approx(1.0)
    assert matrix.shots_by_prepared_state == {
        "00": 100,
        "01": 100,
        "10": 100,
        "11": 100,
    }
    np.testing.assert_allclose(mitigated, observed)


def test_missing_calibration_state_is_rejected() -> None:
    calibrations = {
        "00": {"00": 100},
        "01": {"01": 100},
        "10": {"10": 100},
    }

    try:
        build_readout_confusion_matrix(calibrations, 2)
    except ValueError as exc:
        assert "missing calibration states" in str(exc)
    else:
        raise AssertionError("missing calibration state was accepted")


def test_counts_to_probabilities_rejects_unknown_labels_and_empty_counts() -> None:
    labels = computational_basis_labels(2)

    with pytest.raises(ValueError, match="unknown bitstring"):
        counts_to_probabilities({"00": 1, "20": 1}, labels)
    with pytest.raises(ValueError, match="empty count dictionary"):
        counts_to_probabilities({}, labels)
    with pytest.raises(ValueError, match="counts must be non-negative"):
        counts_to_probabilities({"00": 2, "01": -1}, labels)


def test_bitstring_index_accepts_spaced_labels_and_rejects_unknown_label() -> None:
    labels = computational_basis_labels(3)

    assert bitstring_index("1 0 1", labels) == 5
    with pytest.raises(ValueError, match="unknown bitstring label"):
        bitstring_index("1111", labels)


def test_pseudo_inverse_recovers_known_distribution() -> None:
    calibrations = {
        "0": {"0": 90, "1": 10},
        "1": {"0": 20, "1": 80},
    }
    matrix = build_readout_confusion_matrix(calibrations, 1)
    true = np.array([0.25, 0.75])
    observed = matrix.matrix @ true

    mitigated = mitigate_probabilities(observed, matrix)
    mitigated_from_counts = mitigate_counts({"0": 33, "1": 67}, matrix)

    np.testing.assert_allclose(mitigated, true, atol=1e-12)
    assert mitigated_from_counts.shape == (2,)
    assert float(mitigated_from_counts.sum()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "observed",
    [
        np.array([1.1, -0.1]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
        np.array([0.0, 0.0]),
    ],
)
def test_mitigation_rejects_invalid_observed_probability_vectors(observed) -> None:
    calibrations = {
        "0": {"0": 100},
        "1": {"1": 100},
    }
    matrix = build_readout_confusion_matrix(calibrations, 1)

    with pytest.raises(ValueError, match="observed probabilities"):
        mitigate_probabilities(observed, matrix)


def test_probability_observables_use_target_bitstring() -> None:
    labels = computational_basis_labels(2)
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    assert probability_state_retention(probabilities, labels, "10") == 0.3
    assert probability_parity_leakage(probabilities, labels, "00") == 0.5
    assert probability_magnetisation_leakage(probabilities, labels, "01") == 0.5
    assert probability_mean_magnetisation(probabilities, labels) == pytest.approx(-0.6)
