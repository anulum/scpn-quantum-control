# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — readout matrix tests
# scpn-quantum-control -- readout matrix mitigation tests
"""Tests for full-basis readout matrix mitigation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.mitigation.readout_matrix import (
    bitstring_index,
    build_readout_confusion_matrix,
    computational_basis_labels,
    counts_to_probabilities,
    label_index_map,
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
def test_mitigation_rejects_invalid_observed_probability_vectors(
    observed: NDArray[np.float64],
) -> None:
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


def test_probabilities_follow_the_supplied_label_order() -> None:
    """A permuted label order places each count at its declared position."""
    assert list(counts_to_probabilities({"0": 1}, ("1", "0"))) == [0.0, 1.0]
    assert list(counts_to_probabilities({"1": 1}, ("1", "0"))) == [1.0, 0.0]
    assert list(counts_to_probabilities({"01": 3, "10": 1}, ("11", "10", "01", "00"))) == [
        0.0,
        0.25,
        0.75,
        0.0,
    ]


def test_partial_label_order_is_a_domain_error_not_an_index_error() -> None:
    """A label subset must place its own outcomes instead of overflowing."""
    assert list(counts_to_probabilities({"11": 1}, ("10", "11"))) == [0.0, 1.0]
    assert list(counts_to_probabilities({"10": 2, "11": 2}, ("10", "11"))) == [0.5, 0.5]
    with pytest.raises(ValueError, match="unknown bitstring"):
        counts_to_probabilities({"00": 1}, ("10", "11"))


def test_empty_label_order_is_refused() -> None:
    """A probability vector needs a declared label order to be placed against."""
    with pytest.raises(ValueError, match="labels must not be empty"):
        counts_to_probabilities({"0": 1}, ())
    with pytest.raises(ValueError, match="labels must not be empty"):
        bitstring_index("0", ())
    with pytest.raises(ValueError, match="labels must not be empty"):
        label_index_map(())


def test_duplicate_labels_are_refused_as_ambiguous() -> None:
    """A repeated label cannot resolve to one position and must not be guessed."""
    with pytest.raises(ValueError, match="labels must not repeat"):
        counts_to_probabilities({"0": 1}, ("0", "0"))
    with pytest.raises(ValueError, match="labels must not repeat"):
        bitstring_index("0", ("0", "0"))
    with pytest.raises(ValueError, match="labels must not repeat"):
        probability_state_retention(np.array([1.0, 0.0]), ("0", "0"), "0")


def test_sparse_counts_leave_unreported_outcomes_at_zero() -> None:
    """Absent outcomes stay zero rather than shifting later labels."""
    labels = ("11", "10", "01", "00")
    probabilities = counts_to_probabilities({"10": 4}, labels)
    assert list(probabilities) == [0.0, 1.0, 0.0, 0.0]


def test_bitstring_index_resolves_against_the_supplied_labels() -> None:
    """The index is the declared position, and the numeric value without labels."""
    assert bitstring_index("0", ("1", "0")) == 1
    assert bitstring_index("10", ("11", "10", "01", "00")) == 1
    assert bitstring_index("10") == 2
    assert bitstring_index("1 0", ("11", "10", "01", "00")) == 1
    with pytest.raises(ValueError, match="unknown bitstring"):
        bitstring_index("11", ("0", "1"))


def test_state_retention_follows_the_supplied_label_order() -> None:
    """Retention reads the target's declared position, not its numeric value."""
    labels = ("11", "10", "01", "00")
    probabilities = counts_to_probabilities({"01": 3, "11": 1}, labels)
    assert probability_state_retention(probabilities, labels, "01") == 0.75
    assert probability_state_retention(probabilities, labels, "11") == 0.25
    assert probability_state_retention(probabilities, labels, "00") == 0.0


def test_leakage_helpers_are_order_independent() -> None:
    """Sector helpers pair labels with probabilities positionally in any order."""
    canonical = ("00", "01", "10", "11")
    permuted = ("11", "10", "01", "00")
    counts = {"01": 3, "11": 1}
    canonical_probabilities = counts_to_probabilities(counts, canonical)
    permuted_probabilities = counts_to_probabilities(counts, permuted)
    for target in ("01", "11"):
        assert probability_parity_leakage(
            canonical_probabilities, canonical, target
        ) == pytest.approx(probability_parity_leakage(permuted_probabilities, permuted, target))
        assert probability_magnetisation_leakage(
            canonical_probabilities, canonical, target
        ) == pytest.approx(
            probability_magnetisation_leakage(permuted_probabilities, permuted, target)
        )
    assert probability_mean_magnetisation(canonical_probabilities, canonical) == pytest.approx(
        probability_mean_magnetisation(permuted_probabilities, permuted)
    )
