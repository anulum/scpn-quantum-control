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

from scpn_quantum_control.mitigation.readout_matrix import (
    build_readout_confusion_matrix,
    computational_basis_labels,
    counts_to_probabilities,
    mitigate_probabilities,
    probability_magnetisation_leakage,
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


def test_pseudo_inverse_recovers_known_distribution() -> None:
    calibrations = {
        "0": {"0": 90, "1": 10},
        "1": {"0": 20, "1": 80},
    }
    matrix = build_readout_confusion_matrix(calibrations, 1)
    true = np.array([0.25, 0.75])
    observed = matrix.matrix @ true

    mitigated = mitigate_probabilities(observed, matrix)

    np.testing.assert_allclose(mitigated, true, atol=1e-12)


def test_probability_observables_use_target_bitstring() -> None:
    labels = computational_basis_labels(2)
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    assert probability_state_retention(probabilities, labels, "10") == 0.3
    assert probability_parity_leakage(probabilities, labels, "00") == 0.5
    assert probability_magnetisation_leakage(probabilities, labels, "01") == 0.5
