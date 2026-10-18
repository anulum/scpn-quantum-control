# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Measurement-preserving mitigation conformance
"""Exercise readout correction and extrapolation through their public consumers."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

from scpn_quantum_control.mitigation import (
    build_readout_confusion_matrix,
    gate_fold_circuit,
    mitigate_counts,
    mitigate_counts_with_uncertainty,
    zne_extrapolate_with_readout_uncertainty,
)


def test_partial_permuted_measurement_reaches_corrected_extrapolation() -> None:
    """Preserve an actual nonidentity physical readout through the complete chain."""
    circuit = QuantumCircuit(3, 1)
    circuit.ry(0.9, 2)
    circuit.x(0)
    circuit.measure(2, 0)
    scales = [1, 3, 5]
    counts = []
    for scale in scales:
        folded = gate_fold_circuit(circuit, scale)
        measurements = [item for item in folded.data if item.operation.name == "measure"]
        assert len(measurements) == 1
        assert folded.find_bit(measurements[0].qubits[0]).index == 2
        assert folded.find_bit(measurements[0].clbits[0]).index == 0
        assert folded.num_clbits == 1
        result = StatevectorSampler(seed=100 + scale).run([folded], shots=1000).result()[0]
        counts.append(result.data.c.get_counts())
    original_counts = json.dumps(counts, sort_keys=True)
    calibration = build_readout_confusion_matrix({"0": {"0": 1000}, "1": {"1": 1000}}, 1)
    calibration = replace(calibration, labels=("1", "0"), matrix=calibration.matrix[::-1, ::-1])
    result = zne_extrapolate_with_readout_uncertainty(scales, counts, calibration, [-1.0, 1.0])
    for estimate, raw in zip(result.readout_estimates, counts, strict=True):
        assert estimate.labels == ("1", "0")
        assert estimate.shots == 1000
        np.testing.assert_array_equal(
            estimate.quasiprobabilities, [raw.get("1", 0) / 1000, raw.get("0", 0) / 1000]
        )
    assert result.extrapolation.method == "gls-delta"
    assert result.extrapolation.standard_error > 0
    assert json.dumps(counts, sort_keys=True) == original_counts


def test_negative_quasiprobabilities_preserve_original_calibration_and_counts() -> None:
    """Expose the inverse's signed result while preserving legacy probability output."""
    calibration_counts = {"0": {"0": 900, "1": 100}, "1": {"0": 200, "1": 800}}
    calibration = build_readout_confusion_matrix(calibration_counts, 1)
    original = calibration.matrix.copy()
    counts = {"1": 100}
    result = mitigate_counts_with_uncertainty(counts, calibration)
    np.testing.assert_allclose(result.quasiprobabilities, [-2 / 7, 9 / 7], atol=1e-12)
    np.testing.assert_array_equal(mitigate_counts(counts, calibration), [0.0, 1.0])
    np.testing.assert_array_equal(calibration.matrix, original)
    assert counts == {"1": 100}
    assert calibration_counts == {"0": {"0": 900, "1": 100}, "1": {"0": 200, "1": 800}}
    assert np.trace(result.calibration_covariance) > 0


def test_two_measured_qubits_keep_their_permuted_physical_and_label_order() -> None:
    """Preserve two readout wires and one unmeasured qubit through actual sampling."""
    circuit = QuantumCircuit(3, 2)
    circuit.ry(0.7, 2)
    circuit.ry(1.1, 0)
    circuit.x(1)
    circuit.measure(2, 0)
    circuit.measure(0, 1)
    counts = []
    for scale in [1, 3, 5]:
        folded = gate_fold_circuit(circuit, scale)
        mapping = [
            (folded.find_bit(item.qubits[0]).index, folded.find_bit(item.clbits[0]).index)
            for item in folded.data
            if item.operation.name == "measure"
        ]
        assert mapping == [(2, 0), (0, 1)]
        assert folded.num_qubits == 3 and folded.num_clbits == 2
        sample = StatevectorSampler(seed=200 + scale).run([folded], shots=1000).result()[0]
        counts.append(sample.data.c.get_counts())
    labels = ("00", "01", "10", "11")
    calibration = build_readout_confusion_matrix({label: {label: 1000} for label in labels}, 2)
    permutation = [2, 0, 3, 1]
    permuted = tuple(labels[index] for index in permutation)
    calibration = replace(
        calibration, labels=permuted, matrix=calibration.matrix[np.ix_(permutation, permutation)]
    )
    original = json.dumps(counts, sort_keys=True)
    result = zne_extrapolate_with_readout_uncertainty(
        [1, 3, 5], counts, calibration, [-1.0, 1.0, 1.0, -1.0]
    )
    for raw, estimate in zip(counts, result.readout_estimates, strict=True):
        assert estimate.labels == permuted
        assert estimate.shots == sum(raw.values()) == 1000
        np.testing.assert_array_equal(
            estimate.quasiprobabilities, [raw.get(label, 0) / 1000 for label in permuted]
        )
    assert result.extrapolation.standard_error > 0
    assert json.dumps(counts, sort_keys=True) == original


def test_ill_conditioned_calibration_refuses_without_replacing_original_inputs() -> None:
    """Refuse a genuine indistinguishable-preparation calibration before returning a result."""
    raw = {"0": {"0": 500, "1": 500}, "1": {"0": 500, "1": 500}}
    calibration = build_readout_confusion_matrix(raw, 1)
    original = calibration.matrix.copy()
    with pytest.raises(ValueError, match="condition"):
        mitigate_counts_with_uncertainty({"0": 60, "1": 40}, calibration)
    np.testing.assert_array_equal(calibration.matrix, original)
    assert raw == {"0": {"0": 500, "1": 500}, "1": {"0": 500, "1": 500}}


def test_harmful_calibration_correction_is_retained() -> None:
    """Retain a synthetic calibration-mismatch case that worsens a known zero target."""
    calibration = build_readout_confusion_matrix(
        {"0": {"0": 900, "1": 100}, "1": {"0": 200, "1": 800}}, 1
    )
    counts = {"0": 520, "1": 480}
    result = mitigate_counts_with_uncertainty(counts, calibration)
    corrected = float(np.dot([1.0, -1.0], result.quasiprobabilities))
    assert corrected == pytest.approx(-3 / 35)
    assert abs(corrected) > abs((counts["0"] - counts["1"]) / 1000)


def test_shared_calibration_covariance_reaches_the_extrapolated_error() -> None:
    """Use closed-form Bernoulli derivatives to catch independent-error substitution."""
    calibration = build_readout_confusion_matrix(
        {"0": {"0": 900, "1": 100}, "1": {"0": 200, "1": 800}}, 1
    )
    counts = [{"0": 410, "1": 590}, {"0": 550, "1": 450}]
    result = zne_extrapolate_with_readout_uncertainty([1, 3], counts, calibration, [1.0, -1.0])
    first = np.array([(0.41 - 0.2) / 0.7, (0.55 - 0.2) / 0.7])
    second = 1.0 - first
    expected = (
        4
        / 0.7**2
        * (np.outer(first, first) * 0.9 * 0.1 / 1000 + np.outer(second, second) * 0.2 * 0.8 / 1000)
    )
    expected += np.diag(4 * np.array([0.41 * 0.59, 0.55 * 0.45]) / (1000 * 0.7**2))
    np.testing.assert_allclose(result.observation_covariance, expected, rtol=1e-12, atol=1e-15)
    weights = np.array([1.5, -0.5])
    expected_variance = float(weights @ expected @ weights)
    assert result.extrapolation.zero_noise_estimate == pytest.approx(
        float(weights @ (2 * first - 1)), abs=1e-12
    )
    assert result.extrapolation.standard_error**2 == pytest.approx(expected_variance, rel=1e-12)
    independent_variance = float(weights**2 @ np.diag(expected))
    assert not np.isclose(expected_variance, independent_variance, rtol=1e-3)
