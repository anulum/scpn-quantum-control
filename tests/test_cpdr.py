# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Cpdr
"""Tests for CPDR error mitigation."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.mitigation.cpdr import (
    CPDRResult,
    _nearest_clifford_angle,
    compute_ideal_values,
    compute_noisy_values_from_counts,
    cpdr_full_pipeline,
    cpdr_mitigate,
    fit_regression,
    generate_training_circuits,
)


class TestNearestCliffordAngle:
    def test_zero(self):
        assert _nearest_clifford_angle(0.0) == pytest.approx(0.0)

    def test_pi_over_2(self):
        assert _nearest_clifford_angle(np.pi / 2) == pytest.approx(np.pi / 2)

    def test_pi(self):
        assert _nearest_clifford_angle(np.pi) == pytest.approx(np.pi)

    def test_near_pi(self):
        assert _nearest_clifford_angle(3.0) == pytest.approx(np.pi)

    def test_small_angle(self):
        assert _nearest_clifford_angle(0.1) == pytest.approx(0.0)

    def test_negative_wraps(self):
        # -π/4 mod 2π ≈ 5.50 → nearest is 2π ≈ 6.28 or 3π/2 ≈ 4.71
        result = _nearest_clifford_angle(-np.pi / 4)
        assert result in [pytest.approx(a) for a in [0.0, 3 * np.pi / 2, 2 * np.pi]]


class TestGenerateTrainingCircuits:
    def test_returns_correct_count(self):
        qc = QuantumCircuit(2)
        qc.rz(0.5, 0)
        qc.ry(1.2, 1)
        qc.cx(0, 1)
        qc.measure_all()

        training = generate_training_circuits(qc, n_training=10)
        assert len(training) == 10

    def test_preserves_structure(self):
        qc = QuantumCircuit(2)
        qc.rz(0.5, 0)
        qc.cx(0, 1)
        qc.measure_all()

        training = generate_training_circuits(qc, n_training=5)
        for tc in training:
            assert tc.num_qubits == 2
            assert tc.num_clbits >= 2

    def test_different_from_target(self):
        qc = QuantumCircuit(2)
        qc.rz(0.5, 0)
        qc.ry(1.2, 1)
        qc.measure_all()

        training = generate_training_circuits(qc, n_training=5, seed=42)
        # At least some training circuits should differ from target
        target_params = [float(p) for inst in qc.data for p in inst.operation.params]
        all_same = True
        for tc in training:
            tc_params = [float(p) for inst in tc.data for p in inst.operation.params]
            if tc_params != target_params:
                all_same = False
                break
        assert not all_same

    def test_no_rotation_gates(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        training = generate_training_circuits(qc, n_training=3)
        assert len(training) == 3

    def test_seed_reproducibility(self):
        qc = QuantumCircuit(2)
        qc.rz(0.5, 0)
        qc.measure_all()

        t1 = generate_training_circuits(qc, n_training=3, seed=42)
        t2 = generate_training_circuits(qc, n_training=3, seed=42)
        for a, b in zip(t1, t2):
            p_a = [float(p) for inst in a.data for p in inst.operation.params]
            p_b = [float(p) for inst in b.data for p in inst.operation.params]
            assert p_a == p_b


class TestComputeIdealValues:
    def test_identity_circuit(self):
        qc = QuantumCircuit(2)  # |00⟩ → ⟨Z⟩ = +1
        vals = compute_ideal_values([qc])
        assert len(vals) == 1
        assert vals[0] == pytest.approx(1.0)

    def test_x_gate(self):
        qc = QuantumCircuit(1)
        qc.x(0)  # |1⟩ → ⟨Z⟩ = -1
        vals = compute_ideal_values([qc])
        assert vals[0] == pytest.approx(-1.0)

    def test_hadamard(self):
        qc = QuantumCircuit(1)
        qc.h(0)  # |+⟩ → ⟨Z⟩ = 0
        vals = compute_ideal_values([qc])
        assert abs(vals[0]) < 0.01

    def test_observable_qubits(self):
        qc = QuantumCircuit(2)
        qc.x(0)  # qubit 0 → |1⟩, qubit 1 → |0⟩
        vals = compute_ideal_values([qc], observable_qubits=[0])
        assert vals[0] == pytest.approx(-1.0)
        vals = compute_ideal_values([qc], observable_qubits=[1])
        assert vals[0] == pytest.approx(1.0)


class TestComputeNoisyValues:
    def test_all_zeros(self):
        counts = [{"00": 1000}]
        vals = compute_noisy_values_from_counts(counts, 2)
        assert vals[0] == pytest.approx(1.0)

    def test_all_ones(self):
        counts = [{"11": 1000}]
        vals = compute_noisy_values_from_counts(counts, 2)
        assert vals[0] == pytest.approx(-1.0)

    def test_mixed(self):
        counts = [{"00": 500, "11": 500}]
        vals = compute_noisy_values_from_counts(counts, 2)
        assert vals[0] == pytest.approx(0.0)

    def test_empty_counts(self):
        vals = compute_noisy_values_from_counts([{}], 2)
        assert vals[0] == 0.0

    def test_multiple_circuits(self):
        counts = [{"00": 1000}, {"11": 1000}, {"00": 500, "11": 500}]
        vals = compute_noisy_values_from_counts(counts, 2)
        assert len(vals) == 3
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(-1.0)
        assert vals[2] == pytest.approx(0.0)


class TestFitRegression:
    def test_perfect_linear(self):
        ideal = [0.0, 0.5, 1.0]
        noisy = [0.1, 0.35, 0.6]  # noisy = 0.5 * ideal + 0.1
        slope, intercept, r_sq = fit_regression(ideal, noisy)
        assert slope == pytest.approx(0.5)
        assert intercept == pytest.approx(0.1)
        assert r_sq == pytest.approx(1.0)

    def test_identity_mapping(self):
        ideal = [-1.0, 0.0, 1.0]
        noisy = [-1.0, 0.0, 1.0]
        slope, intercept, r_sq = fit_regression(ideal, noisy)
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(0.0)
        assert r_sq == pytest.approx(1.0)

    def test_single_point_fallback(self):
        slope, intercept, r_sq = fit_regression([1.0], [0.5])
        assert slope == 1.0
        assert intercept == 0.0


class TestCPDRMitigate:
    def test_corrects_depolarization(self):
        # Simulated: ideal values in [-1, 1], noisy compressed to [-0.5, 0.5]
        ideal = [-1.0, -0.5, 0.0, 0.5, 1.0]
        noisy = [-0.5, -0.25, 0.0, 0.25, 0.5]  # noisy = 0.5 * ideal
        raw = 0.3  # true ideal ≈ 0.6
        result = cpdr_mitigate(raw, ideal, noisy)
        assert isinstance(result, CPDRResult)
        assert result.mitigated_value == pytest.approx(0.6)
        assert result.regression_slope == pytest.approx(0.5)
        assert result.regression_r_squared == pytest.approx(1.0)

    def test_with_offset(self):
        # noisy = 0.8 * ideal + 0.1
        ideal = [-1.0, 0.0, 1.0]
        noisy = [-0.7, 0.1, 0.9]
        raw = 0.5
        result = cpdr_mitigate(raw, ideal, noisy)
        expected = (0.5 - 0.1) / 0.8  # = 0.5
        assert result.mitigated_value == pytest.approx(expected)

    def test_n_training(self):
        result = cpdr_mitigate(0.5, [0.0, 1.0], [0.0, 0.8])
        assert result.n_training_circuits == 2


class TestCPDRFullPipeline:
    def test_end_to_end_simulator(self):
        # Build a simple circuit
        qc = QuantumCircuit(2)
        qc.rz(0.5, 0)
        qc.ry(1.2, 1)
        qc.cx(0, 1)
        qc.measure_all()

        # Simulate target measurement (noiseless)
        from qiskit.quantum_info import Statevector

        base = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(base)
        target_counts = sv.sample_counts(4000)

        # Mock backend: return noiseless counts (ideal = noisy)
        def noiseless_backend(circuits):
            results = []
            for c in circuits:
                b = c.remove_final_measurements(inplace=False)
                s = Statevector.from_instruction(b)
                results.append(s.sample_counts(4000))
            return results

        result = cpdr_full_pipeline(
            qc,
            target_counts,
            noiseless_backend,
            n_training=10,
            seed=42,
        )
        assert isinstance(result, CPDRResult)
        # With noiseless backend, mitigated ≈ raw
        assert abs(result.mitigated_value - result.raw_value) < 0.15
