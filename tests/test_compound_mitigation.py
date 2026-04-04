# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for compound error mitigation (CPDR + Symmetry)."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from scpn_quantum_control.mitigation.compound_mitigation import (
    CompoundMitigationResult,
    compound_mitigate_pipeline,
)


class TestCompoundMitigation:
    def test_compound_mitigate_pipeline(self):
        """Verify compound mitigation combines CPDR and Z2 symmetry."""
        qc = QuantumCircuit(2)
        qc.rx(np.pi / 2, 0)
        qc.ry(np.pi / 2, 1)

        # Mock counts: 80% perfect even parity, 20% odd parity noise
        target_counts = {"00": 800, "01": 200}

        def mock_backend(circuits):
            # Same mock counts for training
            return [{"00": 800, "01": 200} for _ in circuits]

        res = compound_mitigate_pipeline(
            target_circuit=qc,
            target_counts=target_counts,
            run_on_backend=mock_backend,
            expected_parity=0,
            n_training=5,
            perturbation_scale=0.01,
            seed=42,
        )

        assert isinstance(res, CompoundMitigationResult)
        assert res.n_training_circuits == 5
        # 20% odd parity should be rejected
        assert abs(res.mean_rejection_rate - 0.2) < 1e-6
        assert np.isfinite(res.mitigated_value)

    def test_odd_parity_expectation(self):
        """Parity postselect correctly filters with expected_parity=1."""
        from scpn_quantum_control.mitigation.symmetry_verification import parity_postselect

        counts = {"01": 700, "10": 200, "00": 100}
        res = parity_postselect(counts, expected_parity=1)
        # "01" and "10" are odd parity → verified
        assert res.verified_shots == 900
        assert res.rejected_shots == 100
        assert abs(res.rejection_rate - 0.1) < 1e-6

    def test_zero_rejection_rate(self):
        """All-correct parity yields zero rejection."""
        qc = QuantumCircuit(2)
        target_counts = {"00": 500, "11": 500}

        def mock_backend(circuits):
            return [{"00": 500, "11": 500} for _ in circuits]

        res = compound_mitigate_pipeline(
            target_circuit=qc,
            target_counts=target_counts,
            run_on_backend=mock_backend,
            expected_parity=0,
            n_training=5,
            seed=42,
        )
        assert res.mean_rejection_rate == 0.0

    def test_full_rejection_rate(self):
        """All-wrong parity yields 100% rejection."""
        qc = QuantumCircuit(2)
        target_counts = {"01": 500, "10": 500}

        def mock_backend(circuits):
            return [{"01": 500, "10": 500} for _ in circuits]

        res = compound_mitigate_pipeline(
            target_circuit=qc,
            target_counts=target_counts,
            run_on_backend=mock_backend,
            expected_parity=0,
            n_training=5,
            seed=42,
        )
        assert res.mean_rejection_rate == 1.0

    def test_result_fields_finite(self):
        """All numeric result fields must be finite."""
        qc = QuantumCircuit(2)
        qc.h(0)
        target_counts = {"00": 400, "01": 100, "10": 100, "11": 400}

        def mock_backend(circuits):
            return [{"00": 400, "01": 100, "10": 100, "11": 400} for _ in circuits]

        res = compound_mitigate_pipeline(
            target_circuit=qc,
            target_counts=target_counts,
            run_on_backend=mock_backend,
            expected_parity=0,
            n_training=10,
            seed=42,
        )
        assert np.isfinite(res.raw_value)
        assert np.isfinite(res.mitigated_value)
        assert np.isfinite(res.regression_r_squared)
        assert np.isfinite(res.regression_slope)
        assert np.isfinite(res.regression_intercept)
