# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

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
