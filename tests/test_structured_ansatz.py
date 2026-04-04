# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for the generalized structured ansatz."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.structured_ansatz import build_structured_ansatz


class TestStructuredAnsatz:
    def test_build_empty_graph(self):
        """No couplings > threshold => no entangling gates."""
        K = np.zeros((3, 3))
        qc = build_structured_ansatz(K, reps=1, threshold=1e-6)

        # 3 ry + 3 rz = 6 gates total, 0 cz
        assert len(qc.data) == 6
        ops = qc.count_ops()
        assert "cz" not in ops
        assert "cx" not in ops

    def test_build_full_graph(self):
        """All-to-all couplings => max entangling gates."""
        K = np.ones((3, 3))
        qc = build_structured_ansatz(K, reps=1, threshold=1e-6)

        # 6 single-qubit gates + 3 CZ gates = 9 gates
        assert len(qc.data) == 9
        ops = qc.count_ops()
        assert ops["cz"] == 3

    def test_custom_entanglement_gate(self):
        K = np.array([[0, 1], [1, 0]])
        qc = build_structured_ansatz(K, reps=2, entanglement_gate="cx")

        ops = qc.count_ops()
        assert "cz" not in ops
        assert ops["cx"] == 2  # 1 pair, 2 reps => 2 CNOTs

    def test_invalid_coupling_matrix(self):
        K = np.ones((3, 4))
        with pytest.raises(ValueError):
            build_structured_ansatz(K)

    def test_invalid_entanglement_gate(self):
        K = np.ones((2, 2))
        with pytest.raises(ValueError):
            build_structured_ansatz(K, entanglement_gate="rx")

    def test_parameter_count(self):
        """Total parameters = n_qubits * 2 * reps."""
        for n in [2, 3, 4]:
            for reps in [1, 2, 3]:
                K = np.ones((n, n))
                qc = build_structured_ansatz(K, reps=reps)
                assert qc.num_parameters == n * 2 * reps

    def test_threshold_boundary(self):
        """Coupling exactly at threshold IS included."""
        K = np.array([[0, 0.5], [0.5, 0]])
        qc = build_structured_ansatz(K, reps=1, threshold=0.5)
        ops = qc.count_ops()
        assert ops.get("cz", 0) == 1

    def test_below_threshold_excluded(self):
        """Coupling just below threshold is excluded."""
        K = np.array([[0, 0.499], [0.499, 0]])
        qc = build_structured_ansatz(K, reps=1, threshold=0.5)
        ops = qc.count_ops()
        assert ops.get("cz", 0) == 0

    def test_asymmetric_matrix_symmetrised(self):
        """Asymmetric input is symmetrised before gate placement."""
        K = np.array([[0, 1.0, 0], [0, 0, 0], [0, 0, 0]])
        qc = build_structured_ansatz(K, reps=1, threshold=0.1)
        ops = qc.count_ops()
        # (K + K.T)/2 gives K[0,1]=K[1,0]=0.5 → above threshold
        assert ops.get("cz", 0) == 1

    def test_multiple_reps_gate_count(self):
        """Each rep adds its own layer of entangling gates."""
        K = np.array([[0, 1], [1, 0]])
        for reps in [1, 2, 4]:
            qc = build_structured_ansatz(K, reps=reps)
            ops = qc.count_ops()
            assert ops["cz"] == reps  # 1 pair × reps layers

    def test_cnot_alias(self):
        """'cnot' is accepted as alias for 'cx'."""
        K = np.array([[0, 1], [1, 0]])
        qc = build_structured_ansatz(K, reps=1, entanglement_gate="cnot")
        ops = qc.count_ops()
        assert ops.get("cx", 0) == 1
