# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

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
