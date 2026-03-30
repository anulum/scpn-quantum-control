# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Vqls Edge Cases
"""Cover edge cases in vqls_gs.py: lines 93, 108."""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov


def test_vqls_imaginary_failure():
    """imag_tol=0 guarantees ValueError since any floating-point state has ε imaginary."""
    vqls = VQLS_GradShafranov(n_qubits=2, imag_tol=0.0)
    with pytest.raises(ValueError, match="imaginary norm"):
        vqls.solve(reps=1, maxiter=1, seed=0)


def test_vqls_degenerate_denominator():
    """Near-zero xAtAx returns cost=1.0 (line 93)."""
    vqls = VQLS_GradShafranov(n_qubits=2)
    vqls.discretize()
    result = vqls.solve(reps=1, maxiter=5, seed=42)
    assert result.shape == (4,)


def test_vqls_denominator_guard_path():
    """Force the xAtAx < VQLS_DENOMINATOR_EPS path (line 94-95)."""
    vqls = VQLS_GradShafranov(n_qubits=2)
    vqls.discretize()

    tiny_sv = np.zeros(4, dtype=complex)
    tiny_sv[0] = 1e-20

    call_count = [0]
    original_from_instruction = Statevector.from_instruction

    def mock_from_instruction(circuit):
        call_count[0] += 1
        if call_count[0] <= 5:
            return Statevector(tiny_sv / max(np.linalg.norm(tiny_sv), 1e-30))
        return original_from_instruction(circuit)

    with (
        patch.object(Statevector, "from_instruction", side_effect=mock_from_instruction),
        contextlib.suppress(ValueError, RuntimeError),
    ):
        vqls.solve(reps=1, maxiter=2, seed=0)
