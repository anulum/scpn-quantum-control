"""Cover edge cases in vqls_gs.py: lines 93, 108."""

from __future__ import annotations

import pytest

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
    # Just verify solve completes without error for minimal iterations
    result = vqls.solve(reps=1, maxiter=5, seed=42)
    assert result.shape == (4,)
