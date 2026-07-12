# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — vqls gs claim boundary tests
# SCPN Quantum Control - VQLS Grad-Shafranov claim-boundary tests
"""Contract tests for the VQLS Grad-Shafranov proxy boundary."""

from __future__ import annotations

from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov


def test_vqls_diagnostics_label_bounded_proxy_model() -> None:
    """Diagnostics state that the solver is not a full equilibrium model."""
    solver = VQLS_GradShafranov(n_qubits=2)

    result = solver.solve_with_diagnostics(reps=1, maxiter=1, seed=11, n_restarts=1)

    assert result.model_boundary == "1d_poisson_laplacian_proxy"
    assert result.is_full_grad_shafranov_equilibrium is False
    assert result.converged is True
