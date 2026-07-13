# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — ADAPT-VQE branch contracts
"""Branch-level contracts for the adaptive layered VQE implementation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_dense_matrix,
)
from scpn_quantum_control.phase.adapt_vqe import (
    _build_operator_pool,
    _plus_reference,
    adapt_vqe,
)


def test_operator_pool_excludes_uncoupled_pairs() -> None:
    """Keep only the two local generators when a two-qubit graph has no edge."""
    coupling = np.zeros((2, 2), dtype=np.float64)

    pool = _build_operator_pool(coupling, 2)

    assert len(pool) == 2


def test_zero_layer_budget_returns_reference_result() -> None:
    """Return the untouched reference result when no ansatz layer is allowed."""
    coupling = build_knm_paper27(L=2)
    frequencies = np.asarray(OMEGA_N_16[:2], dtype=np.float64)
    reference = _plus_reference(2)
    hamiltonian = knm_to_dense_matrix(coupling, frequencies)
    expected_energy = float((reference.conj() @ hamiltonian @ reference).real)

    result = adapt_vqe(coupling, frequencies, max_iterations=0, seed=42)

    assert result.energy == pytest.approx(expected_energy)
    assert result.n_iterations == 0
    assert result.n_parameters == 0
    assert result.gradient_norms == []
    assert result.energies == [pytest.approx(expected_energy)]
    assert result.selected_operators == []
    assert not result.converged
