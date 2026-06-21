# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Edge-case tests for the TCBO quantum observer
"""Degenerate-window edge cases for the TCBO topological observables.

Covers the string-order-parameter early return for a degenerate site window and
the topological-entanglement-entropy early return below the minimum qubit count.
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.tcbo.quantum_observer import (
    _string_order_parameter,
    _topological_entanglement_entropy,
)


def test_string_order_parameter_degenerate_window_is_zero() -> None:
    """With no interior sites between i and j the string order parameter is zero."""
    psi = np.zeros(16, dtype=np.complex128)
    psi[0] = 1.0
    assert _string_order_parameter(psi, 4, i=0, j=1) == 0.0


def test_topological_entropy_below_minimum_qubit_count_is_zero() -> None:
    """Below four qubits the Kitaev-Preskill construction returns zero."""
    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = 1.0
    assert _topological_entanglement_entropy(psi, 3) == 0.0
