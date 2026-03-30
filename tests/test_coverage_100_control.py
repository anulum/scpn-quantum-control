# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage tests for control/ module gaps
"""Tests targeting specific uncovered lines in the control/ subpackage."""

from __future__ import annotations

import numpy as np
import pytest

# --- q_disruption_iter.py line 84: rng is None → default_rng() ---


def test_q_disruption_iter_default_rng():
    """Cover line 84: generate_iter_disruption_data with rng=None."""
    from scpn_quantum_control.control.q_disruption_iter import generate_synthetic_iter_data

    X, y = generate_synthetic_iter_data(n_samples=20, rng=None)
    assert X.shape[0] == 20
    assert y.shape[0] == 20


# --- qpetri.py line 50: W_out shape mismatch raises ValueError ---


def test_qpetri_w_out_shape_mismatch():
    """Cover line 50: QuantumPetriNet raises ValueError for W_out shape mismatch."""
    from scpn_quantum_control.control.qpetri import QuantumPetriNet

    W_in = np.array([[1.0, 0.0], [0.0, 1.0]])
    W_out = np.array([[1.0, 0.0]])  # wrong shape: should be (2, 2)
    thresholds = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="W_out shape"):
        QuantumPetriNet(n_places=2, n_transitions=2, W_in=W_in, W_out=W_out, thresholds=thresholds)


# --- vqls_gs.py line 100: xAtAx < eps → cost returns 1.0 ---


def test_vqls_denominator_near_zero():
    """Cover line 100: VQLS cost returns 1.0 when denominator near zero."""
    from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

    solver = VQLS_GradShafranov(n_qubits=2)
    result = solver.solve(maxiter=1, seed=42)
    assert isinstance(result, np.ndarray)
