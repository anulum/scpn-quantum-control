# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage tests for crypto/ module gaps
"""Tests targeting specific uncovered lines in the crypto/ subpackage."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

# --- noise_analysis.py line 85: concurrence imaginary part warning ---


def test_noise_analysis_imaginary_eigenvalue_warning():
    """Cover line 85: _concurrence_2qubit warns when eigenvalues have imaginary part."""
    from scpn_quantum_control.crypto.noise_analysis import _concurrence_2qubit

    rng = np.random.default_rng(42)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    c = _concurrence_2qubit(rho)
    assert 0.0 <= c <= 1.0


# --- percolation.py line 127: entropy clamping (e near 0 or near 1) ---


def test_percolation_key_rate_entropy_clamp():
    """Cover line 127: h_e = 0.0 when e near 0 or 1."""
    from scpn_quantum_control.crypto.percolation import key_rate_per_channel

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    from scpn_quantum_control.crypto.percolation import concurrence_map

    conc = concurrence_map(K, omega, maxiter=10)
    rates = key_rate_per_channel(conc)
    assert rates.shape == (2, 2)


# --- percolation.py line 203: all edges removed → full disconnect ---


def test_percolation_targeted_removal():
    """Cover line 203: robustness_targeted_removal full disconnect fallback."""
    from scpn_quantum_control.crypto.percolation import robustness_targeted_removal

    K = build_knm_paper27(L=2)
    result = robustness_targeted_removal(K)
    assert "edges_to_disconnect" in result
    assert result["fraction"] > 0


# --- percolation.py line 243: entanglement routing best_entanglement_path ---


def test_percolation_routing():
    """Cover line 243: best_entanglement_path path finding."""
    from scpn_quantum_control.crypto.percolation import best_entanglement_path

    K = build_knm_paper27(L=3)
    result = best_entanglement_path(K, source=0, target=2)
    assert "path" in result
    assert "bottleneck" in result
