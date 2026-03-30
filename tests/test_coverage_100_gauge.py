# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage tests for gauge/ module gaps
"""Tests targeting specific uncovered lines in the gauge/ subpackage."""

from __future__ import annotations

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

# --- confinement.py line 63: no loops at requested length ---


def test_confinement_no_loops():
    """Cover line 63: _average_wilson_by_length returns 0.0 when no loops."""
    from scpn_quantum_control.gauge.confinement import _average_wilson_by_length

    # 2-qubit system has no length-4 loops
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = _average_wilson_by_length(K, omega, length=4)
    assert result == 0.0


# --- confinement.py line 84: extract_string_tension w_large/w_small ratio <= 0 ---


def test_confinement_string_tension_zero():
    """Cover line 84: extract_string_tension returns None for w < eps."""
    from scpn_quantum_control.gauge.confinement import extract_string_tension

    assert extract_string_tension(0.0, 0.5) is None
    assert extract_string_tension(0.5, 0.0) is None
    assert extract_string_tension(0.5, 0.5, area_small=1.0, area_large=1.0) is None


# --- confinement.py line 131: confinement_scan default k_values ---


def test_confinement_scan_default():
    """Cover line 131: k_values defaults to linspace when None."""
    from scpn_quantum_control.gauge.confinement import confinement_vs_coupling

    omega = OMEGA_N_16[:2]
    result = confinement_vs_coupling(omega, k_values=None)
    assert len(result["k_base"]) == 15


# --- universality.py line 146: t_bkt near zero → ratio = 0 ---


def test_universality_zero_tbkt():
    """Cover line 146: check_nelson_kosterlitz ratio = 0 when t_bkt ~ 0."""
    from scpn_quantum_control.gauge.universality import check_nelson_kosterlitz

    # Very weak coupling → t_bkt near zero
    K = build_knm_paper27(L=2) * 0.0001
    omega = OMEGA_N_16[:2]
    ratio, deviation = check_nelson_kosterlitz(K, omega)
    assert isinstance(ratio, float)
    assert isinstance(deviation, float)


# --- vortex_detector.py line 145: default k_base_values ---


def test_vortex_density_scan_default():
    """Cover line 145: k_base_values defaults to linspace when None."""
    from scpn_quantum_control.gauge.vortex_detector import vortex_density_vs_coupling

    omega = OMEGA_N_16[:2]
    result = vortex_density_vs_coupling(omega, k_base_values=None)
    assert len(result["k_base"]) == 20


# --- wilson_loop.py line 100: sparse matrix .toarray() ---


def test_wilson_loop_sparse_matrix():
    """Cover line 100: W_mat.toarray() for sparse Wilson operator."""
    from scpn_quantum_control.gauge.wilson_loop import compute_wilson_loops

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    results = compute_wilson_loops(K, omega, max_length=3, max_loops=5)
    assert isinstance(results, list)
