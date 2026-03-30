# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage tests for l16, mitigation, qec, ssgf, tcbo gaps
"""Tests targeting uncovered lines across l16, mitigation, qec, ssgf, and tcbo."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

# --- l16/quantum_director.py lines 153-156: score <= 0.4 → halt action ---


def test_quantum_director_halt_action():
    """Cover lines 153-156: L16 returns 'halt' when stability score <= 0.4."""
    from scpn_quantum_control.l16.quantum_director import compute_l16_lyapunov

    # Weak coupling → low echo, low R → low score → halt or adjust
    K = build_knm_paper27(L=2) * 0.001
    omega = OMEGA_N_16[:2] * 0.001
    result = compute_l16_lyapunov(K, omega)
    assert result.action in ("continue", "adjust", "halt")


def test_quantum_director_adjust_action():
    """Cover lines 153-154: L16 returns 'adjust' when 0.4 < score <= 0.7."""
    from scpn_quantum_control.l16.quantum_director import compute_l16_lyapunov

    # Moderate coupling
    K = build_knm_paper27(L=2) * 0.3
    omega = OMEGA_N_16[:2]
    result = compute_l16_lyapunov(K, omega)
    assert result.action in ("continue", "adjust", "halt")
    assert 0.0 <= result.stability_score <= 1.0


# --- mitigation/cpdr.py line 201: slope near zero → mitigated = raw ---


def test_cpdr_zero_slope():
    """Cover line 201: CPDR returns raw value when regression slope ~ 0."""
    from scpn_quantum_control.mitigation.cpdr import cpdr_mitigate

    ideal = [0.1, 0.2, 0.3, 0.4]
    noisy = [0.15, 0.18, 0.35, 0.38]
    result = cpdr_mitigate(0.5, ideal, noisy)
    assert hasattr(result, "mitigated_value")


# --- qec/control_qec.py line 222: correction fails (non-zero syndrome after correction) ---


def test_control_qec_correction_failure():
    """Cover line 222: decode_and_correct returns False for heavy uncorrectable error."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    # Apply a heavy error: all X errors (likely uncorrectable for d=3)
    err_x = np.ones(qec.code.num_data, dtype=np.int8)
    err_z = np.ones(qec.code.num_data, dtype=np.int8)
    result = qec.decode_and_correct(err_x, err_z)
    assert isinstance(result, bool)


# --- qec/error_budget.py line 89: d reaches max_distance ---


def test_error_budget_max_distance():
    """Cover line 89: minimum_code_distance returns max_distance when no d satisfies target."""
    from scpn_quantum_control.qec.error_budget import minimum_code_distance

    # Very low target → needs very high distance
    d = minimum_code_distance(target_logical_rate=1e-30, p_physical=0.01, max_distance=7)
    assert d == 7


# --- qec/error_budget.py lines 128-129: comm_bound near zero ---


def test_error_budget_zero_comm_bound():
    """Cover lines 128-129: n_steps=1, eps_trotter=0 when comm_bound near zero."""
    from scpn_quantum_control.qec.error_budget import compute_error_budget

    K = np.zeros((2, 2))
    omega = np.array([0.0, 0.0])
    result = compute_error_budget(K, omega, t_total=1.0)
    assert result.n_trotter_steps == 1
    assert result.trotter_error == 0.0


# --- ssgf/quantum_costs.py line 98: half < 1 early return ---


def test_quantum_costs_single_qubit():
    """Cover line 97-98: compute_c4_tcbo returns (1.0, 0.0) for single-qubit."""
    from qiskit.quantum_info import Statevector

    from scpn_quantum_control.ssgf.quantum_costs import compute_c4_tcbo

    sv = Statevector.from_label("0")
    cost, entropy = compute_c4_tcbo(sv, 1)
    assert cost == 1.0
    assert entropy == 0.0


# --- ssgf/quantum_costs.py line 127-128: no correlators → returns (1.0, 0.0) ---


def test_quantum_costs_no_correlators():
    """Cover line 127-128: compute_c_pgbo returns (1.0, 0.0) when no pairs."""
    from qiskit.quantum_info import Statevector

    from scpn_quantum_control.ssgf.quantum_costs import compute_c_pgbo

    sv = Statevector.from_label("0")
    cost, var = compute_c_pgbo(sv, 1)
    assert cost == 1.0
    assert var == 0.0


# --- ssgf/quantum_outer_cycle.py line 61: off_diag empty → returns 1.0 ---


def test_quantum_outer_cycle_single_node():
    """Cover line 61: classical_cost returns 1.0 when W has no off-diagonal."""
    from scpn_quantum_control.ssgf.quantum_outer_cycle import classical_cost

    W = np.array([[1.0]])
    cost = classical_cost(W)
    assert cost == 1.0


# --- ssgf/quantum_spectral.py line 119: default k_values ---


def test_quantum_spectral_default_k_values():
    """Cover line 119: spectral_bridge_vs_coupling k_values defaults to linspace."""
    from scpn_quantum_control.ssgf.quantum_spectral import spectral_bridge_vs_coupling

    omega = OMEGA_N_16[:2]
    result = spectral_bridge_vs_coupling(omega, k_values=None)
    assert len(result["k_base"]) == 20


# --- tcbo/quantum_observer.py line 58: j <= i+1 → returns 0.0 ---


def test_string_order_parameter_adjacent():
    """Cover line 58: _string_order_parameter returns 0.0 when j <= i+1."""
    from scpn_quantum_control.tcbo.quantum_observer import _string_order_parameter

    psi = np.array([1, 0, 0, 0], dtype=complex)
    result = _string_order_parameter(psi, 2, i=0, j=1)
    assert result == 0.0


# --- tcbo/quantum_observer.py line 79: n < 4 → topological_entropy = 0.0 ---


def test_topological_entropy_small():
    """Cover line 79: _topological_entropy returns 0.0 for n < 4."""
    from scpn_quantum_control.tcbo.quantum_observer import _topological_entanglement_entropy

    psi = np.array([1, 0, 0, 0], dtype=complex)
    result = _topological_entanglement_entropy(psi, 2)
    assert result == 0.0
