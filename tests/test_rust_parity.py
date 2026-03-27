# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Parity tests between Rust engine and Python implementations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

HAS_RUST = importlib.util.find_spec("scpn_quantum_engine") is not None
pytestmark = pytest.mark.skipif(not HAS_RUST, reason="scpn_quantum_engine not installed")


class TestRustPythonParity:
    def test_knm_16_exact_parity(self):
        import scpn_quantum_engine as engine
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K_rust = np.array(engine.build_knm(16, 0.45, 0.3))
        K_python = build_knm_paper27(L=16)
        np.testing.assert_allclose(K_rust, K_python, atol=1e-12)

    def test_knm_7_parity(self):
        import scpn_quantum_engine as engine
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K_rust = np.array(engine.build_knm(7, 0.45, 0.3))
        K_python = build_knm_paper27(L=7)
        np.testing.assert_allclose(K_rust, K_python, atol=1e-12)

    def test_knm_diagonal(self):
        import scpn_quantum_engine as engine

        K = np.array(engine.build_knm(4, 0.45, 0.3))
        assert K[0, 0] == pytest.approx(0.45, abs=1e-12)

    def test_knm_calibration_anchors(self):
        import scpn_quantum_engine as engine

        K = np.array(engine.build_knm(16, 0.45, 0.3))
        assert K[0, 1] == pytest.approx(0.302, abs=1e-12)
        assert K[1, 2] == pytest.approx(0.201, abs=1e-12)
        assert K[2, 3] == pytest.approx(0.252, abs=1e-12)
        assert K[3, 4] == pytest.approx(0.154, abs=1e-12)

    def test_knm_cross_hierarchy_max(self):
        import scpn_quantum_engine as engine

        K = np.array(engine.build_knm(16, 0.45, 0.3))
        # K[4,6] = max(exponential, 0.15) — exponential > 0.15 for these params
        assert K[4, 6] > 0.15
        assert K[0, 15] >= 0.05

    def test_pec_coefficients_parity(self):
        import scpn_quantum_engine as engine
        from scpn_quantum_control.mitigation.pec import pauli_twirl_decompose

        for p in [0.001, 0.01, 0.05, 0.1]:
            rust = engine.pec_coefficients(p)
            python = pauli_twirl_decompose(p)
            np.testing.assert_allclose(rust, python, atol=1e-12)

    def test_order_parameter_parity(self):
        import scpn_quantum_engine as engine

        theta = np.array([0.0, 0.1, 0.2, 0.3])
        rust_R = engine.order_parameter(theta)
        z = np.mean(np.exp(1j * theta))
        python_R = float(np.abs(z))
        assert rust_R == pytest.approx(python_R, abs=1e-10)

    def test_kuramoto_euler_parity(self):
        import scpn_quantum_engine as engine
        from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

        n = 4
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n].copy()
        theta0 = np.array([0.1, 0.2, 0.3, 0.4])
        dt = 0.01

        # One Rust step
        theta_rust = np.array(engine.kuramoto_euler(theta0, omega, K, dt, 1))

        # One Python step (manual Euler)
        dtheta = omega.copy()
        for i in range(n):
            for j in range(n):
                dtheta[i] += K[i, j] * np.sin(theta0[j] - theta0[i])
        theta_python = theta0 + dt * dtheta

        np.testing.assert_allclose(theta_rust, theta_python, atol=1e-10)

    # --- New Rust functions ---

    @pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
    def test_state_order_param_sparse_parity(self, n):
        """Rust state_order_param_sparse matches Python _state_order_param_sparse."""
        import scpn_quantum_engine as engine
        from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16
        from scpn_quantum_control.hardware.classical import (
            _build_initial_state,
        )

        omega = OMEGA_N_16[:n]
        psi = _build_initial_state(n, omega)

        # Rust
        R_rust = engine.state_order_param_sparse(psi.real.copy(), psi.imag.copy(), n)

        # Python (manual, no Rust fallback)
        dim = len(psi)
        indices = np.arange(dim, dtype=np.int64)
        psi_conj = psi.conj()
        z = 0.0 + 0.0j
        for q in range(n):
            mask = 1 << q
            flipped = indices ^ mask
            psi_f = psi[flipped]
            exp_x = np.sum(psi_conj * psi_f).real
            bits = (indices >> q) & 1
            signs = 1.0 - 2.0 * bits
            exp_y = np.sum(psi_conj * (1j * signs) * psi_f).real
            z += exp_x + 1j * exp_y
        z /= n
        R_python = float(abs(z))

        np.testing.assert_allclose(R_rust, R_python, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_state_order_param_random_states(self, n):
        """Rust matches Python on random normalised statevectors."""
        import scpn_quantum_engine as engine

        rng = np.random.default_rng(42)
        dim = 2**n
        for _ in range(5):
            psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            psi /= np.linalg.norm(psi)
            R_rust = engine.state_order_param_sparse(psi.real.copy(), psi.imag.copy(), n)
            # Manually compute Python path (without Rust fallback)
            indices = np.arange(dim, dtype=np.int64)
            psi_conj = psi.conj()
            z = 0.0 + 0.0j
            for q in range(n):
                mask = 1 << q
                flipped = indices ^ mask
                psi_f = psi[flipped]
                exp_x = np.sum(psi_conj * psi_f).real
                bits = (indices >> q) & 1
                signs = 1.0 - 2.0 * bits
                exp_y = np.sum(psi_conj * (1j * signs) * psi_f).real
                z += exp_x + 1j * exp_y
            z /= n
            R_python = float(abs(z))
            np.testing.assert_allclose(R_rust, R_python, atol=1e-12)

    @pytest.mark.parametrize("pauli,pauli_idx", [("X", 0), ("Y", 1), ("Z", 2)])
    @pytest.mark.parametrize("qubit", [0, 1, 2])
    def test_expectation_pauli_fast_parity(self, pauli, pauli_idx, qubit):
        """Rust expectation_pauli_fast matches Python kron-based _expectation_pauli."""
        import scpn_quantum_engine as engine

        n = 3
        rng = np.random.default_rng(100 + qubit)
        dim = 2**n
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)

        R_rust = engine.expectation_pauli_fast(
            psi.real.copy(), psi.imag.copy(), n, qubit, pauli_idx
        )

        # Python: build full Pauli operator via kron
        if pauli == "X":
            p = np.array([[0, 1], [1, 0]], dtype=complex)
        elif pauli == "Y":
            p = np.array([[0, -1j], [1j, 0]], dtype=complex)
        else:
            p = np.array([[1, 0], [0, -1]], dtype=complex)
        kron_pos = n - 1 - qubit
        op = np.array([[1.0]])
        for i in range(n):
            op = np.kron(op, p if i == kron_pos else np.eye(2))
        R_python = float(np.real(psi.conj() @ op @ psi))

        np.testing.assert_allclose(R_rust, R_python, atol=1e-10)

    @pytest.mark.parametrize("horizon", [1, 2, 3, 4])
    def test_brute_mpc_parity(self, horizon):
        """Rust brute_mpc matches Python classical_brute_mpc."""
        import scpn_quantum_engine as engine

        B = np.eye(2)
        target = np.array([0.8, 0.6])

        actions_rs, cost_rs, costs_rs, n_eval_rs = engine.brute_mpc(B.ravel(), target, 2, horizon)
        actions_rs = np.array(actions_rs)
        costs_rs = np.array(costs_rs)

        # Python manual
        n_actions = 2**horizon
        b_norm = float(np.linalg.norm(B))
        t_norm = float(np.linalg.norm(target))
        best_cost = np.inf
        best_actions = np.zeros(horizon, dtype=int)
        all_costs = np.zeros(n_actions)
        for idx in range(n_actions):
            acts = np.array([(idx >> bit) & 1 for bit in range(horizon)])
            cost = sum((b_norm * acts[t] - t_norm / horizon) ** 2 for t in range(horizon))
            all_costs[idx] = cost
            if cost < best_cost:
                best_cost = cost
                best_actions = acts.copy()

        assert n_eval_rs == n_actions
        np.testing.assert_allclose(costs_rs, all_costs, atol=1e-12)
        assert cost_rs == pytest.approx(best_cost, abs=1e-12)
        np.testing.assert_array_equal(actions_rs, best_actions)
