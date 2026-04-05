# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for Dynamic Quantum-Classical Co-Evolution."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.qsnn.dynamic_coupling import DynamicCouplingEngine


class TestDynamicCoupling:
    def test_dynamic_coupling_engine_step(self):
        """Verify one cycle of the strange loop."""
        n = 3
        # Weak initial coupling
        initial_K = np.array([[0.0, 0.1, 0.1], [0.1, 0.0, 0.1], [0.1, 0.1, 0.0]])
        omega = np.array([10.0, 10.0, 10.0])

        engine = DynamicCouplingEngine(
            n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=0.5, decay_rate=0.0
        )

        # We step it
        res = engine.step(dt=0.5)

        assert "K_updated" in res
        assert "correlation_matrix" in res
        assert res["K_updated"].shape == (3, 3)
        # Verify symmetry
        np.testing.assert_allclose(res["K_updated"], res["K_updated"].T)

    def test_run_coevolution(self):
        n = 2
        # Fully disconnected initially
        initial_K = np.zeros((n, n))
        omega = np.array([5.0, 5.0])

        engine = DynamicCouplingEngine(
            n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=1.0, decay_rate=0.1
        )

        history = engine.run_coevolution(steps=3, dt=0.5)
        assert len(history) == 3

    def test_k_symmetry_preserved(self):
        """K must remain symmetric after every step."""
        n = 3
        K0 = np.array([[0, 0.2, 0.1], [0.2, 0, 0.15], [0.1, 0.15, 0]])
        omega = np.array([5.0, 7.0, 9.0])
        engine = DynamicCouplingEngine(n_qubits=n, initial_K=K0, omega=omega)
        history = engine.run_coevolution(steps=5, dt=0.3)
        for step in history:
            np.testing.assert_allclose(step["K_updated"], step["K_updated"].T, atol=1e-12)

    def test_correlation_matrix_symmetric(self):
        """Correlation matrix C_nm must be symmetric."""
        n = 3
        K0 = np.array([[0, 0.3, 0], [0.3, 0, 0.3], [0, 0.3, 0]])
        omega = np.ones(n) * 5.0
        engine = DynamicCouplingEngine(n_qubits=n, initial_K=K0, omega=omega)
        res = engine.step(dt=0.5)
        C = res["correlation_matrix"]
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_k_diagonal_stays_zero(self):
        """Self-coupling K[i,i] must remain zero."""
        n = 3
        K0 = np.ones((n, n)) * 0.1
        np.fill_diagonal(K0, 0)
        omega = np.ones(n) * 5.0
        engine = DynamicCouplingEngine(n_qubits=n, initial_K=K0, omega=omega)
        history = engine.run_coevolution(steps=3, dt=0.5)
        for step in history:
            np.testing.assert_allclose(np.diag(step["K_updated"]), 0.0)

    def test_k_non_negative(self):
        """K entries must remain >= 0 (physical constraint)."""
        n = 2
        K0 = np.array([[0, 0.05], [0.05, 0]])
        omega = np.array([5.0, 10.0])
        engine = DynamicCouplingEngine(
            n_qubits=n, initial_K=K0, omega=omega, learning_rate=0.5, decay_rate=0.9
        )
        history = engine.run_coevolution(steps=10, dt=0.5)
        for step in history:
            assert np.all(step["K_updated"] >= -1e-15)

    def test_decay_drives_k_toward_zero(self):
        """With zero learning rate and high decay, K should shrink."""
        n = 2
        K0 = np.array([[0, 1.0], [1.0, 0]])
        omega = np.array([5.0, 5.0])
        engine = DynamicCouplingEngine(
            n_qubits=n, initial_K=K0, omega=omega, learning_rate=0.0, decay_rate=0.5
        )
        history = engine.run_coevolution(steps=5, dt=0.5)
        K_final = history[-1]["K_updated"]
        assert K_final[0, 1] < K0[0, 1]

    def test_statevector_normalised(self):
        """Returned statevector must be normalised."""
        n = 2
        K0 = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([5.0, 5.0])
        engine = DynamicCouplingEngine(n_qubits=n, initial_K=K0, omega=omega)
        res = engine.step(dt=0.5)
        assert abs(np.linalg.norm(res["statevector"]) - 1.0) < 1e-10

    def test_rust_python_correlation_parity(self):
        """Rust correlation_matrix_xy matches Qiskit expectations."""
        try:
            import scpn_quantum_engine as eng
        except ImportError:
            import pytest

            pytest.skip("Rust engine not available")
        from qiskit.quantum_info import SparsePauliOp, Statevector

        n = 3
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)

        # Rust path
        C_rust = np.array(eng.correlation_matrix_xy(psi.real.copy(), psi.imag.copy(), n))

        # Qiskit path
        sv = Statevector(psi)
        C_py = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                x_str = ["I"] * n
                x_str[i] = "X"
                x_str[j] = "X"
                xx = sv.expectation_value(SparsePauliOp("".join(reversed(x_str)))).real
                y_str = ["I"] * n
                y_str[i] = "Y"
                y_str[j] = "Y"
                yy = sv.expectation_value(SparsePauliOp("".join(reversed(y_str)))).real
                C_py[i, j] = xx + yy
                C_py[j, i] = xx + yy

        np.testing.assert_allclose(C_rust, C_py, atol=1e-10)


class TestDynamicCouplingPythonFallback:
    """Cover lines 76-95: Qiskit fallback when Rust unavailable."""

    def test_measure_correlation_no_rust(self):
        """Mock _HAS_RUST=False → Qiskit SparsePauliOp path executes."""
        import scpn_quantum_control.qsnn.dynamic_coupling as dc_mod

        n = 3
        K = np.array([[0, 0.5, 0.2], [0.5, 0, 0.3], [0.2, 0.3, 0]])
        omega = np.array([1.0, 1.5, 2.0])
        engine = DynamicCouplingEngine(n, K, omega)

        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0  # |000⟩

        orig = dc_mod._HAS_RUST
        try:
            dc_mod._HAS_RUST = False
            C = engine._measure_correlation_matrix(psi)
        finally:
            dc_mod._HAS_RUST = orig

        assert C.shape == (n, n)
        np.testing.assert_allclose(C, C.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(C), 0.0, atol=1e-12)

    def test_step_no_rust(self):
        """Full step via Qiskit fallback path."""
        import scpn_quantum_control.qsnn.dynamic_coupling as dc_mod

        n = 2
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 1.5])
        engine = DynamicCouplingEngine(n, K, omega)

        orig = dc_mod._HAS_RUST
        try:
            dc_mod._HAS_RUST = False
            result = engine.step(dt=0.1)
        finally:
            dc_mod._HAS_RUST = orig

        assert "K_updated" in result
        assert "correlation_matrix" in result
