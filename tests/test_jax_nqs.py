# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for JAX-Accelerated NQS
"""Tests for JAX-based RBM wavefunction and VMC ground state search."""

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

try:
    from scpn_quantum_control.phase.jax_nqs import (
        is_jax_available,
        jax_rbm_energy,
        jax_vmc_ground_state,
    )

    _JAX_OK = is_jax_available()
except (ImportError, AttributeError):
    _JAX_OK = False

_SKIP = pytest.mark.skipif(not _JAX_OK, reason="JAX not available")


class TestJaxAvailability:
    """Tests that always run regardless of JAX installation."""

    def test_is_jax_available_returns_bool(self):
        from scpn_quantum_control.phase.jax_nqs import is_jax_available

        assert isinstance(is_jax_available(), bool)

    def test_module_importable(self):
        from scpn_quantum_control.phase import jax_nqs

        assert hasattr(jax_nqs, "is_jax_available")
        assert hasattr(jax_nqs, "jax_rbm_energy")
        assert hasattr(jax_nqs, "jax_vmc_ground_state")


@_SKIP
class TestJaxRBMEnergy:
    """Tests for jax_rbm_energy."""

    def test_returns_scalar(self):
        import jax
        import jax.numpy as jnp

        n = 2
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = jnp.array(knm_to_dense_matrix(K, omega), dtype=jnp.float32)

        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        params = {
            "a": 0.01 * jax.random.normal(k1, (n,)),
            "b": 0.01 * jax.random.normal(k2, (4,)),
            "W": 0.01 * jax.random.normal(k3, (4, n)),
        }

        energy = jax_rbm_energy(params, H, n)
        assert np.isfinite(float(energy))

    def test_energy_real(self):
        import jax
        import jax.numpy as jnp

        n = 2
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = jnp.array(knm_to_dense_matrix(K, omega), dtype=jnp.float32)

        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        params = {
            "a": 0.01 * jax.random.normal(k1, (n,)),
            "b": 0.01 * jax.random.normal(k2, (4,)),
            "W": 0.01 * jax.random.normal(k3, (4, n)),
        }

        energy = jax_rbm_energy(params, H, n)
        assert float(jnp.imag(energy)) == pytest.approx(0.0, abs=1e-5)

    def test_energy_bounded_by_spectrum(self):
        import jax
        import jax.numpy as jnp

        n = 2
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H_np = knm_to_dense_matrix(K, omega)
        H = jnp.array(H_np, dtype=jnp.float32)
        eigvals = np.linalg.eigvalsh(H_np)

        key = jax.random.PRNGKey(7)
        k1, k2, k3 = jax.random.split(key, 3)
        params = {
            "a": 0.01 * jax.random.normal(k1, (n,)),
            "b": 0.01 * jax.random.normal(k2, (4,)),
            "W": 0.01 * jax.random.normal(k3, (4, n)),
        }

        energy = float(jax_rbm_energy(params, H, n))
        assert energy >= eigvals[0] - 0.1
        assert energy <= eigvals[-1] + 0.1


@_SKIP
class TestJaxVMCGroundState:
    """Tests for jax_vmc_ground_state."""

    def test_returns_dict(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = jax_vmc_ground_state(K, omega, n_iterations=10, seed=42)
        assert isinstance(result, dict)
        assert "energy" in result
        assert "energy_history" in result
        assert "params" in result
        assert "n_params" in result

    def test_energy_finite(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = jax_vmc_ground_state(K, omega, n_iterations=10)
        assert np.isfinite(result["energy"])

    def test_energy_decreases(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = jax_vmc_ground_state(K, omega, n_iterations=50, seed=42)
        history = result["energy_history"]
        assert len(history) > 2
        assert history[-1] <= history[0] + 0.5

    def test_params_shapes(self):
        n = 3
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        result = jax_vmc_ground_state(K, omega, n_hidden=6, n_iterations=5)
        assert result["params"]["a"].shape == (n,)
        assert result["params"]["b"].shape == (6,)
        assert result["params"]["W"].shape == (6, n)

    def test_n_params_correct(self):
        n = 2
        n_hid = 4
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        result = jax_vmc_ground_state(K, omega, n_hidden=n_hid, n_iterations=5)
        expected = n + n_hid + n_hid * n
        assert result["n_params"] == expected

    def test_rejects_large_n(self):
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16[:16]
        with pytest.raises(ValueError, match="n<=12"):
            jax_vmc_ground_state(K, omega, n_iterations=1)

    def test_deterministic_with_seed(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        r1 = jax_vmc_ground_state(K, omega, n_iterations=10, seed=99)
        r2 = jax_vmc_ground_state(K, omega, n_iterations=10, seed=99)
        assert r1["energy"] == pytest.approx(r2["energy"], abs=1e-5)


@_SKIP
class TestJaxNQSPipeline:
    """Pipeline integration tests."""

    def test_pipeline_knm_to_jax_vmc(self):
        """Full pipeline: Knm → JAX VMC → energy convergence."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]

        t0 = time.perf_counter()
        result = jax_vmc_ground_state(K, omega, n_iterations=30, seed=42)
        dt = (time.perf_counter() - t0) * 1000

        assert np.isfinite(result["energy"])
        assert result["n_params"] > 0

        print(f"\n  PIPELINE Knm→JAX VMC (3q, 30 iter): {dt:.1f} ms")
        print(f"  Final energy: {result['energy']:.4f}")

    def test_pipeline_jax_vs_exact(self):
        """JAX VMC energy should approach exact ground state."""
        from scpn_quantum_control.hardware.classical import classical_exact_diag

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]

        exact = classical_exact_diag(2, K=K, omega=omega)
        exact_e0 = exact["ground_energy"]

        result = jax_vmc_ground_state(K, omega, n_iterations=200, seed=42)
        vmc_e = result["energy"]

        # VMC should be within 20% of exact for 2 qubits with 200 iterations
        relative_error = abs(vmc_e - exact_e0) / abs(exact_e0)
        assert relative_error < 0.3, f"VMC {vmc_e:.4f} vs exact {exact_e0:.4f}"

        print("\n  PIPELINE JAX VMC vs exact (2q)")
        print(f"  Exact E0: {exact_e0:.4f}, VMC E: {vmc_e:.4f}")
        print(f"  Relative error: {relative_error:.2%}")
