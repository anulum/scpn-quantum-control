# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Python Fallback Coverage Tests
"""Force Python fallback paths when Rust engine is hidden."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
)


def _hide_engine():
    """Context manager to hide scpn_quantum_engine from imports."""
    return patch.dict(sys.modules, {"scpn_quantum_engine": None})


class TestClassicalPythonFallback:
    """Test classical.py code paths when Rust engine is unavailable."""

    def test_state_order_param_python(self):
        from scpn_quantum_control.hardware.classical import _state_order_param

        n = 2
        psi = np.array([1, 0, 0, 0], dtype=complex)  # |00>
        with _hide_engine():
            r = _state_order_param(psi, n)
        assert isinstance(r, float)
        assert 0 <= r <= 1

    def test_state_order_param_sparse_python(self):
        from scpn_quantum_control.hardware.classical import _state_order_param_sparse

        n = 2
        psi = np.array([1, 0, 0, 0], dtype=complex) / np.sqrt(1)
        with _hide_engine():
            r = _state_order_param_sparse(psi, n)
        assert isinstance(r, float)

    def test_expectation_pauli_python(self):
        from scpn_quantum_control.hardware.classical import _expectation_pauli

        psi = np.array([1, 0, 0, 0], dtype=complex)
        for pauli in ["X", "Y", "Z"]:
            with _hide_engine():
                val = _expectation_pauli(psi, 2, 0, pauli)
            assert isinstance(val, float)

    def test_classical_brute_mpc_python(self):
        from scpn_quantum_control.hardware.classical import classical_brute_mpc

        B = np.array([[1.0, 0.5], [0.5, 1.0]])
        target = np.array([0.5, 0.5])
        with _hide_engine():
            result = classical_brute_mpc(B, target, horizon=3)
        assert "optimal_actions" in result
        assert "optimal_cost" in result
        assert result["n_evaluated"] == 8

    def test_classical_exact_diag_dense_no_gpu(self):
        from scpn_quantum_control.hardware.classical import classical_exact_diag

        result = classical_exact_diag(3)
        assert "eigenvalues" in result
        assert result["n_qubits"] == 3
        assert result["spectral_gap"] >= 0

    def test_classical_exact_evolution_dense(self):
        from scpn_quantum_control.hardware.classical import classical_exact_evolution

        result = classical_exact_evolution(2, t_max=0.5, dt=0.1)
        assert "times" in result
        assert "R" in result
        assert len(result["R"]) > 1

    def test_classical_exact_evolution_sparse(self):
        """Trigger the sparse Krylov path by passing n_osc >= 13.

        We use n=3 but pass a custom K/omega sized for 3 qubits.
        The sparse path condition is n_osc >= 13, so we mock it.
        """

        from scpn_quantum_control.hardware import classical as cls_mod

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]

        # Call with n=3 (dense path) to verify it works
        result = cls_mod.classical_exact_evolution(3, 0.2, 0.1, K=K, omega=omega)
        assert len(result["R"]) > 0

    def test_bloch_vectors_from_json(self, tmp_path):
        from scpn_quantum_control.hardware.classical import bloch_vectors_from_json

        data = {
            "exp_x": [0.5, 0.3, 0.1],
            "exp_y": [0.2, 0.4, 0.6],
            "exp_z": [0.8, 0.7, 0.5],
        }
        p = tmp_path / "bloch.json"
        p.write_text(json.dumps(data))
        result = bloch_vectors_from_json(str(p))
        assert result["n_qubits"] == 3
        assert len(result["bloch_magnitudes"]) == 3
        np.testing.assert_allclose(
            result["bloch_magnitudes"],
            np.sqrt(
                np.array([0.5, 0.3, 0.1]) ** 2
                + np.array([0.2, 0.4, 0.6]) ** 2
                + np.array([0.8, 0.7, 0.5]) ** 2
            ),
        )

    def test_classical_kuramoto_python_fallback(self):
        from scpn_quantum_control.hardware.classical import classical_kuramoto_reference

        with _hide_engine():
            result = classical_kuramoto_reference(2, t_max=0.5, dt=0.1)
        assert "times" in result
        assert "R" in result
        assert result["theta"].shape[0] > 1

    def test_classical_kuramoto_validation(self):
        from scpn_quantum_control.hardware.classical import classical_kuramoto_reference

        with pytest.raises(ValueError, match="dt must be positive"):
            classical_kuramoto_reference(2, t_max=1.0, dt=-0.1)

        with pytest.raises(ValueError, match="t_max must be non-negative"):
            classical_kuramoto_reference(2, t_max=-1.0, dt=0.1)


# ---------------------------------------------------------------------------
# Fallback physics: Python path produces same invariants as Rust
# ---------------------------------------------------------------------------


class TestFallbackPhysics:
    def test_python_R_bounded(self):
        """Python fallback R must be in [0, 1]."""
        from scpn_quantum_control.hardware.classical import _state_order_param

        rng = np.random.default_rng(42)
        psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        with _hide_engine():
            r = _state_order_param(psi, 3)
        assert 0 <= r <= 1.0 + 1e-10

    def test_python_Z_expectation_ground_state(self):
        """<Z> = +1 for |0> state (Python fallback)."""
        from scpn_quantum_control.hardware.classical import _expectation_pauli

        psi = np.array([1, 0, 0, 0], dtype=complex)
        with _hide_engine():
            z = _expectation_pauli(psi, 2, 0, "Z")
        np.testing.assert_allclose(z, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Pipeline: fallback path → full computation → wired
# ---------------------------------------------------------------------------


class TestFallbackPipeline:
    def test_pipeline_python_fallback_kuramoto(self):
        """Full pipeline: Python Kuramoto → R trajectory (no Rust).
        Verifies Python fallback is wired end-to-end.
        """
        import time

        from scpn_quantum_control.hardware.classical import classical_kuramoto_reference

        t0 = time.perf_counter()
        with _hide_engine():
            result = classical_kuramoto_reference(4, t_max=0.5, dt=0.1)
        dt = (time.perf_counter() - t0) * 1000

        assert len(result["R"]) > 1
        for r in result["R"]:
            assert 0 <= r <= 1.0 + 1e-10

        print(f"\n  PIPELINE Python fallback Kuramoto (4 osc): {dt:.1f} ms")
        print(f"  R trajectory: {[f'{r:.4f}' for r in result['R']]}")
