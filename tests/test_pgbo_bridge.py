# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Pgbo Bridge
"""Tests for quantum PGBO bridge."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.pgbo.quantum_bridge import (
    PGBOResult,
    compute_pgbo_tensor,
)


class TestComputePGBOTensor:
    def test_returns_result(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        assert isinstance(result, PGBOResult)

    def test_metric_shape(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        # 3 oscillators → 3 upper-triangle params
        assert result.metric_tensor.shape == (3, 3)
        assert result.berry_curvature.shape == (3, 3)

    def test_metric_symmetric(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        np.testing.assert_allclose(result.metric_tensor, result.metric_tensor.T, atol=1e-8)

    def test_metric_positive_semidefinite(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        eigenvalues = np.linalg.eigvalsh(result.metric_tensor)
        assert np.all(eigenvalues >= -1e-6)

    def test_n_parameters(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_pgbo_tensor(K, omega)
        assert result.n_parameters == 6  # C(4,2) = 6

    def test_parameter_labels(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        assert len(result.parameter_labels) == 3
        assert "K_01" in result.parameter_labels

    def test_total_curvature_non_negative(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        assert result.total_curvature >= 0

    def test_curvature_antisymmetric(self):
        """Berry curvature F_μν = -F_νμ."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        np.testing.assert_allclose(result.berry_curvature, -result.berry_curvature.T, atol=1e-6)

    def test_scpn_pgbo(self):
        """Record PGBO tensor at SCPN defaults."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        print("\n  PGBO (3 osc):")
        print(f"  Metric det: {result.metric_determinant:.6e}")
        print(f"  Total curvature: {result.total_curvature:.6f}")
        print(f"  Metric diag: {np.diag(result.metric_tensor)}")
        assert isinstance(result.metric_determinant, float)


# ---------------------------------------------------------------------------
# PGBO physics: information geometry invariants
# ---------------------------------------------------------------------------


class TestPGBOPhysics:
    def test_metric_determinant_nonnegative(self):
        """det(g) ≥ 0 for PSD metric."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        assert result.metric_determinant >= -1e-10

    def test_different_K_different_metric(self):
        """Different coupling → different quantum geometry."""
        omega = OMEGA_N_16[:3]
        r1 = compute_pgbo_tensor(build_knm_paper27(L=3, K_base=0.1), omega)
        r2 = compute_pgbo_tensor(build_knm_paper27(L=3, K_base=2.0), omega)
        assert not np.allclose(r1.metric_tensor, r2.metric_tensor)

    def test_curvature_traceless(self):
        """Antisymmetric F_μν has zero trace."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = compute_pgbo_tensor(K, omega)
        assert abs(np.trace(result.berry_curvature)) < 1e-10


# ---------------------------------------------------------------------------
# Pipeline: Knm → PGBO tensor → metric → wired
# ---------------------------------------------------------------------------


class TestPGBOPipeline:
    def test_pipeline_knm_to_pgbo(self):
        """Full pipeline: build_knm → PGBO tensor → metric + curvature.
        Verifies PGBO module is wired and produces quantum geometric data.
        """
        import time

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        result = compute_pgbo_tensor(K, omega)
        dt = (time.perf_counter() - t0) * 1000

        assert result.metric_tensor.shape == (6, 6)  # C(4,2) parameters
        assert result.metric_determinant >= -1e-10

        print(f"\n  PIPELINE Knm→PGBO (4q, 6 params): {dt:.1f} ms")
        print(f"  det(g) = {result.metric_determinant:.6e}")
        print(f"  Total curvature = {result.total_curvature:.6f}")
