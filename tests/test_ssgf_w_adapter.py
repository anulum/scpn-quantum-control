# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Ssgf W Adapter
"""Tests for SSGF W adaptation from quantum feedback."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.bridge.ssgf_w_adapter import (
    WAdaptResult,
    adapt_w_from_quantum,
)


def _test_system() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a deterministic three-oscillator geometry and phase vector."""
    W = np.array([[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]])
    theta = np.array([0.0, 0.3, 0.6])
    return W, theta


class TestAdaptWFromQuantum:
    """Exercise the public one-step geometry adaptation contract."""

    def test_returns_result(self) -> None:
        """Return the immutable adaptation-result schema."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        assert isinstance(result, WAdaptResult)

    def test_w_shape_preserved(self) -> None:
        """Preserve the input geometry-matrix shape."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        assert result.W_updated.shape == W.shape

    def test_w_symmetric(self) -> None:
        """Preserve symmetry for a symmetric input geometry."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        np.testing.assert_allclose(result.W_updated, result.W_updated.T, atol=1e-12)

    def test_w_non_negative(self) -> None:
        """Enforce the configured non-negative coupling floor."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta, min_coupling=0.0)
        assert np.all(result.W_updated >= 0)

    def test_w_zero_diagonal(self) -> None:
        """Remove self-couplings from the adapted matrix."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        np.testing.assert_allclose(np.diag(result.W_updated), 0.0)

    def test_r_global_bounded(self) -> None:
        """Return a physical synchronisation order parameter."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        assert 0 <= result.r_global <= 1.0

    def test_correlators_shape(self) -> None:
        """Return one correlator entry for every oscillator pair."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        assert result.correlators.shape == (3, 3)

    def test_small_learning_rate_small_update(self) -> None:
        """Scale the largest update with a very small learning rate."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta, learning_rate=1e-6)
        assert result.max_update < 0.01

    def test_w_changes(self) -> None:
        """Update a coupled geometry when learning is enabled."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta, learning_rate=0.1)
        assert not np.allclose(result.W_updated, W)

    def test_explicit_frequencies_execute_public_branch(self) -> None:
        """Use caller-supplied natural frequencies during evolution."""
        W, theta = _test_system()
        omega = np.array([0.1, -0.2, 0.3])
        result = adapt_w_from_quantum(W, theta, omega=omega)
        assert result.W_updated.shape == W.shape
        assert np.isfinite(result.r_global)


# ---------------------------------------------------------------------------
# W adaptation physics: correlator feedback loop
# ---------------------------------------------------------------------------


class TestWAdaptPhysics:
    """Check physical invariants of the correlator feedback update."""

    def test_correlators_symmetric(self) -> None:
        """⟨XX⟩+⟨YY⟩ correlator matrix must be symmetric."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        np.testing.assert_allclose(result.correlators, result.correlators.T, atol=1e-10)

    def test_zero_learning_rate_no_change(self) -> None:
        """lr=0 → W unchanged."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta, learning_rate=0.0)
        np.testing.assert_allclose(result.W_updated, W, atol=1e-14)

    def test_max_update_nonnegative(self) -> None:
        """Report a non-negative absolute maximum update."""
        W, theta = _test_system()
        result = adapt_w_from_quantum(W, theta)
        assert result.max_update >= 0


# ---------------------------------------------------------------------------
# Pipeline: W → quantum feedback → W_updated → wired
# ---------------------------------------------------------------------------


class TestWAdaptPipeline:
    """Exercise the K_nm-to-SSGF geometry adaptation pipeline."""

    def test_pipeline_knm_to_w_adapt(self) -> None:
        """Full pipeline: Knm as W → quantum correlators → adapt W.

        Verifies SSGF W adapter is wired end-to-end.
        """
        import time

        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K = build_knm_paper27(L=4)
        np.fill_diagonal(K, 0.0)
        theta = np.random.default_rng(42).uniform(0, 2 * np.pi, 4)

        t0 = time.perf_counter()
        result = adapt_w_from_quantum(K, theta, learning_rate=0.1)
        dt = (time.perf_counter() - t0) * 1000

        assert result.W_updated.shape == (4, 4)
        assert np.all(result.W_updated >= 0)

        print(f"\n  PIPELINE Knm→WAdapt (4 osc): {dt:.1f} ms")
        print(f"  R_global = {result.r_global:.4f}, max_update = {result.max_update:.4f}")
