# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for Topological Quantum Reinforcement Learning / Optimizer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import scpn_quantum_control.control.topological_optimizer as topology
from scpn_quantum_control.analysis.quantum_persistent_homology import _RIPSER_AVAILABLE
from scpn_quantum_control.control.topological_optimizer import TopologicalCouplingOptimizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _expected_perturbation(seed: int, n: int) -> NDArray[np.float64]:
    """Re-derive the symmetric zero-diagonal perturbation for a seeded first draw.

    Parameters
    ----------
    seed
        Seed of the optimizer's own generator.
    n
        Number of oscillators.

    Returns
    -------
    numpy.ndarray
        ``(raw + raw.T) / 2`` with a zero diagonal, where ``raw`` is the first
        ``normal(0, 0.05, (n, n))`` draw of ``default_rng(seed)``.

    """
    raw = np.random.default_rng(seed).normal(0.0, 0.05, size=(n, n))
    delta = (raw + raw.T) / 2.0
    np.fill_diagonal(delta, 0.0)
    return delta


def _install_score_fakes(
    monkeypatch: pytest.MonkeyPatch, observed_couplings: list[NDArray[np.float64]]
) -> None:
    """Replace evolution, sampling and homology with a deterministic coupling score.

    Parameters
    ----------
    monkeypatch
        Pytest fixture that restores every replaced attribute.
    observed_couplings
        Receives each measured candidate coupling matrix, in call order.

    """
    monkeypatch.setattr(topology, "_RIPSER_AVAILABLE", True)

    def fake_fast_sparse_evolution(
        K: NDArray[np.float64],
        omega: NDArray[np.float64],
        *,
        t_total: float,
        n_steps: int,
    ) -> dict[str, NDArray[np.complex128]]:
        return {"final_state": np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)}

    def fake_measurement(
        self: TopologicalCouplingOptimizer,
        psi: NDArray[np.complex128],
        shots: int = 5000,
        *,
        K_candidate: NDArray[np.float64] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        assert K_candidate is not None
        observed_couplings.append(np.array(K_candidate, dtype=float).copy())
        score = float(np.sum(K_candidate))
        return {"score": score}, {"score": score}

    monkeypatch.setattr(topology, "fast_sparse_evolution", fake_fast_sparse_evolution)
    monkeypatch.setattr(
        TopologicalCouplingOptimizer, "_simulate_measurement_counts", fake_measurement
    )
    monkeypatch.setattr(
        topology,
        "quantum_persistent_homology",
        lambda x_counts, y_counts, n, persistence_threshold: SimpleNamespace(
            p_h1=x_counts["score"]
        ),
    )


class TestTopologicalOptimizer:
    """Topological coupling optimiser steps, constraints and randomness custody."""

    def test_topological_optimizer_step(self) -> None:
        """Verify one cycle of topological gradient descent on p_h1."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")

        n = 3
        # Incoherent state (all decoupled) -> should yield high p_h1 eventually.
        initial_K = np.zeros((n, n))
        omega = np.array([5.0, 10.0, 15.0])

        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=0.5, dt=0.5
        )

        # We step it
        res = opt.step(n_samples=2)

        assert "K_updated" in res
        assert "p_h1_current" in res
        assert "gradient_norm" in res
        assert res["K_updated"].shape == (n, n)
        # Verify symmetry
        np.testing.assert_allclose(res["K_updated"], res["K_updated"].T)

    def test_optimize_loop(self) -> None:
        """``optimize`` returns one record per requested step."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")

        n = 2
        initial_K = np.ones((n, n)) * 0.1
        omega = np.array([5.0, 5.0])

        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=0.1, dt=0.5
        )

        history = opt.optimize(steps=2, n_samples=1)
        assert len(history) == 2

    def test_k_symmetry_preserved(self) -> None:
        """K must remain symmetric after optimization steps."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 3
        rng = np.random.default_rng(42)
        K0 = rng.random((n, n)) * 0.3
        K0 = (K0 + K0.T) / 2
        np.fill_diagonal(K0, 0)
        omega = np.array([5.0, 7.0, 9.0])
        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=K0, omega=omega, learning_rate=0.1, dt=0.5
        )
        res = opt.step(n_samples=2)
        np.testing.assert_allclose(res["K_updated"], res["K_updated"].T, atol=1e-12)

    def test_k_non_negative(self) -> None:
        """K entries must remain >= 0."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 2
        K0 = np.array([[0, 0.01], [0.01, 0]])
        omega = np.array([5.0, 5.0])
        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=K0, omega=omega, learning_rate=1.0, dt=0.5
        )
        res = opt.step(n_samples=3)
        assert np.all(res["K_updated"] >= -1e-15)

    def test_diagonal_stays_zero(self) -> None:
        """Self-coupling K[i,i] must remain zero."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 3
        K0 = np.ones((n, n)) * 0.2
        np.fill_diagonal(K0, 0)
        omega = np.ones(n) * 5.0
        opt = TopologicalCouplingOptimizer(
            n_qubits=n, initial_K=K0, omega=omega, learning_rate=0.1, dt=0.5
        )
        history = opt.optimize(steps=3, n_samples=1)
        for step in history:
            np.testing.assert_allclose(np.diag(step["K_updated"]), 0.0)

    def test_gradient_norm_finite(self) -> None:
        """Gradient norm must be a finite non-negative number."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 2
        K0 = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([5.0, 10.0])
        opt = TopologicalCouplingOptimizer(n_qubits=n, initial_K=K0, omega=omega)
        res = opt.step(n_samples=2)
        assert np.isfinite(res["gradient_norm"])
        assert res["gradient_norm"] >= 0

    def test_initialiser_enforces_physical_coupling_constraints(self) -> None:
        """The initial couplings are symmetrised with a zero diagonal."""
        K0 = np.array([[1.0, 0.1, 0.4], [0.5, 2.0, 0.3], [0.2, 0.9, 3.0]])
        omega = np.array([5.0, 7.0, 9.0])

        opt = TopologicalCouplingOptimizer(n_qubits=3, initial_K=K0, omega=omega)

        np.testing.assert_allclose(opt.K, opt.K.T)
        np.testing.assert_allclose(np.diag(opt.K), 0.0)
        np.testing.assert_allclose(opt.K[0, 1], 0.3)

    def test_step_requires_persistent_homology_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A step refuses clearly when the persistent-homology backend is missing.

        Parameters
        ----------
        monkeypatch
            Pytest fixture that hides the backend for this test only.

        """
        monkeypatch.setattr(topology, "_RIPSER_AVAILABLE", False)
        opt = TopologicalCouplingOptimizer(
            n_qubits=2,
            initial_K=np.array([[0.0, 0.1], [0.1, 0.0]]),
            omega=np.array([5.0, 10.0]),
        )

        with pytest.raises(ImportError, match="ripser"):
            opt.step(n_samples=1)

    def test_finite_difference_measures_candidate_couplings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each sample measures K, then K plus and minus one seeded direction."""
        observed_couplings: list[NDArray[np.float64]] = []
        _install_score_fakes(monkeypatch, observed_couplings)
        initial_K = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=float)
        delta = _expected_perturbation(7, 2)

        opt = TopologicalCouplingOptimizer(
            n_qubits=2,
            initial_K=initial_K,
            omega=np.array([5.0, 10.0]),
            learning_rate=0.1,
            rng=np.random.default_rng(7),
        )

        result = opt.step(n_samples=1)

        assert len(observed_couplings) == 3
        np.testing.assert_allclose(observed_couplings[0], initial_K)
        np.testing.assert_allclose(observed_couplings[1], np.maximum(initial_K + delta, 0.0))
        np.testing.assert_allclose(observed_couplings[2], np.maximum(initial_K - delta, 0.0))
        np.testing.assert_allclose(result["K_updated"], result["K_updated"].T)
        np.testing.assert_allclose(np.diag(result["K_updated"]), 0.0)

    def test_step_leaves_the_global_numpy_random_state_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Perturbations come from the optimizer's own generator only."""
        _install_score_fakes(monkeypatch, [])
        np.random.seed(1234)
        before = np.random.get_state(legacy=False)
        opt = TopologicalCouplingOptimizer(
            n_qubits=2,
            initial_K=np.array([[0.0, 0.2], [0.2, 0.0]]),
            omega=np.array([5.0, 10.0]),
            rng=np.random.default_rng(3),
        )
        opt.optimize(steps=2, n_samples=2)
        np.testing.assert_equal(np.random.get_state(legacy=False), before)

    def test_seeded_runs_reproduce_and_seeds_differ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same seed gives the same updated couplings; another seed does not."""
        _install_score_fakes(monkeypatch, [])

        def run(seed: int) -> NDArray[np.float64]:
            opt = TopologicalCouplingOptimizer(
                n_qubits=3,
                initial_K=np.full((3, 3), 0.2),
                omega=np.array([5.0, 10.0, 15.0]),
                rng=np.random.default_rng(seed),
            )
            history = opt.optimize(steps=2, n_samples=2)
            return np.asarray(history[-1]["K_updated"], dtype=float)

        np.testing.assert_array_equal(run(11), run(11))
        assert not np.array_equal(run(11), run(12))

    def test_simulated_measurement_sampling_uses_the_optimizer_generator(self) -> None:
        """Seeded optimizers draw identical shots without touching the global state."""
        psi = np.full(4, 0.5, dtype=np.complex128)
        np.random.seed(99)
        before = np.random.get_state(legacy=False)

        def counts(seed: int) -> tuple[dict[str, int], dict[str, int]]:
            opt = TopologicalCouplingOptimizer(
                n_qubits=2,
                initial_K=np.zeros((2, 2)),
                omega=np.array([1.0, 2.0]),
                rng=np.random.default_rng(seed),
            )
            return opt._simulate_measurement_counts(psi, shots=64)

        first = counts(5)
        assert first == counts(5)
        assert sum(first[0].values()) == 64
        assert sum(first[1].values()) == 64
        np.testing.assert_equal(np.random.get_state(legacy=False), before)

    def test_default_generator_is_private_to_each_optimizer(self) -> None:
        """Without ``rng`` each optimizer gets its own new generator."""
        first = TopologicalCouplingOptimizer(2, np.zeros((2, 2)), np.array([1.0, 2.0]))
        second = TopologicalCouplingOptimizer(2, np.zeros((2, 2)), np.array([1.0, 2.0]))
        assert isinstance(first.rng, np.random.Generator)
        assert first.rng is not second.rng
