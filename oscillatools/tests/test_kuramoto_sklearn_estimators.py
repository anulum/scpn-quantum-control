# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scikit-learn-style inference estimator tests
"""Tests for the duck-typed scikit-learn-style Kuramoto inference estimators.

The load-bearing checks are the fit→predict round trips: both estimators are fit
on velocities generated from a known networked Kuramoto model and must
reconstruct those velocities (exactly for the coupling-function estimator given
the topology, and to the sparsity threshold for the sparse estimator). The
scikit-learn protocol (``get_params`` / ``set_params`` / ``score``, the
fit-returns-self contract, and — when scikit-learn is installed — a
``GridSearchCV`` search) is checked directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.kuramoto_sklearn_estimators import (
    CouplingFunctionEstimator,
    SparseDynamicsEstimator,
    _KuramotoInferenceEstimator,
)


def _ring_coupling(size: int, strength: float) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float64)
    for node in range(size):
        matrix[node, (node + 1) % size] = strength
        matrix[node, (node - 1) % size] = strength
    return matrix


def _kuramoto_dataset(
    *, size: int = 4, samples: int = 40, coupling_strength: float = 0.8, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (phases, derivatives, coupling, frequencies) from a networked model.

    Frequencies are kept well above a 0.05 sparsity threshold so the sparse
    estimator recovers the model exactly.
    """

    rng = np.random.default_rng(seed)
    coupling = _ring_coupling(size, coupling_strength)
    frequencies = rng.uniform(0.3, 0.9, size) * rng.choice([-1.0, 1.0], size)
    phases = rng.uniform(-np.pi, np.pi, (samples, size))
    derivatives = np.empty_like(phases)
    for sample, state in enumerate(phases):
        difference = state[None, :] - state[:, None]
        derivatives[sample] = frequencies + np.sum(coupling * np.sin(difference), axis=1)
    return phases, derivatives, coupling, frequencies


# --------------------------------------------------------------------------- #
# Shared protocol base
# --------------------------------------------------------------------------- #
class TestInferenceEstimatorBase:
    def test_base_fit_and_predict_are_abstract(self) -> None:
        base = _KuramotoInferenceEstimator()
        with pytest.raises(NotImplementedError, match="fit"):
            base.fit(np.zeros((2, 2)), np.zeros((2, 2)))
        with pytest.raises(NotImplementedError, match="predict"):
            base.predict(np.zeros((2, 2)))

    def test_set_params_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            SparseDynamicsEstimator().set_params(learning_rate=0.1)

    def test_get_set_params_round_trip(self) -> None:
        estimator = SparseDynamicsEstimator(n_harmonics=2, threshold=0.1, max_iterations=5)
        assert estimator.get_params() == {
            "n_harmonics": 2,
            "threshold": 0.1,
            "max_iterations": 5,
        }
        returned = estimator.set_params(threshold=0.2, n_harmonics=3)
        assert returned is estimator
        assert estimator.threshold == 0.2
        assert estimator.n_harmonics == 3

    def test_score_handles_zero_variance_target(self) -> None:
        class _ConstantPredictor(_KuramotoInferenceEstimator):
            def predict(self, phases: np.ndarray) -> np.ndarray:
                return np.full((phases.shape[0], phases.shape[1]), 5.0)

        estimator = _ConstantPredictor()
        constant = np.full((6, 3), 5.0)
        # Zero-variance target reconstructed exactly -> perfect score.
        assert estimator.score(np.zeros((6, 3)), constant) == 1.0
        # Zero-variance target missed -> zero score (no variance to explain).
        assert estimator.score(np.zeros((6, 3)), np.full((6, 3), 3.0)) == 0.0


# --------------------------------------------------------------------------- #
# SparseDynamicsEstimator
# --------------------------------------------------------------------------- #
class TestSparseDynamicsEstimator:
    def test_fit_returns_self_and_exposes_the_model(self) -> None:
        phases, derivatives, _, _ = _kuramoto_dataset()
        estimator = SparseDynamicsEstimator(n_harmonics=1, threshold=0.05)
        returned = estimator.fit(phases, derivatives)
        assert returned is estimator
        assert estimator.discovered_model_.active_terms > 0

    def test_predict_recovers_the_velocities(self) -> None:
        phases, derivatives, _, _ = _kuramoto_dataset()
        estimator = SparseDynamicsEstimator(n_harmonics=1, threshold=0.05).fit(phases, derivatives)
        prediction = estimator.predict(phases)
        assert prediction.shape == derivatives.shape
        np.testing.assert_allclose(prediction, derivatives, atol=1e-9)
        assert estimator.score(phases, derivatives) == pytest.approx(1.0, abs=1e-9)

    def test_predict_generalises_to_unseen_states(self) -> None:
        phases, derivatives, coupling, frequencies = _kuramoto_dataset()
        estimator = SparseDynamicsEstimator(n_harmonics=1, threshold=0.05).fit(phases, derivatives)
        rng = np.random.default_rng(99)
        held_out = rng.uniform(-np.pi, np.pi, (10, coupling.shape[0]))
        expected = np.empty_like(held_out)
        for sample, state in enumerate(held_out):
            difference = state[None, :] - state[:, None]
            expected[sample] = frequencies + np.sum(coupling * np.sin(difference), axis=1)
        np.testing.assert_allclose(estimator.predict(held_out), expected, atol=1e-9)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(AttributeError, match="must be fitted"):
            SparseDynamicsEstimator().predict(np.zeros((3, 4)))

    def test_rejects_non_two_dimensional_input(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            SparseDynamicsEstimator().fit(np.zeros(4), np.zeros(4))


# --------------------------------------------------------------------------- #
# CouplingFunctionEstimator
# --------------------------------------------------------------------------- #
class TestCouplingFunctionEstimator:
    def test_fit_returns_self_and_exposes_the_estimate(self) -> None:
        phases, derivatives, coupling, _ = _kuramoto_dataset()
        estimator = CouplingFunctionEstimator(coupling=coupling, n_harmonics=1)
        returned = estimator.fit(phases, derivatives)
        assert returned is estimator
        assert estimator.estimate_.frequencies.shape == (coupling.shape[0],)

    def test_predict_reconstructs_the_velocities_exactly(self) -> None:
        phases, derivatives, coupling, _ = _kuramoto_dataset()
        estimator = CouplingFunctionEstimator(coupling=coupling, n_harmonics=1).fit(
            phases, derivatives
        )
        prediction = estimator.predict(phases)
        assert prediction.shape == derivatives.shape
        np.testing.assert_allclose(prediction, derivatives, atol=1e-9)
        assert estimator.score(phases, derivatives) == pytest.approx(1.0, abs=1e-9)

    def test_get_params_includes_the_coupling(self) -> None:
        coupling = _ring_coupling(3, 0.5)
        estimator = CouplingFunctionEstimator(coupling=coupling, n_harmonics=2)
        params = estimator.get_params()
        assert params["n_harmonics"] == 2
        np.testing.assert_array_equal(params["coupling"], coupling)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(AttributeError, match="must be fitted"):
            CouplingFunctionEstimator(coupling=_ring_coupling(3, 0.5)).predict(np.zeros((2, 3)))


# --------------------------------------------------------------------------- #
# scikit-learn interoperability (skipped when scikit-learn is absent)
# --------------------------------------------------------------------------- #
def test_gridsearchcv_compatibility() -> None:
    model_selection = pytest.importorskip("sklearn.model_selection")
    phases, derivatives, _, _ = _kuramoto_dataset(samples=60)
    search = model_selection.GridSearchCV(
        SparseDynamicsEstimator(),
        {"threshold": [0.05, 0.2], "n_harmonics": [1, 2]},
        cv=3,
    )
    search.fit(phases, derivatives)
    assert search.best_params_["threshold"] in (0.05, 0.2)
    assert hasattr(search, "best_score_")


def test_clone_compatibility() -> None:
    base = pytest.importorskip("sklearn.base")
    original = SparseDynamicsEstimator(n_harmonics=2, threshold=0.3, max_iterations=7)
    cloned = base.clone(original)
    assert cloned.get_params() == original.get_params()
    assert cloned is not original
