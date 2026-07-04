# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scikit-learn-style estimator API for the Kuramoto inference side
r"""scikit-learn-style estimators wrapping the Kuramoto inference routines.

The inference routines (sparse governing-equation discovery, coupling-function
inference) already do the science; adoption asks for them behind the familiar
scikit-learn estimator protocol so they drop into cross-validation and pipeline
tooling. These estimators provide that protocol — ``fit(X, y)`` / ``predict(X)`` /
``score(X, y)`` with ``get_params`` / ``set_params`` — over ``(n_samples, N)``
phase snapshots ``X`` and phase-velocity targets ``y``, the PySINDy convention.

The protocol is **duck-typed**: scikit-learn is not a dependency of this package,
so nothing here imports it. An estimator is nonetheless usable with
``sklearn.model_selection.GridSearchCV`` and ``sklearn.base.clone`` when
scikit-learn is installed, because those consume the protocol structurally
(constructor-stored hyperparameters exposed by ``get_params``, a ``fit`` returning
``self``, and a higher-is-better ``score``). No new compute is added: the
estimators delegate discovery to the shipped inference routines and reconstruction
to the discovered model, so this is packaging discipline, not new numerics.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from .kuramoto_coupling_function_inference import (
    CouplingFunctionEstimate,
    infer_coupling_function,
)
from .kuramoto_sparse_identification import SparseDynamicsModel, discover_phase_dynamics


def _as_snapshots(values: NDArray[np.float64], *, name: str) -> NDArray[np.float64]:
    """Return ``values`` as a contiguous ``(n_samples, N)`` float64 snapshot array."""

    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional (n_samples, N) array")
    return array


class _KuramotoInferenceEstimator:
    """The shared scikit-learn-style protocol for the inference estimators.

    Subclasses declare their constructor hyperparameter names in ``_PARAMETERS``
    and implement :meth:`fit` and :meth:`predict`; this base supplies the
    ``get_params`` / ``set_params`` introspection and the coefficient-of-
    determination :meth:`score` that make the estimator cross-validation ready.
    """

    _PARAMETERS: tuple[str, ...] = ()

    def __sklearn_tags__(self) -> object:  # pragma: no cover (scikit-learn optional)
        """Return scikit-learn's regressor tags (only invoked when sklearn drives us).

        scikit-learn 1.6+ reaches for this to classify the estimator (e.g. inside
        ``GridSearchCV``). Building a version-correct tags object requires
        scikit-learn's own machinery, so this borrows it from a throwaway
        ``RegressorMixin`` estimator via a lazy import — the module itself never
        imports scikit-learn, keeping it an optional dependency.
        """

        from sklearn.base import BaseEstimator, RegressorMixin

        # sklearn is an untyped optional dependency, so mypy sees its bases as Any.
        class _RegressorTagSource(RegressorMixin, BaseEstimator):  # type: ignore[misc]
            pass

        return _RegressorTagSource().__sklearn_tags__()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return the constructor hyperparameters (the scikit-learn contract)."""

        return {name: getattr(self, name) for name in self._PARAMETERS}

    def set_params(self, **params: object) -> Self:
        """Set constructor hyperparameters in place and return ``self``.

        Raises
        ------
        ValueError
            If a supplied name is not a declared hyperparameter.
        """

        for name, value in params.items():
            if name not in self._PARAMETERS:
                raise ValueError(f"unknown parameter {name!r}; expected one of {self._PARAMETERS}")
            setattr(self, name, value)
        return self

    def fit(self, phases: NDArray[np.float64], derivatives: NDArray[np.float64]) -> Self:
        """Fit the estimator to phase snapshots and their velocities."""

        raise NotImplementedError("subclasses implement fit")

    def predict(self, phases: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict phase velocities at the given phase snapshots."""

        raise NotImplementedError("subclasses implement predict")

    def score(self, phases: NDArray[np.float64], derivatives: NDArray[np.float64]) -> float:
        r"""Return the aggregate coefficient of determination :math:`R^2`.

        ``R^2 = 1 - \sum (y - \hat y)^2 / \sum (y - \bar y)^2`` over every element of
        the ``(n_samples, N)`` velocity target, with the per-output mean ``\bar y``.
        A perfect prediction scores ``1.0``; scikit-learn's cross-validators
        maximise it.
        """

        target = _as_snapshots(derivatives, name="derivatives")
        prediction = self.predict(phases)
        residual = float(np.sum((target - prediction) ** 2))
        total = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
        if total == 0.0:
            return 1.0 if residual == 0.0 else 0.0
        return 1.0 - residual / total


class SparseDynamicsEstimator(_KuramotoInferenceEstimator):
    """A scikit-learn-style wrapper over :func:`discover_phase_dynamics`.

    ``fit`` discovers the sparse governing equations (directed topology and
    trigonometric functional form) from phase snapshots and their velocities by
    sequentially-thresholded least squares; ``predict`` reconstructs the phase
    velocities from the discovered model. The fitted :class:`SparseDynamicsModel`
    is available as ``discovered_model_``.

    Parameters
    ----------
    n_harmonics : int, optional
        The highest harmonic in the candidate library (``>= 1``); default ``1``.
    threshold : float, optional
        The sparsity threshold (``> 0``); default ``0.05``.
    max_iterations : int, optional
        The maximum sequential-thresholding iterations; default ``10``.
    """

    _PARAMETERS = ("n_harmonics", "threshold", "max_iterations")

    def __init__(
        self, *, n_harmonics: int = 1, threshold: float = 0.05, max_iterations: int = 10
    ) -> None:
        self.n_harmonics = n_harmonics
        self.threshold = threshold
        self.max_iterations = max_iterations

    def fit(self, phases: NDArray[np.float64], derivatives: NDArray[np.float64]) -> Self:
        """Discover the sparse model from ``(n_samples, N)`` snapshots and velocities."""

        snapshots = _as_snapshots(phases, name="phases")
        velocities = _as_snapshots(derivatives, name="derivatives")
        self.discovered_model_: SparseDynamicsModel = discover_phase_dynamics(
            snapshots,
            velocities,
            n_harmonics=self.n_harmonics,
            threshold=self.threshold,
            max_iterations=self.max_iterations,
        )
        return self

    def predict(self, phases: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct phase velocities at ``(n_samples, N)`` snapshots.

        Raises
        ------
        AttributeError
            If called before :meth:`fit`.
        """

        if not hasattr(self, "discovered_model_"):
            raise AttributeError("SparseDynamicsEstimator must be fitted before predict")
        snapshots = _as_snapshots(phases, name="phases")
        return np.stack([self.discovered_model_.field(row) for row in snapshots])


class CouplingFunctionEstimator(_KuramotoInferenceEstimator):
    r"""A scikit-learn-style wrapper over :func:`infer_coupling_function`.

    Given the known coupling topology ``K``, ``fit`` infers the shared coupling
    function :math:`\Gamma` and the natural frequencies by physics-informed
    collocation least-squares; ``predict`` reconstructs the phase velocities
    :math:`\dot\theta_i = \omega_i + \sum_j K_{ij}\,\Gamma(\theta_j - \theta_i)`.
    The fitted :class:`CouplingFunctionEstimate` is available as ``estimate_``.

    Parameters
    ----------
    coupling : numpy.ndarray
        The known ``(N, N)`` coupling matrix ``K``.
    n_harmonics : int, optional
        The number of Fourier harmonics to fit (``>= 1``); default ``1``.
    """

    _PARAMETERS = ("coupling", "n_harmonics")

    def __init__(self, *, coupling: NDArray[np.float64], n_harmonics: int = 1) -> None:
        self.coupling = coupling
        self.n_harmonics = n_harmonics

    def fit(self, phases: NDArray[np.float64], derivatives: NDArray[np.float64]) -> Self:
        """Infer ``Γ`` and ``ω`` from ``(n_samples, N)`` snapshots and velocities."""

        snapshots = _as_snapshots(phases, name="phases")
        velocities = _as_snapshots(derivatives, name="derivatives")
        self.estimate_: CouplingFunctionEstimate = infer_coupling_function(
            snapshots,
            velocities,
            np.ascontiguousarray(self.coupling, dtype=np.float64),
            self.n_harmonics,
        )
        return self

    def predict(self, phases: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct phase velocities at ``(n_samples, N)`` snapshots.

        Raises
        ------
        AttributeError
            If called before :meth:`fit`.
        """

        if not hasattr(self, "estimate_"):
            raise AttributeError("CouplingFunctionEstimator must be fitted before predict")
        snapshots = _as_snapshots(phases, name="phases")
        coupling = np.ascontiguousarray(self.coupling, dtype=np.float64)
        frequencies = self.estimate_.frequencies
        predictions = np.empty_like(snapshots)
        for sample, state in enumerate(snapshots):
            differences = state[None, :] - state[:, None]
            gamma = self.estimate_.evaluate(differences)
            predictions[sample] = frequencies + np.sum(coupling * gamma, axis=1)
        return predictions


__all__ = [
    "CouplingFunctionEstimator",
    "SparseDynamicsEstimator",
]
