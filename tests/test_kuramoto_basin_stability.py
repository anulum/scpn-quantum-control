# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for Menck–Kurths basin stability of synchronisation
"""Module-specific tests for :mod:`kuramoto_basin_stability`.

The headline contract is the higher-order linear-vs-basin divergence: a triadic-coupled in-phase
state is *more* linearly stable than a pairwise one yet has a far smaller basin of attraction, so
the two stability measures disagree. The estimate's binomial standard error shrinks with the
sample count, single-node and whole-network perturbations are both supported, and the input
contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel import (
    mean_field_force,
    mean_field_jacobian,
    simplex_mean_field_force,
    simplex_mean_field_jacobian,
)
from scpn_quantum_control.accel.kuramoto_basin_stability import (
    BasinStabilityEstimate,
    synchronisation_basin_stability,
)

_N = 12


def _second_eigenvalue(jacobian: NDArray[np.float64]) -> float:
    """The leading non-Goldstone (second-largest real-part) eigenvalue of a phase Jacobian."""
    eigenvalues = np.sort(np.linalg.eigvals(jacobian).real)[::-1]
    return float(eigenvalues[1])


def _pairwise_force(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    return mean_field_force(theta, 2.0)


def _higher_order_force(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.3 * mean_field_force(theta, 1.0) + simplex_mean_field_force(theta, 3.0, 2)


def test_higher_order_linear_vs_basin_divergence() -> None:
    omega = np.zeros(_N)
    synced = np.zeros(_N)

    pairwise = synchronisation_basin_stability(
        _pairwise_force,
        omega,
        synced,
        n_samples=150,
        n_perturbed_nodes=_N,
        dt=0.05,
        n_steps=300,
        seed=0,
    )
    higher_order = synchronisation_basin_stability(
        _higher_order_force,
        omega,
        synced,
        n_samples=150,
        n_perturbed_nodes=_N,
        dt=0.05,
        n_steps=300,
        seed=0,
    )

    pairwise_eig = _second_eigenvalue(mean_field_jacobian(synced, 2.0))
    higher_eig = _second_eigenvalue(
        0.3 * mean_field_jacobian(synced, 1.0) + simplex_mean_field_jacobian(synced, 3.0, 2)
    )

    # both in-phase states are linearly stable
    assert pairwise_eig < 0.0
    assert higher_eig < 0.0
    # the higher-order state is *more* linearly stable ...
    assert higher_eig < pairwise_eig
    # ... yet its basin is far smaller — the measures diverge
    assert pairwise.basin_fraction > 0.9
    assert higher_order.basin_fraction < 0.2


def test_standard_error_shrinks_with_sample_count() -> None:
    omega = np.zeros(_N)
    synced = np.zeros(_N)
    shared: dict[str, Any] = dict(n_perturbed_nodes=_N, dt=0.05, n_steps=200, seed=1)
    small = synchronisation_basin_stability(_pairwise_force, omega, synced, n_samples=40, **shared)
    large = synchronisation_basin_stability(
        _pairwise_force, omega, synced, n_samples=400, **shared
    )
    assert large.standard_error <= small.standard_error
    assert large.basin_fraction == pytest.approx(small.basin_fraction, abs=0.1)
    # the reported error is the binomial standard error
    p = large.basin_fraction
    assert large.standard_error == pytest.approx(np.sqrt(p * (1.0 - p) / large.n_samples))


def test_single_node_perturbation_supported() -> None:
    omega = np.zeros(_N)
    synced = np.zeros(_N)
    estimate = synchronisation_basin_stability(
        _pairwise_force,
        omega,
        synced,
        n_samples=60,
        n_perturbed_nodes=1,
        dt=0.05,
        n_steps=200,
        seed=2,
    )
    assert isinstance(estimate, BasinStabilityEstimate)
    assert 0.0 <= estimate.basin_fraction <= 1.0
    assert estimate.n_returned + (estimate.n_samples - estimate.n_returned) == estimate.n_samples
    # single-node perturbation of a strongly pairwise-locked state almost always returns
    assert estimate.basin_fraction > 0.9


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"omega": np.zeros((2, 2))}, "omega must be a non-empty"),
        ({"synchronised_state": np.zeros(3)}, "synchronised_state must have shape"),
        ({"n_samples": 0}, "n_samples must be positive"),
        ({"n_perturbed_nodes": 0}, "n_perturbed_nodes must satisfy"),
        ({"n_perturbed_nodes": _N + 1}, "n_perturbed_nodes must satisfy"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
        ({"perturbation": 0.0}, "perturbation must be positive"),
        ({"return_tolerance": 0.0}, "return_tolerance must be positive"),
    ],
)
def test_validation_errors(mutation: dict[str, Any], message: str) -> None:
    call: dict[str, Any] = {
        "omega": np.zeros(_N),
        "synchronised_state": np.zeros(_N),
        "n_samples": 10,
        "n_perturbed_nodes": _N,
        "dt": 0.05,
        "n_steps": 10,
        "perturbation": np.pi,
        "return_tolerance": 0.02,
        "seed": 0,
    }
    call.update(mutation)
    with pytest.raises(ValueError, match=message):
        synchronisation_basin_stability(
            _pairwise_force,
            call["omega"],
            call["synchronised_state"],
            n_samples=call["n_samples"],
            n_perturbed_nodes=call["n_perturbed_nodes"],
            dt=call["dt"],
            n_steps=call["n_steps"],
            perturbation=call["perturbation"],
            return_tolerance=call["return_tolerance"],
            seed=call["seed"],
        )
