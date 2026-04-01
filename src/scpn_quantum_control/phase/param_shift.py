# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Parameter-Shift Gradient Rule
"""Analytic gradient computation via the parameter-shift rule.

For a parameterised quantum circuit U(θ) and observable H,
the gradient of ⟨H⟩ = ⟨0|U†(θ)HU(θ)|0⟩ is:

  ∂⟨H⟩/∂θ_k = ½[⟨H⟩(θ_k + π/2) - ⟨H⟩(θ_k - π/2)]

This is exact (no finite-difference error) and works on real hardware.

Reference: Mitarai et al., PRA 98, 032309 (2018);
Schuld et al., PRA 99, 032331 (2019).
Tequila and PennyLane implement this natively; we provide a
standalone version compatible with our VQE and NQS pipelines.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def parameter_shift_gradient(
    cost_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Compute gradient of cost_fn via parameter-shift rule.

    Parameters
    ----------
    cost_fn : callable
        Function mapping parameter vector → scalar expectation value.
        Must accept numpy array of shape (n_params,).
    params : array (n_params,)
        Current parameter values.
    shift : float
        Shift amount. Default π/2 for standard Pauli rotation gates.

    Returns
    -------
    grad : array (n_params,)
        Gradient vector.
    """
    n = len(params)
    grad = np.zeros(n)
    for k in range(n):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[k] += shift
        params_minus[k] -= shift
        grad[k] = (cost_fn(params_plus) - cost_fn(params_minus)) / (2 * np.sin(shift))
    return grad


def vqe_with_param_shift(
    cost_fn: Callable[[np.ndarray], float],
    n_params: int,
    learning_rate: float = 0.1,
    n_iterations: int = 100,
    seed: int | None = None,
) -> dict:
    """Run VQE optimisation using parameter-shift gradients.

    Parameters
    ----------
    cost_fn : callable
        Maps parameter vector → energy expectation value.
    n_params : int
        Number of variational parameters.
    learning_rate : float
        Gradient descent step size.
    n_iterations : int
        Number of optimisation steps.
    seed : int or None
        RNG seed.

    Returns
    -------
    dict with keys: optimal_params, energy, energy_history, grad_norms
    """
    rng = np.random.default_rng(seed)
    params = rng.normal(0, 0.1, n_params)
    energy_history = []
    grad_norms = []

    for _step in range(n_iterations):
        energy = cost_fn(params)
        energy_history.append(energy)

        grad = parameter_shift_gradient(cost_fn, params)
        grad_norms.append(float(np.linalg.norm(grad)))
        params -= learning_rate * grad

    final_energy = cost_fn(params)
    energy_history.append(final_energy)

    return {
        "optimal_params": params,
        "energy": final_energy,
        "energy_history": energy_history,
        "grad_norms": grad_norms,
    }
