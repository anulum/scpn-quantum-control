# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX-Accelerated Neural Quantum State
"""JAX-based exact-enumeration RBM wavefunction with automatic differentiation.

This optional research path replaces the NumPy implementation's central finite
differences with JAX ``jit`` and ``grad``. It has no committed isolated speed
benchmark and fails closed with ``ImportError`` when JAX is unavailable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..dense_budget import require_dense_allocation

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def is_jax_available() -> bool:
    """Check if JAX is installed."""
    return _JAX_AVAILABLE


def jax_rbm_energy(
    params: dict[str, Any],
    H: Any,
    n: int,
) -> Any:
    """Compute ⟨ψ|H|ψ⟩ for RBM wavefunction using JAX.

    params: dict with keys 'a' (visible biases), 'b' (hidden biases), 'W' (weights)
    H: dense Hamiltonian as jax array
    n: number of qubits
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX not installed: pip install jax jaxlib")

    a = params["a"]
    b = params["b"]
    W = params["W"]

    dim = 2**n

    def log_psi(sigma: Any) -> Any:
        theta = W @ sigma + b
        return jnp.sum(a * sigma) + jnp.sum(jnp.log(jnp.cosh(theta)))

    # Build all spin configurations
    configs = jnp.array(
        [[1 - 2 * ((k >> i) & 1) for i in range(n)] for k in range(dim)],
        dtype=jnp.float32,
    )

    log_psis = vmap(log_psi)(configs)
    psi = jnp.exp(log_psis)
    psi = psi / jnp.linalg.norm(psi)

    energy = jnp.real(psi.conj() @ H @ psi)
    return energy


def jax_vmc_ground_state(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    n_hidden: int | None = None,
    learning_rate: float = 0.01,
    n_iterations: int = 200,
    seed: int = 42,
    *,
    max_dense_gib: float | None = None,
) -> dict[str, Any]:
    """Exact-enumeration ground-state search with JAX auto-differentiation."""
    if not _JAX_AVAILABLE:
        raise ImportError("JAX not installed: pip install jax jaxlib")

    from ..bridge.knm_hamiltonian import knm_to_dense_matrix

    coupling = np.asarray(K, dtype=float)
    frequencies = np.asarray(omega, dtype=float)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("K must be a square rank-2 coupling matrix")
    n = coupling.shape[0]
    if n < 1:
        raise ValueError("JAX NQS requires at least one visible spin")
    if n > 12:
        raise ValueError(f"Exact JAX NQS for n<=12 (got {n})")
    if frequencies.ndim != 1 or frequencies.shape != (n,):
        raise ValueError("omega must be a rank-1 vector matching K")
    if not np.all(np.isfinite(coupling)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("K and omega must contain only finite values")
    if not np.allclose(coupling, coupling.T, rtol=0.0, atol=1e-12):
        raise ValueError("K must be symmetric")
    if n_hidden is not None and (
        not isinstance(n_hidden, int) or isinstance(n_hidden, bool) or n_hidden <= 0
    ):
        raise ValueError("n_hidden must be a positive integer when supplied")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not isinstance(n_iterations, int) or isinstance(n_iterations, bool) or n_iterations <= 0:
        raise ValueError("n_iterations must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    n_hid = 2 * n if n_hidden is None else n_hidden
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=2,
        object_count=1,
        max_gib=max_dense_gib,
        label="JAX NQS dense Hamiltonian workspace",
    )
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=1,
        object_count=4,
        max_gib=max_dense_gib,
        label="JAX NQS dense exact-enumeration workspace",
    )
    hamiltonian = knm_to_dense_matrix(coupling, frequencies, max_dense_gib=max_dense_gib)
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("JAX NQS Hamiltonian must contain only finite values")
    if not np.allclose(hamiltonian, hamiltonian.conj().T, rtol=0.0, atol=1e-12):
        raise ValueError("JAX NQS Hamiltonian must be Hermitian")
    if not np.allclose(hamiltonian.imag, 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("JAX NQS supports only Hamiltonians with negligible imaginary entries")
    H = jnp.array(hamiltonian.real, dtype=jnp.float32)

    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        "a": 0.01 * jax.random.normal(k1, (n,)),
        "b": 0.01 * jax.random.normal(k2, (n_hid,)),
        "W": 0.01 * jax.random.normal(k3, (n_hid, n)),
    }

    @jit  # type: ignore[untyped-decorator]
    def loss_fn(p: dict[str, Any]) -> Any:
        return jax_rbm_energy(p, H, n)

    grad_fn = jit(grad(loss_fn))

    energy_history = []
    for _step in range(n_iterations):
        e = float(loss_fn(params))
        energy_history.append(e)
        grads = grad_fn(params)
        params = {k: params[k] - learning_rate * grads[k] for k in params}

    final_energy = float(loss_fn(params))
    energy_history.append(final_energy)

    return {
        "energy": final_energy,
        "energy_history": energy_history,
        "params": {k: np.array(v) for k, v in params.items()},
        "n_params": n + n_hid + n_hid * n,
    }
