# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Krylov Complexity
"""Krylov complexity at the synchronization transition.

Krylov complexity K(t) measures operator spreading in Hilbert space
under Heisenberg evolution O(t) = e^{iHt} O e^{-iHt}. The Lanczos
algorithm builds an orthonormal Krylov basis {|O_n)} from repeated
application of the Liouvillian L = [H, ·]:

    L|O_n) = b_{n+1}|O_{n+1}) + b_n|O_{n-1})

The Lanczos coefficients b_n encode the operator growth rate.
Krylov complexity: K(t) = Σ_n n |φ_n(t)|²

For chaotic systems: K(t) grows exponentially then linearly.
For integrable systems: K(t) grows polynomially.

At a QPT, b_n may show universal scaling. For second-order
transitions: del Campo et al. (arXiv:2510.13947) established
Kibble-Zurek scaling of Krylov cumulants. For BKT (infinite-order):
the KZ mechanism breaks down due to the essential singularity in
the correlation length. The Krylov complexity behavior at BKT
is completely open.

Prior art: Krylov + QPT for Ising (del Campo 2025), XXZ chaos
(Afrasiar 2024). Krylov + BKT or synchronization: NONE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


@dataclass
class KrylovResult:
    """Krylov complexity computation result."""

    lanczos_b: np.ndarray  # Lanczos coefficients b_n
    times: np.ndarray
    krylov_complexity: np.ndarray  # K(t) = Σ n |φ_n(t)|²
    peak_complexity: float
    n_lanczos: int  # number of Lanczos steps before convergence


def _liouvillian_action(H: np.ndarray, op: np.ndarray) -> np.ndarray:
    """L(op) = [H, op] = H·op - op·H."""
    result: np.ndarray = H @ op - op @ H
    return result


def _operator_inner_product(A: np.ndarray, B: np.ndarray) -> float:
    """Hilbert-Schmidt inner product (A|B) = Tr(A† B) / d."""
    d = A.shape[0]
    return float(np.real(np.trace(A.conj().T @ B)) / d)


def lanczos_coefficients(
    H: np.ndarray,
    O_init: np.ndarray,
    max_steps: int = 50,
    tol: float = 1e-12,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute Lanczos coefficients b_n and Krylov basis.

    Uses the operator Lanczos algorithm on the Liouvillian L = [H, ·].
    Rust fast path avoids Python per-step overhead for the commutator loop.

    Returns (b_coefficients, krylov_basis).
    """
    try:
        import scpn_quantum_engine as _engine

        dim = H.shape[0]
        H_c = np.asarray(H, dtype=complex)
        O_c = np.asarray(O_init, dtype=complex)
        b_vec = _engine.lanczos_b_coefficients(
            np.ascontiguousarray(H_c.real).ravel(),
            np.ascontiguousarray(H_c.imag).ravel(),
            np.ascontiguousarray(O_c.real).ravel(),
            np.ascontiguousarray(O_c.imag).ravel(),
            dim,
            max_steps,
            tol,
        )
        b_arr = np.array(b_vec)
        # Rust path doesn't return basis — build placeholder list
        dummy_basis = [np.empty(0)] * (len(b_arr) + 1)
        return b_arr, dummy_basis
    except (ImportError, AttributeError):
        pass

    # Normalize initial operator
    norm_0 = np.sqrt(_operator_inner_product(O_init, O_init))
    if norm_0 < tol:
        return np.array([0.0]), [O_init]

    O_prev = np.zeros_like(O_init)
    O_curr = O_init / norm_0

    basis = [O_curr.copy()]
    b_list: list[float] = []

    for _n in range(max_steps):
        # A_next = L(O_curr) - b_n * O_prev
        A_next = _liouvillian_action(H, O_curr)
        if len(b_list) > 0:
            A_next = A_next - b_list[-1] * O_prev

        # Remove component along current basis vector
        a_n = _operator_inner_product(O_curr, A_next)
        A_next = A_next - a_n * O_curr

        # b_{n+1} = ||A_next||
        b_next = np.sqrt(max(0.0, _operator_inner_product(A_next, A_next)))

        if b_next < tol:
            break

        b_list.append(b_next)
        O_prev = O_curr.copy()
        O_curr = A_next / b_next
        basis.append(O_curr.copy())

    return np.array(b_list), basis


def krylov_complexity(
    H: np.ndarray,
    O_init: np.ndarray,
    t_max: float = 10.0,
    n_times: int = 100,
    max_lanczos: int = 50,
) -> KrylovResult:
    """Compute Krylov complexity K(t) for operator O under H evolution.

    K(t) = Σ_n n |φ_n(t)|² where φ_n(t) are expansion coefficients
    in the Krylov basis, obtained by solving the tridiagonal system:
    i dφ_n/dt = b_{n+1} φ_{n+1} + b_n φ_{n-1}
    """
    b_coeffs, basis = lanczos_coefficients(H, O_init, max_lanczos)
    n_basis = len(b_coeffs) + 1  # basis has one more element than b_coeffs

    if n_basis < 2:
        times = np.linspace(0, t_max, n_times)
        return KrylovResult(
            lanczos_b=b_coeffs,
            times=times,
            krylov_complexity=np.zeros(n_times),
            peak_complexity=0.0,
            n_lanczos=0,
        )

    # Build tridiagonal Hamiltonian in Krylov space
    H_krylov = np.zeros((n_basis, n_basis))
    for i in range(len(b_coeffs)):
        H_krylov[i, i + 1] = b_coeffs[i]
        H_krylov[i + 1, i] = b_coeffs[i]

    # Time evolution: |φ(t)⟩ = e^{-iH_krylov t} |0⟩
    times = np.linspace(0, t_max, n_times)
    phi_0 = np.zeros(n_basis)
    phi_0[0] = 1.0

    K_t: np.ndarray = np.zeros(n_times)
    from scipy.linalg import expm

    for idx, t in enumerate(times):
        U = expm(-1j * H_krylov * t)
        phi_t = U @ phi_0
        probs = np.abs(phi_t) ** 2
        K_t[idx] = float(np.sum(np.arange(n_basis) * probs))

    return KrylovResult(
        lanczos_b=b_coeffs,
        times=times,
        krylov_complexity=K_t,
        peak_complexity=float(np.max(K_t)),
        n_lanczos=len(b_coeffs),
    )


def krylov_vs_coupling(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray | None = None,
    t_max: float = 10.0,
    n_times: int = 50,
) -> dict[str, list[float]]:
    """Scan Krylov complexity diagnostics across coupling strength.

    Uses Z_0 (first qubit Pauli-Z) as the probe operator.
    """
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 10)

    n = len(omega)

    # Probe operator: Z on first qubit
    Z0 = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(2**n):
        # Z eigenvalue: +1 if qubit 0 is |0⟩, -1 if |1⟩
        bit = (i >> 0) & 1
        Z0[i, i] = 1.0 - 2.0 * bit

    results: dict[str, list[float]] = {
        "K_base": [],
        "peak_complexity": [],
        "n_lanczos": [],
        "mean_b": [],
    }

    for kb in k_range:
        K = float(kb) * K_topology
        H = knm_to_dense_matrix(K, omega)
        kr = krylov_complexity(H, Z0, t_max, n_times)
        results["K_base"].append(float(kb))
        results["peak_complexity"].append(kr.peak_complexity)
        results["n_lanczos"].append(float(kr.n_lanczos))
        results["mean_b"].append(float(np.mean(kr.lanczos_b)) if len(kr.lanczos_b) > 0 else 0.0)

    return results
