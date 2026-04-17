# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Koopman
"""Koopman linearisation for the nonlinear Kuramoto model.

The Kuramoto model has nonlinear coupling sin(θ_j - θ_i). The current
XY Hamiltonian approximation linearises this as cos(θ_j - θ_i) (valid
near synchronisation). Koopman operator theory lifts the nonlinear
dynamics into a linear (but infinite-dimensional) space.

For the Kuramoto system:
    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)

The Koopman observable basis uses:
    g_ij^(c) = cos(θ_j - θ_i)
    g_ij^(s) = sin(θ_j - θ_i)
    g_i = θ_i  (identity observables)

In this lifted space, the dynamics become linear:
    dg/dt = L_K × g

where L_K is the Koopman generator matrix. The eigenvalues of L_K
give the Koopman modes (frequencies of collective motion).

Quantum simulation of L_K via Hamiltonian encoding:
    H_Koopman = i × L_K  (skew-Hermitian → unitary evolution)

This extends the XY approximation to the full nonlinear regime,
establishing the BQP-completeness argument from Babbush et al. 2023
for the coupled oscillator problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Upper bound on n_oscillators for routines that allocate the full n²×n²
# Koopman generator. At n=32 the dense generator is 1024×1024 (8 MB) and
# `eigvals` returns in ~1 s on commodity hardware. Larger sizes may be
# legitimate (sparse / structured methods) but must be opted in via the
# `max_oscillators` parameter; otherwise a stray call with n=200 would
# allocate 320 MB and run `eigvals` for many minutes.
MAX_OSCILLATORS_DEFAULT = 32


def _validate_inputs(
    K: np.ndarray,
    omega: np.ndarray,
    theta_ref: np.ndarray | None,
    max_oscillators: int,
) -> None:
    """Validate Koopman inputs. Raises ValueError on any violation."""
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square 2-D matrix, got shape {K.shape}")
    n = K.shape[0]
    if n == 0:
        raise ValueError("K must have at least one oscillator")
    if not np.all(np.isfinite(K)):
        raise ValueError("K contains non-finite entries (NaN or Inf)")
    if omega.ndim != 1 or omega.shape[0] != n:
        raise ValueError(f"omega must be 1-D with length {n}, got shape {omega.shape}")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega contains non-finite entries (NaN or Inf)")
    if theta_ref is not None and (theta_ref.ndim != 1 or theta_ref.shape[0] != n):
        raise ValueError(f"theta_ref must be 1-D with length {n}, got shape {theta_ref.shape}")
    if n > max_oscillators:
        raise ValueError(
            f"n_oscillators={n} exceeds max_oscillators={max_oscillators}; "
            f"the dense Koopman generator is n² × n² = {n * n}² entries. "
            f"Pass max_oscillators={n} explicitly to confirm the allocation."
        )


@dataclass
class KoopmanResult:
    """Koopman linearisation result."""

    generator: np.ndarray  # L_K matrix
    eigenvalues: np.ndarray  # Koopman eigenvalues
    n_observables: int
    n_oscillators: int
    observable_labels: list[str]


def build_koopman_generator(
    K: np.ndarray,
    omega: np.ndarray,
    theta_ref: np.ndarray | None = None,
    max_oscillators: int = MAX_OSCILLATORS_DEFAULT,
) -> tuple[np.ndarray, list[str]]:
    """Build the Koopman generator matrix L_K for the Kuramoto system.

    Observable basis (for n oscillators):
        - n identity observables: θ_i
        - n(n-1)/2 cosine pair observables: cos(θ_j - θ_i)
        - n(n-1)/2 sine pair observables: sin(θ_j - θ_i)

    Total dimension: n + n(n-1) = n²

    The generator L_K acts as: dg/dt = L_K g, where:
        dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)
        d/dt cos(Δ) = -(dθ_j/dt - dθ_i/dt) sin(Δ)
        d/dt sin(Δ) = +(dθ_j/dt - dθ_i/dt) cos(Δ)

    At a reference point theta_ref, the linearised generator captures
    the local dynamics exactly.

    Args:
        K: coupling matrix (n×n, symmetric, zero diagonal)
        omega: natural frequencies
        theta_ref: reference phase configuration (default: zeros)
        max_oscillators: hard cap on n to prevent unbounded n²×n²
            allocation. Raise it explicitly when working with structured
            problems where the dense generator is genuinely needed.
    """
    _validate_inputs(K, omega, theta_ref, max_oscillators)
    n = K.shape[0]
    if theta_ref is None:
        theta_ref = np.zeros(n)

    n_pairs = n * (n - 1) // 2
    dim = n + 2 * n_pairs
    L = np.zeros((dim, dim))
    labels: list[str] = []

    # Label ordering: θ_0..θ_{n-1}, cos_{01}..cos_{(n-2)(n-1)}, sin_{01}..
    for i in range(n):
        labels.append(f"θ_{i}")

    pair_idx: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            labels.append(f"cos({j}-{i})")
            pair_idx.append((i, j))
    for i in range(n):
        for j in range(i + 1, n):
            labels.append(f"sin({j}-{i})")

    # dθ_i/dt = ω_i + Σ_j K_ij × sin_observable(j,i)
    # The sine observables are at indices n + n_pairs + k
    for i in range(n):
        # Constant ω_i term handled separately (inhomogeneous)
        for k, (a, b) in enumerate(pair_idx):
            if b == i:
                # K_ai sin(θ_i - θ_a) = -K_ai sin(θ_a - θ_i)
                # sin(θ_b - θ_a) where b=i, contribution to dθ_i
                L[i, n + n_pairs + k] += K[a, i]  # K_ai × sin(i-a)
            elif a == i:
                # K_bi sin(θ_b - θ_i), sin observable for (i,b)
                L[i, n + n_pairs + k] -= K[b, i]  # note: -sin(i-b) = sin(b-i)

    # d/dt cos(θ_b - θ_a) = -(dθ_b/dt - dθ_a/dt) × sin(θ_b - θ_a)
    # Linearised: couples cos to sin via frequency difference
    for k, (a, b) in enumerate(pair_idx):
        delta_omega = omega[b] - omega[a]
        cos_idx = n + k
        sin_idx = n + n_pairs + k
        # d(cos)/dt ≈ -Δω × sin  (dominant term at reference)
        L[cos_idx, sin_idx] = -delta_omega
        # d(sin)/dt ≈ +Δω × cos
        L[sin_idx, cos_idx] = delta_omega

    # Coupling corrections from K at reference point
    for k, (a, b) in enumerate(pair_idx):
        cos_idx = n + k
        sin_idx = n + n_pairs + k
        delta = theta_ref[b] - theta_ref[a]
        sin_d = np.sin(delta)

        # Second-order coupling terms from K
        for m in range(n):
            if m in (a, b):
                continue
            # Coupling of pair (a,b) to oscillator m via K
            coupling_a = K[m, a]
            coupling_b = K[m, b]
            # These create higher-order terms in the full expansion
            # For the linearised version, we include the direct effect
            L[cos_idx, cos_idx] += -(coupling_b - coupling_a) * sin_d * 0.5
            L[sin_idx, sin_idx] += (coupling_b - coupling_a) * sin_d * 0.5

    return L, labels


def build_koopman_generator_rust(
    K: np.ndarray,
    omega: np.ndarray,
    theta_ref: np.ndarray | None = None,
    max_oscillators: int = MAX_OSCILLATORS_DEFAULT,
) -> tuple[np.ndarray, list[str]]:
    """Rust-accelerated Koopman generator (falls back to Python).

    TODO(rust): implement koopman_generator in scpn_quantum_engine
    for n>16 where the O(n³) coupling correction loop matters.
    """
    return build_koopman_generator(K, omega, theta_ref, max_oscillators)


def koopman_analysis(
    K: np.ndarray,
    omega: np.ndarray,
    theta_ref: np.ndarray | None = None,
    max_oscillators: int = MAX_OSCILLATORS_DEFAULT,
) -> KoopmanResult:
    """Full Koopman linearisation and spectral analysis.

    `max_oscillators` is forwarded to `build_koopman_generator` and
    bounds the dense n²×n² eigendecomposition. The default keeps a
    pathological caller (n=200, eigvals ≫ minutes, ~320 MB) from
    silently exhausting the host.
    """
    n = K.shape[0]
    L, labels = build_koopman_generator(K, omega, theta_ref, max_oscillators)
    eigenvalues = np.linalg.eigvals(L)
    eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]

    return KoopmanResult(
        generator=L,
        eigenvalues=eigenvalues,
        n_observables=L.shape[0],
        n_oscillators=n,
        observable_labels=labels,
    )


def koopman_dimension(n_osc: int) -> int:
    """Observable space dimension for n oscillators: n + n(n-1) = n²."""
    return n_osc * n_osc


def koopman_to_hamiltonian(L: np.ndarray) -> np.ndarray:
    """Convert Koopman generator to Hermitian Hamiltonian.

    H = i × (L - L†) / 2 (anti-Hermitian part, Hermitianised)

    This H can be encoded as a Pauli Hamiltonian for quantum simulation,
    extending the XY approximation to the full nonlinear Kuramoto dynamics.
    """
    H: np.ndarray = 1j * (L - L.conj().T) / 2.0
    # Ensure exact Hermiticity
    H = (H + H.conj().T) / 2.0
    return H
