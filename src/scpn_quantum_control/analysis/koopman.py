# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Koopman
"""Finite local Koopman-style closure for the Kuramoto model.

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

The exact Koopman generator acts on an infinite observable space.  This module
constructs a finite, reference-point-dependent local closure:
    dg/dt = L_K × g

where ``L_K`` is the finite matrix implemented here. Its eigenvalues are local
closure diagnostics; they are not certified eigenvalues of an exact invariant
Koopman subspace.

The helper :func:`koopman_to_hamiltonian` keeps only the anti-Hermitian part of
``L_K`` and multiplies it by ``i``. This Hermitian projection is suitable for
matrix experiments, but it is not dynamically equivalent to the discarded
symmetric part.

No function in this module establishes full nonlinear closure,
BQP-completeness, quantum advantage, or a production control route.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._rust_accel import optional_rust_engine

# Upper bound on n_oscillators for routines that allocate the full n²×n²
# Koopman generator. At n=32 the dense generator is 1024×1024 (8 MB) and
# `eigvals` returns in ~1 s on commodity hardware. Larger sizes may be
# legitimate (sparse / structured methods) but must be opted in via the
# `max_oscillators` parameter; otherwise a stray call with n=200 would
# allocate 320 MB and run `eigvals` for many minutes.
MAX_OSCILLATORS_DEFAULT = 32


def _validate_inputs(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    theta_ref: NDArray[np.float64] | None,
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


def _observable_labels(n: int) -> list[str]:
    labels = [f"θ_{i}" for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            labels.append(f"cos({j}-{i})")
    for i in range(n):
        for j in range(i + 1, n):
            labels.append(f"sin({j}-{i})")
    return labels


@dataclass
class KoopmanResult:
    """Finite local closure and its dense spectrum.

    Attributes
    ----------
    generator
        Reference-point-dependent finite observable matrix.
    eigenvalues
        Dense eigenvalues sorted by descending absolute magnitude.
    n_observables
        Matrix dimension, equal to ``n_oscillators**2``.
    n_oscillators
        Number of input oscillators.
    observable_labels
        Labels for identity, cosine-pair, and sine-pair coordinates.

    """

    generator: NDArray[np.float64]  # L_K matrix
    eigenvalues: NDArray[np.complex128]  # Koopman eigenvalues
    n_observables: int
    n_oscillators: int
    observable_labels: list[str]


def build_koopman_generator(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    theta_ref: NDArray[np.float64] | None = None,
    max_oscillators: int = MAX_OSCILLATORS_DEFAULT,
) -> tuple[NDArray[np.float64], list[str]]:
    """Build the finite local observable matrix for a Kuramoto system.

    Observable basis (for n oscillators):
        - n identity observables: θ_i
        - n(n-1)/2 cosine pair observables: cos(θ_j - θ_i)
        - n(n-1)/2 sine pair observables: sin(θ_j - θ_i)

    Total dimension: n + n(n-1) = n²

    The generator L_K acts as: dg/dt = L_K g, where:
        dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)
        d/dt cos(Δ) = -(dθ_j/dt - dθ_i/dt) sin(Δ)
        d/dt sin(Δ) = +(dθ_j/dt - dθ_i/dt) cos(Δ)

    The pair-observable derivatives are truncated to the documented finite
    basis. The matrix is therefore a local closure diagnostic, not an exact
    finite-dimensional representation of the nonlinear flow.

    Parameters
    ----------
    K
        Finite coupling matrix. Shape and finiteness are validated; callers
        remain responsible for any symmetry or diagonal convention.
    omega
        Finite natural-frequency vector with one entry per oscillator.
    theta_ref
        Reference phase configuration. Defaults to the all-zero point.
    max_oscillators
        Hard cap on ``n`` before allocating the dense ``n² × n²`` matrix.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Finite local matrix and matching observable labels.

    Raises
    ------
    ValueError
        If shapes, finiteness, or the allocation cap are invalid.

    """
    _validate_inputs(K, omega, theta_ref, max_oscillators)
    n = K.shape[0]
    if theta_ref is None:
        theta_ref = np.zeros(n)

    n_pairs = n * (n - 1) // 2
    dim = n + 2 * n_pairs
    L = np.zeros((dim, dim))
    labels = _observable_labels(n)

    pair_idx: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_idx.append((i, j))

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
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    theta_ref: NDArray[np.float64] | None = None,
    max_oscillators: int = MAX_OSCILLATORS_DEFAULT,
    *,
    require_rust: bool = False,
) -> tuple[NDArray[np.float64], list[str]]:
    """Rust-preferred Koopman generator with explicit Python fallback.

    When the optional ``scpn_quantum_engine.koopman_generator`` export is
    available, this function routes through it and reconstructs the canonical
    Python observable labels. If the native extension or export is unavailable,
    the function falls back to :func:`build_koopman_generator` unless
    ``require_rust=True`` is set.
    """
    _validate_inputs(K, omega, theta_ref, max_oscillators)
    n = K.shape[0]
    theta = np.zeros(n) if theta_ref is None else theta_ref
    labels = _observable_labels(n)

    engine = optional_rust_engine()
    rust_builder = None if engine is None else getattr(engine, "koopman_generator", None)
    if callable(rust_builder):
        generator = np.asarray(
            rust_builder(
                np.ascontiguousarray(K, dtype=np.float64),
                np.ascontiguousarray(omega, dtype=np.float64),
                np.ascontiguousarray(theta, dtype=np.float64),
            ),
            dtype=np.float64,
        )
        expected_shape = (n * n, n * n)
        if generator.shape != expected_shape:
            raise ValueError(
                "scpn_quantum_engine.koopman_generator returned shape "
                f"{generator.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(generator)):
            raise ValueError("scpn_quantum_engine.koopman_generator returned non-finite entries")
        return generator, labels

    if require_rust:
        raise ImportError("scpn_quantum_engine.koopman_generator is unavailable")

    return build_koopman_generator(K, omega, theta_ref, max_oscillators)


def koopman_analysis(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    theta_ref: NDArray[np.float64] | None = None,
    max_oscillators: int = MAX_OSCILLATORS_DEFAULT,
) -> KoopmanResult:
    """Build and diagonalize the finite local observable closure.

    Parameters are forwarded to :func:`build_koopman_generator`.
    ``max_oscillators`` bounds the dense ``n² × n²`` eigendecomposition.

    Returns
    -------
    KoopmanResult
        Closure matrix, sorted spectrum, dimensions, and labels.

    Notes
    -----
    The returned spectrum characterizes this finite local closure only.

    """
    n = K.shape[0]
    L, labels = build_koopman_generator(K, omega, theta_ref, max_oscillators)
    eigenvalues = np.linalg.eigvals(L).astype(np.complex128)
    eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]

    return KoopmanResult(
        generator=L,
        eigenvalues=eigenvalues,
        n_observables=L.shape[0],
        n_oscillators=n,
        observable_labels=labels,
    )


def koopman_dimension(n_osc: int) -> int:
    """Return the finite basis dimension ``n + n(n-1) = n²``.

    Parameters
    ----------
    n_osc
        Number of oscillators.

    """
    return n_osc * n_osc


def koopman_to_hamiltonian(L: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Project a real closure matrix onto a Hermitian matrix.

    H = i × (L - L†) / 2 (anti-Hermitian part, Hermitianised)

    Parameters
    ----------
    L
        Finite local closure matrix.

    Returns
    -------
    numpy.ndarray
        ``i(L-L†)/2``, symmetrized to exact numerical Hermiticity.

    Notes
    -----
    The projection discards the symmetric part of ``L``. It is not a proof of
    dynamical equivalence to the nonlinear Kuramoto system.

    """
    H: NDArray[np.complex128] = (1j * (L - L.conj().T) / 2.0).astype(np.complex128)
    # Ensure exact Hermiticity
    H = ((H + H.conj().T) / 2.0).astype(np.complex128)
    return H
