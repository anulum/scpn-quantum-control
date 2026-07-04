# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Linear stability spectrum of a networked Kuramoto state
"""Linear stability spectrum of a networked Kuramoto phase configuration.

Linearising the networked Kuramoto dynamics ``θ̇_j = ω_j + Σ_k K_jk sin(θ_k − θ_j)`` about a
configuration ``θ*`` gives the stability Jacobian ``J_jl = K_jl cos(θ*_l − θ*_j)`` (``l ≠ j``)
with diagonal ``J_jj = −Σ_{k≠j} K_jk cos(θ*_k − θ*_j)``. Its spectrum classifies the linear
stability of synchronisation on the network: an eigenvalue with a positive real part is a growing
perturbation, so a configuration is linearly stable when every transverse eigenvalue has a
non-positive real part.

Every row of ``J`` sums to zero, so the uniform shift ``(1, …, 1)/√N`` is always an exact
eigenvector with eigenvalue zero — the **Goldstone mode** of the global rotational symmetry
``θ_j ↦ θ_j + c``. This neutral direction is not an instability; it is projected out (identified
and excluded) before the configuration is classified, and the **spectral gap** between it and the
nearest transverse mode is the relaxation rate onto the synchronisation manifold.

This is an analysis layer over the differentiable simulation lane: it composes the multi-language
:func:`~oscillatools.accel.networked_kuramoto.networked_kuramoto_jacobian` (Rust → Julia →
Python floor) with a dense eigendecomposition. The eigendecomposition delegates to
``numpy.linalg`` (LAPACK) — :func:`numpy.linalg.eigh` for a symmetric Jacobian (an undirected
coupling matrix) and :func:`numpy.linalg.eig` for a non-symmetric one (a directed network) —
following the established engine convention (the OTOC kernel likewise diagonalises in NumPy and
accelerates only the downstream work), so the module adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_jacobian


@dataclass(frozen=True)
class StabilitySpectrum:
    """The eigen-spectrum and stability classification of a networked Kuramoto state.

    Attributes
    ----------
    eigenvalues : numpy.ndarray
        The ``(N,)`` complex eigenvalues of the stability Jacobian, ordered by descending real
        part (the most unstable first). A symmetric Jacobian yields real eigenvalues carried with
        a zero imaginary part.
    eigenvectors : numpy.ndarray
        The ``(N, N)`` complex eigenvectors as columns, aligned to ``eigenvalues``.
    goldstone_index : int
        The column index of the Goldstone (uniform-shift) eigenvector, the neutral mode of the
        global phase-rotation symmetry.
    goldstone_eigenvalue : complex
        The Goldstone eigenvalue, zero to numerical precision.
    leading_nontrivial_eigenvalue : complex
        The transverse eigenvalue of largest real part (the Goldstone mode excluded); its real
        part is the stability margin — negative for a stable configuration.
    spectral_gap : float
        ``Re(goldstone) − Re(leading_nontrivial)``, the relaxation rate of the slowest transverse
        mode onto the synchronisation manifold. Positive for a stable configuration; non-positive
        when a transverse mode grows.
    is_linearly_stable : bool
        ``True`` when every transverse eigenvalue has a real part below ``stability_tolerance``.
    is_symmetric : bool
        ``True`` when the Jacobian is symmetric (an undirected coupling matrix), in which case the
        eigendecomposition used :func:`numpy.linalg.eigh`.
    """

    eigenvalues: NDArray[np.complex128]
    eigenvectors: NDArray[np.complex128]
    goldstone_index: int
    goldstone_eigenvalue: complex
    leading_nontrivial_eigenvalue: complex
    spectral_gap: float
    is_linearly_stable: bool
    is_symmetric: bool


def _stability_jacobian(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the ``(N, N)`` networked stability Jacobian, validating the inputs.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N`` or fewer than two oscillators are
        given (a transverse mode needs at least two oscillators).
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count < 2:
        raise ValueError(f"stability spectrum needs at least 2 oscillators, got {count}")
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if matrix.shape != (count, count):
        raise ValueError(
            f"coupling must be a square matrix of order {count}, got shape {matrix.shape}"
        )
    return np.asarray(networked_kuramoto_jacobian(phases, matrix), dtype=np.float64)


def stability_spectrum(
    theta: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    symmetry_tolerance: float = 1e-12,
    stability_tolerance: float = 1e-9,
) -> StabilitySpectrum:
    r"""Eigen-spectrum and linear-stability classification of a networked Kuramoto state.

    Builds the stability Jacobian ``J(θ*)`` for the coupling matrix ``K`` and diagonalises it,
    identifies and projects out the Goldstone (uniform-shift) mode, and classifies the transverse
    spectrum. A symmetric Jacobian (undirected ``K``) is diagonalised with
    :func:`numpy.linalg.eigh` for real eigenvalues and an orthonormal eigenbasis; a non-symmetric
    Jacobian (directed ``K``) is diagonalised with :func:`numpy.linalg.eig`.

    Parameters
    ----------
    theta : numpy.ndarray
        The ``(N,)`` phase configuration ``θ*`` in radians (``N ≥ 2``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    symmetry_tolerance : float, optional
        The absolute tolerance for judging ``J`` symmetric (so :func:`numpy.linalg.eigh` applies).
    stability_tolerance : float, optional
        A transverse eigenvalue is treated as growing when its real part is at or above this value;
        the configuration is linearly stable when none is.

    Returns
    -------
    StabilitySpectrum
        The ordered eigenvalues and eigenvectors, the Goldstone mode, the leading transverse
        eigenvalue, the spectral gap, and the stability and symmetry flags.

    Raises
    ------
    ValueError
        If ``coupling`` is not a square matrix of order ``N`` or ``N < 2``.

    Notes
    -----
    The eigenvalues match a direct :func:`numpy.linalg.eig` of the Jacobian (a symmetric Jacobian
    additionally matches :func:`numpy.linalg.eigh`); the Goldstone eigenvalue is zero to numerical
    precision because every row of ``J`` sums to zero.
    """
    jacobian = _stability_jacobian(theta, coupling)
    count = jacobian.shape[0]
    is_symmetric = bool(np.allclose(jacobian, jacobian.T, atol=symmetry_tolerance, rtol=0.0))

    if is_symmetric:
        real_values, real_vectors = np.linalg.eigh(jacobian)
        values = real_values.astype(np.complex128)
        vectors = real_vectors.astype(np.complex128)
    else:
        values, vectors = np.linalg.eig(jacobian)
        values = values.astype(np.complex128)
        vectors = vectors.astype(np.complex128)

    order = np.argsort(-values.real, kind="stable")
    values = np.ascontiguousarray(values[order])
    vectors = np.ascontiguousarray(vectors[:, order])

    # The Goldstone mode is the uniform-shift direction; it has the largest overlap with the
    # normalised all-ones vector and an eigenvalue that is zero to numerical precision.
    uniform = np.full(count, 1.0 / np.sqrt(count), dtype=np.complex128)
    overlaps = np.abs(uniform.conj() @ vectors)
    goldstone_index = int(np.argmax(overlaps))

    transverse = np.delete(values.real, goldstone_index)
    leading_position = int(np.argmax(transverse))
    # Map the position within the Goldstone-excluded array back to an index in ``values``.
    leading_index = leading_position + (1 if leading_position >= goldstone_index else 0)
    leading_eigenvalue = complex(values[leading_index])
    goldstone_eigenvalue = complex(values[goldstone_index])
    spectral_gap = float(goldstone_eigenvalue.real - leading_eigenvalue.real)
    is_linearly_stable = bool(leading_eigenvalue.real < stability_tolerance)

    return StabilitySpectrum(
        eigenvalues=values,
        eigenvectors=vectors,
        goldstone_index=goldstone_index,
        goldstone_eigenvalue=goldstone_eigenvalue,
        leading_nontrivial_eigenvalue=leading_eigenvalue,
        spectral_gap=spectral_gap,
        is_linearly_stable=is_linearly_stable,
        is_symmetric=is_symmetric,
    )


def synchronisation_rate(
    theta: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    symmetry_tolerance: float = 1e-12,
    stability_tolerance: float = 1e-9,
) -> float:
    """Relaxation rate of the slowest transverse mode onto the synchronisation manifold.

    The spectral gap of :func:`stability_spectrum` — positive when the configuration is linearly
    stable (perturbations decay at this rate) and non-positive when a transverse mode grows.
    Parameters are as for :func:`stability_spectrum`.
    """
    spectrum = stability_spectrum(
        theta,
        coupling,
        symmetry_tolerance=symmetry_tolerance,
        stability_tolerance=stability_tolerance,
    )
    return spectrum.spectral_gap


def is_synchronisation_stable(
    theta: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    symmetry_tolerance: float = 1e-12,
    stability_tolerance: float = 1e-9,
) -> bool:
    """Whether the configuration is linearly stable (every transverse mode decays).

    The ``is_linearly_stable`` flag of :func:`stability_spectrum`. Parameters are as there.
    """
    spectrum = stability_spectrum(
        theta,
        coupling,
        symmetry_tolerance=symmetry_tolerance,
        stability_tolerance=stability_tolerance,
    )
    return spectrum.is_linearly_stable


__all__ = [
    "StabilitySpectrum",
    "is_synchronisation_stable",
    "stability_spectrum",
    "synchronisation_rate",
]
