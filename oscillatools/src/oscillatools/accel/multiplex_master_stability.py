# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Master stability function and the multiplex decomposition
r"""Master stability function and the multiplex stability decomposition.

The master stability function (Pecora & Carroll, 1998) decouples the linear stability of a
synchronised network: around the synchronous manifold the variational equation block-diagonalises,
and each block is the *same* small system parametrised by a coupling eigenvalue ``α``,

.. math::

    \Lambda(\alpha) = \max\operatorname{Re}\,\operatorname{spec}\bigl(J_{\text{node}}
        + \alpha\,J_{\text{coupling}}\bigr),

so the high-dimensional stability problem reduces to scanning ``\Lambda`` over the network's coupling
eigenvalues. For a **multiplex** network of identical oscillators that share an intra-layer graph,
the in-phase synchronous state's ``LN \times LN`` Jacobian is a Kronecker sum of the intra-layer and
inter-layer Jacobians, so its spectrum is *exactly* the pairwise sums of the per-structure
eigenvalues (Berner/Mehrmann/Schöll/Yanchuk, *SIAM J. Appl. Dyn. Syst.*, 2021):

.. math::

    \operatorname{spec}(J_{\text{multiplex}}) = \{\gamma_k + \mu_a\},

with ``\gamma_k`` the intra-layer eigenvalues and ``\mu_a`` the inter-layer eigenvalues. The
synchronous state is stable when every transverse mode (all combinations except the single global
phase-shift zero mode) decays. This extends the toolkit's single-layer stability spectrum to the
multilayer setting, replacing an ``LN``-dimensional eigenproblem with an ``N``- plus ``L``-dimensional
one. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_jacobian


@dataclass(frozen=True)
class MultiplexSynchronisationStability:
    """The decomposed stability of a multiplex in-phase synchronous state.

    Attributes
    ----------
    spectrum : numpy.ndarray
        The ``(L·N,)`` eigenvalues of the synchronous-state Jacobian, obtained by the decomposition
        ``γ_k + μ_a`` rather than a full ``LN`` eigensolve.
    intra_eigenvalues : numpy.ndarray
        The ``(N,)`` intra-layer Jacobian eigenvalues ``γ_k``.
    inter_eigenvalues : numpy.ndarray
        The ``(L,)`` inter-layer Jacobian eigenvalues ``μ_a``.
    transverse_decay : float
        The largest real part among the transverse modes (the spectrum with the single global
        zero mode removed); negative means the synchronous state is linearly stable.
    is_stable : bool
        Whether every transverse mode decays (``transverse_decay`` below the tolerance).
    """

    spectrum: NDArray[np.complex128]
    intra_eigenvalues: NDArray[np.complex128]
    inter_eigenvalues: NDArray[np.complex128]
    transverse_decay: float
    is_stable: bool


def master_stability_function(
    node_jacobian: NDArray[np.float64],
    coupling_jacobian: NDArray[np.float64],
    coupling_eigenvalue: complex,
) -> float:
    r"""The Pecora–Carroll master stability function ``Λ(α)``.

    Returns the largest real part of the spectrum of ``J_node + α J_coupling`` — the transverse
    growth rate of the synchronous manifold for a coupling eigenvalue ``α``. The synchronous state of
    a network is stable when ``Λ`` is negative at every (rescaled) coupling eigenvalue.

    Parameters
    ----------
    node_jacobian : numpy.ndarray
        The node Jacobian ``J_node`` evaluated on the synchronous solution (square, ``(m, m)``).
    coupling_jacobian : numpy.ndarray
        The coupling-function Jacobian ``J_coupling`` (square, same shape).
    coupling_eigenvalue : complex
        The coupling eigenvalue ``α`` (real or complex).

    Returns
    -------
    float
        ``Λ(α) = max Re spec(J_node + α J_coupling)``.

    Raises
    ------
    ValueError
        If the Jacobians are not matching square matrices.
    """
    node = np.ascontiguousarray(node_jacobian, dtype=np.float64)
    coupling = np.ascontiguousarray(coupling_jacobian, dtype=np.float64)
    if node.ndim != 2 or node.shape[0] != node.shape[1] or node.shape[0] < 1:
        raise ValueError("node_jacobian must be a non-empty square matrix")
    if coupling.shape != node.shape:
        raise ValueError(f"coupling_jacobian must have shape {node.shape}, got {coupling.shape}")
    spectrum = np.linalg.eigvals(node + coupling_eigenvalue * coupling)
    return float(np.max(spectrum.real))


def multiplex_synchronisation_stability(
    intra_coupling: NDArray[np.float64],
    inter_coupling: NDArray[np.float64],
    *,
    stability_tolerance: float = 1e-9,
) -> MultiplexSynchronisationStability:
    r"""The multiplex decomposition of the in-phase synchronous-state stability.

    For identical oscillators that share the intra-layer graph ``intra_coupling`` across all layers
    and couple between layers through ``inter_coupling``, the in-phase synchronous Jacobian is the
    Kronecker sum of the intra-layer and inter-layer Jacobians; its spectrum is therefore the pairwise
    sums ``γ_k + μ_a`` of the per-structure eigenvalues, computed here without forming the full
    ``LN`` matrix.

    Parameters
    ----------
    intra_coupling : numpy.ndarray
        The shared intra-layer coupling graph ``A`` (square, ``(N, N)``, ``N ≥ 2``).
    inter_coupling : numpy.ndarray
        The inter-layer coupling ``B`` over the layers (square, ``(L, L)``, ``L ≥ 2``).
    stability_tolerance : float, optional
        The transverse-decay threshold below which the state counts as stable; defaults to ``1e-9``.

    Returns
    -------
    MultiplexSynchronisationStability
        The decomposed spectrum, the per-structure eigenvalues, the transverse decay rate and the
        stability verdict.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    intra = np.ascontiguousarray(intra_coupling, dtype=np.float64)
    inter = np.ascontiguousarray(inter_coupling, dtype=np.float64)
    if intra.ndim != 2 or intra.shape[0] != intra.shape[1] or intra.shape[0] < 2:
        raise ValueError("intra_coupling must be an (N >= 2, N) square matrix")
    if inter.ndim != 2 or inter.shape[0] != inter.shape[1] or inter.shape[0] < 2:
        raise ValueError("inter_coupling must be an (L >= 2, L) square matrix")
    if not (np.all(np.isfinite(intra)) and np.all(np.isfinite(inter))):
        raise ValueError("the coupling matrices must be finite")
    if stability_tolerance <= 0.0:
        raise ValueError(f"stability_tolerance must be positive, got {stability_tolerance}")

    nodes = intra.shape[0]
    layers = inter.shape[0]
    intra_eigenvalues = np.linalg.eigvals(
        networked_kuramoto_jacobian(np.zeros(nodes, dtype=np.float64), intra)
    )
    inter_eigenvalues = np.linalg.eigvals(
        networked_kuramoto_jacobian(np.zeros(layers, dtype=np.float64), inter)
    )
    spectrum = (intra_eigenvalues[:, None] + inter_eigenvalues[None, :]).reshape(-1)

    real_parts = np.sort(spectrum.real)
    transverse_decay = float(real_parts[-2])  # drop the single global phase-shift zero mode
    return MultiplexSynchronisationStability(
        spectrum=np.ascontiguousarray(spectrum, dtype=np.complex128),
        intra_eigenvalues=np.ascontiguousarray(intra_eigenvalues, dtype=np.complex128),
        inter_eigenvalues=np.ascontiguousarray(inter_eigenvalues, dtype=np.complex128),
        transverse_decay=transverse_decay,
        is_stable=bool(transverse_decay < stability_tolerance),
    )


__all__ = [
    "MultiplexSynchronisationStability",
    "master_stability_function",
    "multiplex_synchronisation_stability",
]
