# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — k-simplex mean-field Kuramoto force and Jacobian, arbitrary interaction order
r"""All-to-all ``k``-simplex Kuramoto mean field — higher-order coupling at arbitrary order.

A ``p``-simplex couples ``p + 1`` oscillators at once. For the all-to-all (mean-field) higher-order
Kuramoto model the order-``p`` force on oscillator ``i`` is

.. math::

    F_i = K\,r^{p}\,\sin\!\bigl(p\,ψ - p\,θ_i\bigr) = K\,\operatorname{Im}\!\bigl(z^{p}\,e^{-i p θ_i}\bigr),

where ``z = r e^{iψ} = N^{-1} Σ_j e^{iθ_j}`` is the Kuramoto order parameter. The ``r^{p}`` scaling
is what makes higher-order coupling produce explosive (abrupt, hysteretic) synchronisation. The
form follows from the all-to-all simplex sum collapsing onto a power of the order parameter,
``N^{-p} Σ_{j_1\dots j_p} \sin(θ_{j_1} + \dots + θ_{j_p} - p θ_i) = r^{p} \sin(p ψ - p θ_i)``.

The order recovers the lower-order models exactly:

- ``p = 1`` is the classic Kuramoto mean field ``K r \sin(ψ - θ_i)``;
- ``p = 2`` is the triadic (2-simplex) force ``K r^2 \sin(2ψ - 2θ_i)`` of
  :func:`~scpn_quantum_control.accel.triadic_mean_field.triadic_mean_field_force`;
- ``p = 3`` is the 3-simplex (four-body) force.

The stability Jacobian is obtained by differentiating ``F_i = K\,\operatorname{Im}(z^{p} e^{-i p θ_i})``
through the order parameter ``z`` (:func:`simplex_mean_field_jacobian`); it is non-symmetric (the
higher-order mean field is non-variational) yet every row sums to zero — the global-phase
Goldstone mode. This is an analysis layer evaluated directly from the order parameter, so it adds
no compute kernel and generalises the polyglot triadic mean field to any interaction order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_order(order: int) -> None:
    """Validate that the simplex order is a positive integer."""
    if order < 1:
        raise ValueError(f"order must be a positive integer, got {order}")


def simplex_mean_field_force(
    theta: NDArray[np.float64], coupling: float, order: int
) -> NDArray[np.float64]:
    r"""Return the all-to-all ``p``-simplex mean-field force ``F_i = K r^p sin(p ψ − p θ_i)``.

    The order-``order`` higher-order Kuramoto mean field, equal to
    ``K Im(z^{order} e^{-i·order·θ_i})`` with ``z`` the order parameter. ``order = 1`` is the
    classic Kuramoto mean field, ``order = 2`` the triadic force, ``order = 3`` the four-body
    3-simplex force.

    Parameters
    ----------
    theta : numpy.ndarray
        The oscillator phases ``θ`` (one-dimensional). An empty array yields an empty force.
    coupling : float
        The simplex coupling strength ``K`` (any real value).
    order : int
        The simplex order ``p`` (``≥ 1``); the interaction involves ``p + 1`` oscillators.

    Returns
    -------
    numpy.ndarray
        The force on each oscillator, of the same length as ``theta``.

    Raises
    ------
    ValueError
        If ``theta`` is not one-dimensional or ``order`` is below ``1``.
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.ndim != 1:
        raise ValueError("theta must be a one-dimensional array")
    _validate_order(order)
    if phases.size == 0:
        return np.empty(0, dtype=np.float64)
    mean_field = np.mean(np.exp(1j * phases))
    force = coupling * np.imag(mean_field**order * np.exp(-1j * order * phases))
    return np.asarray(force, dtype=np.float64)


def simplex_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float, order: int
) -> NDArray[np.float64]:
    r"""Return the ``(N, N)`` stability Jacobian of the ``p``-simplex mean-field force.

    Differentiating ``F_i = K Im(z^{p} e^{-i p θ_i})`` through ``z = N^{-1} Σ_j e^{iθ_j}`` gives

    .. math::

        \frac{∂F_i}{∂θ_l} = K\,\operatorname{Im}\!\Bigl[
            \frac{i p}{N}\,z^{p-1}\,e^{i(θ_l - p θ_i)}
            - i p\,δ_{il}\,z^{p}\,e^{-i p θ_i}\Bigr].

    The matrix is non-symmetric but every row sums to zero (the global-phase Goldstone mode). At
    ``order = 2`` it equals
    :func:`~scpn_quantum_control.accel.triadic_mean_field.triadic_mean_field_jacobian`.

    Parameters
    ----------
    theta : numpy.ndarray
        The oscillator phases ``θ`` (one-dimensional). An empty array yields a ``(0, 0)`` matrix.
    coupling : float
        The simplex coupling strength ``K``.
    order : int
        The simplex order ``p`` (``≥ 1``).

    Returns
    -------
    numpy.ndarray
        The ``(N, N)`` Jacobian matrix.

    Raises
    ------
    ValueError
        If ``theta`` is not one-dimensional or ``order`` is below ``1``.
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.ndim != 1:
        raise ValueError("theta must be a one-dimensional array")
    _validate_order(order)
    count = phases.size
    if count == 0:
        return np.empty((0, 0), dtype=np.float64)
    mean_field = np.mean(np.exp(1j * phases))
    row_factor = np.exp(-1j * order * phases)  # e^{-i p θ_i}, indexed by i
    column_factor = np.exp(1j * phases)  # e^{i θ_l}, indexed by l
    matrix = (1j * order / count) * mean_field ** (order - 1) * np.outer(row_factor, column_factor)
    diagonal = 1j * order * mean_field**order * row_factor
    matrix[np.diag_indices(count)] -= diagonal
    return np.asarray(coupling * np.imag(matrix), dtype=np.float64)


__all__ = [
    "simplex_mean_field_force",
    "simplex_mean_field_jacobian",
]
