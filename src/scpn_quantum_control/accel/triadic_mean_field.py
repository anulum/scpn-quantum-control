# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Triadic (2-simplex) Kuramoto mean-field force and stability Jacobian
"""Triadic (2-simplex) higher-order Kuramoto mean-field force and its stability Jacobian.

The all-to-all 2-simplex (three-body) coupling of the Skardal–Arenas higher-order Kuramoto
model drives each oscillator through the *squared* first moment: ``F_j = K r² sin(2ψ − 2θ_j)``,
with order parameter ``r`` and mean phase ``ψ``. Because it scales as ``r²`` it produces the
abrupt (explosive) synchronisation transitions and bistability that pairwise coupling cannot.
It is distinct from the second Daido harmonic, which uses the second moment
``r_2 e^{iψ_2} = ⟨e^{2iθ}⟩`` rather than the squared first moment ``r² e^{2iψ} = ⟨e^{iθ}⟩²``.
Its Jacobian ``J_jl = (2K/N) r cos(2θ_j − θ_l − ψ) − δ_jl 2K r² cos(2ψ − 2θ_j)`` is
non-symmetric (the higher-order mean field is non-variational) yet every row sums to zero (the
global-phase Goldstone mode).

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~scpn_quantum_control.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher


def _rust_triadic_mean_field_force(
    theta: NDArray[np.float64], coupling: float
) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_force = getattr(engine, "triadic_mean_field_force", None)
    if not callable(rust_force):
        raise ImportError("scpn_quantum_engine.triadic_mean_field_force is unavailable")

    return np.asarray(
        rust_force(np.ascontiguousarray(theta, dtype=np.float64), float(coupling)),
        dtype=np.float64,
    )


def _julia_triadic_mean_field_force(
    theta: NDArray[np.float64], coupling: float
) -> NDArray[np.float64]:
    from .julia import triadic_mean_field_force as julia_force

    return julia_force(theta, coupling)


def _python_triadic_mean_field_force(
    theta: NDArray[np.float64], coupling: float
) -> NDArray[np.float64]:
    # Correctness floor — F_j = K r² sin(2ψ − 2θ_j) =
    # K [2 C S cos 2θ_j − (C² − S²) sin 2θ_j], with C = ⟨cos θ⟩, S = ⟨sin θ⟩. An empty input
    # yields an empty array.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.size == 0:
        return np.zeros(0, dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    double = 2.0 * phases
    force = coupling * (
        2.0 * cos_mean * sin_mean * np.cos(double)
        - (cos_mean * cos_mean - sin_mean * sin_mean) * np.sin(double)
    )
    return np.ascontiguousarray(force, dtype=np.float64)


_TRIADIC_MEAN_FIELD_FORCE_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]]
] = [
    ("rust", _rust_triadic_mean_field_force),
    ("julia", _julia_triadic_mean_field_force),
    ("python", _python_triadic_mean_field_force),
]


def _rust_triadic_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float
) -> NDArray[np.float64]:
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_jacobian = getattr(engine, "triadic_mean_field_jacobian", None)
    if not callable(rust_jacobian):
        raise ImportError("scpn_quantum_engine.triadic_mean_field_jacobian is unavailable")

    return np.asarray(
        rust_jacobian(np.ascontiguousarray(theta, dtype=np.float64), float(coupling)),
        dtype=np.float64,
    )


def _julia_triadic_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float
) -> NDArray[np.float64]:
    from .julia import triadic_mean_field_jacobian as julia_jacobian

    return julia_jacobian(theta, coupling)


def _python_triadic_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float
) -> NDArray[np.float64]:
    # Correctness floor — J_jl = (2K/N) r cos(2θ_j − θ_l − ψ) − δ_jl 2K r² cos(2ψ − 2θ_j) =
    # (2K/N) (C cos(2θ_j − θ_l) + S sin(2θ_j − θ_l)) on the off-diagonal, with the
    # −2K (2 C S sin 2θ_j + (C² − S²) cos 2θ_j) curvature added on the diagonal. Non-symmetric,
    # every row sums to zero. C = ⟨cos θ⟩, S = ⟨sin θ⟩.
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    count = phases.size
    if count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cos_mean = float(np.mean(np.cos(phases)))
    sin_mean = float(np.mean(np.sin(phases)))
    offset = 2.0 * phases[:, None] - phases[None, :]
    jacobian = (2.0 * coupling / count) * (cos_mean * np.cos(offset) + sin_mean * np.sin(offset))
    double = 2.0 * phases
    diagonal = (
        -2.0
        * coupling
        * (
            2.0 * cos_mean * sin_mean * np.sin(double)
            + (cos_mean * cos_mean - sin_mean * sin_mean) * np.cos(double)
        )
    )
    jacobian[np.diag_indices(count)] += diagonal
    return np.ascontiguousarray(jacobian, dtype=np.float64)


# The Jacobian chain mirrors the force chain (Rust → Julia → Python floor). Their
# micro-benchmark (at K = 1) is recorded in
# ``docs/benchmarks/triadic_mean_field_tiers.json``; rerun
# ``python scripts/bench_triadic_mean_field_tiers.py`` when these chains are edited.
_TRIADIC_MEAN_FIELD_JACOBIAN_CHAIN: list[
    tuple[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]]
] = [
    ("rust", _rust_triadic_mean_field_jacobian),
    ("julia", _julia_triadic_mean_field_jacobian),
    ("python", _python_triadic_mean_field_jacobian),
]


_triadic_mean_field_force_dispatcher = MultiLangDispatcher(_TRIADIC_MEAN_FIELD_FORCE_CHAIN)
_triadic_mean_field_jacobian_dispatcher = MultiLangDispatcher(_TRIADIC_MEAN_FIELD_JACOBIAN_CHAIN)


def triadic_mean_field_force(theta: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
    r"""Triadic (2-simplex) Kuramoto mean-field force with multi-language dispatch.

    Returns :math:`F_j = K r^2 \sin(2\psi - 2\theta_j)`, the all-to-all three-body coupling of
    the Skardal–Arenas higher-order Kuramoto model, with order parameter :math:`r` and mean
    phase :math:`\psi`. The :math:`r^2` scaling produces explosive (abrupt) synchronisation.

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The triadic coupling strength ``K`` (any real value).

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 force array of the same length as ``theta``. An empty input
        yields an empty array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_triadic_mean_field_force_tier_used`.
    """
    return np.asarray(_triadic_mean_field_force_dispatcher(theta, coupling), dtype=np.float64)


def triadic_mean_field_jacobian(
    theta: NDArray[np.float64], coupling: float
) -> NDArray[np.float64]:
    r"""Triadic (2-simplex) Kuramoto mean-field stability Jacobian with multi-language dispatch.

    Returns :math:`J_{jl} = (2K/N) r \cos(2\theta_j - \theta_l - \psi) - \delta_{jl} 2K r^2
    \cos(2\psi - 2\theta_j)`, the linearisation of the triadic mean-field force. The matrix is
    non-symmetric (the higher-order mean field is non-variational) yet every row sums to zero
    (the global-phase Goldstone mode).

    Parameters
    ----------
    theta : numpy.ndarray
        One-dimensional array of oscillator phases in radians.
    coupling : float
        The triadic coupling strength ``K``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(N, N)`` float64 Jacobian matrix. An empty input yields a
        ``(0, 0)`` array.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_triadic_mean_field_jacobian_tier_used`.
    """
    return np.asarray(_triadic_mean_field_jacobian_dispatcher(theta, coupling), dtype=np.float64)


def last_triadic_mean_field_force_tier_used() -> str | None:
    """Return the tier that served the most recent ``triadic_mean_field_force``."""
    return _triadic_mean_field_force_dispatcher.last_tier


def last_triadic_mean_field_jacobian_tier_used() -> str | None:
    """Return the tier that served the most recent ``triadic_mean_field_jacobian``."""
    return _triadic_mean_field_jacobian_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("triadic_mean_field_force", _triadic_mean_field_force_dispatcher)
register_dispatcher("triadic_mean_field_jacobian", _triadic_mean_field_jacobian_dispatcher)


__all__ = [
    "last_triadic_mean_field_force_tier_used",
    "last_triadic_mean_field_jacobian_tier_used",
    "triadic_mean_field_force",
    "triadic_mean_field_jacobian",
]
