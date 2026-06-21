# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Wirtinger (CR) calculus for complex-valued objectives
r"""Wirtinger calculus: holomorphic and non-holomorphic complex derivatives.

The registered Phase-QNode statevector engine differentiates real rotation
angles, so its complex-derivative contract is fail-closed. This module supplies
the complementary surface the differentiable lane previously left unsupported:
the Wirtinger (Cauchy-Riemann) calculus for an arbitrary complex callable
``f: C^n -> C``.

Writing ``z = x + i y``, the Wirtinger partials are

    df/dz      = 1/2 (df/dx - i df/dy)
    df/dconj_z = 1/2 (df/dx + i df/dy).

A function is holomorphic exactly when ``df/dconj_z = 0`` (the Cauchy-Riemann
equations); then ``df/dz`` is the ordinary complex derivative. For a
real-valued loss ``L: C^n -> R`` the steepest-descent direction is
``df/dconj_z`` (equal to ``conj(df/dz)``), the standard CR-calculus gradient used
to optimise complex parameters.

Partials are evaluated by central differences in the independent real
directions ``x`` and ``y``, which is exact (to rounding) for both holomorphic and
non-holomorphic functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
ComplexObjective = Callable[[NDArray[np.complex128]], complex]
RealObjective = Callable[[NDArray[np.complex128]], float]


@dataclass(frozen=True)
class WirtingerDerivative:
    """Wirtinger partials of ``f`` at a point, plus a holomorphicity residual."""

    df_dz: NDArray[np.complex128]
    df_dconj_z: NDArray[np.complex128]
    holomorphic_residual: float


@dataclass(frozen=True)
class WirtingerOptimisationResult:
    """Trajectory of a complex steepest-descent run on a real-valued objective."""

    parameters: NDArray[np.complex128]
    loss_history: NDArray[np.float64]
    final_loss: float
    provenance: dict[str, Any] = field(default_factory=dict)


def _as_complex_vector(z: NDArray[np.complex128]) -> NDArray[np.complex128]:
    array = np.asarray(z, dtype=np.complex128)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("z must be a non-empty one-dimensional complex vector")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError("z must be finite")
    return array


def _validate_step(step: float) -> float:
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a positive finite value")
    return float(step)


def wirtinger_partials(
    f: ComplexObjective,
    z: NDArray[np.complex128],
    *,
    step: float = 1e-6,
) -> WirtingerDerivative:
    """Return the Wirtinger partials ``df/dz`` and ``df/dconj_z`` of ``f`` at ``z``.

    Args:
        f: a complex callable taking a length-``n`` complex vector.
        z: the evaluation point (length-``n`` complex vector).
        step: the central-difference step in the real and imaginary directions.

    Returns:
        A :class:`WirtingerDerivative` with both partial-derivative vectors and
        the holomorphicity residual ``max|df/dconj_z|``.
    """
    point = _as_complex_vector(z)
    h = _validate_step(step)
    n = point.size
    df_dz = np.empty(n, dtype=np.complex128)
    df_dconj_z = np.empty(n, dtype=np.complex128)
    for k in range(n):
        basis = np.zeros(n, dtype=np.complex128)
        basis[k] = 1.0
        df_dx = (complex(f(point + h * basis)) - complex(f(point - h * basis))) / (2.0 * h)
        df_dy = (complex(f(point + 1j * h * basis)) - complex(f(point - 1j * h * basis))) / (
            2.0 * h
        )
        df_dz[k] = 0.5 * (df_dx - 1j * df_dy)
        df_dconj_z[k] = 0.5 * (df_dx + 1j * df_dy)
    residual = float(np.max(np.abs(df_dconj_z))) if n else 0.0
    return WirtingerDerivative(df_dz=df_dz, df_dconj_z=df_dconj_z, holomorphic_residual=residual)


def is_holomorphic(
    f: ComplexObjective,
    z: NDArray[np.complex128],
    *,
    tolerance: float = 1e-6,
    step: float = 1e-6,
) -> bool:
    """Test the Cauchy-Riemann condition ``df/dconj_z = 0`` at ``z``."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    return wirtinger_partials(f, z, step=step).holomorphic_residual <= tolerance


def holomorphic_gradient(
    f: ComplexObjective,
    z: NDArray[np.complex128],
    *,
    tolerance: float = 1e-6,
    step: float = 1e-6,
) -> NDArray[np.complex128]:
    """Return the complex derivative ``df/dz`` of a holomorphic ``f`` at ``z``.

    Raises ``ValueError`` (fail-closed) when ``f`` is not holomorphic at ``z``, so
    the ordinary complex derivative is not defined.
    """
    derivative = wirtinger_partials(f, z, step=step)
    if derivative.holomorphic_residual > tolerance:
        raise ValueError(
            "f is not holomorphic at z (Cauchy-Riemann residual "
            f"{derivative.holomorphic_residual:.3e} exceeds tolerance {tolerance:.3e}); "
            "use wirtinger_partials for the non-holomorphic derivative"
        )
    return derivative.df_dz


def real_objective_gradient(
    loss: RealObjective,
    z: NDArray[np.complex128],
    *,
    step: float = 1e-6,
) -> NDArray[np.complex128]:
    """Return the CR steepest-descent gradient ``dL/dconj_z`` of a real loss.

    For ``L: C^n -> R`` the update ``z <- z - eta * gradient`` reduces ``L``; the
    gradient equals ``conj(dL/dz)``.
    """

    def _complex_loss(vector: NDArray[np.complex128]) -> complex:
        return complex(float(np.real(loss(vector))), 0.0)

    return wirtinger_partials(_complex_loss, z, step=step).df_dconj_z


def minimise_real_objective(
    loss: RealObjective,
    z0: NDArray[np.complex128],
    *,
    learning_rate: float = 0.1,
    steps: int = 100,
    step: float = 1e-6,
) -> WirtingerOptimisationResult:
    """Minimise a real-valued objective of complex parameters by CR descent.

    Reports the observed local loss trajectory only; it makes no claim of global
    convergence for non-convex complex objectives.
    """
    point = _as_complex_vector(z0).copy()
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a positive finite value")
    if steps < 1:
        raise ValueError("steps must be a positive integer")

    history = np.empty(steps + 1, dtype=np.float64)
    history[0] = float(np.real(loss(point)))
    for index in range(steps):
        gradient = real_objective_gradient(loss, point, step=step)
        point = point - learning_rate * gradient
        history[index + 1] = float(np.real(loss(point)))
    return WirtingerOptimisationResult(
        parameters=point,
        loss_history=history,
        final_loss=float(history[-1]),
        provenance={
            "n_parameters": int(point.size),
            "learning_rate": learning_rate,
            "steps": steps,
            "gradient": "conjugate_wirtinger_cr_calculus",
            "claim_boundary": (
                "local CR steepest descent on a real-valued objective of complex "
                "parameters; observed local loss decrease only; no global "
                "convergence, provider, or hardware claim"
            ),
        },
    )
