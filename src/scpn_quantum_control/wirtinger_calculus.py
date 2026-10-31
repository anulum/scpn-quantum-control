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
directions ``x`` and ``y``, for holomorphic and non-holomorphic functions
alike. Coordinate ``k`` moves by ``step * max(1, |z_k|)`` and each difference is
divided by the step actually realised in floating point, so the relative
accuracy does not degrade as ``|z|`` grows. The result carries the central
difference's ``O(h**2)`` truncation error plus rounding of order
``eps * |f| / h``; it is not exact.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
ComplexObjective = Callable[[NDArray[np.complex128]], complex]
RealObjective = Callable[[NDArray[np.complex128]], float | complex]
"""Real loss; a complex return may carry only an imaginary rounding residue."""


@dataclass(frozen=True)
class WirtingerDerivative:
    """Wirtinger partials of ``f`` at a point, plus a holomorphicity residual."""

    df_dz: NDArray[np.complex128]
    df_dconj_z: NDArray[np.complex128]
    holomorphic_residual: float
    difference_noise: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    """Per-coordinate error floor of ``df/dconj_z``: the rounding floor
    ``4 eps max|f| / h_k`` plus twice the Richardson estimate of the ``O(h**2)``
    truncation from a second difference at ``2 h_k``. A residual below it is
    indistinguishable from zero."""


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


_REAL_LOSS_IMAGINARY_TOLERANCE = 1.0e-12
"""Imaginary part of a real loss, relative to the loss's largest magnitude over
the evaluated points, that is treated as complex rounding."""


def _validate_step(step: float) -> float:
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a positive finite value")
    return float(step)


def _displaced(
    point: NDArray[np.complex128], k: int, direction: complex, step: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    """Return the points either side of ``z_k`` and the step realised between them.

    The step is ``step * max(1, |z_k|)`` along ``direction`` (``1`` or ``1j``),
    and the realised step is measured after rounding, so dividing by it keeps
    the difference quotient consistent with the points actually evaluated.
    """
    h = step * max(1.0, abs(complex(point[k])))
    upper = point.copy()
    lower = point.copy()
    upper[k] += direction * h
    lower[k] -= direction * h
    delta = complex(upper[k] - lower[k])
    realised = delta.real if direction == 1.0 else delta.imag
    if not realised > 0.0:
        raise ValueError("step is below the floating-point resolution at z")
    return upper, lower, realised


def _wirtinger_pair(
    f: ComplexObjective,
    point: NDArray[np.complex128],
    k: int,
    step: float,
    values: list[complex],
) -> tuple[complex, complex, float]:
    """Return ``df/dz_k``, ``df/dconj_z_k`` and the smaller realised step.

    Every evaluated value is appended to ``values`` for the rounding floor.
    """
    upper, lower, realised_x = _displaced(point, k, 1.0, step)
    plus_x, minus_x = complex(f(upper)), complex(f(lower))
    upper, lower, realised_y = _displaced(point, k, 1.0j, step)
    plus_y, minus_y = complex(f(upper)), complex(f(lower))
    values.extend((plus_x, minus_x, plus_y, minus_y))
    df_dx = (plus_x - minus_x) / realised_x
    df_dy = (plus_y - minus_y) / realised_y
    return 0.5 * (df_dx - 1j * df_dy), 0.5 * (df_dx + 1j * df_dy), min(realised_x, realised_y)


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
        step: the relative central-difference step; coordinate ``k`` moves by
            ``step * max(1, |z_k|)`` in the real and imaginary directions.

    Returns
    -------
        A :class:`WirtingerDerivative` with both partial-derivative vectors and
        the holomorphicity residual ``max|df/dconj_z|``.

    """
    point = _as_complex_vector(z)
    h = _validate_step(step)
    n = point.size
    df_dz = np.empty(n, dtype=np.complex128)
    df_dconj_z = np.empty(n, dtype=np.complex128)
    noise = np.empty(n, dtype=np.float64)
    eps = float(np.finfo(np.float64).eps)
    for k in range(n):
        values: list[complex] = []
        df_dz[k], df_dconj_z[k], realised = _wirtinger_pair(f, point, k, h, values)
        _, coarse_conj, _ = _wirtinger_pair(f, point, k, 2.0 * h, values)
        # Rounding floor of the differences, plus the Richardson estimate of the
        # O(h**2) truncation in df/dconj_z_k (twice it, for safety).
        rounding = 4.0 * eps * max(abs(value) for value in values) / realised
        truncation = 2.0 * abs(coarse_conj - df_dconj_z[k]) / 3.0
        noise[k] = rounding + truncation
    residual = float(np.max(np.abs(df_dconj_z))) if n else 0.0
    return WirtingerDerivative(
        df_dz=df_dz,
        df_dconj_z=df_dconj_z,
        holomorphic_residual=residual,
        difference_noise=noise,
    )


def is_holomorphic(
    f: ComplexObjective,
    z: NDArray[np.complex128],
    *,
    tolerance: float = 1e-6,
    step: float = 1e-6,
) -> bool:
    """Test the Cauchy-Riemann condition ``df/dconj_z = 0`` at ``z``.

    ``tolerance`` is relative and applies per coordinate: each
    ``|df/dconj_z_k|`` must not exceed ``tolerance * |df/dz_k|`` plus the
    rounding floor of that coordinate's central differences. The test
    therefore depends neither on the size of ``f`` or ``z`` nor on how large
    other coordinates' derivatives are, and a critical point is not refused
    for rounding noise.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    return _cauchy_riemann_holds(wirtinger_partials(f, z, step=step), tolerance)


def _cauchy_riemann_holds(derivative: WirtingerDerivative, tolerance: float) -> bool:
    """Compare each coordinate's residual with its own ``|df/dz_k|`` and noise floor."""
    noise = (
        derivative.difference_noise
        if derivative.difference_noise.shape == derivative.df_dz.shape
        else np.zeros(derivative.df_dz.shape, dtype=np.float64)
    )
    limit = tolerance * np.abs(derivative.df_dz) + noise
    return bool(np.all(np.abs(derivative.df_dconj_z) <= limit))


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
    if not _cauchy_riemann_holds(derivative, tolerance):
        raise ValueError(
            "f is not holomorphic at z (Cauchy-Riemann residual "
            f"{derivative.holomorphic_residual:.3e} exceeds relative tolerance "
            f"{tolerance:.3e}); use wirtinger_partials for the non-holomorphic derivative"
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


ComplexVectorFunction = Callable[[NDArray[np.complex128]], Any]
FrameworkConvention = Literal["torch", "jax"]
MatrixInput = NDArray[np.complexfloating[Any, Any]] | NDArray[np.floating[Any]]
"""Real symmetric or complex Hermitian matrix accepted by the spectral derivatives."""


def _complex_output(value: Any) -> NDArray[np.complex128]:
    array = np.atleast_1d(np.asarray(value, dtype=np.complex128))
    if array.ndim != 1 or array.size == 0:
        raise ValueError("f must return a complex scalar or a non-empty complex vector")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError("f must return finite values")
    return array


def real_block_jacobian(
    f: ComplexVectorFunction,
    z: NDArray[np.complex128],
    *,
    step: float = 1e-6,
) -> NDArray[np.float64]:
    """Return the authoritative real-block Jacobian of ``f: C^n -> C^m`` at ``z``.

    Parameters
    ----------
    f
        Callable taking a length-``n`` complex vector and returning a complex
        scalar or a length-``m`` complex vector.
    z
        Evaluation point.
    step
        Relative central-difference step: coordinate ``k`` moves by
        ``step * max(1, |z_k|)`` along its real and imaginary directions, and
        each difference is divided by the step realised after rounding.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(2m, 2n)``, ordered
        ``[[dRe f/dx, dRe f/dy], [dIm f/dx, dIm f/dy]]``. It is defined for every
        real-differentiable ``f``, holomorphic or not, and is the convention every
        framework gradient in this module is converted from.

    Raises
    ------
    ValueError
        If ``z``, ``step`` or the returned values are malformed, the step is
        below the floating-point resolution at ``z``, or the output size
        changes between evaluations.

    """
    point = _as_complex_vector(z)
    h = _validate_step(step)
    size = _complex_output(f(point)).size
    columns: list[NDArray[np.float64]] = []
    for direction in (1.0 + 0.0j, 1.0j):
        for k in range(point.size):
            upper_point, lower_point, realised = _displaced(point, k, direction, h)
            upper = _complex_output(f(upper_point))
            lower = _complex_output(f(lower_point))
            if upper.size != size or lower.size != size:
                raise ValueError("f must return the same output size at every point")
            derivative = (upper - lower) / realised
            columns.append(np.concatenate((derivative.real, derivative.imag)))
    return np.stack(columns, axis=1).astype(np.float64)


def real_block_gradient(
    loss: RealObjective,
    z: NDArray[np.complex128],
    *,
    step: float = 1e-6,
    scale: float | None = None,
) -> NDArray[np.float64]:
    """Return ``(dL/dx, dL/dy)`` of a real loss as one ``float64`` vector of length ``2n``.

    Parameters
    ----------
    loss
        Real-valued objective of a complex vector.
    z
        Evaluation point.
    step
        Relative central-difference step, as in :func:`real_block_jacobian`.
    scale
        Magnitude of the terms the loss is computed from, such as ``|H|`` for
        ``vdot(v, H @ v)``. Its rounding residue is judged against it. Defaults
        to the largest ``|L|`` over the evaluated points, which refuses a real
        loss that passes near zero while summing large terms; give the scale
        explicitly there.

    Returns
    -------
    numpy.ndarray
        The first row of :func:`real_block_jacobian` of ``loss``.

    Raises
    ------
    ValueError
        If ``scale`` is not finite and non-negative, or the loss returns an
        imaginary part above rounding: above ``1e-12`` times ``scale``. A real
        loss computed in complex arithmetic, such as ``vdot(v, H @ v)``,
        carries such rounding and is admitted; only the real part is
        differentiated. The scale is the loss's own, so a small loss with a
        comparable imaginary part is refused, whatever its units.

    """
    seen: list[complex] = []

    def _recorded(vector: NDArray[np.complex128]) -> complex:
        value = complex(loss(vector))
        seen.append(value)
        return value

    if scale is not None and (isinstance(scale, bool) or not np.isfinite(scale) or scale < 0.0):
        raise ValueError("scale must be a finite non-negative value")
    gradient: NDArray[np.float64] = real_block_jacobian(_recorded, z, step=step)[0]
    reference = max(abs(value) for value in seen) if scale is None else float(scale)
    residue = max(abs(value.imag) for value in seen)
    if residue > _REAL_LOSS_IMAGINARY_TOLERANCE * reference:
        raise ValueError(
            f"loss must be real-valued: imaginary residue {residue:.3e} exceeds "
            f"1e-12 * {reference:.3e}; pass scale= when a real loss sums large terms"
        )
    return gradient


def framework_complex_gradient(
    loss: RealObjective,
    z: NDArray[np.complex128],
    *,
    convention: FrameworkConvention,
    step: float = 1e-6,
    scale: float | None = None,
) -> NDArray[np.complex128]:
    """Express a real loss's gradient in a named framework's complex convention.

    Parameters
    ----------
    loss
        Real-valued objective of a complex vector.
    z
        Evaluation point.
    convention
        ``"torch"`` gives ``dL/dx + i dL/dy`` (``2 dL/dconj_z``, the direction
        PyTorch autograd returns); ``"jax"`` gives ``dL/dx - i dL/dy``
        (``2 dL/dz``, the conjugate that ``jax.grad`` returns).
    step
        Central-difference step of the underlying real-block gradient.
    scale
        Magnitude of the loss's terms, as in :func:`real_block_gradient`.

    Returns
    -------
    numpy.ndarray
        Complex gradient in the requested convention, converted explicitly from
        the real-block gradient.

    Raises
    ------
    ValueError
        If the convention is not ``"torch"`` or ``"jax"``.

    """
    if convention not in ("torch", "jax"):
        raise ValueError("convention must be 'torch' or 'jax'")
    gradient = real_block_gradient(loss, z, step=step, scale=scale)
    half = gradient.size // 2
    sign = 1.0 if convention == "torch" else -1.0
    return (gradient[:half] + sign * 1j * gradient[half:]).astype(np.complex128)


class SpectralDegeneracyError(ValueError):
    """Raised when a spectral derivative is undefined at a repeated eigenvalue."""


def _hermitian_matrix(matrix: MatrixInput, name: str) -> NDArray[np.complex128]:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} must be finite")
    # Relative to the matrix's own size, with no unit floor: a matrix in small
    # physical units, or a small tangent, is judged at its own scale.
    scale = float(np.max(np.abs(array)))
    if float(np.max(np.abs(array - array.conj().T))) > 1e-12 * scale:
        raise ValueError(f"{name} must be Hermitian")
    return array


def _gap_tolerance(eigenvalues: NDArray[np.float64], gap_tolerance: float | None) -> float:
    if gap_tolerance is None:
        return 1e-10 * float(np.max(np.abs(eigenvalues)))
    if not np.isfinite(gap_tolerance) or gap_tolerance <= 0.0:
        raise ValueError("gap_tolerance must be a positive finite value")
    return float(gap_tolerance)


@dataclass(frozen=True)
class EigenpairDerivative:
    """First-order change of one simple Hermitian eigenpair along a tangent.

    The eigenvector derivative is stated in the gauge ``<v, dv> = 0``; any other
    phase convention differs from it by ``i * c * v`` for a real ``c``.
    """

    eigenvalue: float
    eigenvalue_derivative: float
    eigenvector: NDArray[np.complex128]
    eigenvector_derivative: NDArray[np.complex128]
    minimum_gap: float
    rounding_error_estimate: float = 0.0
    """First-order estimate, with a safety factor of two, of the Euclidean norm of
    the error in ``eigenvector_derivative`` from rounding in the
    eigendecomposition: ``2 eps max|lambda| (|dv| / gap + 2 |dA|_2 / gap**2)``.
    The second term dominates when the tangent splits a near-degenerate pair
    without mixing it. It is an estimate, not a proven bound."""


def hermitian_eigenpair_derivative(
    matrix: MatrixInput,
    tangent: MatrixInput,
    index: int,
    *,
    gap_tolerance: float | None = None,
) -> EigenpairDerivative:
    """Differentiate one eigenpair of a Hermitian matrix, refusing at degeneracy.

    Parameters
    ----------
    matrix
        Hermitian matrix ``A``.
    tangent
        Hermitian direction ``dA`` of the same shape.
    index
        Eigenpair index in ascending eigenvalue order.
    gap_tolerance
        Smallest admitted distance to every other eigenvalue; defaults to
        ``1e-10 * max|lambda|`` over the whole spectrum.

    Returns
    -------
    EigenpairDerivative
        ``dlambda = v^H dA v`` and ``dv = sum_j v_j (v_j^H dA v) / (lambda - lambda_j)``.
        Rounding in the eigendecomposition perturbs ``dv`` by up to about
        ``eps * max|lambda| * (|dv| / gap + 2 * |dA|_2 / gap**2)`` in norm,
        reported as ``rounding_error_estimate``. The ``1 / gap**2`` term means
        a tangent that splits a near-degenerate pair can make ``dv`` inaccurate
        long before the gap reaches the tolerance; compare the estimate with
        ``|dv|`` before relying on the result.

    Raises
    ------
    SpectralDegeneracyError
        If another eigenvalue lies within ``gap_tolerance``. The individual
        eigenvector derivative is then undefined; no denominator is clipped.
        Use :func:`spectral_projector_derivative` for the invariant subspace.
    ValueError
        If the matrices, index or tolerance are malformed.

    """
    a = _hermitian_matrix(matrix, "matrix")
    d = _hermitian_matrix(tangent, "tangent")
    if d.shape != a.shape:
        raise ValueError("tangent must have the shape of matrix")
    if (
        isinstance(index, bool | np.bool_)
        or not isinstance(index, int | np.integer)
        or not 0 <= int(index) < a.shape[0]
    ):
        raise ValueError("index must address one eigenpair")
    index = int(index)
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    others = np.delete(eigenvalues, index)
    gap = float(np.min(np.abs(others - eigenvalues[index]))) if others.size else math.inf
    tolerance = _gap_tolerance(eigenvalues, gap_tolerance)
    if gap <= tolerance:
        raise SpectralDegeneracyError(
            f"eigenvalue {index} is within {tolerance:.3e} of another eigenvalue "
            f"(gap {gap:.3e}); its individual eigenvector derivative is undefined"
        )
    vector = eigenvectors[:, index]
    coupling = eigenvectors.conj().T @ d @ vector
    differences = eigenvalues[index] - eigenvalues
    differences[index] = 1.0
    weights = coupling / differences
    weights[index] = 0.0
    derivative = eigenvectors @ weights
    return EigenpairDerivative(
        eigenvalue=float(eigenvalues[index]),
        eigenvalue_derivative=float(np.real(coupling[index])),
        eigenvector=vector.astype(np.complex128),
        eigenvector_derivative=derivative.astype(np.complex128),
        minimum_gap=gap,
        # Divided step by step so that neither tiny nor huge matrices overflow.
        rounding_error_estimate=float(
            2.0
            * np.finfo(np.float64).eps
            * (np.max(np.abs(eigenvalues)) / gap)
            * (np.linalg.norm(derivative) + 2.0 * (np.linalg.norm(d, 2) / gap))
        ),
    )


@dataclass(frozen=True)
class SpectralProjectorDerivative:
    """Projector onto an invariant subspace and its first-order change."""

    projector: NDArray[np.complex128]
    projector_derivative: NDArray[np.complex128]
    indices: tuple[int, ...]
    separating_gap: float


def spectral_projector_derivative(
    matrix: MatrixInput,
    tangent: MatrixInput,
    indices: tuple[int, ...],
    *,
    gap_tolerance: float | None = None,
) -> SpectralProjectorDerivative:
    """Differentiate the projector onto the eigenspace of chosen eigenvalues.

    Parameters
    ----------
    matrix
        Hermitian matrix ``A``.
    tangent
        Hermitian direction ``dA`` of the same shape.
    indices
        Eigenvalue indices, ascending order, spanning the subspace. Repeated
        eigenvalues inside the subspace are supported.
    gap_tolerance
        Smallest admitted gap between the chosen and the remaining eigenvalues.

    Returns
    -------
    SpectralProjectorDerivative
        ``P = sum_{i in S} v_i v_i^H`` and
        ``dP = sum_{i in S, j not in S} (v_j v_j^H dA v_i v_i^H + h.c.) / (lambda_i - lambda_j)``.
        Both are independent of every eigenvector phase and of the basis chosen
        inside a degenerate subspace.

    Raises
    ------
    SpectralDegeneracyError
        If a remaining eigenvalue lies within ``gap_tolerance`` of a chosen one,
        so the subspace itself is not separated.
    ValueError
        If the matrices, indices or tolerance are malformed.

    """
    a = _hermitian_matrix(matrix, "matrix")
    d = _hermitian_matrix(tangent, "tangent")
    if d.shape != a.shape:
        raise ValueError("tangent must have the shape of matrix")
    size = a.shape[0]
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            isinstance(i, bool | np.bool_)
            or not isinstance(i, int | np.integer)
            or not 0 <= int(i) < size
            for i in indices
        )
    ):
        raise ValueError("indices must be distinct eigenpair indices")
    chosen = tuple(sorted(int(i) for i in indices))
    rest = [i for i in range(size) if i not in chosen]
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    gap = (
        float(np.min(np.abs(eigenvalues[list(chosen)][:, None] - eigenvalues[rest][None, :])))
        if rest
        else math.inf
    )
    tolerance = _gap_tolerance(eigenvalues, gap_tolerance)
    if gap <= tolerance:
        raise SpectralDegeneracyError(
            f"the chosen eigenvalues are within {tolerance:.3e} of the rest "
            f"(gap {gap:.3e}); the invariant subspace is not separated"
        )
    inside = eigenvectors[:, list(chosen)]
    outside = eigenvectors[:, rest]
    projector = inside @ inside.conj().T
    weights = 1.0 / (eigenvalues[list(chosen)][None, :] - eigenvalues[rest][:, None])
    half = outside @ ((outside.conj().T @ d @ inside) * weights) @ inside.conj().T
    derivative = half + half.conj().T
    return SpectralProjectorDerivative(
        projector=projector.astype(np.complex128),
        projector_derivative=derivative.astype(np.complex128),
        indices=chosen,
        separating_gap=gap,
    )


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
