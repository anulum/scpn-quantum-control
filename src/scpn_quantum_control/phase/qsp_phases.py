# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QSP phase-factor synthesis
r"""Certified quantum-signal-processing phase-factor synthesis.

The reflection (``Wx``) convention of Gilyén, Su, Low and Wiebe (STOC 2019) is
used throughout. For a signal operator

.. math::

    W(x) = \begin{pmatrix} x & i\sqrt{1-x^2} \\
                            i\sqrt{1-x^2} & x \end{pmatrix},
    \qquad x = \cos\theta,

and phase factors :math:`\Phi = (\phi_0, \dots, \phi_d)` the QSP unitary is

.. math::

    U(x, \Phi) = e^{i\phi_0 Z} \prod_{k=1}^{d} W(x)\, e^{i\phi_k Z},

whose top-left entry is :math:`P(x) + i\sqrt{1-x^2}\,\cdot\,` a polynomial in
the structure theorem. This module synthesises :math:`\Phi` so that

.. math::

    \operatorname{Re}\langle 0 | U(x, \Phi) | 0 \rangle = f(x)

for a real target polynomial :math:`f` of definite parity with
:math:`|f(x)| \le 1` on :math:`[-1, 1]`.

Synthesis follows the symmetric-phase Newton method of Dong, Meng, Whaley and
Lin, *Efficient phase-factor evaluation in quantum signal processing*,
Phys. Rev. A **103**, 042419 (2021): symmetric phase factors
:math:`\phi_k = \phi_{d-k}` reduce the problem to
:math:`\lceil (d+1)/2 \rceil` unknowns, which are matched against the target
at the same number of Chebyshev nodes and solved by Newton iteration with the
exact Jacobian of the matrix product. The published initial guess
:math:`\Phi^0 = (\pi/4, 0, \dots, 0, \pi/4)` is used.

Every synthesis is certified before it is returned: the realised response is
compared with the target on a dense grid, and the complementary polynomial is
extracted and checked against the QSP completion identity. A synthesis that
cannot be certified within tolerance raises rather than returning phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any, Final

import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.typing import NDArray
from scipy.special import jv

_PAULI_Z: Final[NDArray[np.complex128]] = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

DEFAULT_TOLERANCE: Final[float] = 1e-12
"""Largest certified supremum error a synthesis may return."""

DEFAULT_MAX_ITERATIONS: Final[int] = 200
"""Newton iteration budget before a synthesis is refused."""

VERIFICATION_GRID_FACTOR: Final[int] = 64
"""Dense verification grid size as a multiple of the target degree."""

MINIMUM_VERIFICATION_GRID: Final[int] = 512
"""Smallest dense verification grid, so low degrees are still measured."""

INITIAL_EDGE_PHASE: Final[float] = float(np.pi / 4.0)
"""Published initial guess for the outermost symmetric phase factor."""


class QSPSynthesisError(RuntimeError):
    """Raised when phase factors cannot be certified against the target."""


@dataclass(frozen=True)
class QSPPhaseFactors:
    """Certified QSP phase factors and the evidence that certifies them.

    Parameters
    ----------
    phases
        ``degree + 1`` symmetric phase factors in the ``Wx`` convention.
    degree
        Degree of the realised target polynomial.
    parity
        ``0`` for an even target, ``1`` for an odd target.
    newton_iterations
        Newton steps taken before the node residual met the tolerance.
    node_residual
        Largest absolute residual at the Chebyshev collocation nodes.
    supremum_error
        Largest absolute difference between the realised response and the
        target over the dense verification grid. This is the certificate.
    completion_residual
        Largest absolute value of ``P(x)^2 + (1 - x^2) Q(x)^2 - 1`` over the
        same grid, where ``Q`` is the extracted complementary polynomial.
    verification_grid
        Number of points in the dense verification grid.

    """

    phases: NDArray[np.float64]
    degree: int
    parity: int
    newton_iterations: int
    node_residual: float
    supremum_error: float
    completion_residual: float
    verification_grid: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping of the factors and evidence."""
        return {
            "phases": [float(value) for value in self.phases],
            "degree": self.degree,
            "parity": self.parity,
            "newton_iterations": self.newton_iterations,
            "node_residual": self.node_residual,
            "supremum_error": self.supremum_error,
            "completion_residual": self.completion_residual,
            "verification_grid": self.verification_grid,
        }


def qsp_unitary(phases: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.complex128]:
    r"""Evaluate the QSP unitary at each signal value.

    Parameters
    ----------
    phases
        Phase factors :math:`(\phi_0, \dots, \phi_d)`.
    x
        Signal values in ``[-1, 1]``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(x.size, 2, 2)`` holding :math:`U(x, \Phi)`.

    Raises
    ------
    ValueError
        If no phase is supplied or a signal value lies outside ``[-1, 1]``.

    """
    return _qsp_factor_product(*_qsp_factors(phases, x))


def qsp_response(phases: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.complex128]:
    r"""Return the QSP response :math:`\langle 0 | U(x, \Phi) | 0 \rangle`.

    Parameters
    ----------
    phases
        Phase factors :math:`(\phi_0, \dots, \phi_d)`.
    x
        Signal values in ``[-1, 1]``.

    Returns
    -------
    numpy.ndarray
        Complex response at each signal value.

    Raises
    ------
    ValueError
        If no phase is supplied or a signal value lies outside ``[-1, 1]``.

    """
    unitary = qsp_unitary(phases, x)
    response: NDArray[np.complex128] = unitary[:, 0, 0]
    return response


def complementary_polynomial(
    phases: NDArray[np.float64], *, samples: int | None = None
) -> NDArray[np.float64]:
    r"""Extract the complementary polynomial realised by the phase factors.

    The QSP structure theorem states that the off-diagonal entry of
    :math:`U(x, \Phi)` equals :math:`i\sqrt{1-x^2}\,Q(x)` for a polynomial
    ``Q`` of degree ``d - 1`` and parity opposite to the target. This routine
    recovers ``Q`` by Chebyshev projection, so the returned coefficients can be
    checked against the completion identity rather than assumed to satisfy it.

    Parameters
    ----------
    phases
        Phase factors :math:`(\phi_0, \dots, \phi_d)`.
    samples
        Number of Chebyshev sample points. Defaults to a grid comfortably
        above the Nyquist limit for degree ``d``.

    Returns
    -------
    numpy.ndarray
        Chebyshev coefficients of ``Q``, of length ``max(d, 1)``.

    Raises
    ------
    ValueError
        If fewer than two phase factors are supplied, or ``samples`` is too
        small to determine the coefficients.

    """
    phase_values = _as_phase_vector(phases)
    degree = phase_values.size - 1
    if degree < 1:
        raise ValueError("complementary polynomial requires at least two phase factors")
    count = _resolve_sample_count(samples, degree)
    nodes = np.cos(np.pi * (2.0 * np.arange(count) + 1.0) / (2.0 * count))
    unitary = qsp_unitary(phase_values, nodes)
    sine = np.sqrt(np.maximum(0.0, 1.0 - nodes * nodes))
    values = (unitary[:, 0, 1] / 1j).real / sine
    fitted = cheb.chebfit(nodes, values, degree - 1)
    coefficients: NDArray[np.float64] = np.asarray(fitted, dtype=np.float64)
    return coefficients


def jacobi_anger_cosine_coefficients(tau: float, degree: int) -> NDArray[np.float64]:
    r"""Return Chebyshev coefficients of the truncated :math:`\cos(\tau x)`.

    The Jacobi--Anger expansion
    :math:`\cos(\tau x) = J_0(\tau) + 2\sum_{k\ge1} (-1)^k J_{2k}(\tau)
    T_{2k}(x)` is the even target used for QSVT Hamiltonian simulation.

    Parameters
    ----------
    tau
        Finite evolution parameter.
    degree
        Even truncation degree.

    Returns
    -------
    numpy.ndarray
        ``degree + 1`` Chebyshev coefficients of even parity.

    Raises
    ------
    ValueError
        If ``tau`` is not finite, or ``degree`` is negative or odd.

    """
    return _jacobi_anger_coefficients(tau, degree, parity=0)


def jacobi_anger_sine_coefficients(tau: float, degree: int) -> NDArray[np.float64]:
    r"""Return Chebyshev coefficients of the truncated :math:`\sin(\tau x)`.

    The Jacobi--Anger expansion
    :math:`\sin(\tau x) = 2\sum_{k\ge0} (-1)^k J_{2k+1}(\tau) T_{2k+1}(x)`
    is the odd target used for QSVT Hamiltonian simulation.

    Parameters
    ----------
    tau
        Finite evolution parameter.
    degree
        Odd truncation degree.

    Returns
    -------
    numpy.ndarray
        ``degree + 1`` Chebyshev coefficients of odd parity.

    Raises
    ------
    ValueError
        If ``tau`` is not finite, or ``degree`` is negative or even.

    """
    return _jacobi_anger_coefficients(tau, degree, parity=1)


def synthesise_qsp_phases(
    chebyshev_coefficients: NDArray[np.float64],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> QSPPhaseFactors:
    r"""Synthesise certified QSP phase factors for a real target polynomial.

    Parameters
    ----------
    chebyshev_coefficients
        Chebyshev coefficients of the target ``f``, of definite parity, with
        :math:`|f(x)| \le 1` on ``[-1, 1]``.
    tolerance
        Largest certified supremum error the result may carry.
    max_iterations
        Newton iteration budget.

    Returns
    -------
    QSPPhaseFactors
        Phase factors together with the residuals that certify them.

    Raises
    ------
    ValueError
        If the coefficients are empty, non-finite, of mixed parity, exceed
        unit magnitude on ``[-1, 1]``, or the tolerance or iteration budget is
        not positive.
    QSPSynthesisError
        If Newton iteration does not reach the tolerance, or the certified
        supremum error or completion residual exceeds it. No uncertified phase
        factors are returned.

    """
    coefficients = _validate_target(chebyshev_coefficients)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number")
    budget = _positive_integer(max_iterations, "max_iterations")

    degree = coefficients.size - 1
    parity = degree % 2
    free, iterations, node_residual = _newton_solve(coefficients, tolerance, budget)
    phases = _expand_symmetric(free, degree)

    grid_size = max(MINIMUM_VERIFICATION_GRID, VERIFICATION_GRID_FACTOR * (degree + 1))
    grid = np.linspace(-1.0, 1.0, grid_size)
    realised = qsp_response(phases, grid).real
    target_values = cheb.chebval(grid, coefficients)
    supremum_error = float(np.max(np.abs(realised - target_values)))

    completion_residual = _completion_residual(phases, grid, realised)

    if supremum_error > tolerance:
        raise QSPSynthesisError(
            f"certified supremum error {supremum_error:.3e} exceeds tolerance {tolerance:.3e}"
        )
    if completion_residual > tolerance:
        raise QSPSynthesisError(
            f"completion identity residual {completion_residual:.3e} "
            f"exceeds tolerance {tolerance:.3e}"
        )
    return QSPPhaseFactors(
        phases=phases,
        degree=degree,
        parity=parity,
        newton_iterations=iterations,
        node_residual=node_residual,
        supremum_error=supremum_error,
        completion_residual=completion_residual,
        verification_grid=grid_size,
    )


def _completion_residual(
    phases: NDArray[np.float64], grid: NDArray[np.float64], realised: NDArray[np.float64]
) -> float:
    """Return the largest completion-identity residual over the grid."""
    if phases.size < 2:
        return 0.0
    complementary = complementary_polynomial(phases)
    unitary = qsp_unitary(phases, grid)
    imaginary = unitary[:, 0, 0].imag
    magnitude = realised * realised + imaginary * imaginary
    quadrature = cheb.chebval(grid, complementary)
    identity = magnitude + (1.0 - grid * grid) * quadrature * quadrature
    return float(np.max(np.abs(identity - 1.0)))


def _newton_solve(
    coefficients: NDArray[np.float64], tolerance: float, max_iterations: int
) -> tuple[NDArray[np.float64], int, float]:
    """Solve the square symmetric-phase collocation system by Newton iteration."""
    degree = coefficients.size - 1
    half = (degree + 2) // 2
    nodes = np.cos(np.pi * (2.0 * np.arange(1, half + 1) - 1.0) / (4.0 * half))
    target_values = cheb.chebval(nodes, coefficients)

    free = np.zeros(half, dtype=np.float64)
    free[0] = INITIAL_EDGE_PHASE
    residual = np.inf
    for iteration in range(max_iterations):
        phases = _expand_symmetric(free, degree)
        values, gradient = _response_and_gradient(phases, nodes)
        difference = values - target_values
        residual = float(np.max(np.abs(difference)))
        if residual <= tolerance:
            return free, iteration, residual
        jacobian = np.zeros((half, half), dtype=np.float64)
        for k in range(degree + 1):
            jacobian[:, k if k < half else degree - k] += gradient[k]
        try:
            step = np.linalg.solve(jacobian, difference)
        except np.linalg.LinAlgError as error:
            raise QSPSynthesisError(
                f"Newton step failed at iteration {iteration}: singular Jacobian"
            ) from error
        free = free - step
    raise QSPSynthesisError(
        f"Newton iteration did not converge in {max_iterations} steps; "
        f"largest node residual {residual:.3e}"
    )


def _response_and_gradient(
    phases: NDArray[np.float64], x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the real response and its exact derivative in every phase."""
    factors, owners = _qsp_factors(phases, x)
    count = len(factors)
    identity = np.broadcast_to(np.eye(2, dtype=np.complex128), (x.size, 2, 2))

    prefix: list[NDArray[np.complex128]] = [identity]
    for factor in factors:
        prefix.append(prefix[-1] @ factor)
    suffix: list[NDArray[np.complex128]] = [identity] * (count + 1)
    accumulated = identity
    for position in range(count - 1, -1, -1):
        accumulated = factors[position] @ accumulated
        suffix[position] = accumulated

    values: NDArray[np.float64] = prefix[count][:, 0, 0].real
    gradient = np.zeros((phases.size, x.size), dtype=np.float64)
    for position, owner in enumerate(owners):
        if owner < 0:
            continue
        derivative = prefix[position] @ (1j * _PAULI_Z @ factors[position]) @ suffix[position + 1]
        gradient[owner] = derivative[:, 0, 0].real
    return values, gradient


def _qsp_factors(
    phases: NDArray[np.float64], x: NDArray[np.float64]
) -> tuple[list[NDArray[np.complex128]], list[int]]:
    """Return the ordered QSP matrix factors and the phase each one carries."""
    phase_values = _as_phase_vector(phases)
    signal = _as_signal_vector(x)
    sine = np.sqrt(np.maximum(0.0, 1.0 - signal * signal))
    rotation = np.zeros((signal.size, 2, 2), dtype=np.complex128)
    rotation[:, 0, 0] = signal
    rotation[:, 1, 1] = signal
    rotation[:, 0, 1] = 1j * sine
    rotation[:, 1, 0] = 1j * sine

    exponentials = np.exp(1j * phase_values)
    factors: list[NDArray[np.complex128]] = []
    owners: list[int] = []

    def phase_factor(index_: int) -> NDArray[np.complex128]:
        matrix = np.zeros((signal.size, 2, 2), dtype=np.complex128)
        matrix[:, 0, 0] = exponentials[index_]
        matrix[:, 1, 1] = np.conj(exponentials[index_])
        return matrix

    factors.append(phase_factor(0))
    owners.append(0)
    for position in range(1, phase_values.size):
        factors.append(rotation)
        owners.append(-1)
        factors.append(phase_factor(position))
        owners.append(position)
    return factors, owners


def _qsp_factor_product(
    factors: list[NDArray[np.complex128]], owners: list[int]
) -> NDArray[np.complex128]:
    """Multiply the ordered QSP factors into one unitary per signal value."""
    del owners
    product = factors[0]
    for factor in factors[1:]:
        product = product @ factor
    result: NDArray[np.complex128] = product
    return result


def _expand_symmetric(free: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """Mirror the free half of a symmetric phase vector into the full vector."""
    half = free.size
    phases = np.zeros(degree + 1, dtype=np.float64)
    phases[:half] = free
    phases[degree + 1 - half :] = free[::-1][:half]
    return phases


def _jacobi_anger_coefficients(tau: float, degree: int, *, parity: int) -> NDArray[np.float64]:
    """Return truncated Jacobi--Anger Chebyshev coefficients of one parity."""
    tau_value = float(tau)
    if not np.isfinite(tau_value):
        raise ValueError("tau must be finite")
    degree_value = _non_negative_integer(degree, "degree")
    if degree_value % 2 != parity:
        expected = "even" if parity == 0 else "odd"
        raise ValueError(f"degree must be {expected} for this expansion, got {degree_value}")
    coefficients = np.zeros(degree_value + 1, dtype=np.float64)
    if parity == 0:
        coefficients[0] = float(jv(0, tau_value))
        for order in range(1, degree_value // 2 + 1):
            coefficients[2 * order] = 2.0 * (-1.0) ** order * float(jv(2 * order, tau_value))
    else:
        for order in range((degree_value - 1) // 2 + 1):
            coefficients[2 * order + 1] = (
                2.0 * (-1.0) ** order * float(jv(2 * order + 1, tau_value))
            )
    return coefficients


def _validate_target(chebyshev_coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return validated target coefficients of definite parity within unit norm."""
    coefficients = np.asarray(chebyshev_coefficients, dtype=np.float64).reshape(-1)
    if coefficients.size == 0:
        raise ValueError("target requires at least one Chebyshev coefficient")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("target coefficients must all be finite")
    degree = coefficients.size - 1
    if coefficients[degree] == 0.0 and degree > 0:
        raise ValueError("leading target coefficient must be non-zero")
    wrong_parity = coefficients[(degree % 2 + 1) % 2 :: 2] if degree >= 0 else coefficients
    offending = np.flatnonzero(wrong_parity)
    if offending.size:
        raise ValueError("target coefficients must have definite parity")
    grid = np.linspace(-1.0, 1.0, max(MINIMUM_VERIFICATION_GRID, 8 * (degree + 1)))
    supremum = float(np.max(np.abs(cheb.chebval(grid, coefficients))))
    if supremum > 1.0:
        raise ValueError(f"target must satisfy |f(x)| <= 1 on [-1, 1], measured {supremum:.6f}")
    return coefficients


def _as_phase_vector(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a validated one-dimensional phase vector."""
    values = np.asarray(phases, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one phase factor is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("phase factors must all be finite")
    return values


def _as_signal_vector(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return validated signal values confined to the QSP domain."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one signal value is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal values must all be finite")
    if np.any(np.abs(values) > 1.0):
        raise ValueError("signal values must lie in [-1, 1]")
    return values


def _resolve_sample_count(samples: int | None, degree: int) -> int:
    """Return a Chebyshev sample count sufficient to determine degree ``d - 1``."""
    if samples is None:
        return max(4 * (degree + 1), MINIMUM_VERIFICATION_GRID)
    count = _positive_integer(samples, "samples")
    if count < degree + 1:
        raise ValueError(f"samples must be at least {degree + 1} to determine the coefficients")
    return count


def _non_negative_integer(value: Any, name: str) -> int:
    """Return an integer-like non-negative value without boolean coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        integer_value = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative, got {integer_value}")
    return int(integer_value)


def _positive_integer(value: Any, name: str) -> int:
    """Return an integer-like positive value without boolean coercion."""
    integer_value = _non_negative_integer(value, name)
    if integer_value == 0:
        raise ValueError(f"{name} must be positive")
    return integer_value


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOLERANCE",
    "INITIAL_EDGE_PHASE",
    "MINIMUM_VERIFICATION_GRID",
    "VERIFICATION_GRID_FACTOR",
    "QSPPhaseFactors",
    "QSPSynthesisError",
    "complementary_polynomial",
    "jacobi_anger_cosine_coefficients",
    "jacobi_anger_sine_coefficients",
    "qsp_response",
    "qsp_unitary",
    "synthesise_qsp_phases",
]
