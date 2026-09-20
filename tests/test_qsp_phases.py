# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — QSP phase-factor synthesis tests
"""Certified synthesis, refusal and convention contracts for QSP phase factors."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import chebyshev as cheb
from numpy.typing import NDArray

from scpn_quantum_control.phase.qsp_phases import (
    DEFAULT_TOLERANCE,
    INITIAL_EDGE_PHASE,
    QSPPhaseFactors,
    QSPSynthesisError,
    complementary_polynomial,
    jacobi_anger_cosine_coefficients,
    jacobi_anger_sine_coefficients,
    qsp_response,
    qsp_unitary,
    synthesise_qsp_phases,
)

DENSE_GRID = np.linspace(-1.0, 1.0, 1201)


def _chebyshev_basis(degree: int) -> NDArray[np.float64]:
    coefficients = np.zeros(degree + 1, dtype=np.float64)
    coefficients[degree] = 1.0
    return coefficients


def test_zero_phases_realise_the_chebyshev_polynomial_exactly() -> None:
    """The convention anchor: all-zero phases give ``T_d`` to machine precision.

    ``W(x) = exp(i theta X)`` with ``x = cos theta``, so the product of ``d``
    signal operators is ``exp(i d theta X)`` and its top-left entry is
    ``cos(d theta) = T_d(x)``. Any convention error breaks this identity.
    """
    for degree in (1, 2, 3, 5, 8, 13):
        response = qsp_response(np.zeros(degree + 1), DENSE_GRID)
        expected = cheb.chebval(DENSE_GRID, _chebyshev_basis(degree))

        assert np.max(np.abs(response.real - expected)) < 1e-13
        assert np.max(np.abs(response.imag)) < 1e-13


def test_qsp_unitary_is_unitary_on_the_domain() -> None:
    """Every evaluated QSP matrix must be unitary."""
    phases = np.array([0.3, -1.1, 0.7, 2.0, -0.4])
    unitary = qsp_unitary(phases, np.linspace(-1.0, 1.0, 41))
    adjoint = np.conjugate(np.transpose(unitary, (0, 2, 1)))
    products = adjoint @ unitary

    assert np.max(np.abs(products - np.eye(2))) < 1e-13


@pytest.mark.parametrize(("tau", "degree"), [(2.0, 12), (5.0, 24), (10.0, 40)])
def test_cosine_targets_are_certified_to_tolerance(tau: float, degree: int) -> None:
    """Synthesis must realise the Jacobi--Anger cosine target within tolerance."""
    coefficients = jacobi_anger_cosine_coefficients(tau, degree)

    factors = synthesise_qsp_phases(coefficients)

    assert factors.degree == degree
    assert factors.parity == 0
    assert factors.phases.size == degree + 1
    assert factors.supremum_error <= DEFAULT_TOLERANCE
    assert factors.completion_residual <= DEFAULT_TOLERANCE
    realised = qsp_response(factors.phases, DENSE_GRID).real
    assert np.max(np.abs(realised - cheb.chebval(DENSE_GRID, coefficients))) <= DEFAULT_TOLERANCE


@pytest.mark.parametrize(("tau", "degree"), [(2.0, 13), (5.0, 25)])
def test_sine_targets_are_certified_to_tolerance(tau: float, degree: int) -> None:
    """Odd-parity targets must synthesise as well as even ones."""
    coefficients = jacobi_anger_sine_coefficients(tau, degree)

    factors = synthesise_qsp_phases(coefficients)

    assert factors.parity == 1
    assert factors.supremum_error <= DEFAULT_TOLERANCE
    realised = qsp_response(factors.phases, DENSE_GRID).real
    assert np.max(np.abs(realised - cheb.chebval(DENSE_GRID, coefficients))) <= DEFAULT_TOLERANCE


def test_synthesised_phases_are_symmetric() -> None:
    """The symmetric-phase method must return a palindromic phase vector."""
    factors = synthesise_qsp_phases(jacobi_anger_cosine_coefficients(4.0, 16))

    assert factors.phases == pytest.approx(factors.phases[::-1], abs=1e-12)


def test_constant_target_needs_no_signal_operator() -> None:
    """A degree-zero target is realised by the single leading phase."""
    factors = synthesise_qsp_phases(np.array([1.0]))

    assert factors.degree == 0
    assert factors.phases.size == 1
    assert factors.completion_residual == 0.0
    assert qsp_response(factors.phases, DENSE_GRID).real == pytest.approx(1.0, abs=1e-11)


def test_complementary_polynomial_satisfies_the_completion_identity() -> None:
    """The extracted ``Q`` must complete ``P`` to unit magnitude across the domain."""
    factors = synthesise_qsp_phases(jacobi_anger_cosine_coefficients(3.0, 14))
    quadrature = complementary_polynomial(factors.phases)
    response = qsp_response(factors.phases, DENSE_GRID)
    magnitude = response.real**2 + response.imag**2
    identity = magnitude + (1.0 - DENSE_GRID**2) * cheb.chebval(DENSE_GRID, quadrature) ** 2

    assert quadrature.size == factors.degree
    assert np.max(np.abs(identity - 1.0)) < 1e-11


def test_complementary_polynomial_has_opposite_parity() -> None:
    """``Q`` must carry the parity the structure theorem assigns it."""
    factors = synthesise_qsp_phases(jacobi_anger_cosine_coefficients(3.0, 10))
    quadrature = complementary_polynomial(factors.phases)

    assert np.max(np.abs(quadrature[0::2])) < 1e-10


def test_complementary_polynomial_accepts_an_explicit_sample_count() -> None:
    """An explicit sample count must reproduce the default determination."""
    factors = synthesise_qsp_phases(jacobi_anger_cosine_coefficients(2.0, 8))
    default = complementary_polynomial(factors.phases)
    explicit = complementary_polynomial(factors.phases, samples=600)

    assert explicit == pytest.approx(default, abs=1e-10)


def test_complementary_polynomial_refuses_insufficient_evidence() -> None:
    """Too few phases or samples cannot determine the complementary polynomial."""
    with pytest.raises(ValueError, match="at least two phase factors"):
        complementary_polynomial(np.array([0.2]))

    with pytest.raises(ValueError, match="samples must be at least"):
        complementary_polynomial(np.zeros(9), samples=4)


def test_jacobi_anger_coefficients_reproduce_the_analytic_functions() -> None:
    """The truncated expansions must match the functions they name."""
    cosine = jacobi_anger_cosine_coefficients(3.0, 24)
    sine = jacobi_anger_sine_coefficients(3.0, 25)

    assert np.max(np.abs(cheb.chebval(DENSE_GRID, cosine) - np.cos(3.0 * DENSE_GRID))) < 1e-12
    assert np.max(np.abs(cheb.chebval(DENSE_GRID, sine) - np.sin(3.0 * DENSE_GRID))) < 1e-12
    assert np.max(np.abs(cosine[1::2])) == 0.0
    assert np.max(np.abs(sine[0::2])) == 0.0


@pytest.mark.parametrize(
    ("builder", "degree", "message"),
    [
        (jacobi_anger_cosine_coefficients, 11, "degree must be even"),
        (jacobi_anger_sine_coefficients, 12, "degree must be odd"),
    ],
)
def test_jacobi_anger_refuses_the_wrong_parity(builder: object, degree: int, message: str) -> None:
    """Each expansion owns one parity and must refuse the other."""
    with pytest.raises(ValueError, match=message):
        builder(2.0, degree)  # type: ignore[operator]


def test_jacobi_anger_refuses_non_finite_and_invalid_degrees() -> None:
    """A non-finite parameter or an invalid degree is refused."""
    with pytest.raises(ValueError, match="tau must be finite"):
        jacobi_anger_cosine_coefficients(float("inf"), 4)

    with pytest.raises(ValueError, match="degree must be non-negative"):
        jacobi_anger_cosine_coefficients(2.0, -2)

    with pytest.raises(ValueError, match="degree must be a non-negative integer"):
        jacobi_anger_cosine_coefficients(2.0, True)

    with pytest.raises(ValueError, match="degree must be a non-negative integer"):
        jacobi_anger_cosine_coefficients(2.0, "4")  # type: ignore[arg-type]


def test_target_validation_refuses_unrealisable_inputs() -> None:
    """A target outside the QSP structure theorem must never reach Newton."""
    with pytest.raises(ValueError, match="at least one Chebyshev coefficient"):
        synthesise_qsp_phases(np.array([]))

    with pytest.raises(ValueError, match="must all be finite"):
        synthesise_qsp_phases(np.array([0.0, float("nan")]))

    with pytest.raises(ValueError, match="leading target coefficient"):
        synthesise_qsp_phases(np.array([0.5, 0.0]))

    with pytest.raises(ValueError, match="definite parity"):
        synthesise_qsp_phases(np.array([0.2, 0.3]))

    with pytest.raises(ValueError, match=r"\|f\(x\)\| <= 1"):
        synthesise_qsp_phases(np.array([0.0, 2.0]))


def test_synthesis_refuses_an_invalid_budget() -> None:
    """A non-positive tolerance or iteration budget is refused before work starts."""
    coefficients = jacobi_anger_cosine_coefficients(2.0, 8)

    with pytest.raises(ValueError, match="tolerance must be a positive finite number"):
        synthesise_qsp_phases(coefficients, tolerance=0.0)

    with pytest.raises(ValueError, match="tolerance must be a positive finite number"):
        synthesise_qsp_phases(coefficients, tolerance=float("nan"))

    with pytest.raises(ValueError, match="max_iterations must be positive"):
        synthesise_qsp_phases(coefficients, max_iterations=0)


def test_synthesis_refuses_an_exhausted_iteration_budget() -> None:
    """An unconverged Newton run must raise rather than return its last iterate."""
    coefficients = jacobi_anger_cosine_coefficients(10.0, 40)

    with pytest.raises(QSPSynthesisError, match="did not converge in 2 steps"):
        synthesise_qsp_phases(coefficients, max_iterations=2)


def test_synthesis_refuses_a_tolerance_it_cannot_certify() -> None:
    """A tolerance below attainable precision must fail closed, not round down."""
    coefficients = jacobi_anger_cosine_coefficients(5.0, 24)

    with pytest.raises(QSPSynthesisError, match="did not converge"):
        synthesise_qsp_phases(coefficients, tolerance=1e-18)


def test_response_and_unitary_refuse_invalid_arguments() -> None:
    """Empty, non-finite or out-of-domain arguments are refused."""
    with pytest.raises(ValueError, match="at least one phase factor"):
        qsp_response(np.array([]), np.array([0.0]))

    with pytest.raises(ValueError, match="phase factors must all be finite"):
        qsp_response(np.array([float("inf")]), np.array([0.0]))

    with pytest.raises(ValueError, match="at least one signal value"):
        qsp_response(np.zeros(3), np.array([]))

    with pytest.raises(ValueError, match="signal values must all be finite"):
        qsp_response(np.zeros(3), np.array([float("nan")]))

    with pytest.raises(ValueError, match=r"signal values must lie in \[-1, 1\]"):
        qsp_unitary(np.zeros(3), np.array([1.5]))


def test_phase_factors_serialise_their_evidence() -> None:
    """The record must expose the certificate, not only the phases."""
    factors = synthesise_qsp_phases(jacobi_anger_cosine_coefficients(2.0, 8))
    payload = factors.to_dict()

    assert isinstance(factors, QSPPhaseFactors)
    assert payload["degree"] == 8
    assert payload["parity"] == 0
    assert len(payload["phases"]) == 9  # type: ignore[arg-type]
    assert payload["supremum_error"] <= DEFAULT_TOLERANCE  # type: ignore[operator]
    assert payload["completion_residual"] <= DEFAULT_TOLERANCE  # type: ignore[operator]
    assert payload["verification_grid"] >= 512  # type: ignore[operator]
    assert payload["newton_iterations"] >= 1  # type: ignore[operator]


def test_published_initial_guess_is_not_a_solution() -> None:
    """The documented starting point must not be mistaken for synthesised phases."""
    degree = 12
    guess = np.zeros(degree + 1)
    guess[0] = INITIAL_EDGE_PHASE
    guess[-1] = INITIAL_EDGE_PHASE
    coefficients = jacobi_anger_cosine_coefficients(2.0, degree)
    factors = synthesise_qsp_phases(coefficients)

    guess_error = np.max(
        np.abs(qsp_response(guess, DENSE_GRID).real - cheb.chebval(DENSE_GRID, coefficients))
    )
    assert guess_error > 1e-3
    assert factors.supremum_error < guess_error


def test_certification_refuses_a_response_that_misses_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response that drifts after Newton must be refused, not returned.

    The dense certificate is the authority. Newton converging at its
    collocation nodes is necessary but not sufficient, so the guard is
    exercised by perturbing only the certification evaluation.
    """
    import scpn_quantum_control.phase.qsp_phases as module

    genuine = module.qsp_response

    def drifted(phases: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.complex128]:
        return genuine(phases, x) + 1e-3

    monkeypatch.setattr(module, "qsp_response", drifted)

    with pytest.raises(QSPSynthesisError, match="certified supremum error"):
        synthesise_qsp_phases(jacobi_anger_cosine_coefficients(2.0, 8))


def test_certification_refuses_a_broken_completion_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phases whose complementary polynomial fails the identity are refused."""
    import scpn_quantum_control.phase.qsp_phases as module

    monkeypatch.setattr(
        module,
        "_completion_residual",
        lambda phases, grid, realised: 1.0,
    )

    with pytest.raises(QSPSynthesisError, match="completion identity residual"):
        synthesise_qsp_phases(jacobi_anger_cosine_coefficients(2.0, 8))


def test_newton_refuses_a_singular_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Newton step that cannot be solved must fail closed with its iteration."""

    def singular(*args: object, **kwargs: object) -> NDArray[np.float64]:
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(np.linalg, "solve", singular)

    with pytest.raises(QSPSynthesisError, match="singular Jacobian"):
        synthesise_qsp_phases(jacobi_anger_cosine_coefficients(2.0, 8))
