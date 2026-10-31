# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — spectrum-bound quantum gradient estimator qualification
"""Generator spectra, shared parameters, refusals and finite-shot coverage.

Expectations come from explicit two-level state vectors. Derivatives are
compared with closed forms under the frozen float64 budget
``1e-10 + 1e-8*|reference|``. The finite-shot study follows the frozen
protocol: seeds 0..199, 1000 shots per shifted expectation, every replicate
retained, and the nominal level checked against the Clopper-Pearson interval
of the observed coverage.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import beta

from scpn_quantum_control import (
    GateOccurrence,
    Parameter,
    QuantumGradientEstimatorRefusal,
    execute_quantum_gradient_estimator,
    execute_quantum_gradient_estimator_with_uncertainty,
    multi_frequency_parameter_shift_rule,
    parameter_shift_gradient_with_uncertainty,
    plan_quantum_gradient_estimator,
    value_and_parameter_shift_grad,
)

Matrix = NDArray[np.complex128]
_ZERO = np.array([1.0, 0.0], dtype=np.complex128)
_PAULI_Z = np.diag([1.0, -1.0]).astype(np.complex128)
_CALIBRATION: dict[str, object] = {"device": "statevector", "readout_error": 0.0}
_BUDGET_ABSOLUTE = 1.0e-10
_BUDGET_RELATIVE = 1.0e-8


def _within_budget(actual: object, reference: object) -> bool:
    """Apply the frozen float64 test budget elementwise."""
    actual_array = np.asarray(actual, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    return bool(
        np.all(
            np.abs(actual_array - reference_array)
            <= _BUDGET_ABSOLUTE + _BUDGET_RELATIVE * np.abs(reference_array)
        )
    )


def _ry(angle: float) -> Matrix:
    """Return ``exp(-i * angle * Y / 2)``."""
    cosine, sine = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


def _rx(angle: float) -> Matrix:
    """Return ``exp(-i * angle * X / 2)``."""
    cosine, sine = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return np.array([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=np.complex128)


def _expectation(state: NDArray[np.complex128]) -> float:
    """Return the real expectation of Pauli Z."""
    return float(np.real(np.conj(state) @ _PAULI_Z @ state))


def _single_rotation(values: NDArray[np.float64]) -> float:
    """Return Z after ``RY(theta)|0>``, which equals ``cos(theta)``."""
    return _expectation(_ry(float(values[0])) @ _ZERO)


def _shared_rotation(values: NDArray[np.float64]) -> float:
    """Return Z after ``RY(theta) RY(theta)|0>``, i.e. ``cos(2*theta)``."""
    angle = float(values[0])
    return _expectation(_ry(angle) @ _ry(angle) @ _ZERO)


def _mixed_rotations(values: NDArray[np.float64]) -> float:
    """Return Z after ``RX(2*a) RY(a) RY(b)|0>``.

    It equals ``cos(2*a) * cos(a + b)``: the RY pair turns the Bloch vector to
    polar angle ``a + b`` in the x-z plane, and the RX rotation scales its z
    component by ``cos(2*a)``.
    """
    first, second = float(values[0]), float(values[1])
    return _expectation(_rx(2.0 * first) @ _ry(first) @ _ry(second) @ _ZERO)


def _plan(occurrences: list[GateOccurrence], count: int, **options: Any) -> Any:
    """Plan with the shared calibration and observable."""
    return plan_quantum_gradient_estimator(
        occurrences, count, observable="Z", calibration=_CALIBRATION, **options
    )


@pytest.mark.parametrize("theta", [-1.1, 0.0, 0.4, 2.5])
def test_quantum_gradient_estimators_01(theta: float) -> None:
    """``RY(theta)|0>`` with Z has value ``cos(theta)`` and slope ``-sin(theta)``.

    Parameters
    ----------
    theta
        Rotation angle.

    """
    plan = _plan([GateOccurrence("ry", 0)], 1)
    assert plan.frequencies == ((1.0,),)
    rule = plan.rules[0]
    assert rule is not None and rule.terms == ((math.pi / 2.0, 0.5),)
    assert plan.evaluations == 3
    result = execute_quantum_gradient_estimator(
        plan, _single_rotation, [theta], calibration=_CALIBRATION
    )
    assert _within_budget(result.value, math.cos(theta))
    assert _within_budget(result.gradient, [-math.sin(theta)])
    assert result.evaluations == plan.evaluations
    assert result.method == "spectrum_bound_parameter_shift"


@pytest.mark.parametrize("theta", [-0.9, 0.4, 1.3])
def test_quantum_gradient_estimators_02(theta: float) -> None:
    """A parameter shared by two rotations needs both chain contributions.

    The standard two-term rule sees ``cos(2*theta)`` only through frequency 1
    and returns zero. The plan derives the frequencies ``{1, 2}`` and its rule
    returns ``-2*sin(2*theta)``.

    Parameters
    ----------
    theta
        Shared rotation angle.

    """
    exact = -2.0 * math.sin(2.0 * theta)
    generic = value_and_parameter_shift_grad(_shared_rotation, [theta])
    assert abs(float(generic.gradient[0])) < 1.0e-12
    assert abs(float(generic.gradient[0]) - exact) > 0.1

    plan = _plan([GateOccurrence("ry", 0), GateOccurrence("ry", 0)], 1)
    assert plan.frequencies == ((1.0, 2.0),)
    result = execute_quantum_gradient_estimator(
        plan, _shared_rotation, [theta], calibration=_CALIBRATION
    )
    assert _within_budget(result.value, math.cos(2.0 * theta))
    assert _within_budget(result.gradient, [exact])


def test_quantum_gradient_estimators_02_mixed_scales_and_two_parameters() -> None:
    """Different gates and scales on one parameter combine into one spectrum."""
    first, second = 0.35, -0.8
    plan = _plan(
        [
            GateOccurrence("ry", 1),
            GateOccurrence("ry", 0),
            GateOccurrence("rx", 0, 2.0),
        ],
        2,
    )
    assert plan.frequencies == ((1.0, 2.0, 3.0), (1.0,))
    result = execute_quantum_gradient_estimator(
        plan, _mixed_rotations, [first, second], calibration=_CALIBRATION
    )
    phase = first + second
    expected = (
        -2.0 * math.sin(2.0 * first) * math.cos(phase) - math.cos(2.0 * first) * math.sin(phase),
        -math.cos(2.0 * first) * math.sin(phase),
    )
    assert _within_budget(result.value, math.cos(2.0 * first) * math.cos(phase))
    assert _within_budget(result.gradient, expected)
    assert result.evaluations == plan.evaluations == 1 + 2 * 3 + 2


@pytest.mark.parametrize(
    ("occurrences", "frequencies"),
    [
        ([GateOccurrence("ry", 0, 0.5)], (0.5,)),
        ([GateOccurrence("phase_rotation", 0)], (1.0,)),
        ([GateOccurrence("controlled_phase", 0), GateOccurrence("rz", 0)], (1.0, 2.0)),
        ([GateOccurrence("rz", 0, np.float64(-3.0))], (3.0,)),
    ],
    ids=["half-scale", "phase", "controlled-phase-and-rz", "numpy-scale"],
)
def test_registered_generators_give_their_difference_spectra(
    occurrences: list[GateOccurrence], frequencies: tuple[float, ...]
) -> None:
    """Frequencies are the positive differences of scaled eigenvalue sums.

    Parameters
    ----------
    occurrences
        Gate occurrences of the single parameter.
    frequencies
        Independently derived difference spectrum.

    """
    plan = _plan(occurrences, 1)
    assert plan.frequencies == (frequencies,)

    def objective(values: NDArray[np.float64]) -> float:
        # A pure trigonometric polynomial with exactly these frequencies.
        return sum(math.cos(omega * float(values[0]) + omega) for omega in frequencies)

    theta = 0.6
    result = execute_quantum_gradient_estimator(plan, objective, [theta], calibration=_CALIBRATION)
    expected = -sum(omega * math.sin(omega * theta + omega) for omega in frequencies)
    assert _within_budget(result.gradient, [expected])


def test_unused_and_frozen_parameters_have_exact_zero_gradients() -> None:
    """A parameter without occurrences, or a frozen one, is never shifted."""
    calls: list[tuple[float, ...]] = []

    def objective(values: NDArray[np.float64]) -> float:
        calls.append(tuple(float(value) for value in values))
        return math.cos(float(values[0])) + math.sin(float(values[2]))

    plan = _plan([GateOccurrence("ry", 0), GateOccurrence("ry", 2)], 3)
    assert plan.frequencies[1] == () and plan.rules[1] is None
    result = execute_quantum_gradient_estimator(
        plan,
        objective,
        [0.2, 0.9, -0.4],
        calibration=_CALIBRATION,
        parameters=[Parameter("a"), Parameter("b"), Parameter("c", trainable=False)],
    )
    assert _within_budget(result.gradient, [-math.sin(0.2), 0.0, 0.0])
    assert all(call[1] == 0.9 and call[2] == -0.4 for call in calls)
    assert result.evaluations == len(calls) == 3


@pytest.mark.parametrize(
    ("occurrences", "operations", "reason"),
    [
        ([GateOccurrence("u3", 0)], (), "unsupported_generator"),
        ([GateOccurrence("ry", 0)], ("measure",), "non_smooth_operation"),
        ([GateOccurrence("ry", 0)], ("reset", "cx"), "non_smooth_operation"),
        ([GateOccurrence("ry", 0)], ("postselect",), "non_smooth_operation"),
        ([GateOccurrence("ry", 0)], ("jump",), "non_smooth_operation"),
        ([GateOccurrence("ry", 3)], (), "invalid_occurrence"),
        ([GateOccurrence("ry", cast(int, 0.0))], (), "invalid_occurrence"),
        ([GateOccurrence("ry", 0, 0.0)], (), "invalid_occurrence"),
        ([GateOccurrence("ry", 0, math.nan)], (), "invalid_occurrence"),
        ([GateOccurrence("ry", 0, cast(float, True))], (), "invalid_occurrence"),
        ([GateOccurrence("ry", 0, cast(float, "1"))], (), "invalid_occurrence"),
        ([cast(GateOccurrence, ("ry", 0))], (), "invalid_occurrence"),
        (
            [GateOccurrence("ry", 0), GateOccurrence("ry", 0, 1.0e-9)],
            (),
            "ill_conditioned_spectrum",
        ),
        (
            [GateOccurrence("ry", 0, math.sqrt(prime)) for prime in (1, 2, 3, 5, 7, 11, 13)],
            (),
            "spectrum_too_large",
        ),
    ],
    ids=[
        "unregistered-gate",
        "measure",
        "reset",
        "postselect",
        "jump",
        "index-out-of-range",
        "index-not-int",
        "zero-scale",
        "nan-scale",
        "bool-scale",
        "text-scale",
        "not-an-occurrence",
        "ill-conditioned",
        "too-many-sums",
    ],
)
def test_quantum_gradient_estimators_03(
    occurrences: list[GateOccurrence], operations: tuple[str, ...], reason: str
) -> None:
    """Unsupported generators and non-smooth operations refuse at planning.

    Parameters
    ----------
    occurrences
        Requested gate occurrences.
    operations
        Other circuit operations.
    reason
        Expected refusal code.

    """
    with pytest.raises(QuantumGradientEstimatorRefusal) as refusal:
        _plan(occurrences, 1, operations=operations)
    assert refusal.value.reason == reason


@pytest.mark.parametrize(
    ("count", "observable", "shots", "message"),
    [
        (0, "Z", None, "parameter_count"),
        (cast(int, True), "Z", None, "parameter_count"),
        (1, "", None, "observable"),
        (1, "Z", 0, "shots"),
        (1, "Z", cast(int, 2.0), "shots"),
    ],
)
def test_plan_refuses_malformed_requests(
    count: int, observable: str, shots: int | None, message: str
) -> None:
    """Counts, observable names and shot counts are validated before planning.

    Parameters
    ----------
    count
        Parameter count.
    observable
        Observable name.
    shots
        Shots per evaluation.
    message
        Expected field named by the refusal.

    """
    with pytest.raises(ValueError, match=message):
        plan_quantum_gradient_estimator(
            [GateOccurrence("ry", 0)],
            count,
            observable=observable,
            calibration=_CALIBRATION,
            shots=shots,
        )


def test_changed_calibration_or_budget_invalidates_the_plan_before_execution() -> None:
    """A drifted calibration or shot budget refuses without evaluating anything."""
    calls: list[str] = []

    def objective(values: NDArray[np.float64]) -> float:
        calls.append("evaluated")
        return _single_rotation(values)

    plan = _plan([GateOccurrence("ry", 0)], 1, shots=1000)
    assert plan.shot_budget == 3000
    changed = {**_CALIBRATION, "readout_error": 0.01}
    with pytest.raises(QuantumGradientEstimatorRefusal) as refusal:
        execute_quantum_gradient_estimator(plan, objective, [0.3], calibration=changed, shots=1000)
    assert refusal.value.reason == "calibration_changed"
    for drifted in (None, 2000):
        with pytest.raises(QuantumGradientEstimatorRefusal) as refusal:
            execute_quantum_gradient_estimator(
                plan, objective, [0.3], calibration=_CALIBRATION, shots=drifted
            )
        assert refusal.value.reason == "budget_drift"
    assert calls == []
    assert _plan([GateOccurrence("ry", 0)], 1).shot_budget is None

    result = execute_quantum_gradient_estimator(
        plan, objective, [0.3], calibration=dict(_CALIBRATION), shots=1000
    )
    assert _within_budget(result.gradient, [-math.sin(0.3)])
    assert len(calls) == 3


def test_plan_digest_binds_every_planned_input() -> None:
    """Observable, shots, calibration and occurrences each change the digest."""
    base = _plan([GateOccurrence("ry", 0)], 1, shots=100)
    variants = [
        plan_quantum_gradient_estimator(
            [GateOccurrence("ry", 0)], 1, observable="X", calibration=_CALIBRATION, shots=100
        ),
        _plan([GateOccurrence("ry", 0)], 1, shots=200),
        plan_quantum_gradient_estimator(
            [GateOccurrence("ry", 0)], 1, observable="Z", calibration={"device": "qpu"}, shots=100
        ),
        _plan([GateOccurrence("rx", 0)], 1, shots=100),
        _plan([GateOccurrence("ry", 0, 2.0)], 1, shots=100),
    ]
    digests = {base.digest, *(variant.digest for variant in variants)}
    assert len(digests) == 1 + len(variants)
    assert _plan([GateOccurrence("ry", 0)], 1, shots=100).digest == base.digest


def test_execution_refuses_malformed_plan_or_values() -> None:
    """Execution requires a plan and values of the planned width."""
    plan = _plan([GateOccurrence("ry", 0)], 1)
    with pytest.raises(ValueError, match="QuantumGradientEstimatorPlan"):
        execute_quantum_gradient_estimator(
            cast(Any, plan.digest), _single_rotation, [0.1], calibration=_CALIBRATION
        )
    with pytest.raises(ValueError, match="planned parameter count"):
        execute_quantum_gradient_estimator(
            plan, _single_rotation, [0.1, 0.2], calibration=_CALIBRATION
        )


def _binomial_expectation(
    angle: float, shots: int, generator: np.random.Generator
) -> tuple[float, float]:
    """Sample Z after ``RY(angle)|0>`` and return its mean and unbiased variance."""
    probability_zero = math.cos(angle / 2.0) ** 2
    zeros = int(generator.binomial(shots, probability_zero))
    mean = (2.0 * zeros - shots) / shots
    return mean, (1.0 - mean * mean) * shots / (shots - 1)


def test_quantum_gradient_estimators_04() -> None:
    """Finite-shot intervals cover the exact derivative at the nominal rate.

    Every replicate of the frozen protocol is retained. The planned rule
    shifts each angle by ``pi/2``; the plus and minus expectations use
    independent streams spawned from the replicate seed, as the variance
    formula requires. The estimand is the exact derivative of row 01.
    """
    theta, shots, confidence = 0.7, 1000, 0.95
    plan = _plan([GateOccurrence("ry", 0)], 1, shots=shots)
    rule = plan.rules[0]
    assert rule is not None
    ((shift, _coefficient),) = rule.terms
    exact = -math.sin(theta)
    replicates: list[tuple[int, float, float, float, bool]] = []
    for seed in range(200):
        plus_stream, minus_stream = (
            np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
        )
        plus_mean, plus_variance = _binomial_expectation(theta + shift, shots, plus_stream)
        minus_mean, minus_variance = _binomial_expectation(theta - shift, shots, minus_stream)
        estimate = parameter_shift_gradient_with_uncertainty(
            [plus_mean],
            [minus_mean],
            [plus_variance],
            [minus_variance],
            [shots],
            sample_provenance={
                "sample_seed": seed,
                "shot_batch_id": f"replicate-{seed}",
                "source_class": "local_simulator",
            },
            rule=rule,
            confidence_level=confidence,
        )
        interval = estimate.confidence_interval
        assert interval is not None
        lower, upper = float(interval.lower[0]), float(interval.upper[0])
        replicates.append(
            (seed, float(estimate.gradient[0]), lower, upper, lower <= exact <= upper)
        )

    assert len(replicates) == 200
    covered = sum(row[4] for row in replicates)
    observed = covered / len(replicates)
    low = float(beta.ppf(0.025, covered, len(replicates) - covered + 1))
    high = float(beta.ppf(0.975, covered + 1, len(replicates) - covered))
    assert low <= confidence <= high, (observed, low, high)
    estimates = np.array([row[1] for row in replicates])
    # The estimator is unbiased for the same estimand: the replicate mean is
    # within four standard errors of the mean of the exact derivative.
    standard_error = float(np.std(estimates, ddof=1)) / math.sqrt(len(estimates))
    assert abs(float(np.mean(estimates)) - exact) <= 4.0 * standard_error


def test_quantum_gradient_estimators_04_runs_the_same_plan_exactly() -> None:
    """The finite-shot plan's rule gives the exact derivative on exact expectations."""
    plan = _plan([GateOccurrence("ry", 0)], 1)
    objective: Callable[[NDArray[np.float64]], float] = _single_rotation
    result = execute_quantum_gradient_estimator(plan, objective, [0.7], calibration=_CALIBRATION)
    assert _within_budget(result.gradient, [-math.sin(0.7)])


@pytest.mark.parametrize("frequency", [3.0, 6.0, 9.0])
def test_single_frequency_rule_avoids_a_vanishing_sine(frequency: float) -> None:
    """A lone frequency whose default shift lands on a sine zero gets another shift.

    For ``w = 3`` the first candidate shift ``pi/3`` makes ``2*sin(w*s)``
    vanish to rounding, and the one-term system still has condition number
    one; its coefficient was about ``1e16`` and the derivative was wrong.

    Parameters
    ----------
    frequency
        Integer multiple of three.

    """
    rule = multi_frequency_parameter_shift_rule([frequency])
    ((shift, coefficient),) = rule.terms
    assert abs(2.0 * math.sin(frequency * shift)) >= 1.0e-6
    assert abs(coefficient) < 10.0
    theta = 0.6
    result = value_and_parameter_shift_grad(
        lambda values: math.cos(frequency * float(values[0]) + 1.0), [theta], rule=rule
    )
    assert _within_budget(result.gradient, [-frequency * math.sin(frequency * theta + 1.0)])


def test_explicit_shift_on_a_sine_zero_is_refused() -> None:
    """An explicit shift that zeroes the one-term sine system refuses."""
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        multi_frequency_parameter_shift_rule([3.0], shifts=[math.pi / 3.0])


def _trig_polynomial_derivative_error(frequencies: list[float], seed: int) -> float:
    """Return the planned rule's error on a random trigonometric polynomial."""
    rule = multi_frequency_parameter_shift_rule(frequencies)
    generator = np.random.default_rng(seed)
    weights = generator.normal(size=(2, len(frequencies)))
    spectrum = np.asarray(frequencies, dtype=np.float64)

    def polynomial(angle: float) -> float:
        return float(weights[0] @ np.cos(spectrum * angle) + weights[1] @ np.sin(spectrum * angle))

    theta = 0.37
    estimate = sum(
        coefficient * (polynomial(theta + shift) - polynomial(theta - shift))
        for shift, coefficient in rule.terms
    )
    exact = float(
        spectrum @ (-weights[0] * np.sin(spectrum * theta) + weights[1] * np.cos(spectrum * theta))
    )
    return abs(estimate - exact)


@pytest.mark.parametrize(
    ("frequencies", "largest_factor"),
    [
        ([2.9999995], 4.6),
        ([3.0000005], 4.6),
        ([1.0, 2.0, 3.0], 3.2),
        ([3.0, 6.0, 9.0], 28.6),
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.2),
        ([float(order) for order in range(1, 11)], 34.0),
        ([1.0, 3.0], 22.0),
        ([0.5, 1.5, 2.5], 37.0),
    ],
    ids=[
        "just-below-three",
        "just-above-three",
        "one-two-three",
        "three-six-nine",
        "tenths",
        "one-to-ten",
        "one-three",
        "odd-halves",
    ],
)
def test_planned_rules_are_exact_and_cheap(
    frequencies: list[float], largest_factor: float
) -> None:
    """Equidistant spectra get the closed-form rule; others the least-variance grid.

    A frequency a hair from an integer multiple of pi's denominators used to
    pick a shift on a sine zero, with coefficients near ``3e6``.

    Parameters
    ----------
    frequencies
        Generator frequency set.
    largest_factor
        Upper bound on ``2 * sum c**2``.

    """
    rule = multi_frequency_parameter_shift_rule(frequencies)
    assert rule.shot_variance_factor <= largest_factor
    assert _trig_polynomial_derivative_error(frequencies, 7) <= 1e-12
    if len(frequencies) == 1:
        ((shift, coefficient),) = rule.terms
        assert _within_budget(
            [shift, coefficient], [math.pi / (2 * frequencies[0]), frequencies[0] / 2]
        )


def test_scaled_and_long_equidistant_spectra_are_planned_quickly() -> None:
    """Units do not decide refusal, and planning time stays bounded."""
    tenths = _plan(
        [GateOccurrence("rz", 0, 0.1), GateOccurrence("rx", 0, 0.2), GateOccurrence("ry", 0, 0.3)],
        1,
    )
    assert tenths.frequencies[0] == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6), rel=1e-12)
    ten = _plan([GateOccurrence("ry", 0)] * 10, 1)
    assert ten.frequencies[0] == tuple(float(order) for order in range(1, 11))
    started = time.perf_counter()
    irrational = [
        GateOccurrence("ry", 0, scale) for scale in (1.0, 2**0.5, 3**0.5, 5**0.5, 7**0.5)
    ]
    with pytest.raises(QuantumGradientEstimatorRefusal) as refusal:
        _plan(irrational, 1)
    assert refusal.value.reason == "spectrum_too_large"
    powers = [GateOccurrence("ry", 0, float(2**power)) for power in range(5)]
    assert len(_plan(powers, 1).frequencies[0]) == 31
    assert time.perf_counter() - started < 5.0


@pytest.mark.parametrize("operation", ["MEASURE", "Reset", "measurement", "mid_circuit_measure"])
def test_non_smooth_operations_are_refused_in_any_spelling(operation: str) -> None:
    """The deny-list matches words in any case, not exact names only.

    Parameters
    ----------
    operation
        Variant spelling of a non-smooth operation.

    """
    with pytest.raises(QuantumGradientEstimatorRefusal) as refusal:
        _plan([GateOccurrence("ry", 0)], 1, operations=[operation])
    assert refusal.value.reason == "non_smooth_operation"


def test_numpy_parameter_index_and_variance_factors() -> None:
    """``np.int64`` indices plan like ints, and each rule reports its shot cost."""
    plan = _plan([GateOccurrence("ry", cast(int, np.int64(0))), GateOccurrence("ry", 0)], 2)
    assert plan.frequencies == ((1.0, 2.0), ())
    factors = plan.shot_variance_factors
    assert factors[1] is None and factors[0] is not None
    assert _within_budget(factors[0], 1.5)


@pytest.mark.parametrize("field", ["rules", "shots", "evaluations"])
def test_a_changed_plan_is_refused_before_evaluation(field: str) -> None:
    """Editing a plan after planning breaks its digest, and nothing is evaluated.

    Parameters
    ----------
    field
        Planned field that is replaced.

    """
    plan = _plan([GateOccurrence("ry", 0)], 1, shots=100)
    replacement: dict[str, object] = {
        "rules": (multi_frequency_parameter_shift_rule([1.0], shifts=[0.3]),),
        "shots": 10,
        "evaluations": 1,
    }
    changed = dataclasses.replace(plan, **{field: replacement[field]})
    calls: list[object] = []

    def objective(values: NDArray[np.float64]) -> float:
        calls.append(values)
        return 0.0

    with pytest.raises(QuantumGradientEstimatorRefusal) as refusal:
        execute_quantum_gradient_estimator(
            changed, objective, [0.4], calibration=_CALIBRATION, shots=changed.shots
        )
    assert refusal.value.reason == "plan_changed"
    assert calls == []


def _shared_sampler(shots: int, seed: int) -> Callable[[NDArray[np.float64]], tuple[float, float]]:
    """Sample Z after ``RY(a) RY(a) RY(b)|0>`` with a fresh stream per call."""
    streams = iter(np.random.SeedSequence(seed).spawn(64))

    def sampler(values: NDArray[np.float64]) -> tuple[float, float]:
        angle = 2.0 * float(values[0]) + float(values[1])
        return _binomial_expectation(angle, shots, np.random.default_rng(next(streams)))

    return sampler


def test_finite_shot_execution_uses_each_parameter_rule() -> None:
    """A shared and a single parameter use their own rules, signs included."""
    plan = _plan(
        [GateOccurrence("ry", 0), GateOccurrence("ry", 0), GateOccurrence("ry", 1)], 2, shots=1000
    )
    shared, single = plan.rules
    assert shared is not None and single is not None
    assert len(shared.terms) == 2 and min(c for _, c in shared.terms) < 0.0
    values = [0.4, 0.3]

    def exact(points: NDArray[np.float64]) -> tuple[float, float]:
        return math.cos(2.0 * float(points[0]) + float(points[1])), 0.0

    result = execute_quantum_gradient_estimator_with_uncertainty(
        plan,
        exact,
        values,
        calibration=_CALIBRATION,
        shots=1000,
        sample_provenance={
            "sample_seed": 0,
            "shot_batch_id": "exact",
            "source_class": "synthetic_fixture",
        },
        observable_range=(-1.0, 1.0),
    )
    angle = 2.0 * values[0] + values[1]
    assert _within_budget(result.gradient, [-2.0 * math.sin(angle), -math.sin(angle)])
    assert result.evaluations == plan.evaluations
    assert {record.parameter_index for record in result.records} == {0, 1}
    frozen = execute_quantum_gradient_estimator_with_uncertainty(
        plan,
        exact,
        values,
        calibration=_CALIBRATION,
        shots=1000,
        sample_provenance={
            "sample_seed": 0,
            "shot_batch_id": "frozen",
            "source_class": "synthetic_fixture",
        },
        observable_range=(-1.0, 1.0),
        parameters=[Parameter("a"), Parameter("b", trainable=False)],
    )
    assert frozen.gradient[1] == 0.0 and frozen.evaluations == plan.evaluations - 2


def test_finite_shot_execution_refuses_before_sampling() -> None:
    """Calibration, budget, exact plans and bad samples refuse; nothing is sampled first."""
    plan = _plan([GateOccurrence("ry", 0)], 1, shots=100)
    calls: list[object] = []

    def sampler(values: NDArray[np.float64]) -> tuple[float, float]:
        calls.append(values)
        return 0.0, -1.0

    provenance = {"sample_seed": 1, "shot_batch_id": "refusal", "source_class": "local_simulator"}
    for calibration, shots, reason in (
        ({"device": "other"}, 100, "calibration_changed"),
        (_CALIBRATION, 50, "budget_drift"),
    ):
        with pytest.raises(QuantumGradientEstimatorRefusal) as refusal:
            execute_quantum_gradient_estimator_with_uncertainty(
                plan,
                sampler,
                [0.2],
                calibration=calibration,
                shots=shots,
                sample_provenance=provenance,
                observable_range=(-1.0, 1.0),
            )
        assert refusal.value.reason == reason
    assert calls == []
    exact_plan = _plan([GateOccurrence("ry", 0)], 1)
    with pytest.raises(ValueError, match="exact plan"):
        execute_quantum_gradient_estimator_with_uncertainty(
            exact_plan,
            sampler,
            [0.2],
            calibration=_CALIBRATION,
            shots=cast(int, None),
            sample_provenance=provenance,
            observable_range=(-1.0, 1.0),
        )
    with pytest.raises(ValueError, match="non-negative"):
        execute_quantum_gradient_estimator_with_uncertainty(
            plan,
            sampler,
            [0.2],
            calibration=_CALIBRATION,
            shots=100,
            sample_provenance=provenance,
            observable_range=(-1.0, 1.0),
        )
    with pytest.raises(ValueError, match="values length"):
        execute_quantum_gradient_estimator_with_uncertainty(
            plan,
            sampler,
            [0.2, 0.1],
            calibration=_CALIBRATION,
            shots=100,
            sample_provenance=provenance,
            observable_range=(-1.0, 1.0),
        )
    for level, z_value, message in ((1.0, 1.96, "confidence_level"), (0.95, -1.0, "confidence_z")):
        with pytest.raises(ValueError, match=message):
            execute_quantum_gradient_estimator_with_uncertainty(
                plan,
                sampler,
                [0.2],
                calibration=_CALIBRATION,
                shots=100,
                sample_provenance=provenance,
                observable_range=(-1.0, 1.0),
                confidence_level=level,
                confidence_z=z_value,
            )


def test_multi_frequency_finite_shot_intervals_cover_at_the_nominal_rate() -> None:
    """The two-term rule of a shared parameter covers the exact derivative.

    Same frozen protocol as row 04: seeds 0..199, 1000 shots per evaluation,
    every replicate retained, nominal 0.95 inside the Clopper-Pearson interval
    of the observed coverage. With 200 replicates the accepted counts are
    184..196, so a true coverage of 0.93 is accepted with probability about
    0.76 and 0.90 with about 0.21: the study detects shortfalls of about 0.05
    or more, not small ones.

    The point puts the total angle at ``pi/4``, so every shifted expectation is
    ``+-1/sqrt(2)``. Where a shifted expectation sits near ``+-1`` the sampled
    variance collapses towards zero and the normal-approximation interval
    under-covers (112 of 200 at a total angle of 1.6); that limit is
    documented with the estimator, not hidden by this choice.
    """
    plan = _plan(
        [GateOccurrence("ry", 0), GateOccurrence("ry", 0), GateOccurrence("ry", 1)], 2, shots=1000
    )
    values = [0.3, math.pi / 4.0 - 0.6]
    exact = -2.0 * math.sin(2.0 * values[0] + values[1])
    covered = 0
    for seed in range(200):
        result = execute_quantum_gradient_estimator_with_uncertainty(
            plan,
            _shared_sampler(1000, seed),
            values,
            calibration=_CALIBRATION,
            shots=1000,
            sample_provenance={
                "sample_seed": seed,
                "shot_batch_id": f"multi-{seed}",
                "source_class": "local_simulator",
            },
            observable_range=(-1.0, 1.0),
            parameters=[Parameter("a"), Parameter("b", trainable=False)],
        )
        interval = result.confidence_interval
        assert interval is not None
        covered += bool(float(interval.lower[0]) <= exact <= float(interval.upper[0]))
    low = float(beta.ppf(0.025, covered, 200 - covered + 1))
    high = float(beta.ppf(0.975, covered + 1, 200 - covered))
    assert low <= 0.95 <= high, (covered, low, high)


def test_signed_rule_runs_through_the_sample_route() -> None:
    """A planned rule with a negative coefficient no longer crashes the sample route."""
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    assert min(coefficient for _, coefficient in rule.terms) < 0.0
    single = multi_frequency_parameter_shift_rule([2.0], shifts=[3.0 * math.pi / 4.0])
    ((_shift, coefficient),) = single.terms
    assert coefficient < 0.0
    estimate = parameter_shift_gradient_with_uncertainty(
        [[0.1]],
        [[0.3]],
        [[0.5]],
        [[0.5]],
        [[100]],
        sample_provenance={
            "sample_seed": 0,
            "shot_batch_id": "signed",
            "source_class": "synthetic_fixture",
        },
        rule=single,
    )
    assert _within_budget(estimate.gradient, [coefficient * (0.1 - 0.3)])


def test_pole_samples_give_conservative_intervals() -> None:
    """Where every shifted sample sits near +-1 the floored variance does not under-cover.

    At a total angle of 1.6 the two-term rule's shifted points all lie near the
    poles. The plug-in variance covered 112 of 200 replicates there; with the
    Agresti-Coull floor the observed coverage must not fall below the nominal
    level (it may exceed it, since the floor is conservative).
    """
    plan = _plan(
        [GateOccurrence("ry", 0), GateOccurrence("ry", 0), GateOccurrence("ry", 1)], 2, shots=1000
    )
    values = [0.7, 0.2]
    exact = -2.0 * math.sin(2.0 * values[0] + values[1])
    covered = 0
    for seed in range(200):
        result = execute_quantum_gradient_estimator_with_uncertainty(
            plan,
            _shared_sampler(1000, seed),
            values,
            calibration=_CALIBRATION,
            shots=1000,
            sample_provenance={
                "sample_seed": seed,
                "shot_batch_id": f"pole-{seed}",
                "source_class": "local_simulator",
            },
            observable_range=(-1.0, 1.0),
            parameters=[Parameter("a"), Parameter("b", trainable=False)],
        )
        interval = result.confidence_interval
        assert interval is not None
        covered += bool(float(interval.lower[0]) <= exact <= float(interval.upper[0]))
    high = float(beta.ppf(0.975, covered + 1, 200 - covered)) if covered < 200 else 1.0
    assert high >= 0.95, covered


@pytest.mark.parametrize(
    ("observable_range", "mean", "message"),
    [((1.0, -1.0), 0.0, "low < high"), ((-1.0, 1.0), 1.5, "outside observable_range")],
    ids=["reversed-range", "mean-outside"],
)
def test_observable_range_is_checked(
    observable_range: tuple[float, float], mean: float, message: str
) -> None:
    """A reversed range or a mean outside it refuses.

    Parameters
    ----------
    observable_range
        Declared single-shot range.
    mean
        Sampled mean returned by the sampler.
    message
        Expected refusal text.

    """
    plan = _plan([GateOccurrence("ry", 0)], 1, shots=100)
    with pytest.raises(ValueError, match=message):
        execute_quantum_gradient_estimator_with_uncertainty(
            plan,
            lambda values: (mean, 0.1),
            [0.2],
            calibration=_CALIBRATION,
            shots=100,
            sample_provenance={
                "sample_seed": 0,
                "shot_batch_id": "range",
                "source_class": "synthetic_fixture",
            },
            observable_range=observable_range,
        )


def _first_admissible_cost(frequencies: list[float]) -> float | None:
    """Return the shot cost of the earlier first-admissible grid, or ``None`` if it refused.

    This restates the earlier default independently: grids ``k pi / D`` at the
    original scale from ``D = 2n + 1``, the first with condition number at most
    ``1e10`` and smallest singular value at least ``1e-6``.
    """
    spectrum = np.asarray(frequencies, dtype=np.float64)
    count = spectrum.size
    start = 2 * count + 1
    stop = max(start + 64, math.ceil(4.0 * float(spectrum.max()) * count) + 65)
    steps = np.arange(1, count + 1, dtype=np.float64)
    for denominator in range(start, stop):
        system = 2.0 * np.sin(np.outer(spectrum, steps * math.pi / denominator))
        singular = np.linalg.svd(system, compute_uv=False)
        if singular[-1] >= 1e-6 and singular[0] / singular[-1] <= 1e10:
            coefficients = np.linalg.solve(system, spectrum)
            return 2.0 * float(np.sum(coefficients**2)) * count
    return None


@pytest.mark.parametrize(
    "frequencies",
    [
        [1.0, 3.0],
        [0.5, 1.5, 2.5],
        [0.75, 2.25, 3.0, 3.75],
        [2**0.5 - 1.0, 1.0, 2**0.5, 1.0 + 2**0.5],
        [1.0, 2.0, 3.0 * (1.0 + 1.1e-9)],
        [float(k) * (1.0 + 1e-7 * (k % 3)) for k in range(1, 17)],
    ],
    ids=["one-three", "odd-halves", "superset", "root-two", "past-tolerance", "near-integer-16"],
)
def test_non_equidistant_rules_never_cost_more_than_before(frequencies: list[float]) -> None:
    """The search includes the earlier grid, an equidistant superset and wider phases.

    Cost is ``2 * sum c**2`` times the number of terms, the shot cost for a
    fixed total number of shots. Spectra the earlier search refused must now
    be planned exactly.

    Parameters
    ----------
    frequencies
        Non-equidistant generator frequencies.

    """
    rule = multi_frequency_parameter_shift_rule(frequencies)
    cost = rule.shot_variance_factor * len(rule.terms)
    before = _first_admissible_cost(frequencies)
    assert before is None or cost <= before * (1.0 + 1e-12)
    assert _trig_polynomial_derivative_error(frequencies, 3) <= 1e-12


def test_one_third_scales_keep_exact_frequencies() -> None:
    """Three ``RY(theta/3)`` gates plan ``{1/3, 2/3, 1}`` without decimal rounding."""
    plan = _plan([GateOccurrence("ry", 0, 1.0 / 3.0)] * 3, 1)
    assert plan.frequencies[0][-1] == 1.0
    rule = plan.rules[0]
    assert rule is not None
    theta = 0.8
    estimate = sum(
        coefficient * (math.cos(theta + shift) - math.cos(theta - shift))
        for shift, coefficient in rule.terms
    )
    assert abs(estimate + math.sin(theta)) <= 1e-14
