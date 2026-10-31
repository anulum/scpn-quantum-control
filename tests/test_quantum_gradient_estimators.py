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

import math
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
