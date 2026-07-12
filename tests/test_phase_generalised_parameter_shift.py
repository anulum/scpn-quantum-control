# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase generalised parameter shift tests
"""Tests for finite-spectrum phase parameter-shift evidence."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import phase
from scpn_quantum_control.differentiable import (
    FiniteShotSampleProvenance,
    GradientFailurePolicy,
    Parameter,
)
from scpn_quantum_control.phase.generalised_parameter_shift import (
    GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY,
    GeneralisedParameterShiftPlan,
    GeneralisedParameterShiftResult,
    GeneralisedParameterShiftTerm,
    GeneralisedStochasticParameterShiftResult,
    estimate_generalised_parameter_shift_shot_noise,
    generalised_parameter_shift_gradient,
    plan_generalised_parameter_shift,
    value_and_generalised_parameter_shift_grad,
)


def _construct_term(**overrides: Any) -> GeneralisedParameterShiftTerm:
    return GeneralisedParameterShiftTerm(
        term_index=overrides.get("term_index", 0),
        parameter_index=overrides.get("parameter_index", 0),
        parameter_name=overrides.get("parameter_name", "theta"),
        frequency=overrides.get("frequency", 1.0),
        shift=overrides.get("shift", math.pi / 2.0),
        coefficient=overrides.get("coefficient", 0.5),
    )


def _construct_plan(
    term: GeneralisedParameterShiftTerm,
    **overrides: Any,
) -> GeneralisedParameterShiftPlan:
    return GeneralisedParameterShiftPlan(
        parameter_names=overrides.get("parameter_names", ("theta",)),
        trainable=overrides.get("trainable", (True,)),
        spectra=overrides.get("spectra", ((1.0,),)),
        terms=overrides.get("terms", (term,)),
        method=overrides.get("method", "generalised_parameter_shift"),
        claim_boundary=overrides.get(
            "claim_boundary",
            GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY,
        ),
    )


def _construct_stochastic_wrapper(
    plan: GeneralisedParameterShiftPlan,
    base_stochastic_gradient: Any,
    **overrides: Any,
) -> GeneralisedStochasticParameterShiftResult:
    return GeneralisedStochasticParameterShiftResult(
        plan=plan,
        stochastic_gradient=overrides.get("stochastic_gradient", base_stochastic_gradient),
        envelope=overrides.get(
            "envelope",
            "independent_shifted_mean_normal_approximation",
        ),
        claim_boundary=overrides.get(
            "claim_boundary",
            GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY,
        ),
    )


def _objective(values: NDArray[np.float64]) -> float:
    return float(
        math.sin(values[0])
        + 0.2 * math.cos(2.0 * values[0])
        + 0.05 * math.sin(3.0 * values[0])
        + 0.3 * math.sin(2.0 * values[1])
        + 0.4 * values[2]
    )


def test_generalised_plan_records_per_parameter_spectra_and_exports() -> None:
    values = np.array([0.23, -0.4, 0.7], dtype=np.float64)
    parameters = (
        Parameter("theta"),
        Parameter("phi"),
        Parameter("frozen_bias", trainable=False),
    )

    plan = plan_generalised_parameter_shift(
        values,
        [[1.0, 2.0, 3.0], [2.0], [1.0]],
        parameters=parameters,
    )
    payload = plan.to_dict()

    assert phase.plan_generalised_parameter_shift is plan_generalised_parameter_shift
    assert phase.GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY == (
        GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY
    )
    assert isinstance(plan, GeneralisedParameterShiftPlan)
    assert plan.parameter_count == 3
    assert plan.shifted_evaluations == 8
    assert plan.parameter_names == ("theta", "phi", "frozen_bias")
    assert plan.trainable == (True, True, False)
    assert plan.spectra == ((1.0, 2.0, 3.0), (2.0,), (1.0,))
    assert len(plan.terms_for_parameter(0)) == 3
    assert len(plan.terms_for_parameter(1)) == 1
    assert plan.terms_for_parameter(2) == ()
    assert payload["shifted_evaluations"] == 8
    assert "provider callback" in str(payload["claim_boundary"])


def test_generalised_parameter_shift_matches_exact_reference() -> None:
    values = np.array([0.23, -0.4, 0.7], dtype=np.float64)
    parameters = (
        Parameter("theta"),
        Parameter("phi"),
        Parameter("frozen_bias", trainable=False),
    )

    result = value_and_generalised_parameter_shift_grad(
        _objective,
        values,
        [[1.0, 2.0, 3.0], [2.0], [1.0]],
        parameters=parameters,
    )
    direct_gradient = generalised_parameter_shift_gradient(
        _objective,
        values,
        [[1.0, 2.0, 3.0], [2.0], [1.0]],
        parameters=parameters,
    )
    expected = np.array(
        [
            math.cos(values[0])
            - 0.4 * math.sin(2.0 * values[0])
            + 0.15 * math.cos(3.0 * values[0]),
            0.6 * math.cos(2.0 * values[1]),
            0.0,
        ],
        dtype=np.float64,
    )
    payload = result.to_dict()

    assert isinstance(result, GeneralisedParameterShiftResult)
    assert result.value == pytest.approx(_objective(values))
    assert result.evaluations == 1 + result.plan.shifted_evaluations
    assert payload["method"] == "generalised_parameter_shift"
    np.testing.assert_allclose(result.gradient, expected, atol=1.0e-12)
    np.testing.assert_allclose(direct_gradient, expected, atol=1.0e-12)


def test_generalised_term_shifted_parameters_and_custom_shift() -> None:
    values = np.array([0.3], dtype=np.float64)
    plan = plan_generalised_parameter_shift(
        values,
        [[1.0]],
        shifts=[[math.pi / 3.0]],
    )
    term = plan.terms[0]

    assert isinstance(term, GeneralisedParameterShiftTerm)
    assert term.to_dict()["frequency"] == 1.0
    np.testing.assert_allclose(term.shifted_parameters(values, sign=1), [0.3 + math.pi / 3.0])
    np.testing.assert_allclose(term.shifted_parameters(values, sign=-1), [0.3 - math.pi / 3.0])
    with pytest.raises(ValueError, match="sign"):
        term.shifted_parameters(values, sign=0)

    gradient = generalised_parameter_shift_gradient(
        lambda vector: float(math.sin(vector[0])),
        values,
        [[1.0]],
        shifts=[[math.pi / 3.0]],
    )

    np.testing.assert_allclose(gradient, [math.cos(0.3)], atol=1.0e-12)


def test_generalised_stochastic_shift_estimator_builds_shot_noise_envelope() -> None:
    plan = plan_generalised_parameter_shift(
        [0.2, -0.1],
        [[1.0, 2.0], [1.0]],
        parameters=[Parameter("theta"), Parameter("phi")],
    )
    plus = np.array([1.4, 0.9, 0.3], dtype=np.float64)
    minus = np.array([0.2, 0.1, -0.5], dtype=np.float64)
    plus_variance = np.array([0.20, 0.40, 0.25], dtype=np.float64)
    minus_variance = np.array([0.10, 0.30, 0.15], dtype=np.float64)
    plus_shots = np.array([100, 200, 160], dtype=np.int64)
    minus_shots = np.array([120, 220, 180], dtype=np.int64)
    provenance = FiniteShotSampleProvenance(
        sample_seed="seed-7",
        shot_batch_id="batch-3",
        source_class="local_simulator",
    )

    result = estimate_generalised_parameter_shift_shot_noise(
        plan,
        plus,
        minus,
        plus_variance,
        minus_variance,
        plus_shots,
        minus_shots,
        value=0.75,
        sample_provenance=provenance,
        confidence_z=2.0,
        failure_policy=GradientFailurePolicy(max_standard_error=0.2),
    )
    payload = result.to_dict()

    expected_gradient = np.zeros(2, dtype=np.float64)
    expected_variance = np.zeros(2, dtype=np.float64)
    for row, term in enumerate(plan.terms):
        expected_gradient[term.parameter_index] += term.coefficient * (plus[row] - minus[row])
        expected_variance[term.parameter_index] += term.coefficient**2 * (
            plus_variance[row] / float(plus_shots[row])
            + minus_variance[row] / float(minus_shots[row])
        )

    assert isinstance(result, GeneralisedStochasticParameterShiftResult)
    assert result.stochastic_gradient.method == "generalised_parameter_shift_shot_noise"
    assert result.stochastic_gradient.hardware_execution is False
    assert result.stochastic_gradient.failure_policy_status == "passed"
    assert result.stochastic_gradient.evaluations == plan.shifted_evaluations
    assert len(result.stochastic_gradient.records) == len(plan.terms)
    assert payload["envelope"] == "independent_shifted_mean_normal_approximation"
    np.testing.assert_allclose(result.gradient, expected_gradient)
    np.testing.assert_allclose(result.standard_error, np.sqrt(expected_variance))
    np.testing.assert_allclose(result.confidence_radius, 2.0 * np.sqrt(expected_variance))
    assert result.stochastic_gradient.records[0].sample_seed == "seed-7"


def test_generalised_stochastic_estimator_reports_failed_uncertainty_policy() -> None:
    plan = plan_generalised_parameter_shift([0.2], [[1.0]])

    result = estimate_generalised_parameter_shift_shot_noise(
        plan,
        [1.0],
        [0.0],
        [9.0],
        [9.0],
        [10],
        sample_provenance={
            "sample_seed": "seed",
            "shot_batch_id": "batch",
            "source_class": "synthetic_fixture",
        },
        failure_policy=GradientFailurePolicy(max_confidence_radius=0.01),
    )

    assert result.stochastic_gradient.failure_policy_status == "failed"
    assert result.stochastic_gradient.failure_reasons
    assert "confidence_radius exceeds" in result.stochastic_gradient.failure_reasons[0]


def test_generalised_parameter_shift_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="generator_frequencies length"):
        plan_generalised_parameter_shift([0.1, 0.2], [[1.0]])
    with pytest.raises(ValueError, match="positive"):
        plan_generalised_parameter_shift([0.1], [[0.0]])
    with pytest.raises(ValueError, match="unique"):
        plan_generalised_parameter_shift([0.1], [[1.0, 1.0]])
    with pytest.raises(ValueError, match="shifts length"):
        plan_generalised_parameter_shift([0.1], [[1.0]], shifts=[[0.2], [0.3]])
    with pytest.raises(ValueError, match="single-frequency shifts"):
        plan_generalised_parameter_shift([0.1], [[1.0]], shifts=[[0.2, 0.3]])
    with pytest.raises(ValueError, match="single-frequency shift"):
        plan_generalised_parameter_shift([0.1], [[1.0]], shifts=[[-0.2]])
    with pytest.raises(ValueError, match="singular"):
        plan_generalised_parameter_shift([0.1], [[1.0]], shifts=[[math.pi]])
    with pytest.raises(ValueError, match="at least one"):
        plan_generalised_parameter_shift([0.1], [[]])

    plan = plan_generalised_parameter_shift([0.1], [[1.0]])
    with pytest.raises(ValueError, match="term vector"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [[1.0]],
            [0.0],
            [0.1],
            [0.1],
            [100],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": "batch",
                "source_class": "synthetic_fixture",
            },
        )
    with pytest.raises(ValueError, match="length"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [1.0, 2.0],
            [0.0],
            [0.1],
            [0.1],
            [100],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": "batch",
                "source_class": "synthetic_fixture",
            },
        )
    with pytest.raises(ValueError, match="finite"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [float("nan")],
            [0.0],
            [0.1],
            [0.1],
            [100],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": "batch",
                "source_class": "synthetic_fixture",
            },
        )
    with pytest.raises(ValueError, match="non-negative"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [1.0],
            [0.0],
            [-0.1],
            [0.1],
            [100],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": "batch",
                "source_class": "synthetic_fixture",
            },
        )
    with pytest.raises(ValueError, match="integer"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [100.5],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": "batch",
                "source_class": "synthetic_fixture",
            },
        )
    with pytest.raises(ValueError, match="shot counts"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [0],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": "batch",
                "source_class": "synthetic_fixture",
            },
        )
    with pytest.raises(ValueError, match="sample_provenance"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [100],
        )
    with pytest.raises(ValueError, match="sample_seed"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [100],
            sample_provenance={
                "sample_seed": object(),
                "shot_batch_id": "batch",
                "source_class": "synthetic_fixture",
            },
        )
    with pytest.raises(ValueError, match="shot_batch_id"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [100],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": object(),
                "source_class": "synthetic_fixture",
            },
        )
    with pytest.raises(ValueError, match="confidence_level"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [100],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": "batch",
                "source_class": "synthetic_fixture",
            },
            confidence_level=1.0,
        )
    with pytest.raises(ValueError, match="confidence_z"):
        estimate_generalised_parameter_shift_shot_noise(
            plan,
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [100],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": "batch",
                "source_class": "synthetic_fixture",
            },
            confidence_z=0.0,
        )


def test_generalised_contract_records_reject_invalid_direct_construction() -> None:
    term = GeneralisedParameterShiftTerm(
        term_index=0,
        parameter_index=0,
        parameter_name="theta",
        frequency=1.0,
        shift=math.pi / 2.0,
        coefficient=0.5,
    )
    plan = GeneralisedParameterShiftPlan(
        parameter_names=("theta",),
        trainable=(True,),
        spectra=((1.0,),),
        terms=(term,),
    )

    invalid_terms: tuple[dict[str, Any], ...] = (
        {"term_index": True},
        {"parameter_index": True},
        {"term_index": -1},
        {"parameter_index": -1},
        {"parameter_name": ""},
        {"frequency": 0.0},
        {"shift": 0.0},
    )
    for override in invalid_terms:
        with pytest.raises(ValueError):
            _construct_term(**override)

    with pytest.raises(ValueError, match="out of range"):
        term.shifted_parameters([], sign=1)

    invalid_plans: tuple[dict[str, Any], ...] = (
        {"parameter_names": (), "trainable": (), "spectra": (), "terms": ()},
        {"trainable": (True, False)},
        {"spectra": ((1.0,), (2.0,))},
        {"parameter_names": ("",)},
        {"trainable": (object(),)},
        {"terms": (replace(term, term_index=1),)},
        {"terms": (replace(term, parameter_index=1),)},
        {"terms": (replace(term, parameter_name="phi"),)},
        {"trainable": (False,), "terms": (term,)},
        {"terms": (replace(term, frequency=2.0),)},
        {"spectra": ((),)},
        {"claim_boundary": ""},
        {"method": ""},
    )
    for override in invalid_plans:
        with pytest.raises(ValueError):
            _construct_plan(term, **override)

    with pytest.raises(ValueError, match="parameter_index"):
        plan.terms_for_parameter(-1)
    with pytest.raises(ValueError, match="gradient length"):
        GeneralisedParameterShiftResult(
            value=0.0,
            gradient=np.array([0.0, 1.0], dtype=np.float64),
            plan=plan,
            evaluations=3,
        )
    with pytest.raises(ValueError, match="evaluations"):
        GeneralisedParameterShiftResult(
            value=0.0,
            gradient=np.array([0.0], dtype=np.float64),
            plan=plan,
            evaluations=2,
        )
    with pytest.raises(ValueError, match="method"):
        GeneralisedParameterShiftResult(
            value=0.0,
            gradient=np.array([0.0], dtype=np.float64),
            plan=plan,
            evaluations=3,
            method="different",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        GeneralisedParameterShiftResult(
            value=0.0,
            gradient=np.array([0.0], dtype=np.float64),
            plan=plan,
            evaluations=3,
            claim_boundary="",
        )

    valid_stochastic = estimate_generalised_parameter_shift_shot_noise(
        plan,
        [1.0],
        [0.0],
        [0.1],
        [0.1],
        [100],
        sample_provenance={
            "sample_seed": "seed",
            "shot_batch_id": "batch",
            "source_class": "synthetic_fixture",
        },
    ).stochastic_gradient
    phi_term = replace(term, parameter_name="phi")
    phi_plan = GeneralisedParameterShiftPlan(
        parameter_names=("phi",),
        trainable=(True,),
        spectra=((1.0,),),
        terms=(phi_term,),
    )
    frozen_plan = GeneralisedParameterShiftPlan(
        parameter_names=("theta",),
        trainable=(False,),
        spectra=((1.0,),),
        terms=(),
    )

    with pytest.raises(ValueError, match="parameter_names"):
        _construct_stochastic_wrapper(phi_plan, valid_stochastic)
    with pytest.raises(ValueError, match="trainable"):
        _construct_stochastic_wrapper(frozen_plan, valid_stochastic)

    invalid_stochastic_wrappers: tuple[dict[str, Any], ...] = (
        {"stochastic_gradient": replace(valid_stochastic, evaluations=4)},
        {"envelope": ""},
        {"claim_boundary": ""},
    )
    for override in invalid_stochastic_wrappers:
        with pytest.raises(ValueError):
            _construct_stochastic_wrapper(plan, valid_stochastic, **override)
