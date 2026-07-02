# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- parameter-shift differentiable transform tests
"""Tests for parameter-shift and scalar-gradient differentiable diagnostics."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
from scpn_quantum_control import differentiable as differentiable_module
from scpn_quantum_control import differentiable_parameter_shift as parameter_shift_module
from scpn_quantum_control.differentiable import (
    FiniteShotSampleProvenance,
    GradientCheckResult,
    GradientResult,
    Parameter,
    ParameterShiftRule,
    ParameterShiftSampleRecord,
    ShotAllocationResult,
    StochasticGradientResult,
    allocate_parameter_shift_shots,
    batch_complex_step_gradient,
    batch_parameter_shift_gradient,
    batch_value_and_complex_step_grad,
    batch_value_and_finite_difference_grad,
    batch_value_and_parameter_shift_grad,
    check_parameter_shift_consistency,
    multi_frequency_parameter_shift_rule,
    parameter_shift_gradient,
    parameter_shift_gradient_with_uncertainty,
    value_and_parameter_shift_grad,
)

FloatArray = NDArray[np.float64]
SAMPLE_PROVENANCE = {
    "sample_seed": "parameter-shift-test-seed",
    "shot_batch_id": "parameter-shift-test-batch",
    "source_class": "caller_supplied",
}


def test_facade_and_package_root_reuse_extracted_parameter_shift_helpers() -> None:
    """Facade and package-root exports should reuse the extracted helpers."""

    moved_helpers = (
        "parameter_shift_gradient",
        "batch_parameter_shift_gradient",
        "batch_value_and_parameter_shift_grad",
        "value_and_parameter_shift_grad",
        "parameter_shift_gradient_with_uncertainty",
    )

    for helper_name in moved_helpers:
        extracted = getattr(parameter_shift_module, helper_name)
        assert getattr(differentiable_module, helper_name) is extracted
        assert getattr(scpn, helper_name) is extracted
    assert differentiable_module.FiniteShotSampleProvenance is FiniteShotSampleProvenance
    assert scpn.FiniteShotSampleProvenance is FiniteShotSampleProvenance


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across differentiable parameter-shift payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def _single_parameter_shift_record(
    *,
    trainable: bool = True,
    gradient_contribution: float = 0.5,
    variance_contribution: float = 0.001,
) -> ParameterShiftSampleRecord:
    """Build a one-parameter shifted-sample record with known shot variance."""

    return ParameterShiftSampleRecord(
        term_index=0,
        parameter_index=0,
        parameter_name="theta",
        trainable=trainable,
        shift=math.pi / 2.0,
        coefficient=0.5,
        plus_value=1.0,
        minus_value=0.0,
        plus_variance=0.2,
        minus_variance=0.2,
        plus_shots=100,
        minus_shots=100,
        sample_seed=SAMPLE_PROVENANCE["sample_seed"],
        shot_batch_id=SAMPLE_PROVENANCE["shot_batch_id"],
        source_class=SAMPLE_PROVENANCE["source_class"],
        gradient_contribution=gradient_contribution,
        variance_contribution=variance_contribution,
    )


def test_parameter_shift_matches_sine_derivative() -> None:
    """Single-parameter shift should recover the exact derivative of sin."""

    theta = 0.37
    rule = ParameterShiftRule()

    gradient = parameter_shift_gradient(
        lambda values: math.sin(values[0]),
        [theta],
        rule=rule,
    )

    _assert_allclose(gradient, [math.cos(theta)], atol=1.0e-12)


def test_value_and_parameter_shift_grad_returns_metadata() -> None:
    """The public helper should return value, gradient, and explicit provenance."""

    result = value_and_parameter_shift_grad(
        lambda values: math.sin(values[0]) + math.cos(values[1]),
        [0.1, -0.2],
        parameters=[Parameter("theta"), Parameter("phi", trainable=False)],
    )

    assert isinstance(result, GradientResult)
    assert result.value == pytest.approx(math.sin(0.1) + math.cos(-0.2))
    _assert_allclose(result.gradient, [math.cos(0.1), 0.0], atol=1.0e-12)
    assert result.method == "parameter_shift"
    assert result.trainable == (True, False)
    assert result.parameter_names == ("theta", "phi")
    assert result.evaluations == 3


def test_parameter_shift_rejects_non_scalar_objective() -> None:
    """Gradient objectives must be scalar-valued to avoid silent shape bugs."""

    def non_scalar_objective(_values: FloatArray) -> Any:
        return np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="scalar"):
        parameter_shift_gradient(non_scalar_objective, [0.1])


def test_parameter_shift_rejects_implicit_parameter_coercion() -> None:
    """Differentiable parameters must be explicit real numeric values."""

    with pytest.raises(ValueError, match="parameters must contain real numeric scalars"):
        parameter_shift_gradient(lambda values: math.sin(values[0]), cast(Any, ["0.1"]))

    with pytest.raises(ValueError, match="parameters must contain real numeric scalars"):
        parameter_shift_gradient(lambda values: math.sin(values[0]), cast(Any, [True]))


def test_parameter_shift_rejects_non_real_objective_scalar() -> None:
    """Objective return values must be explicit finite real scalars."""

    def string_objective(_values: FloatArray) -> Any:
        return "1.0"

    def complex_objective(_values: FloatArray) -> Any:
        return 1.0 + 0.0j

    with pytest.raises(ValueError, match="differentiable objective must return a scalar"):
        parameter_shift_gradient(string_objective, [0.1])

    with pytest.raises(ValueError, match="differentiable objective must return a scalar"):
        parameter_shift_gradient(complex_objective, [0.1])


def test_parameter_metadata_validation_and_custom_rule() -> None:
    """Parameter metadata and custom rules should fail closed."""

    with pytest.raises(ValueError, match="non-empty"):
        Parameter("")
    with pytest.raises(ValueError, match="boolean"):
        Parameter("theta", trainable=cast(Any, np.bool_(True)))
    with pytest.raises(ValueError, match="unique"):
        value_and_parameter_shift_grad(
            lambda values: math.sin(values[0]) + math.sin(values[1]),
            [0.1, 0.2],
            parameters=[Parameter("theta"), Parameter("theta")],
        )
    with pytest.raises(ValueError, match="positive"):
        ParameterShiftRule(shift=0.0)
    with pytest.raises(ValueError, match="coefficient must be a real numeric scalar"):
        ParameterShiftRule(coefficient=cast(Any, "0.5"))

    result = value_and_parameter_shift_grad(
        lambda values: math.sin(2.0 * values[0]),
        [0.3],
        rule=ParameterShiftRule(shift=math.pi / 4.0, coefficient=1.0),
    )

    _assert_allclose(result.gradient, [2.0 * math.cos(0.6)], atol=1.0e-12)


def test_parameter_shift_consistency_passes_for_shift_compatible_objective() -> None:
    """Gradient checks should pass for a standard sinusoidal generator rule."""

    result = check_parameter_shift_consistency(
        lambda values: math.sin(values[0]) + math.cos(values[1]),
        [0.3, -0.2],
        tolerance=1.0e-5,
    )

    assert isinstance(result, GradientCheckResult)
    assert result.passed
    assert result.candidate.method == "parameter_shift"
    assert result.reference.method == "finite_difference_central"
    assert result.max_abs_error <= 1.0e-5
    assert result.value_delta == pytest.approx(0.0)


def test_parameter_shift_consistency_detects_wrong_rule_coefficient() -> None:
    """Gradient checks should fail closed when a rule coefficient is invalid."""

    result = check_parameter_shift_consistency(
        lambda values: math.sin(values[0]),
        [0.3],
        rule=ParameterShiftRule(coefficient=0.25),
        tolerance=1.0e-5,
    )

    assert not result.passed
    assert result.max_abs_error > result.tolerance


def test_parameter_shift_consistency_rejects_invalid_tolerance() -> None:
    """Gradient-check tolerances must be explicit non-negative real scalars."""

    with pytest.raises(ValueError, match="gradient check tolerance must be a real numeric scalar"):
        check_parameter_shift_consistency(
            lambda values: math.sin(values[0]), [0.3], tolerance=cast(Any, "1e-5")
        )
    with pytest.raises(
        ValueError, match="gradient check tolerance must be finite and non-negative"
    ):
        check_parameter_shift_consistency(
            lambda values: math.sin(values[0]), [0.3], tolerance=-1.0
        )


def test_multi_frequency_parameter_shift_rule_matches_analytic_reference() -> None:
    """Multi-frequency rules should differentiate wider generator spectra exactly."""

    theta = 0.23
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0, 3.0])

    def objective(values: FloatArray) -> float:
        return float(
            math.sin(values[0])
            + 0.2 * math.cos(2.0 * values[0])
            + 0.05 * math.sin(3.0 * values[0])
        )

    result = value_and_parameter_shift_grad(objective, [theta], rule=rule)
    expected = math.cos(theta) - 0.4 * math.sin(2.0 * theta) + 0.15 * math.cos(3.0 * theta)

    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift is None
    assert result.coefficient is None
    assert result.evaluations == 1 + 2 * len(rule.terms)
    assert rule.frequencies == (1.0, 2.0, 3.0)
    _assert_allclose(result.gradient, [expected], atol=1.0e-12)


def test_multi_frequency_parameter_shift_preserves_trainable_metadata() -> None:
    """Multi-term rules should skip frozen parameters without spending probes."""

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    result = value_and_parameter_shift_grad(
        lambda values: math.sin(values[0]) + math.sin(2.0 * values[1]),
        [0.2, -0.3],
        parameters=[Parameter("theta"), Parameter("frozen_phi", trainable=False)],
        rule=rule,
    )

    _assert_allclose(result.gradient, [math.cos(0.2), 0.0], atol=1.0e-12)
    assert result.parameter_names == ("theta", "frozen_phi")
    assert result.trainable == (True, False)
    assert result.evaluations == 1 + 2 * len(rule.terms)


def test_multi_frequency_parameter_shift_rule_rejects_invalid_systems() -> None:
    """Multi-frequency rule construction must fail closed for ambiguous systems."""

    with pytest.raises(ValueError, match="positive"):
        multi_frequency_parameter_shift_rule([0.0, 1.0])
    with pytest.raises(ValueError, match="unique"):
        multi_frequency_parameter_shift_rule([1.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        multi_frequency_parameter_shift_rule([1.0, 2.0], shifts=[0.1])
    with pytest.raises(ValueError, match="ill-conditioned"):
        multi_frequency_parameter_shift_rule([1.0, 2.0], shifts=[math.pi, 2.0 * math.pi])


def test_multi_frequency_parameter_shift_propagates_per_term_shot_noise() -> None:
    """Multi-frequency shot noise should propagate independent per-term records."""

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    plus = np.array([[1.4, 10.0], [0.9, 20.0]], dtype=np.float64)
    minus = np.array([[0.2, 11.0], [0.1, 19.0]], dtype=np.float64)
    plus_variance = np.array([[0.20, 0.30], [0.40, 0.50]], dtype=np.float64)
    minus_variance = np.array([[0.10, 0.20], [0.30, 0.40]], dtype=np.float64)
    plus_shots = np.array([[100, 101], [200, 201]], dtype=np.float64)
    minus_shots = np.array([[120, 121], [220, 221]], dtype=np.float64)

    result = parameter_shift_gradient_with_uncertainty(
        plus,
        minus,
        plus_variance,
        minus_variance,
        plus_shots,
        minus_shots,
        sample_provenance=SAMPLE_PROVENANCE,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
        rule=rule,
    )

    expected_gradient = np.zeros(2, dtype=np.float64)
    expected_variance = np.zeros(2, dtype=np.float64)
    for term_index, (_shift, coefficient) in enumerate(rule.terms):
        expected_gradient[0] += coefficient * (plus[term_index, 0] - minus[term_index, 0])
        expected_variance[0] += coefficient**2 * (
            plus_variance[term_index, 0] / plus_shots[term_index, 0]
            + minus_variance[term_index, 0] / minus_shots[term_index, 0]
        )

    assert result.method == "multi_frequency_parameter_shift_shot_noise"
    assert result.shift is None
    assert result.coefficient is None
    assert result.evaluations == 2 * len(rule.terms)
    assert result.shots.shape == (len(rule.terms), 2, 2)
    assert len(result.records) == len(rule.terms) * 2
    assert all(isinstance(record, ParameterShiftSampleRecord) for record in result.records)
    assert all(record.trainable == (record.parameter_name == "theta") for record in result.records)
    assert all(
        record.gradient_contribution == 0.0
        for record in result.records
        if record.parameter_name == "frozen"
    )
    _assert_allclose(result.gradient, expected_gradient)
    _assert_allclose(result.standard_error, np.sqrt(expected_variance))
    _assert_allclose(np.diag(result.covariance), expected_variance)
    _assert_allclose(result.shots[:, 0, :], plus_shots)
    _assert_allclose(result.shots[:, 1, :], minus_shots)


def test_multi_frequency_parameter_shift_allocates_per_term_shots() -> None:
    """Shot allocation should support all multi-frequency plus/minus terms."""

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    allocation = allocate_parameter_shift_shots(
        [[0.20, 0.30], [0.40, 0.50]],
        [[0.10, 0.20], [0.30, 0.40]],
        target_standard_error=0.15,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
        rule=rule,
        min_shots=7,
    )

    assert allocation.method == "multi_frequency_parameter_shift_target_se"
    assert allocation.shots.shape == (len(rule.terms), 2, 2)
    assert allocation.total_shots == int(np.sum(allocation.shots))
    assert allocation.predicted_standard_error[0] <= 0.15
    assert allocation.predicted_standard_error[1] == 0.0
    assert np.all(allocation.shots[:, :, 1] == 7)


def test_multi_frequency_parameter_shift_rejects_ambiguous_shot_tensors() -> None:
    """Multi-frequency shot-noise inputs must expose the term dimension."""

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    with pytest.raises(ValueError, match="shape"):
        parameter_shift_gradient_with_uncertainty(
            [1.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.1, 0.1],
            [100, 100],
            rule=rule,
        )
    with pytest.raises(ValueError, match="first dimension"):
        allocate_parameter_shift_shots(
            [[0.1, 0.1, 0.1]],
            [[0.1, 0.1, 0.1]],
            target_standard_error=0.1,
            rule=rule,
        )


def test_parameter_shift_gradient_with_uncertainty_propagates_shot_noise() -> None:
    """Independent plus/minus shot noise should propagate into gradient variance."""

    result = parameter_shift_gradient_with_uncertainty(
        plus_values=[0.8, 0.1],
        minus_values=[0.2, -0.3],
        plus_variances=[0.36, 0.25],
        minus_variances=[0.16, 0.09],
        plus_shots=[900, 400],
        minus_shots=[400, 100],
        sample_provenance=SAMPLE_PROVENANCE,
        value=0.5,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )

    assert isinstance(result, StochasticGradientResult)
    assert result.method == "parameter_shift_shot_noise"
    assert result.claim_boundary == differentiable_module.STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY
    assert result.hardware_execution is False
    assert result.evaluations == 2
    assert result.parameter_names == ("theta", "frozen")
    assert result.trainable == (True, False)
    assert len(result.records) == 2
    active_record, frozen_record = result.records
    assert active_record.parameter_name == "theta"
    assert active_record.shift == pytest.approx(math.pi / 2.0)
    assert active_record.coefficient == pytest.approx(0.5)
    assert active_record.plus_shots == 900
    assert active_record.minus_shots == 400
    assert active_record.sample_seed == SAMPLE_PROVENANCE["sample_seed"]
    assert active_record.shot_batch_id == SAMPLE_PROVENANCE["shot_batch_id"]
    assert active_record.source_class == SAMPLE_PROVENANCE["source_class"]
    assert active_record.gradient_contribution == pytest.approx(0.3)
    assert active_record.variance_contribution > 0.0
    assert frozen_record.parameter_name == "frozen"
    assert frozen_record.trainable is False
    assert frozen_record.gradient_contribution == pytest.approx(0.0)
    assert frozen_record.variance_contribution == pytest.approx(0.0)
    _assert_allclose(result.gradient, [0.3, 0.0])
    expected_variance = 0.5**2 * (0.36 / 900.0 + 0.16 / 400.0)
    _assert_allclose(result.standard_error, [math.sqrt(expected_variance), 0.0])
    _assert_allclose(result.covariance, np.diag([expected_variance, 0.0]))
    _assert_allclose(result.confidence_radius, 1.959963984540054 * result.standard_error)
    _assert_allclose(result.shots, [[900.0, 400.0], [400.0, 100.0]])
    evidence = cast(dict[str, Any], result.to_dict())
    assert (
        evidence["claim_boundary"]
        == differentiable_module.STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY
    )
    assert evidence["hardware_execution"] is False
    evidence_records = cast(list[dict[str, object]], evidence["records"])
    assert len(evidence_records) == 2
    assert evidence_records[0]["sample_seed"] == SAMPLE_PROVENANCE["sample_seed"]
    assert evidence_records[0]["shot_batch_id"] == SAMPLE_PROVENANCE["shot_batch_id"]
    assert evidence_records[0]["source_class"] == SAMPLE_PROVENANCE["source_class"]


def test_parameter_shift_uncertainty_requires_sample_provenance() -> None:
    """Materialised finite-shot gradients must identify the sample source."""

    with pytest.raises(ValueError, match="sample provenance"):
        parameter_shift_gradient_with_uncertainty(
            plus_values=[0.8],
            minus_values=[0.2],
            plus_variances=[0.36],
            minus_variances=[0.16],
            plus_shots=[900],
            parameters=[Parameter("theta")],
        )


def test_parameter_shift_uncertainty_accepts_typed_sample_provenance() -> None:
    """Typed finite-shot provenance should normalise integer seed tokens."""

    result = parameter_shift_gradient_with_uncertainty(
        plus_values=[0.8],
        minus_values=[0.2],
        plus_variances=[0.36],
        minus_variances=[0.16],
        plus_shots=[900],
        sample_provenance=FiniteShotSampleProvenance(
            sample_seed=17,
            shot_batch_id="typed-provenance-batch",
            source_class="caller_supplied",
        ),
        parameters=[Parameter("theta")],
    )

    record = result.records[0]
    assert record.sample_seed == "17"
    assert record.shot_batch_id == "typed-provenance-batch"
    assert record.source_class == "caller_supplied"


def test_parameter_shift_sample_record_rejects_inconsistent_shift_math() -> None:
    """Shifted-sample records must match their finite-shot contribution math."""

    with pytest.raises(ValueError, match="gradient_contribution"):
        _single_parameter_shift_record(gradient_contribution=0.25)
    with pytest.raises(ValueError, match="variance_contribution"):
        _single_parameter_shift_record(variance_contribution=0.002)
    with pytest.raises(ValueError, match="non-trainable"):
        _single_parameter_shift_record(
            trainable=False,
            gradient_contribution=0.5,
            variance_contribution=0.001,
        )


def test_stochastic_gradient_records_must_reconstruct_result() -> None:
    """Parameter-shift evidence must be reconstructable from shifted records."""

    record = _single_parameter_shift_record()
    standard_error = np.array([math.sqrt(0.001)], dtype=np.float64)

    with pytest.raises(ValueError, match="reconstruct gradient"):
        StochasticGradientResult(
            value=0.0,
            gradient=np.array([0.25], dtype=np.float64),
            standard_error=standard_error,
            covariance=np.array([[0.001]], dtype=np.float64),
            confidence_radius=1.959963984540054 * standard_error,
            shots=np.array([[100.0], [100.0]], dtype=np.float64),
            confidence_level=0.95,
            method="parameter_shift_shot_noise",
            shift=math.pi / 2.0,
            coefficient=0.5,
            evaluations=2,
            parameter_names=("theta",),
            trainable=(True,),
            records=(record,),
        )
    with pytest.raises(ValueError, match="reconstruct covariance"):
        StochasticGradientResult(
            value=0.0,
            gradient=np.array([0.5], dtype=np.float64),
            standard_error=standard_error,
            covariance=np.array([[0.002]], dtype=np.float64),
            confidence_radius=1.959963984540054 * standard_error,
            shots=np.array([[100.0], [100.0]], dtype=np.float64),
            confidence_level=0.95,
            method="parameter_shift_shot_noise",
            shift=math.pi / 2.0,
            coefficient=0.5,
            evaluations=2,
            parameter_names=("theta",),
            trainable=(True,),
            records=(record,),
        )


def test_parameter_shift_uncertainty_reuses_plus_shots_when_minus_shots_omitted() -> None:
    """Shot-noise gradients should default minus shots to plus shots."""

    result = parameter_shift_gradient_with_uncertainty(
        plus_values=[0.8],
        minus_values=[0.2],
        plus_variances=[0.36],
        minus_variances=[0.16],
        plus_shots=[900],
        sample_provenance=SAMPLE_PROVENANCE,
        parameters=[Parameter("theta")],
    )

    _assert_allclose(result.gradient, [0.3])
    _assert_allclose(result.shots, [[900.0], [900.0]])
    assert result.records[0].plus_shots == 900
    assert result.records[0].minus_shots == 900


def test_parameter_shift_gradient_with_uncertainty_rejects_invalid_inputs() -> None:
    """Shot-noise gradients must fail closed on impossible measurement contracts."""

    with pytest.raises(ValueError, match="minus_values shape"):
        parameter_shift_gradient_with_uncertainty([1.0], [0.0, 0.0], [0.1], [0.1], [10])
    with pytest.raises(ValueError, match="variance shapes"):
        parameter_shift_gradient_with_uncertainty([1.0], [0.0], [0.1, 0.2], [0.1], [10])
    with pytest.raises(ValueError, match="shot-count shapes"):
        parameter_shift_gradient_with_uncertainty([1.0], [0.0], [0.1], [0.1], [10, 11])
    with pytest.raises(ValueError, match="shot variances"):
        parameter_shift_gradient_with_uncertainty([1.0], [0.0], [-0.1], [0.1], [10])
    with pytest.raises(ValueError, match="shot counts"):
        parameter_shift_gradient_with_uncertainty([1.0], [0.0], [0.1], [0.1], [0])
    with pytest.raises(ValueError, match="confidence_level"):
        parameter_shift_gradient_with_uncertainty(
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [10],
            confidence_level=1.0,
        )
    with pytest.raises(ValueError, match="confidence_z"):
        parameter_shift_gradient_with_uncertainty(
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [10],
            confidence_z=0.0,
        )
    with pytest.raises(ValueError, match="missing shot_batch_id"):
        parameter_shift_gradient_with_uncertainty(
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [10],
            sample_provenance={"sample_seed": "seed", "source_class": "caller_supplied"},
        )
    with pytest.raises(ValueError, match="source_class"):
        parameter_shift_gradient_with_uncertainty(
            [1.0],
            [0.0],
            [0.1],
            [0.1],
            [10],
            sample_provenance={
                "sample_seed": "seed",
                "shot_batch_id": "batch",
                "source_class": "unknown",
            },
        )
    with pytest.raises(ValueError, match="hardware execution"):
        StochasticGradientResult(
            value=0.0,
            gradient=np.array([1.0]),
            standard_error=np.array([0.1]),
            covariance=np.eye(1),
            confidence_radius=np.array([0.2]),
            shots=np.array([[10.0], [10.0]]),
            confidence_level=0.95,
            method="parameter_shift_shot_noise",
            shift=math.pi / 2.0,
            coefficient=0.5,
            evaluations=2,
            parameter_names=("theta",),
            trainable=(True,),
            hardware_execution=True,
        )


def test_allocate_parameter_shift_shots_meets_target_standard_error() -> None:
    """Shot allocation should conservatively meet target gradient uncertainty."""

    allocation = allocate_parameter_shift_shots(
        plus_variances=[0.36, 0.25],
        minus_variances=[0.16, 0.09],
        target_standard_error=0.02,
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )

    assert isinstance(allocation, ShotAllocationResult)
    assert allocation.method == "parameter_shift_target_se"
    assert allocation.parameter_names == ("theta", "frozen")
    assert allocation.trainable == (True, False)
    assert allocation.shots.shape == (2, 2)
    assert allocation.shots[0, 0] >= 1.0
    assert allocation.shots[1, 0] >= 1.0
    assert allocation.shots[0, 1] == 1.0
    assert allocation.shots[1, 1] == 1.0
    assert allocation.predicted_standard_error[0] <= 0.02
    assert allocation.predicted_standard_error[1] == pytest.approx(0.0)
    assert allocation.total_shots == int(np.sum(allocation.shots))
    _assert_allclose(allocation.covariance, np.diag(allocation.predicted_standard_error**2))


def test_allocate_parameter_shift_shots_respects_caps() -> None:
    """Shot allocation should report capped uncertainty when budgets are bounded."""

    allocation = allocate_parameter_shift_shots(
        plus_variances=[1.0],
        minus_variances=[1.0],
        target_standard_error=1.0e-3,
        min_shots=4,
        max_shots_per_evaluation=10,
    )

    _assert_allclose(allocation.shots, [[10.0], [10.0]])
    assert allocation.predicted_standard_error[0] > allocation.target_standard_error


def test_allocate_parameter_shift_shots_rejects_invalid_inputs() -> None:
    """Shot allocation must fail closed on impossible planning contracts."""

    with pytest.raises(ValueError, match="minus_variances shape"):
        allocate_parameter_shift_shots([0.1], [0.1, 0.2], target_standard_error=0.1)
    with pytest.raises(ValueError, match="shot variances"):
        allocate_parameter_shift_shots([-0.1], [0.1], target_standard_error=0.1)
    with pytest.raises(ValueError, match="target_standard_error"):
        allocate_parameter_shift_shots([0.1], [0.1], target_standard_error=0.0)
    with pytest.raises(ValueError, match="min_shots"):
        allocate_parameter_shift_shots([0.1], [0.1], target_standard_error=0.1, min_shots=0)
    with pytest.raises(ValueError, match="max_shots_per_evaluation"):
        allocate_parameter_shift_shots(
            [0.1],
            [0.1],
            target_standard_error=0.1,
            min_shots=10,
            max_shots_per_evaluation=5,
        )


def test_batch_parameter_shift_gradient_stacks_independent_objectives() -> None:
    """Batch helper should produce one gradient row per scalar objective."""

    gradients = batch_parameter_shift_gradient(
        [
            lambda values: math.sin(values[0]),
            lambda values: math.cos(values[0]),
        ],
        [0.25],
    )

    _assert_allclose(gradients[:, 0], [math.cos(0.25), -math.sin(0.25)])

    with pytest.raises(ValueError, match="objectives"):
        batch_parameter_shift_gradient([], [0.25])


def test_batch_value_gradient_results_preserve_metadata() -> None:
    """Batch value APIs should preserve objective values and provenance."""

    parameter_shift_results = batch_value_and_parameter_shift_grad(
        [
            lambda values: math.sin(values[0]),
            lambda values: math.cos(values[0]),
        ],
        [0.25],
        parameters=[Parameter("theta")],
    )
    finite_difference_results = batch_value_and_finite_difference_grad(
        [
            lambda values: values[0] ** 2,
            lambda values: 3.0 * values[0],
        ],
        [0.25],
        parameters=[Parameter("theta")],
    )
    complex_step_results = batch_value_and_complex_step_grad(
        [
            lambda values: np.sin(values[0]),
            lambda values: values[0] ** 3,
        ],
        [0.25],
        parameters=[Parameter("theta")],
    )

    assert len(parameter_shift_results) == 2
    assert len(complex_step_results) == 2
    assert parameter_shift_results[0].parameter_names == ("theta",)
    assert parameter_shift_results[0].method == "parameter_shift"
    assert complex_step_results[0].method == "complex_step"
    assert finite_difference_results[0].method == "finite_difference_central"
    _assert_allclose(
        [result.value for result in parameter_shift_results],
        [math.sin(0.25), math.cos(0.25)],
        atol=1.0e-12,
    )
    _assert_allclose(finite_difference_results[0].gradient, [0.5], atol=1.0e-6)
    _assert_allclose(finite_difference_results[1].gradient, [3.0], atol=1.0e-6)
    _assert_allclose(
        [result.gradient[0] for result in complex_step_results],
        [math.cos(0.25), 3.0 * 0.25**2],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_batch_value_gradient_results_reject_empty_objectives() -> None:
    """Batch value APIs must fail closed on empty objective lists."""

    with pytest.raises(ValueError, match="objectives"):
        batch_value_and_parameter_shift_grad([], [0.25])
    with pytest.raises(ValueError, match="objectives"):
        batch_value_and_finite_difference_grad([], [0.25])
    with pytest.raises(ValueError, match="objectives"):
        batch_value_and_complex_step_grad([], [0.25])


def test_batch_complex_step_gradient_matches_analytic_derivatives() -> None:
    """Batched complex-step gradients should stack analytic scalar results."""

    gradients = batch_complex_step_gradient(
        [
            lambda values: np.sin(values[0]) + values[1] ** 2,
            lambda values: values[0] * values[1],
        ],
        [0.25, -0.5],
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
    )

    assert gradients.shape == (2, 2)
    _assert_allclose(
        gradients,
        [[math.cos(0.25), 0.0], [-0.5, 0.0]],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    with pytest.raises(ValueError, match="objectives"):
        batch_complex_step_gradient([], [0.25])
