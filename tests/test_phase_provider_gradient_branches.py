# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — provider-gradient validation branch tests
"""Exercise provider-gradient validation and provenance branches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.provider_gradient as provider_gradient
from scpn_quantum_control.phase import (
    ProviderExpectationSample,
    execute_provider_parameter_shift_gradient,
    multi_frequency_parameter_shift_rule,
    prepare_provider_hardware_parameter_shift_gradient,
)

FloatArray = NDArray[np.float64]

_HARDWARE_EVIDENCE_IDS = {
    "backend_calibration_id": "cal-provider-branch",
    "no_qpu_gate_id": "no-qpu-provider-branch",
    "claim_boundary_id": "claim-boundary-provider-branch",
    "cost_budget_id": "budget-provider-branch",
}


def _statevector_result() -> provider_gradient.ProviderGradientExecutionResult:
    """Return one valid single-parameter statevector result."""

    def sampler(values: FloatArray, shots: int | None) -> ProviderExpectationSample:
        assert shots is None
        return ProviderExpectationSample(value=float(np.cos(values[0])))

    return execute_provider_parameter_shift_gradient(sampler, np.array([0.2]))


def _finite_metadata(**overrides: object) -> dict[str, object]:
    """Return valid finite-shot provenance with selected overrides."""
    metadata: dict[str, object] = {
        "sample_seed": "seed",
        "shot_batch_id": "batch",
        "source_class": "synthetic_fixture",
    }
    metadata.update(overrides)
    return metadata


def test_sample_normalizes_nested_metadata_and_fills_default_shots() -> None:
    """Normalize recursive JSON metadata and fill an omitted shot count."""
    sample = ProviderExpectationSample(
        value=1,
        variance=0,
        metadata={"nested": [1, {"finite": 2.0}], "enabled": True},
    )
    defaulted = sample.with_default_shots(64)

    assert defaulted.shots == 64
    assert defaulted.metadata == sample.metadata
    assert sample.with_default_shots(None) is sample


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"value": True}, "real numeric scalar"),
        ({"value": complex(1.0, 1.0)}, "real numeric scalar"),
        ({"value": float("inf")}, "must be finite"),
        ({"value": 0.0, "variance": -1.0}, "must be non-negative"),
        ({"value": 0.0, "shots": True}, "positive integer"),
        ({"value": 0.0, "shots": 0}, "positive integer"),
    ],
)
def test_sample_rejects_invalid_scalar_and_shot_fields(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    """Reject invalid numeric, variance, and shot representations."""
    with pytest.raises(ValueError, match=message):
        ProviderExpectationSample(**kwargs)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (cast(Mapping[str, object], {1: "bad"}), "metadata keys must be strings"),
        ({"bad": float("inf")}, "must be finite"),
        ({"bad": {1: "nested"}}, "keys must be strings"),
        ({"bad": object()}, "must be JSON-compatible"),
    ],
)
def test_sample_rejects_non_json_metadata(
    metadata: Mapping[str, object],
    message: str,
) -> None:
    """Reject metadata that cannot be represented as deterministic JSON."""
    with pytest.raises(ValueError, match=message):
        ProviderExpectationSample(value=0.0, metadata=metadata)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parameter_index", -1, "parameter_index"),
        ("parameter_index", True, "parameter_index"),
        ("shift_index", -1, "shift_index"),
        ("shift_index", True, "shift_index"),
        ("shift", 0.0, "must be positive"),
        ("coefficient", float("nan"), "must be finite"),
        ("gradient", float("nan"), "must be finite"),
        ("standard_error", -1.0, "must be non-negative"),
        ("confidence_radius", -1.0, "must be non-negative"),
    ],
)
def test_shift_record_rejects_invalid_scalar_fields(
    field: str,
    value: Any,
    message: str,
) -> None:
    """Reject malformed indices and scalar statistics in shift records."""
    record = _statevector_result().records[0]
    with pytest.raises(ValueError, match=message):
        replace(record, **{field: value})


def test_shift_record_rejects_invalid_parameter_vectors() -> None:
    """Reject malformed, non-finite, or shape-mismatched shifted vectors."""
    record = _statevector_result().records[0]
    with pytest.raises(ValueError, match="one-dimensional"):
        replace(record, plus_parameters=np.array([[0.1]]))
    with pytest.raises(ValueError, match="finite values"):
        replace(record, minus_parameters=np.array([np.inf]))
    with pytest.raises(ValueError, match="matching shapes"):
        replace(record, minus_parameters=np.array([0.1, 0.2]))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"gradient": np.array([0.1, 0.2])}, "shapes must match"),
        ({"total_evaluations": 0}, "total_evaluations must be positive"),
        ({"total_shots": 0}, "total_shots must be positive"),
        ({"claim_boundary": ""}, "claim_boundary must be non-empty"),
    ],
)
def test_execution_result_rejects_inconsistent_envelope(
    updates: dict[str, Any],
    message: str,
) -> None:
    """Reject inconsistent shapes, counts, and claim boundaries."""
    result = _statevector_result()
    with pytest.raises(ValueError, match=message):
        replace(result, **updates)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider", " ", "provider must be non-empty"),
        ("backend", " ", "backend must be non-empty"),
        ("method", " ", "method must be non-empty"),
        ("mode", " ", "mode must be non-empty"),
        ("rule_terms", 0, "rule_terms must be positive"),
        ("total_evaluations", 0, "total_evaluations must be positive"),
        ("estimated_total_shots", 0, "estimated_total_shots must be positive"),
        ("claim_boundary", " ", "claim_boundary must be non-empty"),
    ],
)
def test_hardware_preparation_result_rejects_invalid_envelope(
    field: str,
    value: Any,
    message: str,
) -> None:
    """Reject blank labels and non-positive hardware-planning counts."""
    result = prepare_provider_hardware_parameter_shift_gradient(
        np.array([0.2]),
        evidence_ids=_HARDWARE_EVIDENCE_IDS,
    )
    with pytest.raises(ValueError, match=message):
        replace(result, **{field: value})


def test_hardware_preparation_handles_multi_frequency_and_missing_plan() -> None:
    """Describe multi-term preparation and serialize an unavailable plan."""
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    result = prepare_provider_hardware_parameter_shift_gradient(
        np.array([0.2]),
        evidence_ids=_HARDWARE_EVIDENCE_IDS,
        rule=rule,
    )
    without_plan = replace(result, plan=None)

    assert result.method == "hardware_policy_dry_run_multi_frequency_parameter_shift"
    assert without_plan.to_dict()["plan"] is None
    assert (
        provider_gradient._provider_hardware_plan(
            backend="statevector",
            n_params=1,
            shift_terms=1,
            shots=0,
            confidence_level=0.95,
            allow_hardware=False,
        )
        is None
    )


def test_sampler_contract_and_parameter_vectors_fail_closed() -> None:
    """Reject wrong sampler returns and malformed parameter vectors."""

    def wrong_sampler(values: FloatArray, shots: int | None) -> ProviderExpectationSample:
        return cast(ProviderExpectationSample, 1.0)

    with pytest.raises(ValueError, match="must return ProviderExpectationSample"):
        execute_provider_parameter_shift_gradient(wrong_sampler, np.array([0.2]))
    with pytest.raises(ValueError, match="one-dimensional"):
        execute_provider_parameter_shift_gradient(wrong_sampler, np.array([[0.2]]))
    with pytest.raises(ValueError, match="finite values"):
        execute_provider_parameter_shift_gradient(wrong_sampler, np.array([np.inf]))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (_finite_metadata(source_class=1), "source_class must be one of"),
        (_finite_metadata(source_class="unknown"), "source_class must be one of"),
        (_finite_metadata(sample_seed=True), "non-empty string or integer token"),
        (_finite_metadata(sample_seed=[]), "non-empty string or integer token"),
        (_finite_metadata(sample_seed=" "), "must be non-empty"),
    ],
)
def test_finite_shot_provenance_rejects_invalid_tokens(
    metadata: Mapping[str, object],
    message: str,
) -> None:
    """Reject invalid source classes and sample provenance tokens."""

    def sampler(values: FloatArray, shots: int | None) -> ProviderExpectationSample:
        return ProviderExpectationSample(value=float(values[0]), variance=0.1, metadata=metadata)

    with pytest.raises(ValueError, match=message):
        execute_provider_parameter_shift_gradient(
            sampler,
            np.array([0.2]),
            backend="qasm_simulator",
            shots=32,
        )


def test_finite_shot_integer_tokens_receive_default_shots() -> None:
    """Accept integer provenance tokens and executor-supplied shot counts."""

    def sampler(values: FloatArray, shots: int | None) -> ProviderExpectationSample:
        return ProviderExpectationSample(
            value=float(values[0]),
            variance=0.1,
            metadata=_finite_metadata(sample_seed=1, shot_batch_id=2),
        )

    result = execute_provider_parameter_shift_gradient(
        sampler,
        np.array([0.2]),
        backend="qasm_simulator",
        shots=32,
    )
    assert result.total_shots == 64


def test_multi_frequency_statevector_and_invalid_shift_paths() -> None:
    """Label multi-term statevector results and reject singular shifts."""

    def sampler(values: FloatArray, shots: int | None) -> ProviderExpectationSample:
        return ProviderExpectationSample(value=float(np.sin(values[0])))

    result = execute_provider_parameter_shift_gradient(
        sampler,
        np.array([0.2]),
        rule=multi_frequency_parameter_shift_rule([1.0, 2.0]),
    )
    assert result.method == "multi_frequency_parameter_shift"
    assert provider_gradient._result_method("custom", 2) == "custom"
    with pytest.raises(ValueError, match="must be positive"):
        execute_provider_parameter_shift_gradient(sampler, np.array([0.2]), shift=0.0)
    with pytest.raises(ValueError, match="denominator singular"):
        execute_provider_parameter_shift_gradient(sampler, np.array([0.2]), shift=np.pi)


def test_standard_error_defensive_branches_fail_closed() -> None:
    """Exercise variance and shot guards beneath the public prechecks."""
    empty = ProviderExpectationSample(value=0.0)
    variance_only = ProviderExpectationSample(value=0.0, variance=0.1)

    with pytest.raises(ValueError, match="require sample variance"):
        provider_gradient._standard_error(
            empty,
            empty,
            coefficient=0.5,
            require_variance=True,
        )
    with pytest.raises(ValueError, match="require sample shots"):
        provider_gradient._standard_error(
            variance_only,
            variance_only,
            coefficient=0.5,
            require_variance=True,
        )
    assert (
        provider_gradient._standard_error(
            variance_only,
            variance_only,
            coefficient=0.5,
            require_variance=False,
        )
        == 0.0
    )
