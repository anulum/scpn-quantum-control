# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native semantic source binding tests
"""Bind real native owner records to their retained semantic source bytes."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    GradientResult,
    Parameter,
    StochasticGradientResult,
    value_and_grad,
)
from scpn_quantum_control.hardware.hal import (
    BackendProfile,
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumJobRef,
    QuantumJobResult,
    QuantumWorkload,
)
from scpn_quantum_control.native_semantic_binding import (
    capture_native_source,
    validate_native_source_record,
)
from scpn_quantum_control.program_ad_registry import PrimitiveContract, primitive_contract_for
from scpn_quantum_control.stable_core_product import digest_stable_core_payload


def _native_sources() -> tuple[
    PrimitiveContract,
    GradientResult,
    StochasticGradientResult,
    QuantumJobResult,
    BackendProfile,
    QuantumWorkload,
    QuantumJobRef,
    Parameter,
]:
    """Create one actual public producer object per native source family."""
    contract = primitive_contract_for("scpn.program_ad.elementwise:sin@1")
    assert contract is not None
    gradient = value_and_grad(lambda values: values[0] ** 2, [2.0], method="reverse_mode")
    assert isinstance(gradient, GradientResult)
    stochastic = StochasticGradientResult(
        value=1.0,
        gradient=np.array([0.5]),
        standard_error=np.array([0.1]),
        covariance=np.array([[0.01]]),
        confidence_radius=np.array([0.2]),
        shots=np.array([[32.0], [32.0]]),
        confidence_level=0.95,
        method="shot_noise_parameter_shift",
        shift=None,
        coefficient=None,
        evaluations=2,
        parameter_names=("x",),
        trainable=(True,),
    )
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = LocalDeterministicSimulator(hal.profile("local_statevector"))
    hal.register_backend(backend)
    job = hal.submit(
        backend.backend_id,
        QuantumWorkload("native-source", "mlir", "module {}", 2, shots=16),
    )
    return (
        contract,
        gradient,
        stochastic,
        hal.result(job),
        backend.profile,
        QuantumWorkload("native-source-request", "mlir", "module {}", 2, shots=16),
        job,
        Parameter("theta", trainable=False),
    )


@pytest.mark.parametrize("index", range(8))
def test_real_native_source_round_trip_preserves_owner_bytes(index: int) -> None:
    """A retained projection matches the real owner and keeps its native fields."""
    source = _native_sources()[index]
    retained = capture_native_source(source)
    expected = source.to_semantic_source()

    assert retained["record"] == expected
    assert retained["record_sha256"] == digest_stable_core_payload(expected)
    assert validate_native_source_record(source, retained).matched
    assert "unit" not in expected
    if isinstance(source, QuantumJobResult):
        assert "statevector_amplitudes" not in retained["record"]
        assert "hardware_execution" not in retained["record"]
    if isinstance(source, StochasticGradientResult):
        assert retained["record"]["standard_error"] == [0.1]
        assert retained["record"]["covariance"] == [[0.01]]
    if isinstance(source, QuantumWorkload):
        assert retained["record"]["requested_shots"] == 16
    if isinstance(source, BackendProfile):
        assert retained["record"]["capabilities"]["supports_statevector"] is True
    if isinstance(source, Parameter):
        assert retained["record"]["trainable"] is False


def test_rehashed_substitution_cannot_borrow_real_gradient_identity() -> None:
    """A changed result with a matching self-hash is still not the source object."""
    source = _native_sources()[1]
    retained = capture_native_source(source)
    record = retained["record"]
    assert isinstance(record, dict)
    record["gradient"] = [999.0]
    retained["record_sha256"] = digest_stable_core_payload(record)

    binding = validate_native_source_record(source, retained)

    assert not binding.matched
    assert "source_content_mismatch" in binding.reasons
    np.testing.assert_array_equal(source.gradient, [4.0])


@pytest.mark.parametrize("field", ["schema", "producer_identity", "record_sha256"])
def test_native_source_rejects_version_identity_and_digest_changes(field: str) -> None:
    """Version, producer and digest are each checked independently."""
    source = _native_sources()[0]
    retained: dict[str, Any] = capture_native_source(source)
    retained[field] = "substituted"

    binding = validate_native_source_record(source, retained)

    assert not binding.matched
    assert binding.reasons


def test_native_source_refuses_unknown_python_object() -> None:
    """An unrelated object cannot acquire a quantum producer schema."""
    with pytest.raises(ValueError, match="unsupported native semantic source"):
        capture_native_source(object())


def test_real_studio_plan_source_is_optional_and_bound_to_preview() -> None:
    """An actual no-submit Studio preview binds as a versioned source."""
    pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")
    from scpn_quantum_control.studio.executive import (
        ActionRegistry,
        ExecutiveRequest,
        preview_action,
    )
    from scpn_quantum_control.studio.executive_execute import ExecuteActionHandler

    registry = ActionRegistry()
    registry.register(ExecuteActionHandler())
    request = ExecutiveRequest(
        verb="execute",
        action_id="native-source-preview",
        parameters={
            "provider": "ibm-quantum",
            "endpoint": "ibm_brisbane",
            "circuit_digest": "sha256:abc123",
            "circuit_ref": "data/studio/xy_compile_recompute_unit_20260708.json",
            "shots": 4096,
        },
    )
    plan = preview_action(request, registry=registry)

    retained = capture_native_source(plan)

    assert retained["schema"] == "studio.execution_plan.v1"
    assert retained["record"]["plan"]["parameters"]["shots"] == 4096
    assert validate_native_source_record(plan, retained).matched


def test_native_source_rejects_studio_name_impersonation() -> None:
    """A same-named Python class cannot borrow the optional Studio schema."""
    pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")
    impostor_type = type(
        "ExecutionPlan", (), {"__module__": "scpn_quantum_control.studio.executive"}
    )

    with pytest.raises(ValueError, match="impersonating Studio plan"):
        capture_native_source(impostor_type())


def test_optional_studio_source_unavailable_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable optional Studio module cannot grant a schema by name."""
    impostor_type = type(
        "ExecutionPlan", (), {"__module__": "scpn_quantum_control.studio.executive"}
    )
    monkeypatch.setitem(sys.modules, "scpn_quantum_control.studio.executive", None)

    with pytest.raises(ValueError, match="optional Studio plan source is unavailable"):
        capture_native_source(impostor_type())


def test_native_source_projection_cannot_impersonate_another_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compromised producer projection does not define its own identity."""
    source = _native_sources()[1]
    monkeypatch.setattr(
        GradientResult,
        "to_semantic_source",
        lambda self: {"producer_identity": "another.module.Result"},
    )
    with pytest.raises(ValueError, match="changed producer identity"):
        capture_native_source(source)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("not_mapping", "source_record_malformed"),
        ("extra_field", "source_record_malformed"),
        ("missing_record", "source_content_mismatch"),
        ("non_json_number", "source_content_mismatch"),
    ],
)
def test_native_source_refuses_malformed_or_unserialisable_retained_evidence(
    mutation: str, reason: str
) -> None:
    """Malformed retained evidence never borrows a valid native owner digest."""
    source = _native_sources()[0]
    retained: object = capture_native_source(source)
    if mutation == "not_mapping":
        retained = ["not a record"]
    else:
        assert isinstance(retained, dict)
        if mutation == "extra_field":
            retained["invented"] = True
        elif mutation == "missing_record":
            retained["record"] = None
        else:
            retained["record"] = {"bad": np.nan}

    binding = validate_native_source_record(source, retained)

    assert not binding.matched
    assert reason in binding.reasons
