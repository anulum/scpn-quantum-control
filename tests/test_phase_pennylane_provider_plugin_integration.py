# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PennyLane Provider Plugin Integration Tests
"""Integration tests for PennyLane provider evidence and maturity aggregation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from _phase_pennylane_bridge_test_helpers import (
    _closed_form_gradient,
    _FakePennyLane,
    _objective,
)
from numpy.typing import NDArray

import scpn_quantum_control.phase.pennylane_bridge as pennylane_bridge
from scpn_quantum_control.phase import (
    PauliTerm,
    PennyLaneGradientAgreementResult,
    PennyLaneHardwarePluginExecutionArtifact,
    PennyLaneMaturityAuditResult,
    PennyLanePluginMatrixResult,
    PennyLanePluginMatrixRoute,
    PennyLaneProviderGradientParityArtifact,
    PennyLaneProviderPluginExecutionArtifact,
    PennyLaneRoundTripResult,
    PhaseQNodeCircuit,
    run_pennylane_maturity_audit,
    run_pennylane_plugin_matrix,
)

FloatArray = NDArray[np.float64]


@pytest.mark.parametrize("shots", [cast(Any, 1.5), cast(Any, "4096")])
def test_pennylane_provider_plugin_execution_artifact_rejects_non_integer_shots(
    shots: Any,
) -> None:
    """Provider-plugin execution artifacts must not coerce shot counts."""

    with pytest.raises(ValueError, match="shots"):
        PennyLaneProviderPluginExecutionArtifact(
            artifact_id="pl-provider-sim-20260616",
            plugin_name="pennylane-provider-simulator",
            provider_name="example-provider",
            device_name="example.simulator",
            backend_name="example_sim_v1",
            circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
            execution_mode="provider_simulator",
            interface="autograd",
            diff_method="parameter-shift",
            shot_policy="finite_shot",
            shots=shots,
            result_digest="sha256:" + "a" * 64,
            metadata_digest="sha256:" + "b" * 64,
            hardware_execution=False,
            raw_result_replay_artifact_id="pl-provider-replay-20260616",
        )


def test_pennylane_maturity_audit_records_export_metadata_and_provider_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0), ("rx", (0,), 1)),
        PauliTerm(1.0, ((0, "z"),)),
    )
    values = np.array([0.37, -0.29], dtype=float)

    result = run_pennylane_maturity_audit(
        objective=_objective,
        pennylane_objective=_objective,
        pennylane_gradient=_closed_form_gradient,
        values=np.array([0.2, -0.4], dtype=float),
        circuit=circuit,
        phase_qnode_values=values,
        value_tolerance=1e-12,
        gradient_tolerance=1e-12,
    )

    assert isinstance(result, PennyLaneMaturityAuditResult)
    assert not result.identical_circuit_ready
    assert not result.ready_for_provider_exceedance
    gradient_agreement = cast(
        PennyLaneGradientAgreementResult,
        result.evidence["gradient_agreement"],
    )
    caller_qnode_round_trip = cast(
        PennyLaneRoundTripResult,
        result.evidence["caller_qnode_round_trip"],
    )
    phase_qnode_export_round_trip = cast(
        PennyLaneRoundTripResult,
        result.evidence["phase_qnode_export_round_trip"],
    )
    assert gradient_agreement.passed
    assert caller_qnode_round_trip.passed
    assert phase_qnode_export_round_trip.passed
    assert result.evidence["phase_qnode_import_round_trip"] is None
    plugin_matrix = result.evidence["pennylane_plugin_matrix"]
    assert isinstance(plugin_matrix, PennyLanePluginMatrixResult)
    assert plugin_matrix.local_plugin_parity_ready
    assert not plugin_matrix.provider_plugin_execution_ready
    assert not plugin_matrix.ready_for_provider_exceedance
    assert plugin_matrix.route_status("default_qubit_exact_state") == "passed"
    assert plugin_matrix.route_status("provider_plugin_execution") == "blocked"
    assert result.required_capabilities["phase_qnode_import_round_trip"] == "blocked"
    assert result.required_capabilities["pennylane_plugin_matrix"] == "passed"
    assert result.promotion_metadata["device_name"] == "default.qubit"
    assert result.promotion_metadata["shots"] is None
    assert result.promotion_metadata["diff_method"] == "parameter-shift"
    assert result.promotion_metadata["phase_qnode_parameter_shift_evaluations"] == 4
    evaluation_groups = cast(
        list[dict[str, object]],
        result.promotion_metadata["phase_qnode_evaluation_groups"],
    )
    assert len(evaluation_groups) == 2
    assert "provider_plugin_execution" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    required_capabilities = cast(dict[str, str], payload["required_capabilities"])
    assert payload["claim_boundary"] == "bounded_pennylane_provider_maturity_audit"
    assert required_capabilities["device_metadata"] == "passed"
    assert required_capabilities["pennylane_plugin_matrix"] == "passed"


def test_pennylane_plugin_matrix_fails_closed_for_provider_plugins() -> None:
    result = run_pennylane_plugin_matrix()

    assert isinstance(result, PennyLanePluginMatrixResult)
    assert result.local_plugin_parity_ready
    assert not result.provider_plugin_execution_ready
    assert not result.hardware_plugin_execution_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("default_qubit_exact_state") == "passed"
    assert result.route_status("phase_qnode_export_default_qubit") == "passed"
    assert result.route_status("provider_plugin_execution") == "blocked"
    assert result.route_status("hardware_plugin_execution") == "blocked"
    assert "provider_plugin_execution" in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    assert result.claim_boundary == "bounded_pennylane_plugin_matrix"


def test_pennylane_plugin_matrix_route_normalises_metadata() -> None:
    route = PennyLanePluginMatrixRoute(
        name="  provider_plugin_execution  ",
        status="blocked",
        reason="  artefact missing  ",
        requires=(" provider_execution_artifact ",),
    )

    assert route.name == "provider_plugin_execution"
    assert route.reason == "artefact missing"
    assert route.requires == ("provider_execution_artifact",)
    assert route.to_dict() == {
        "name": "provider_plugin_execution",
        "status": "blocked",
        "reason": "artefact missing",
        "requires": ["provider_execution_artifact"],
    }


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: PennyLanePluginMatrixRoute(name=" ", status="blocked", reason="missing"),
            "route name",
        ),
        (
            lambda: PennyLanePluginMatrixRoute(
                name="bad\nname",
                status="blocked",
                reason="missing",
            ),
            "route name",
        ),
        (
            lambda: PennyLanePluginMatrixRoute(
                name="provider_plugin_execution",
                status="unknown",
                reason="missing",
            ),
            "route status",
        ),
        (
            lambda: PennyLanePluginMatrixRoute(
                name="provider_plugin_execution",
                status="blocked",
                reason="",
            ),
            "route reason",
        ),
        (
            lambda: PennyLanePluginMatrixRoute(
                name="provider_plugin_execution",
                status="blocked",
                reason="missing",
                requires=(" ",),
            ),
            "route requirement",
        ),
    ],
)
def test_pennylane_plugin_matrix_route_rejects_malformed_metadata(
    factory: Callable[[], PennyLanePluginMatrixRoute],
    match: str,
) -> None:
    """Plugin matrix routes must not carry malformed evidence states."""

    with pytest.raises(ValueError, match=match):
        factory()


def _provider_plugin_execution_artifact(
    *,
    interface: str = "autograd",
    diff_method: str = "parameter-shift",
    shot_policy: str = "finite_shot",
    shots: int | None = 4096,
) -> PennyLaneProviderPluginExecutionArtifact:
    return PennyLaneProviderPluginExecutionArtifact(
        artifact_id="pl-provider-sim-20260616",
        plugin_name="pennylane-provider-simulator",
        provider_name="example-provider",
        device_name="example.simulator",
        backend_name="example_sim_v1",
        circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
        execution_mode="provider_simulator",
        interface=interface,
        diff_method=diff_method,
        shot_policy=shot_policy,
        shots=shots,
        result_digest="sha256:" + "a" * 64,
        metadata_digest="sha256:" + "b" * 64,
        hardware_execution=False,
        raw_result_replay_artifact_id="pl-provider-replay-20260616",
    )


def _provider_gradient_parity_artifact(
    *,
    circuit_fingerprint: str = "phase-qnode:ry-rx-pauli-z:v1",
    interface: str = "autograd",
    diff_method: str = "parameter-shift",
    shot_policy: str = "finite_shot",
    max_abs_error: float = 1e-9,
    tolerance: float = 1e-6,
    shots: int | None = 4096,
) -> PennyLaneProviderGradientParityArtifact:
    return PennyLaneProviderGradientParityArtifact(
        artifact_id="pl-provider-gradient-20260616",
        provider_execution_artifact_id="pl-provider-sim-20260616",
        plugin_name="pennylane-provider-simulator",
        provider_name="example-provider",
        device_name="example.simulator",
        backend_name="example_sim_v1",
        circuit_fingerprint=circuit_fingerprint,
        interface=interface,
        diff_method=diff_method,
        shot_policy=shot_policy,
        gradient_digest="sha256:" + "c" * 64,
        reference_gradient_digest="sha256:" + "d" * 64,
        max_abs_error=max_abs_error,
        l2_error=2e-9,
        tolerance=tolerance,
        shots=shots,
        replay_artifact_id="pl-provider-gradient-replay-20260616",
        hardware_execution=False,
    )


def _hardware_plugin_execution_artifact(
    *,
    execution_mode: str = "provider_live_qpu",
    shots: int = 4096,
    calibration_captured_at_utc: str = "2026-06-20T12:05:00Z",
    calibration_valid_until_utc: str = "2026-07-20T00:00:00Z",
) -> PennyLaneHardwarePluginExecutionArtifact:
    return PennyLaneHardwarePluginExecutionArtifact(
        artifact_id="pl-hardware-exec-20260616",
        plugin_name="pennylane-provider-hardware",
        provider_name="example-provider",
        device_name="example.qpu",
        backend_name="example_qpu_v1",
        circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
        execution_mode=execution_mode,
        shots=shots,
        live_execution_ticket="ticket-pl-hw-20260616",
        provider_allowlist_id="allowlist-pl-hw-20260616",
        shot_budget_id="shot-budget-pl-hw-20260616",
        hardware_evidence_id="hardware-evidence-pl-hw-20260616",
        result_digest="sha256:" + "e" * 64,
        raw_counts_digest="sha256:" + "f" * 64,
        calibration_snapshot_digest="sha256:" + "1" * 64,
        calibration_captured_at_utc=calibration_captured_at_utc,
        calibration_valid_until_utc=calibration_valid_until_utc,
        metadata_digest="sha256:" + "2" * 64,
        hardware_execution=True,
    )


def test_pennylane_plugin_matrix_accepts_provider_execution_artifact_without_promotion() -> None:
    artifact = _provider_plugin_execution_artifact()

    result = run_pennylane_plugin_matrix(provider_execution_artifact=artifact)

    assert result.provider_plugin_execution_ready
    assert result.route_status("provider_plugin_execution") == "passed"
    assert result.provider_execution_artifact is artifact
    assert not result.hardware_plugin_execution_ready
    assert not result.ready_for_provider_exceedance
    assert "provider_plugin_execution" not in result.open_gaps
    assert "provider_plugin_gradient_parity" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    provider_payload = cast(dict[str, object], payload["provider_execution_artifact"])
    assert provider_payload["artifact_id"] == artifact.artifact_id
    assert provider_payload["execution_mode"] == "provider_simulator"
    assert provider_payload["interface"] == "autograd"
    assert provider_payload["diff_method"] == "parameter-shift"
    assert provider_payload["shot_policy"] == "finite_shot"


def test_pennylane_plugin_matrix_accepts_provider_gradient_parity_without_promotion() -> None:
    execution_artifact = _provider_plugin_execution_artifact(
        interface="tf",
        diff_method="adjoint",
    )
    gradient_artifact = _provider_gradient_parity_artifact(
        interface="tf",
        diff_method="adjoint",
    )

    result = run_pennylane_plugin_matrix(
        provider_execution_artifact=execution_artifact,
        provider_gradient_parity_artifact=gradient_artifact,
    )

    assert result.provider_plugin_execution_ready
    assert result.provider_plugin_gradient_parity_ready
    assert result.route_status("provider_plugin_gradient_parity") == "passed"
    assert result.provider_gradient_parity_artifact is gradient_artifact
    assert not result.hardware_plugin_execution_ready
    assert not result.ready_for_provider_exceedance
    assert "provider_plugin_gradient_parity" not in result.open_gaps
    assert "hardware_plugin_execution" in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    gradient_payload = cast(dict[str, object], payload["provider_gradient_parity_artifact"])
    assert gradient_payload["artifact_id"] == gradient_artifact.artifact_id
    assert gradient_payload["interface"] == "tf"
    assert gradient_payload["diff_method"] == "adjoint"
    assert gradient_payload["shot_policy"] == "finite_shot"
    assert gradient_payload["max_abs_error"] == gradient_artifact.max_abs_error


def test_pennylane_plugin_matrix_accepts_hardware_execution_without_promotion() -> None:
    hardware_artifact = _hardware_plugin_execution_artifact()

    result = run_pennylane_plugin_matrix(hardware_execution_artifact=hardware_artifact)

    assert result.hardware_plugin_execution_ready
    assert result.route_status("hardware_plugin_execution") == "passed"
    assert result.hardware_execution_artifact is hardware_artifact
    assert not result.provider_plugin_execution_ready
    assert not result.provider_plugin_gradient_parity_ready
    assert not result.ready_for_provider_exceedance
    assert "hardware_plugin_execution" not in result.open_gaps
    assert "provider_plugin_execution" in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    hardware_payload = cast(dict[str, object], payload["hardware_execution_artifact"])
    assert hardware_payload["artifact_id"] == hardware_artifact.artifact_id
    assert hardware_payload["hardware_execution"] is True


def test_pennylane_plugin_matrix_rejects_unpaired_provider_gradient_parity() -> None:
    gradient_artifact = _provider_gradient_parity_artifact()

    with pytest.raises(ValueError, match="provider execution artefact"):
        run_pennylane_plugin_matrix(provider_gradient_parity_artifact=gradient_artifact)


def test_pennylane_plugin_matrix_rejects_mismatched_provider_gradient_parity() -> None:
    execution_artifact = _provider_plugin_execution_artifact()
    gradient_artifact = _provider_gradient_parity_artifact(
        circuit_fingerprint="phase-qnode:other:v1",
    )

    with pytest.raises(ValueError, match="circuit_fingerprint"):
        run_pennylane_plugin_matrix(
            provider_execution_artifact=execution_artifact,
            provider_gradient_parity_artifact=gradient_artifact,
        )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: _provider_gradient_parity_artifact(max_abs_error=2e-6, tolerance=1e-6),
            "max_abs_error",
        ),
        (
            lambda: _provider_gradient_parity_artifact(shots=cast(Any, True)),
            "shots",
        ),
        (
            lambda: PennyLaneProviderGradientParityArtifact(
                artifact_id="pl-provider-gradient-20260616",
                provider_execution_artifact_id="pl-provider-sim-20260616",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                interface="autograd",
                diff_method="parameter-shift",
                shot_policy="finite_shot",
                gradient_digest="sha256:" + "c" * 63,
                reference_gradient_digest="sha256:" + "d" * 64,
                max_abs_error=1e-9,
                l2_error=2e-9,
                tolerance=1e-6,
                shots=4096,
                replay_artifact_id="pl-provider-gradient-replay-20260616",
                hardware_execution=False,
            ),
            "gradient_digest",
        ),
        (
            lambda: PennyLaneProviderGradientParityArtifact(
                artifact_id="pl-provider-gradient-20260616",
                provider_execution_artifact_id="pl-provider-sim-20260616",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                interface="autograd",
                diff_method="parameter-shift",
                shot_policy="finite_shot",
                gradient_digest="sha256:" + "c" * 64,
                reference_gradient_digest="sha256:" + "d" * 64,
                max_abs_error=1e-9,
                l2_error=2e-9,
                tolerance=1e-6,
                shots=4096,
                replay_artifact_id="pl-provider-gradient-replay-20260616",
                hardware_execution=True,
            ),
            "hardware execution",
        ),
    ],
)
def test_pennylane_provider_gradient_parity_artifact_rejects_malformed_evidence(
    factory: Callable[[], PennyLaneProviderGradientParityArtifact],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: _hardware_plugin_execution_artifact(execution_mode="provider_simulator"),
            "live hardware execution",
        ),
        (
            lambda: _hardware_plugin_execution_artifact(shots=cast(Any, True)),
            "shots",
        ),
        (
            lambda: PennyLaneHardwarePluginExecutionArtifact(
                artifact_id="pl-hardware-exec-20260616",
                plugin_name="pennylane-provider-hardware",
                provider_name="example-provider",
                device_name="example.qpu",
                backend_name="example_qpu_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_live_qpu",
                shots=4096,
                live_execution_ticket=" ",
                provider_allowlist_id="allowlist-pl-hw-20260616",
                shot_budget_id="shot-budget-pl-hw-20260616",
                hardware_evidence_id="hardware-evidence-pl-hw-20260616",
                result_digest="sha256:" + "e" * 64,
                raw_counts_digest="sha256:" + "f" * 64,
                calibration_snapshot_digest="sha256:" + "1" * 64,
                calibration_captured_at_utc="2026-06-20T12:05:00Z",
                calibration_valid_until_utc="2026-07-20T00:00:00Z",
                metadata_digest="sha256:" + "2" * 64,
                hardware_execution=True,
            ),
            "live_execution_ticket",
        ),
        (
            lambda: PennyLaneHardwarePluginExecutionArtifact(
                artifact_id="pl-hardware-exec-20260616",
                plugin_name="pennylane-provider-hardware",
                provider_name="example-provider",
                device_name="example.qpu",
                backend_name="example_qpu_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_live_qpu",
                shots=4096,
                live_execution_ticket="ticket-pl-hw-20260616",
                provider_allowlist_id="allowlist-pl-hw-20260616",
                shot_budget_id="shot-budget-pl-hw-20260616",
                hardware_evidence_id="hardware-evidence-pl-hw-20260616",
                result_digest="sha256:" + "e" * 63,
                raw_counts_digest="sha256:" + "f" * 64,
                calibration_snapshot_digest="sha256:" + "1" * 64,
                calibration_captured_at_utc="2026-06-20T12:05:00Z",
                calibration_valid_until_utc="2026-07-20T00:00:00Z",
                metadata_digest="sha256:" + "2" * 64,
                hardware_execution=True,
            ),
            "result_digest",
        ),
        (
            lambda: PennyLaneHardwarePluginExecutionArtifact(
                artifact_id="pl-hardware-exec-20260616",
                plugin_name="pennylane-provider-hardware",
                provider_name="example-provider",
                device_name="example.qpu",
                backend_name="example_qpu_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_live_qpu",
                shots=4096,
                live_execution_ticket="ticket-pl-hw-20260616",
                provider_allowlist_id="allowlist-pl-hw-20260616",
                shot_budget_id="shot-budget-pl-hw-20260616",
                hardware_evidence_id="hardware-evidence-pl-hw-20260616",
                result_digest="sha256:" + "e" * 64,
                raw_counts_digest="sha256:" + "f" * 64,
                calibration_snapshot_digest="sha256:" + "1" * 64,
                calibration_captured_at_utc="2026-06-20T12:05:00Z",
                calibration_valid_until_utc="2026-07-20T00:00:00Z",
                metadata_digest="sha256:" + "2" * 64,
                hardware_execution=False,
            ),
            "hardware execution",
        ),
    ],
)
def test_pennylane_hardware_plugin_execution_artifact_rejects_malformed_evidence(
    factory: Callable[[], PennyLaneHardwarePluginExecutionArtifact],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize("execution_mode", ["simulator", "local_simulator", "offline_replay"])
def test_pennylane_provider_plugin_execution_artifact_requires_provider_mode(
    execution_mode: str,
) -> None:
    """Provider artefacts must identify provider-plugin execution explicitly."""

    with pytest.raises(ValueError, match="provider-plugin execution"):
        PennyLaneProviderPluginExecutionArtifact(
            artifact_id="pl-provider-sim-20260616",
            plugin_name="pennylane-provider-simulator",
            provider_name="example-provider",
            device_name="example.simulator",
            backend_name="example_sim_v1",
            circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
            execution_mode=execution_mode,
            interface="autograd",
            diff_method="parameter-shift",
            shot_policy="finite_shot",
            shots=4096,
            result_digest="sha256:" + "a" * 64,
            metadata_digest="sha256:" + "b" * 64,
            hardware_execution=False,
            raw_result_replay_artifact_id="pl-provider-replay-20260616",
        )


def test_pennylane_provider_plugin_execution_artifact_rejects_hardware_claim() -> None:
    with pytest.raises(ValueError, match="must not claim hardware execution"):
        PennyLaneProviderPluginExecutionArtifact(
            artifact_id="pl-provider-sim-20260616",
            plugin_name="pennylane-provider-simulator",
            provider_name="example-provider",
            device_name="example.simulator",
            backend_name="example_sim_v1",
            circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
            execution_mode="provider_simulator",
            interface="autograd",
            diff_method="parameter-shift",
            shot_policy="finite_shot",
            shots=4096,
            result_digest="sha256:" + "a" * 64,
            metadata_digest="sha256:" + "b" * 64,
            hardware_execution=True,
        )


@pytest.mark.parametrize("case", ["plugin_name", "execution_mode", "shots", "replay"])
def test_pennylane_provider_plugin_execution_artifact_rejects_malformed_metadata(
    case: str,
) -> None:
    if case == "plugin_name":
        with pytest.raises(ValueError, match="plugin_name"):
            PennyLaneProviderPluginExecutionArtifact(
                artifact_id="pl-provider-sim-20260616",
                plugin_name="provider\nplugin",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_simulator",
                interface="autograd",
                diff_method="parameter-shift",
                shot_policy="finite_shot",
                shots=4096,
                result_digest="sha256:" + "a" * 64,
                metadata_digest="sha256:" + "b" * 64,
                hardware_execution=False,
                raw_result_replay_artifact_id="pl-provider-replay-20260616",
            )
    elif case == "execution_mode":
        with pytest.raises(ValueError, match="execution_mode"):
            PennyLaneProviderPluginExecutionArtifact(
                artifact_id="pl-provider-sim-20260616",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="qpu",
                interface="autograd",
                diff_method="parameter-shift",
                shot_policy="finite_shot",
                shots=4096,
                result_digest="sha256:" + "a" * 64,
                metadata_digest="sha256:" + "b" * 64,
                hardware_execution=False,
                raw_result_replay_artifact_id="pl-provider-replay-20260616",
            )
    elif case == "shots":
        with pytest.raises(ValueError, match="shots"):
            PennyLaneProviderPluginExecutionArtifact(
                artifact_id="pl-provider-sim-20260616",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_simulator",
                interface="autograd",
                diff_method="parameter-shift",
                shot_policy="finite_shot",
                shots=True,
                result_digest="sha256:" + "a" * 64,
                metadata_digest="sha256:" + "b" * 64,
                hardware_execution=False,
                raw_result_replay_artifact_id="pl-provider-replay-20260616",
            )
    else:
        with pytest.raises(ValueError, match="raw_result_replay_artifact_id"):
            PennyLaneProviderPluginExecutionArtifact(
                artifact_id="pl-provider-sim-20260616",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_simulator",
                interface="autograd",
                diff_method="parameter-shift",
                shot_policy="finite_shot",
                shots=4096,
                result_digest="sha256:" + "a" * 64,
                metadata_digest="sha256:" + "b" * 64,
                hardware_execution=False,
                raw_result_replay_artifact_id="  ",
            )


def test_pennylane_maturity_audit_records_provider_execution_artifact_without_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0), ("rx", (0,), 1)),
        PauliTerm(1.0, ((0, "z"),)),
    )
    artifact = _provider_plugin_execution_artifact()

    result = run_pennylane_maturity_audit(
        objective=_objective,
        pennylane_objective=_objective,
        pennylane_gradient=_closed_form_gradient,
        values=np.array([0.4, -0.2], dtype=float),
        circuit=circuit,
        phase_qnode_values=np.array([0.37, -0.29], dtype=float),
        provider_execution_artifact=artifact,
    )

    plugin_matrix = cast(PennyLanePluginMatrixResult, result.evidence["pennylane_plugin_matrix"])
    assert result.required_capabilities["provider_plugin_execution"] == "passed"
    assert plugin_matrix.provider_execution_artifact is artifact
    assert not result.ready_for_provider_exceedance
    assert "provider_plugin_execution" not in result.open_gaps
    assert "provider_plugin_gradient_parity" in result.open_gaps
    assert "hardware_execution" in result.open_gaps


def test_pennylane_maturity_audit_records_provider_gradient_parity_without_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0), ("rx", (0,), 1)),
        PauliTerm(1.0, ((0, "z"),)),
    )
    execution_artifact = _provider_plugin_execution_artifact(
        interface="tf",
        diff_method="spsa",
    )
    gradient_artifact = _provider_gradient_parity_artifact(
        interface="tf",
        diff_method="spsa",
    )

    result = run_pennylane_maturity_audit(
        objective=_objective,
        pennylane_objective=_objective,
        pennylane_gradient=_closed_form_gradient,
        values=np.array([0.4, -0.2], dtype=float),
        circuit=circuit,
        phase_qnode_values=np.array([0.37, -0.29], dtype=float),
        provider_execution_artifact=execution_artifact,
        provider_gradient_parity_artifact=gradient_artifact,
    )

    plugin_matrix = cast(PennyLanePluginMatrixResult, result.evidence["pennylane_plugin_matrix"])
    assert result.required_capabilities["provider_plugin_execution"] == "passed"
    assert result.required_capabilities["provider_plugin_gradient_parity"] == "passed"
    provider_artifact = plugin_matrix.provider_execution_artifact
    parity_artifact = plugin_matrix.provider_gradient_parity_artifact
    assert provider_artifact is execution_artifact
    assert parity_artifact is gradient_artifact
    assert provider_artifact.interface == "tf"
    assert parity_artifact.interface == "tf"
    assert provider_artifact.diff_method == "spsa"
    assert parity_artifact.diff_method == "spsa"
    assert not result.ready_for_provider_exceedance
    assert "provider_plugin_gradient_parity" not in result.open_gaps
    assert "hardware_execution" in result.open_gaps


def test_pennylane_maturity_audit_records_hardware_execution_without_benchmark_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0), ("rx", (0,), 1)),
        PauliTerm(1.0, ((0, "z"),)),
    )
    hardware_artifact = _hardware_plugin_execution_artifact()

    result = run_pennylane_maturity_audit(
        objective=_objective,
        pennylane_objective=_objective,
        pennylane_gradient=_closed_form_gradient,
        values=np.array([0.4, -0.2], dtype=float),
        circuit=circuit,
        phase_qnode_values=np.array([0.37, -0.29], dtype=float),
        hardware_execution_artifact=hardware_artifact,
    )

    plugin_matrix = cast(PennyLanePluginMatrixResult, result.evidence["pennylane_plugin_matrix"])
    assert result.required_capabilities["hardware_execution"] == "passed"
    assert plugin_matrix.hardware_execution_artifact is hardware_artifact
    assert not result.ready_for_provider_exceedance
    assert "hardware_execution" not in result.open_gaps
    assert "promotion_grade_isolated_benchmarks" in result.open_gaps
