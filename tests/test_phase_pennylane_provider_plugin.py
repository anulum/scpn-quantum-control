# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for PennyLane provider-plugin evidence
"""Tests for scpn_quantum_control.phase.pennylane_provider_plugin."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from scpn_quantum_control.phase import (
    PennyLaneHardwarePluginExecutionArtifact,
    PennyLanePluginMatrixResult,
    PennyLaneProviderEvidenceBundle,
    PennyLaneProviderGradientParityArtifact,
    PennyLaneProviderPluginExecutionArtifact,
)
from scpn_quantum_control.phase import (
    run_pennylane_plugin_matrix as facade_run_pennylane_plugin_matrix,
)
from scpn_quantum_control.phase.pennylane_bridge import (
    run_pennylane_plugin_matrix as bridge_run_pennylane_plugin_matrix,
)
from scpn_quantum_control.phase.pennylane_provider_plugin import (
    PennyLanePluginMatrixRoute,
    run_pennylane_plugin_matrix,
)


def _provider_plugin_execution_artifact() -> PennyLaneProviderPluginExecutionArtifact:
    return PennyLaneProviderPluginExecutionArtifact(
        artifact_id="pl-provider-sim-20260620",
        plugin_name="pennylane-provider-simulator",
        provider_name="example-provider",
        device_name="example.simulator",
        backend_name="example_sim_v1",
        circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
        execution_mode="provider_simulator",
        shots=4096,
        result_digest="sha256:" + "a" * 64,
        metadata_digest="sha256:" + "b" * 64,
        raw_result_replay_artifact_id="pl-provider-replay-20260620",
    )


def _provider_gradient_parity_artifact(
    *,
    circuit_fingerprint: str = "phase-qnode:ry-rx-pauli-z:v1",
    max_abs_error: float = 1e-9,
    tolerance: float = 1e-6,
) -> PennyLaneProviderGradientParityArtifact:
    return PennyLaneProviderGradientParityArtifact(
        artifact_id="pl-provider-gradient-20260620",
        provider_execution_artifact_id="pl-provider-sim-20260620",
        plugin_name="pennylane-provider-simulator",
        provider_name="example-provider",
        device_name="example.simulator",
        backend_name="example_sim_v1",
        circuit_fingerprint=circuit_fingerprint,
        gradient_digest="sha256:" + "c" * 64,
        reference_gradient_digest="sha256:" + "d" * 64,
        max_abs_error=max_abs_error,
        l2_error=2e-9,
        tolerance=tolerance,
        shots=4096,
        replay_artifact_id="pl-provider-gradient-replay-20260620",
    )


def _hardware_plugin_execution_artifact() -> PennyLaneHardwarePluginExecutionArtifact:
    return PennyLaneHardwarePluginExecutionArtifact(
        artifact_id="pl-hardware-exec-20260620",
        plugin_name="pennylane-provider-hardware",
        provider_name="example-provider",
        device_name="example.qpu",
        backend_name="example_qpu_v1",
        circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
        execution_mode="provider_live_qpu",
        shots=4096,
        live_execution_ticket="ticket-pl-hw-20260620",
        provider_allowlist_id="allowlist-pl-hw-20260620",
        shot_budget_id="shot-budget-pl-hw-20260620",
        hardware_evidence_id="hardware-evidence-pl-hw-20260620",
        result_digest="sha256:" + "e" * 64,
        raw_counts_digest="sha256:" + "f" * 64,
        calibration_snapshot_digest="sha256:" + "1" * 64,
        metadata_digest="sha256:" + "2" * 64,
    )


def _provider_evidence_bundle(
    *,
    valid_until_utc: str = "2026-07-20T00:00:00Z",
    hardware_execution_artifact: PennyLaneHardwarePluginExecutionArtifact | None = None,
) -> PennyLaneProviderEvidenceBundle:
    return PennyLaneProviderEvidenceBundle(
        artifact_id="pl-provider-evidence-bundle-20260620",
        provider_execution_artifact=_provider_plugin_execution_artifact(),
        captured_at_utc="2026-06-20T12:00:00Z",
        valid_until_utc=valid_until_utc,
        provider_gradient_parity_artifact=_provider_gradient_parity_artifact(),
        hardware_execution_artifact=hardware_execution_artifact,
    )


def test_pennylane_provider_plugin_matrix_fails_closed_without_artifacts() -> None:
    """Provider-plugin routes remain blocked without explicit evidence artifacts."""
    result = run_pennylane_plugin_matrix()

    assert isinstance(result, PennyLanePluginMatrixResult)
    assert result.local_plugin_parity_ready
    assert not result.provider_plugin_execution_ready
    assert not result.provider_plugin_gradient_parity_ready
    assert not result.hardware_plugin_execution_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("provider_plugin_execution") == "blocked"
    assert result.route_status("isolated_benchmark_artifact") == "blocked"
    assert result.claim_boundary == "bounded_pennylane_plugin_matrix"


def test_pennylane_provider_plugin_matrix_accepts_paired_provider_artifacts() -> None:
    """Same-circuit provider execution and gradient parity artifacts pass their routes."""
    execution_artifact = _provider_plugin_execution_artifact()
    gradient_artifact = _provider_gradient_parity_artifact()

    result = run_pennylane_plugin_matrix(
        provider_execution_artifact=execution_artifact,
        provider_gradient_parity_artifact=gradient_artifact,
    )

    assert result.provider_execution_artifact is execution_artifact
    assert result.provider_gradient_parity_artifact is gradient_artifact
    assert result.provider_plugin_execution_ready
    assert result.provider_plugin_gradient_parity_ready
    assert not result.hardware_plugin_execution_ready
    assert not result.ready_for_provider_exceedance
    assert "provider_plugin_gradient_parity" not in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    payload = result.to_dict()
    provider_payload = cast(dict[str, object], payload["provider_execution_artifact"])
    gradient_payload = cast(dict[str, object], payload["provider_gradient_parity_artifact"])
    assert provider_payload["hardware_execution"] is False
    assert gradient_payload["max_abs_error"] == pytest.approx(1e-9)


def test_pennylane_provider_plugin_matrix_accepts_provider_evidence_bundle() -> None:
    """Fresh bundled provider evidence passes provider execution and parity routes."""
    bundle = _provider_evidence_bundle()

    result = run_pennylane_plugin_matrix(provider_evidence_bundle=bundle)

    assert result.provider_evidence_bundle is bundle
    assert result.provider_execution_artifact is bundle.provider_execution_artifact
    assert result.provider_gradient_parity_artifact is bundle.provider_gradient_parity_artifact
    assert result.provider_plugin_execution_ready
    assert result.provider_plugin_gradient_parity_ready
    assert not result.ready_for_provider_exceedance
    payload = result.to_dict()
    bundle_payload = cast(dict[str, object], payload["provider_evidence_bundle"])
    assert bundle_payload["valid_until_utc"] == "2026-07-20T00:00:00Z"


def test_pennylane_provider_plugin_matrix_rejects_stale_provider_evidence_bundle() -> None:
    """Expired provider evidence bundles fail closed before route promotion."""
    bundle = _provider_evidence_bundle(valid_until_utc="2026-06-21T00:00:00Z")

    with pytest.raises(ValueError, match="provider_evidence_bundle.valid_until_utc"):
        run_pennylane_plugin_matrix(provider_evidence_bundle=bundle)


def test_pennylane_provider_evidence_bundle_rejects_mismatched_hardware_chain() -> None:
    """Bundled hardware evidence must cite the same provider chain and circuit."""
    hardware_artifact = PennyLaneHardwarePluginExecutionArtifact(
        artifact_id="pl-hardware-exec-20260620",
        plugin_name="pennylane-provider-hardware",
        provider_name="other-provider",
        device_name="example.qpu",
        backend_name="example_qpu_v1",
        circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
        execution_mode="provider_live_qpu",
        shots=4096,
        live_execution_ticket="ticket-pl-hw-20260620",
        provider_allowlist_id="allowlist-pl-hw-20260620",
        shot_budget_id="shot-budget-pl-hw-20260620",
        hardware_evidence_id="hardware-evidence-pl-hw-20260620",
        result_digest="sha256:" + "e" * 64,
        raw_counts_digest="sha256:" + "f" * 64,
        calibration_snapshot_digest="sha256:" + "1" * 64,
        metadata_digest="sha256:" + "2" * 64,
    )

    with pytest.raises(ValueError, match="hardware_execution_artifact.provider_name"):
        _provider_evidence_bundle(hardware_execution_artifact=hardware_artifact)


def test_pennylane_provider_plugin_matrix_rejects_bundle_mixed_with_artifacts() -> None:
    """Provider evidence bundles cannot be mixed with individual attachments."""
    with pytest.raises(ValueError, match="provider_evidence_bundle"):
        run_pennylane_plugin_matrix(
            provider_evidence_bundle=_provider_evidence_bundle(),
            provider_execution_artifact=_provider_plugin_execution_artifact(),
        )


def test_pennylane_provider_plugin_matrix_accepts_ticketed_hardware_artifact() -> None:
    """Hardware plugin execution can be recorded without benchmark promotion."""
    hardware_artifact = _hardware_plugin_execution_artifact()

    result = run_pennylane_plugin_matrix(hardware_execution_artifact=hardware_artifact)

    assert result.hardware_execution_artifact is hardware_artifact
    assert result.hardware_plugin_execution_ready
    assert not result.provider_plugin_execution_ready
    assert not result.ready_for_provider_exceedance
    assert "hardware_plugin_execution" not in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    assert hardware_artifact.to_dict()["hardware_execution"] is True


def test_pennylane_provider_plugin_matrix_rejects_unknown_route() -> None:
    """Unknown plugin matrix routes fail closed instead of returning defaults."""
    with pytest.raises(KeyError, match="unknown PennyLane plugin route"):
        run_pennylane_plugin_matrix().route_status("missing_route")


def test_pennylane_provider_plugin_matrix_rejects_unpaired_or_mismatched_parity() -> None:
    """Gradient parity evidence must match the provider execution artifact exactly."""
    with pytest.raises(ValueError, match="provider execution artefact"):
        run_pennylane_plugin_matrix(
            provider_gradient_parity_artifact=_provider_gradient_parity_artifact()
        )

    with pytest.raises(ValueError, match="circuit_fingerprint"):
        run_pennylane_plugin_matrix(
            provider_execution_artifact=_provider_plugin_execution_artifact(),
            provider_gradient_parity_artifact=_provider_gradient_parity_artifact(
                circuit_fingerprint="phase-qnode:other:v1"
            ),
        )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: PennyLanePluginMatrixRoute(name="bad\nname", status="blocked", reason="x"),
            "route name",
        ),
        (
            lambda: PennyLanePluginMatrixRoute(name="route", status="unknown", reason="x"),
            "route status",
        ),
        (
            lambda: PennyLaneProviderPluginExecutionArtifact(
                artifact_id="pl-provider-sim-20260620",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_simulator",
                shots=True,
                result_digest="sha256:" + "a" * 64,
                metadata_digest="sha256:" + "b" * 64,
            ),
            "shots",
        ),
        (
            lambda: PennyLaneProviderPluginExecutionArtifact(
                artifact_id="pl-provider-sim-20260620",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_simulator",
                shots=4096,
                result_digest="sha256:" + "a" * 64,
                metadata_digest="sha256:" + "b" * 64,
                hardware_execution=True,
            ),
            "must not claim hardware execution",
        ),
        (
            lambda: PennyLaneProviderPluginExecutionArtifact(
                artifact_id="pl-provider-sim-20260620",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="simulator",
                shots=4096,
                result_digest="sha256:" + "a" * 64,
                metadata_digest="sha256:" + "b" * 64,
            ),
            "provider-plugin execution",
        ),
        (
            lambda: PennyLaneProviderPluginExecutionArtifact(
                artifact_id="pl-provider-sim-20260620",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_qpu",
                shots=4096,
                result_digest="sha256:" + "a" * 64,
                metadata_digest="sha256:" + "b" * 64,
            ),
            "hardware execution",
        ),
        (
            lambda: _provider_gradient_parity_artifact(max_abs_error=2e-6, tolerance=1e-6),
            "max_abs_error",
        ),
        (
            lambda: _provider_gradient_parity_artifact(max_abs_error=-1.0, tolerance=1e-6),
            "max_abs_error",
        ),
        (
            lambda: PennyLaneProviderGradientParityArtifact(
                artifact_id="pl-provider-gradient-20260620",
                provider_execution_artifact_id="pl-provider-sim-20260620",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                gradient_digest="sha256:" + "c" * 64,
                reference_gradient_digest="sha256:" + "d" * 64,
                max_abs_error=1e-9,
                l2_error=2e-9,
                tolerance=1e-6,
                shots=True,
                replay_artifact_id="pl-provider-gradient-replay-20260620",
            ),
            "shots",
        ),
        (
            lambda: PennyLaneProviderGradientParityArtifact(
                artifact_id="pl-provider-gradient-20260620",
                provider_execution_artifact_id="pl-provider-sim-20260620",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                gradient_digest="sha256:" + "c" * 64,
                reference_gradient_digest="sha256:" + "d" * 64,
                max_abs_error=1e-9,
                l2_error=2e-9,
                tolerance=1e-6,
                shots=4096,
                replay_artifact_id="pl-provider-gradient-replay-20260620",
                hardware_execution=True,
            ),
            "must not claim hardware execution",
        ),
        (
            lambda: PennyLaneProviderGradientParityArtifact(
                artifact_id="pl-provider-gradient-20260620",
                provider_execution_artifact_id="pl-provider-sim-20260620",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                gradient_digest="not-a-digest",
                reference_gradient_digest="sha256:" + "d" * 64,
                max_abs_error=1e-9,
                l2_error=2e-9,
                tolerance=1e-6,
                shots=4096,
                replay_artifact_id="pl-provider-gradient-replay-20260620",
            ),
            "gradient_digest",
        ),
        (
            lambda: PennyLaneProviderPluginExecutionArtifact(
                artifact_id="",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_simulator",
                shots=4096,
                result_digest="sha256:" + "a" * 64,
                metadata_digest="sha256:" + "b" * 64,
            ),
            "artifact_id",
        ),
        (
            lambda: PennyLaneProviderPluginExecutionArtifact(
                artifact_id="pl-provider-sim-20260620",
                plugin_name="pennylane-provider-simulator",
                provider_name="example-provider",
                device_name="example.simulator",
                backend_name="example_sim_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_simulator",
                shots=4096,
                result_digest="not-a-digest",
                metadata_digest="sha256:" + "b" * 64,
            ),
            "result_digest",
        ),
        (
            lambda: PennyLaneHardwarePluginExecutionArtifact(
                artifact_id="pl-hardware-exec-20260620",
                plugin_name="pennylane-provider-hardware",
                provider_name="example-provider",
                device_name="example.qpu",
                backend_name="example_qpu_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="offline_replay",
                shots=4096,
                live_execution_ticket="ticket-pl-hw-20260620",
                provider_allowlist_id="allowlist-pl-hw-20260620",
                shot_budget_id="shot-budget-pl-hw-20260620",
                hardware_evidence_id="hardware-evidence-pl-hw-20260620",
                result_digest="sha256:" + "e" * 64,
                raw_counts_digest="sha256:" + "f" * 64,
                calibration_snapshot_digest="sha256:" + "1" * 64,
                metadata_digest="sha256:" + "2" * 64,
            ),
            "live hardware execution",
        ),
        (
            lambda: PennyLaneHardwarePluginExecutionArtifact(
                artifact_id="pl-hardware-exec-20260620",
                plugin_name="pennylane-provider-hardware",
                provider_name="example-provider",
                device_name="example.qpu",
                backend_name="example_qpu_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="live_qpu_simulator",
                shots=4096,
                live_execution_ticket="ticket-pl-hw-20260620",
                provider_allowlist_id="allowlist-pl-hw-20260620",
                shot_budget_id="shot-budget-pl-hw-20260620",
                hardware_evidence_id="hardware-evidence-pl-hw-20260620",
                result_digest="sha256:" + "e" * 64,
                raw_counts_digest="sha256:" + "f" * 64,
                calibration_snapshot_digest="sha256:" + "1" * 64,
                metadata_digest="sha256:" + "2" * 64,
            ),
            "simulator or replay",
        ),
        (
            lambda: PennyLaneHardwarePluginExecutionArtifact(
                artifact_id="pl-hardware-exec-20260620",
                plugin_name="pennylane-provider-hardware",
                provider_name="example-provider",
                device_name="example.qpu",
                backend_name="example_qpu_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_live_qpu",
                shots=True,
                live_execution_ticket="ticket-pl-hw-20260620",
                provider_allowlist_id="allowlist-pl-hw-20260620",
                shot_budget_id="shot-budget-pl-hw-20260620",
                hardware_evidence_id="hardware-evidence-pl-hw-20260620",
                result_digest="sha256:" + "e" * 64,
                raw_counts_digest="sha256:" + "f" * 64,
                calibration_snapshot_digest="sha256:" + "1" * 64,
                metadata_digest="sha256:" + "2" * 64,
            ),
            "shots",
        ),
        (
            lambda: PennyLaneHardwarePluginExecutionArtifact(
                artifact_id="pl-hardware-exec-20260620",
                plugin_name="pennylane-provider-hardware",
                provider_name="example-provider",
                device_name="example.qpu",
                backend_name="example_qpu_v1",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                execution_mode="provider_live_qpu",
                shots=4096,
                live_execution_ticket="ticket-pl-hw-20260620",
                provider_allowlist_id="allowlist-pl-hw-20260620",
                shot_budget_id="shot-budget-pl-hw-20260620",
                hardware_evidence_id="hardware-evidence-pl-hw-20260620",
                result_digest="sha256:" + "e" * 64,
                raw_counts_digest="sha256:" + "f" * 64,
                calibration_snapshot_digest="sha256:" + "1" * 64,
                metadata_digest="sha256:" + "2" * 64,
                hardware_execution=False,
            ),
            "must claim hardware execution",
        ),
    ],
)
def test_pennylane_provider_plugin_artifacts_reject_malformed_evidence(
    factory: Callable[[], object],
    match: str,
) -> None:
    """Provider-plugin evidence objects reject malformed or promoted metadata."""
    with pytest.raises(ValueError, match=match):
        factory()


def test_pennylane_provider_plugin_exports_remain_compatible() -> None:
    """Old bridge and package-level imports keep resolving to the extracted builder."""
    assert bridge_run_pennylane_plugin_matrix is run_pennylane_plugin_matrix
    assert facade_run_pennylane_plugin_matrix is run_pennylane_plugin_matrix
