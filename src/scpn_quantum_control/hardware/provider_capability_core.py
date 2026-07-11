# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Capability Core
"""Provider-neutral no-submit capability contracts and readiness decisions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .aggregators import ResolvedAggregatorProviderRoute, resolve_aggregator_provider_route
from .openpulse_control import (
    OpenPulseCalibrationWorkflow,
    build_rabi_amplitude_calibration_workflow,
)

CapabilityDecisionStatus = Literal["ready", "blocked", "unknown"]


ProviderMetadataProbe = Callable[[ResolvedAggregatorProviderRoute], "ProviderCapabilitySnapshot"]


@dataclass(frozen=True)
class ProviderCapabilitySnapshot:
    """Provider target metadata collected without submitting a workload."""

    route_id: str
    aggregator: str
    provider: str
    backend_id: str
    target_name: str
    n_qubits: int
    supported_ir_formats: tuple[str, ...]
    basis_gates: tuple[str, ...] = field(default_factory=tuple)
    native_features: tuple[str, ...] = field(default_factory=tuple)
    online: bool | None = None
    simulator: bool = False
    no_submit: bool = True
    max_shots: int | None = None
    max_circuits: int | None = None
    queue_depth: int | None = None
    calibration_timestamp: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("route_id", "aggregator", "provider", "backend_id", "target_name"):
            _require_text(getattr(self, field_name), field_name)
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if not self.supported_ir_formats:
            raise ValueError("supported_ir_formats must not be empty")
        _require_string_tuple(self.supported_ir_formats, "supported_ir_formats")
        _require_string_tuple(self.basis_gates, "basis_gates")
        _require_string_tuple(self.native_features, "native_features")
        if self.no_submit is not True:
            raise ValueError("capability snapshots must be no-submit metadata")
        for field_name in ("max_shots", "max_circuits"):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be positive when provided")
        if self.queue_depth is not None and self.queue_depth < 0:
            raise ValueError("queue_depth must be non-negative when provided")
        if self.calibration_timestamp is not None:
            _require_text(self.calibration_timestamp, "calibration_timestamp")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")


@dataclass(frozen=True)
class ProviderCapabilityDecision:
    """Readiness decision for one no-submit provider capability snapshot."""

    snapshot: ProviderCapabilitySnapshot
    status: CapabilityDecisionStatus
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    required_ir_format: str | None
    min_qubits: int | None
    no_submit: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise the provider capability decision."""
        return {
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "required_ir_format": self.required_ir_format,
            "min_qubits": self.min_qubits,
            "no_submit": self.no_submit,
            "snapshot": {
                "route_id": self.snapshot.route_id,
                "aggregator": self.snapshot.aggregator,
                "provider": self.snapshot.provider,
                "backend_id": self.snapshot.backend_id,
                "target_name": self.snapshot.target_name,
                "n_qubits": self.snapshot.n_qubits,
                "supported_ir_formats": list(self.snapshot.supported_ir_formats),
                "basis_gates": list(self.snapshot.basis_gates),
                "native_features": list(self.snapshot.native_features),
                "online": self.snapshot.online,
                "simulator": self.snapshot.simulator,
                "max_shots": self.snapshot.max_shots,
                "max_circuits": self.snapshot.max_circuits,
                "queue_depth": self.snapshot.queue_depth,
                "calibration_timestamp": self.snapshot.calibration_timestamp,
                "metadata": dict(self.snapshot.metadata),
            },
        }


@dataclass(frozen=True)
class OpenPulseControlReadiness:
    """No-submit readiness surface for pulse-level OpenPulse calibration lanes."""

    snapshot: ProviderCapabilitySnapshot
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    workflow: OpenPulseCalibrationWorkflow | None
    required_ir_formats: tuple[str, ...]
    required_native_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the OpenPulse readiness decision."""
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "required_ir_formats": list(self.required_ir_formats),
            "required_native_features": list(self.required_native_features),
            "snapshot": {
                "route_id": self.snapshot.route_id,
                "provider": self.snapshot.provider,
                "backend_id": self.snapshot.backend_id,
                "target_name": self.snapshot.target_name,
                "supported_ir_formats": list(self.snapshot.supported_ir_formats),
                "native_features": list(self.snapshot.native_features),
                "n_qubits": self.snapshot.n_qubits,
                "online": self.snapshot.online,
                "simulator": self.snapshot.simulator,
                "calibration_timestamp": self.snapshot.calibration_timestamp,
            },
            "workflow": self.workflow.to_payload() if self.workflow is not None else None,
            "hardware_submission": False,
        }


def build_openpulse_control_readiness(
    snapshot: ProviderCapabilitySnapshot,
    *,
    qubit: int,
    dt: float,
    amplitude_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
    shots: int = 4096,
) -> OpenPulseControlReadiness:
    """Build no-submit readiness for OpenPulse control and calibration workflows."""
    blockers: list[str] = []
    warnings: list[str] = []
    required_ir_formats = ("openpulse", "qiskit_qpy", "qiskit")
    required_native_features = ("pulse_control", "drive_channel_access")

    if snapshot.online is False:
        blockers.append("target is offline")
    if snapshot.online is None:
        warnings.append("target online status is unknown")
    if snapshot.n_qubits <= qubit:
        blockers.append(
            f"target exposes {snapshot.n_qubits} qubits, requested qubit index {qubit}"
        )
    if not any(fmt in snapshot.supported_ir_formats for fmt in required_ir_formats):
        blockers.append(
            "target does not advertise an OpenPulse-compatible IR route "
            f"(required one of: {', '.join(required_ir_formats)})"
        )

    feature_set = set(snapshot.native_features)
    if "dynamic_circuits" not in feature_set:
        warnings.append("target does not advertise dynamic_circuits")
    missing_features = [
        feature for feature in required_native_features if feature not in feature_set
    ]
    if missing_features:
        blockers.append(
            "target is missing pulse native features: " + ", ".join(sorted(missing_features))
        )

    workflow: OpenPulseCalibrationWorkflow | None = None
    if not blockers:
        workflow = build_rabi_amplitude_calibration_workflow(
            backend_name=snapshot.target_name,
            qubit=qubit,
            amplitude_grid=amplitude_grid,
            shots=shots,
            dt=dt,
        )

    return OpenPulseControlReadiness(
        snapshot=snapshot,
        ready=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        workflow=workflow,
        required_ir_formats=required_ir_formats,
        required_native_features=required_native_features,
    )


def probe_aggregator_provider_capability(
    *,
    aggregator: str,
    provider: str,
    metadata_probe: ProviderMetadataProbe,
    ir_format: str | None = None,
    route_id: str | None = None,
    min_qubits: int | None = None,
) -> ProviderCapabilityDecision:
    """Resolve a route, collect provider metadata, and assess it without submission."""
    resolved = resolve_aggregator_provider_route(
        aggregator=aggregator,
        provider=provider,
        ir_format=ir_format,
        route_id=route_id,
    )
    snapshot = metadata_probe(resolved)
    return assess_provider_capability_snapshot(
        snapshot,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        route_id=resolved.route.route_id,
        required_ir_format=ir_format,
        min_qubits=min_qubits,
    )


def assess_provider_capability_snapshot(
    snapshot: ProviderCapabilitySnapshot,
    *,
    aggregator: str,
    provider: str,
    backend_id: str,
    route_id: str | None = None,
    required_ir_format: str | None = None,
    min_qubits: int | None = None,
) -> ProviderCapabilityDecision:
    """Assess route-level provider metadata without submitting work."""
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot.no_submit is not True:
        blockers.append("provider capability metadata is not no-submit")
    if route_id is not None and snapshot.route_id != route_id:
        blockers.append(f"route mismatch: expected {route_id}, got {snapshot.route_id}")
    if snapshot.aggregator != aggregator:
        blockers.append(f"aggregator mismatch: expected {aggregator}, got {snapshot.aggregator}")
    if snapshot.provider != provider:
        blockers.append(f"provider mismatch: expected {provider}, got {snapshot.provider}")
    if snapshot.backend_id != backend_id:
        blockers.append(f"backend mismatch: expected {backend_id}, got {snapshot.backend_id}")
    if snapshot.online is False:
        blockers.append("provider target is offline")
    if snapshot.online is None:
        warnings.append("provider target online status was not reported")
    if min_qubits is not None and snapshot.n_qubits < min_qubits:
        blockers.append(
            f"target has {snapshot.n_qubits} qubits but route requires at least {min_qubits}"
        )
    if required_ir_format is not None and required_ir_format not in snapshot.supported_ir_formats:
        blockers.append(f"target does not support required IR format: {required_ir_format}")

    if blockers:
        status: CapabilityDecisionStatus = "blocked"
    elif warnings:
        status = "unknown"
    else:
        status = "ready"
    return ProviderCapabilityDecision(
        snapshot=snapshot,
        status=status,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        required_ir_format=required_ir_format,
        min_qubits=min_qubits,
    )


def _require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty text")


def _require_string_tuple(value: tuple[str, ...], field_name: str) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field_name} must be a tuple of non-empty strings")


__all__ = [
    "CapabilityDecisionStatus",
    "OpenPulseControlReadiness",
    "ProviderCapabilityDecision",
    "ProviderCapabilitySnapshot",
    "ProviderMetadataProbe",
    "assess_provider_capability_snapshot",
    "build_openpulse_control_readiness",
    "probe_aggregator_provider_capability",
]
