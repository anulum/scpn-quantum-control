# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — provider capability discovery
"""No-submit capability discovery contracts for aggregator/provider routes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .aggregators import ResolvedAggregatorProviderRoute, resolve_aggregator_provider_route

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


def snapshot_from_azure_target(
    resolved: ResolvedAggregatorProviderRoute,
    target: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from Azure Quantum target metadata."""

    capability = _first_available_attr(
        target,
        names=("capability", "capabilities", "target_capability", "target_capabilities"),
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            target,
            capability,
            names=("name", "target_id", "target", "id"),
            field_name="Azure target name",
        ),
        n_qubits=_first_positive_int_attr(
            capability,
            target,
            names=("num_qubits", "n_qubits", "qubits", "qubit_count", "qubitCount"),
            field_name="Azure qubit count",
        ),
        supported_ir_formats=_azure_supported_ir_formats(resolved, target, capability),
        basis_gates=_first_string_tuple_attr(
            capability,
            target,
            names=("basis_gates", "native_gates", "gates", "supported_operations"),
        ),
        native_features=_azure_native_features(target, capability),
        online=_azure_online_state(target, capability),
        simulator=_first_bool_attr(
            target,
            capability,
            names=("simulator", "is_simulator"),
        )
        or False,
        max_shots=_first_optional_int_attr(
            capability,
            target,
            names=("max_shots", "shots_limit", "maxShots", "max_execution_shots"),
        ),
        max_circuits=_first_optional_int_attr(
            capability,
            target,
            names=("max_experiments", "max_circuits", "max_jobs"),
        ),
        queue_depth=_first_optional_int_attr(
            target,
            capability,
            names=("average_queue_time", "queue_depth", "pending_jobs", "queue_size"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            target,
            capability,
            names=(
                "latest_calibration",
                "calibration_timestamp",
                "last_calibration",
                "last_updated",
                "lastUpdated",
            ),
        ),
        metadata={
            "adapter": "azure_target_no_submit",
            "provider_id": _first_optional_text_attr(
                target,
                capability,
                names=("provider_id", "provider", "provider_name"),
            ),
        },
    )


def snapshot_from_ionq_backend(
    resolved: ResolvedAggregatorProviderRoute,
    backend: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from direct IonQ backend metadata."""

    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            backend,
            names=("backend", "backend_id", "id", "name", "target", "target_name"),
            field_name="IonQ backend name",
        ),
        n_qubits=_first_positive_int_attr(
            backend,
            names=("qubits", "n_qubits", "num_qubits", "qubit_count", "qubitCount"),
            field_name="IonQ qubit count",
        ),
        supported_ir_formats=_ionq_supported_ir_formats(resolved, backend),
        basis_gates=_first_string_tuple_attr(
            backend,
            names=("basis_gates", "native_gates", "gates", "supported_operations"),
        ),
        native_features=_ionq_native_features(backend),
        online=_ionq_online_state(backend),
        simulator=_first_bool_attr(backend, names=("simulator", "is_simulator")) or False,
        max_shots=_first_optional_int_attr(
            backend,
            names=("max_shots", "shots_limit", "maxShots", "max_execution_shots"),
        ),
        max_circuits=_first_optional_int_attr(
            backend,
            names=("max_circuits", "max_jobs", "max_experiments"),
        ),
        queue_depth=_first_optional_int_attr(
            backend,
            names=("queue_depth", "pending_jobs", "queue_size", "average_queue_time"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            backend,
            names=(
                "last_calibration",
                "calibration_timestamp",
                "latest_calibration",
                "last_updated",
                "lastUpdated",
            ),
        ),
        metadata={
            "adapter": "ionq_backend_no_submit",
            "gateset": _first_optional_text_attr(backend, names=("gateset", "gate_set")),
        },
    )


def snapshot_from_quantinuum_backend(
    resolved: ResolvedAggregatorProviderRoute,
    backend: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from direct Quantinuum metadata."""

    backend_info = _first_available_attr(
        backend,
        names=("backend_info", "_backend_info", "info", "metadata"),
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            backend,
            backend_info,
            names=("machine", "machine_name", "backend", "backend_id", "name", "id"),
            field_name="Quantinuum backend name",
        ),
        n_qubits=_first_positive_int_attr(
            backend_info,
            backend,
            names=("n_qubits", "num_qubits", "qubits", "qubit_count", "qubitCount"),
            field_name="Quantinuum qubit count",
        ),
        supported_ir_formats=_quantinuum_supported_ir_formats(
            resolved,
            backend,
            backend_info,
        ),
        basis_gates=_first_string_tuple_attr(
            backend_info,
            backend,
            names=("gate_set", "basis_gates", "native_gates", "gates", "supported_operations"),
        ),
        native_features=_quantinuum_native_features(backend, backend_info),
        online=_quantinuum_online_state(backend, backend_info),
        simulator=_first_bool_attr(backend, backend_info, names=("simulator", "is_simulator"))
        or False,
        max_shots=_first_optional_int_attr(
            backend_info,
            backend,
            names=("max_n_shots", "max_shots", "shots_limit", "maxShots"),
        ),
        max_circuits=_first_optional_int_attr(
            backend_info,
            backend,
            names=("max_batch_circuits", "max_circuits", "max_jobs", "max_experiments"),
        ),
        queue_depth=_first_optional_int_attr(
            backend_info,
            backend,
            names=("queue_depth", "pending_jobs", "queue_size", "average_queue_time"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            backend_info,
            backend,
            names=(
                "last_calibration",
                "calibration_timestamp",
                "latest_calibration",
                "last_updated",
                "lastUpdated",
            ),
        ),
        metadata={
            "adapter": "quantinuum_backend_no_submit",
            "machine": _first_optional_text_attr(
                backend,
                backend_info,
                names=("machine", "machine_name", "backend", "backend_id", "name", "id"),
            ),
        },
    )


def snapshot_from_rigetti_qcs(
    resolved: ResolvedAggregatorProviderRoute,
    quantum_computer: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from direct Rigetti QCS metadata."""

    compiler = _first_available_attr(
        quantum_computer,
        names=("compiler", "quantum_processor_compiler", "qpu_compiler"),
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            quantum_computer,
            names=(
                "quantum_computer",
                "quantum_computer_name",
                "backend",
                "backend_id",
                "name",
                "id",
            ),
            field_name="Rigetti QCS target name",
        ),
        n_qubits=_first_positive_int_attr(
            quantum_computer,
            names=("num_qubits", "n_qubits", "qubits", "qubit_count", "qubitCount"),
            field_name="Rigetti QCS qubit count",
        ),
        supported_ir_formats=_rigetti_supported_ir_formats(resolved, quantum_computer),
        basis_gates=_first_string_tuple_attr(
            quantum_computer,
            names=("basis_gates", "native_gates", "gates", "supported_operations"),
        ),
        native_features=_rigetti_native_features(quantum_computer),
        online=_rigetti_online_state(quantum_computer),
        simulator=_first_bool_attr(quantum_computer, names=("simulator", "is_simulator")) or False,
        max_shots=_first_optional_int_attr(
            quantum_computer,
            names=("max_shots", "shots_limit", "maxShots", "max_execution_shots"),
        ),
        max_circuits=_first_optional_int_attr(
            quantum_computer,
            names=("max_circuits", "max_jobs", "max_experiments"),
        ),
        queue_depth=_first_optional_int_attr(
            quantum_computer,
            names=("queue_depth", "pending_jobs", "queue_size", "average_queue_time"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            quantum_computer,
            names=(
                "last_calibration",
                "calibration_timestamp",
                "latest_calibration",
                "last_updated",
                "lastUpdated",
            ),
        ),
        metadata={
            "adapter": "rigetti_qcs_no_submit",
            "quilc_version": _first_optional_text_attr(compiler, names=("quilc_version",)),
            "qpu_compiler_version": _first_optional_text_attr(
                compiler,
                names=("qpu_compiler_version", "compiler_version"),
            ),
        },
    )


def snapshot_from_braket_device(
    resolved: ResolvedAggregatorProviderRoute,
    device: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from AWS Braket device metadata."""

    properties = _optional_attr(device, "properties")
    service = _optional_attr(properties, "service")
    paradigm = _optional_attr(properties, "paradigm")
    action = _optional_attr(properties, "action")
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            device,
            properties,
            names=("name", "deviceName", "device_name", "arn", "deviceArn"),
            field_name="Braket device name",
        ),
        n_qubits=_first_positive_int_attr(
            paradigm,
            properties,
            device,
            names=("qubitCount", "qubit_count", "num_qubits", "n_qubits", "qubits"),
            field_name="Braket qubit count",
        ),
        supported_ir_formats=_braket_supported_ir_formats(resolved, action, device, properties),
        basis_gates=_braket_basis_gates(action, properties, device),
        native_features=_braket_native_features(action, properties, device),
        online=_first_online_attr(device, properties, service),
        simulator=_first_bool_attr(
            device,
            properties,
            service,
            names=("simulator", "is_simulator"),
        )
        or False,
        max_shots=_braket_max_shots(service, properties, device),
        max_circuits=_first_optional_int_attr(
            service,
            properties,
            device,
            names=("max_circuits", "max_experiments", "max_jobs"),
        ),
        queue_depth=_braket_queue_depth(device),
        calibration_timestamp=_first_optional_text_attr(
            properties,
            device,
            names=("lastUpdated", "last_updated", "calibration_timestamp", "last_calibration"),
        ),
        metadata={
            "adapter": "braket_device_no_submit",
            "device_arn": _first_optional_text_attr(
                device, properties, names=("arn", "deviceArn")
            ),
            "action_names": _braket_action_names(action),
        },
    )


def snapshot_from_qiskit_runtime_backend(
    resolved: ResolvedAggregatorProviderRoute,
    backend: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from IBM/Qiskit backend metadata."""

    configuration = _optional_noarg_call(backend, "configuration")
    status = _optional_noarg_call(backend, "status")
    properties = _optional_noarg_call(backend, "properties")
    target = _optional_attr(backend, "target")
    basis_gates = _first_string_tuple_attr(
        configuration,
        target,
        backend,
        names=("basis_gates", "operation_names", "operations", "native_gates", "gates"),
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            backend,
            configuration,
            names=("name", "backend_name", "backend_id"),
            field_name="Qiskit backend name",
        ),
        n_qubits=_first_positive_int_attr(
            backend,
            configuration,
            target,
            names=("num_qubits", "n_qubits", "qubits"),
            field_name="Qiskit qubit count",
        ),
        supported_ir_formats=_qiskit_supported_ir_formats(resolved, backend, configuration),
        basis_gates=basis_gates,
        native_features=_qiskit_native_features(backend, target, basis_gates),
        online=_qiskit_online_state(backend, status),
        simulator=_first_bool_attr(
            backend,
            configuration,
            names=("simulator", "is_simulator"),
        )
        or False,
        max_shots=_first_optional_int_attr(
            configuration,
            backend,
            names=("max_shots", "shots_limit", "max_execution_shots"),
        ),
        max_circuits=_first_optional_int_attr(
            configuration,
            backend,
            names=("max_experiments", "max_circuits", "max_jobs"),
        ),
        queue_depth=_first_optional_int_attr(
            status,
            backend,
            names=("pending_jobs", "queue_depth", "queue_size"),
            minimum=0,
        ),
        calibration_timestamp=_qiskit_calibration_timestamp(properties, backend),
        metadata={
            "adapter": "qiskit_runtime_backend_no_submit",
            "metadata_calls": ("configuration", "status", "properties"),
        },
    )


def snapshot_from_qbraid_device(
    resolved: ResolvedAggregatorProviderRoute,
    device: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from qBraid device metadata."""

    profile = _optional_attr(device, "profile")
    target_name = _first_text_attr(
        profile,
        device,
        names=("device_id", "backend_id", "id", "name"),
        field_name="qBraid target name",
    )
    n_qubits = _first_positive_int_attr(
        profile,
        device,
        names=("num_qubits", "n_qubits", "qubits"),
        field_name="qBraid qubit count",
    )
    supported_ir_formats = _declared_ir_formats(
        profile,
        device,
        names=(
            "supported_ir_formats",
            "ir_formats",
            "input_formats",
            "program_formats",
            "supported_program_formats",
        ),
        field_name="qBraid IR formats",
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=target_name,
        n_qubits=n_qubits,
        supported_ir_formats=supported_ir_formats,
        basis_gates=_first_string_tuple_attr(
            profile,
            device,
            names=("basis_gates", "native_gates", "gates"),
        ),
        native_features=_first_string_tuple_attr(
            profile,
            device,
            names=("native_features", "features", "capabilities"),
        ),
        online=_first_online_attr(device, profile),
        simulator=_first_bool_attr(profile, device, names=("simulator", "is_simulator")) or False,
        max_shots=_first_optional_int_attr(device, profile, names=("max_shots", "shots_limit")),
        max_circuits=_first_optional_int_attr(
            device,
            profile,
            names=("max_circuits", "max_experiments", "max_jobs"),
        ),
        queue_depth=_first_optional_int_attr(
            device,
            profile,
            names=("queue_depth", "pending_jobs", "queue_size"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            device,
            profile,
            names=("calibration_timestamp", "last_calibration", "calibrated_at"),
        ),
        metadata={
            "adapter": "qbraid_device_no_submit",
            "provider_name": _first_optional_text_attr(
                profile,
                device,
                names=("provider_name", "provider", "vendor"),
            ),
        },
    )


def snapshot_from_strangeworks_backend(
    resolved: ResolvedAggregatorProviderRoute,
    backend: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from Strangeworks backend metadata."""

    target_name = _first_text_attr(
        backend,
        names=("id", "backend_id", "device_id", "name"),
        field_name="Strangeworks target name",
    )
    n_qubits = _first_positive_int_attr(
        backend,
        names=("n_qubits", "num_qubits", "qubits"),
        field_name="Strangeworks qubit count",
    )
    supported_ir_formats = _declared_ir_formats(
        backend,
        names=(
            "input_formats",
            "supported_ir_formats",
            "ir_formats",
            "program_formats",
            "supported_program_formats",
        ),
        field_name="Strangeworks IR formats",
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=target_name,
        n_qubits=n_qubits,
        supported_ir_formats=supported_ir_formats,
        basis_gates=_first_string_tuple_attr(
            backend,
            names=("basis_gates", "native_gates", "gates"),
        ),
        native_features=_first_string_tuple_attr(
            backend,
            names=("native_features", "features", "capabilities"),
        ),
        online=_first_online_attr(backend),
        simulator=_first_bool_attr(backend, names=("simulator", "is_simulator")) or False,
        max_shots=_first_optional_int_attr(backend, names=("max_shots", "shots_limit")),
        max_circuits=_first_optional_int_attr(
            backend,
            names=("max_circuits", "max_experiments", "max_jobs"),
        ),
        queue_depth=_first_optional_int_attr(
            backend,
            names=("queue_depth", "pending_jobs", "queue_size"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            backend,
            names=("calibration_timestamp", "last_calibration", "calibrated_at"),
        ),
        metadata={"adapter": "strangeworks_backend_no_submit"},
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


def _optional_attr(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    try:
        return getattr(source, name)
    except Exception:
        return None


def _optional_noarg_call(source: Any, name: str) -> Any:
    candidate = _optional_attr(source, name)
    if not callable(candidate):
        return candidate
    try:
        return candidate()
    except Exception:
        return None


def _first_available_attr(*sources: Any, names: tuple[str, ...]) -> Any:
    for value in _attr_candidates(*sources, names=names):
        if value is not None:
            return value
    return None


def _attr_candidates(*sources: Any, names: tuple[str, ...]) -> list[Any]:
    candidates: list[Any] = []
    for source in sources:
        if source is None:
            continue
        for name in names:
            candidates.append(_optional_attr(source, name))
    return candidates


def _first_text_attr(*sources: Any, names: tuple[str, ...], field_name: str) -> str:
    value = _first_optional_text_attr(*sources, names=names)
    if value is None:
        raise ValueError(f"{field_name} must be provided by provider metadata")
    return value


def _first_optional_text_attr(*sources: Any, names: tuple[str, ...]) -> str | None:
    for value in _attr_candidates(*sources, names=names):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_positive_int_attr(*sources: Any, names: tuple[str, ...], field_name: str) -> int:
    value = _first_optional_int_attr(*sources, names=names)
    if value is None:
        raise ValueError(f"{field_name} must be provided by provider metadata")
    return value


def _first_optional_int_attr(
    *sources: Any,
    names: tuple[str, ...],
    minimum: int = 1,
) -> int | None:
    for value in _attr_candidates(*sources, names=names):
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= minimum:
            return value
    return None


def _first_bool_attr(*sources: Any, names: tuple[str, ...]) -> bool | None:
    for value in _attr_candidates(*sources, names=names):
        if isinstance(value, bool):
            return value
    return None


def _first_online_attr(*sources: Any) -> bool | None:
    for value in _attr_candidates(
        *sources,
        names=("online", "is_online", "available", "status"),
    ):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"online", "available", "active", "ready", "operational"}:
                return True
            if normalized in {
                "offline",
                "unavailable",
                "inactive",
                "retired",
                "maintenance",
            }:
                return False
    return None


def _qiskit_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    *sources: Any,
) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        *sources,
        names=(
            "supported_ir_formats",
            "ir_formats",
            "input_formats",
            "program_formats",
            "supported_program_formats",
        ),
    )
    return declared or resolved.route.ir_formats


def _azure_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    *sources: Any,
) -> tuple[str, ...]:
    declared = _azure_declared_ir_formats(*sources)
    if not declared:
        return resolved.route.ir_formats
    return declared


def _azure_declared_ir_formats(*sources: Any) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        *sources,
        names=(
            "supported_ir_formats",
            "ir_formats",
            "input_formats",
            "input_data_formats",
            "program_formats",
        ),
    )
    return tuple(_azure_ir_format_token(item) for item in declared)


def _azure_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"qasm.v3", "qasm3", "openqasm", "openqasm_3", "openqasm3"}:
        return "openqasm3"
    if normalized in {"qiskit", "qiskit_qpy"}:
        return "qiskit"
    if normalized in {"qir", "qir.v1"}:
        return "qir"
    if normalized in {"quil", "rigetti.quil"}:
        return "quil"
    if normalized in {"pasqal_ir", "pasqal.ir"}:
        return "pasqal_ir"
    return normalized


def _azure_native_features(*sources: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(*sources, names=("native_features", "features")))
    formats = _azure_declared_ir_formats(*sources)
    if "openqasm3" in formats or "qiskit" in formats or "qir" in formats or "quil" in formats:
        features.add("gate_model")
    if "pasqal_ir" in formats:
        features.add("neutral_atom")
    return tuple(sorted(features))


def _azure_online_state(*sources: Any) -> bool | None:
    explicit = _first_online_attr(*sources)
    if explicit is not None:
        return explicit
    text_status = _first_optional_text_attr(
        *sources,
        names=("current_availability", "availability", "target_status", "status"),
    )
    if text_status is None:
        return None
    normalized = text_status.strip().lower()
    if normalized in {"available", "online", "ready", "active", "enabled"}:
        return True
    if normalized in {"unavailable", "offline", "disabled", "retired", "maintenance"}:
        return False
    return None


def _ionq_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    backend: Any,
) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        backend,
        names=(
            "supported_ir_formats",
            "ir_formats",
            "input_formats",
            "input_data_formats",
            "program_formats",
        ),
    )
    if not declared:
        return resolved.route.ir_formats
    return tuple(_ionq_ir_format_token(item) for item in declared)


def _ionq_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"ionq_json", "ionq.circuit.v1", "ionq_circuit_v1", "qis"}:
        return "ionq_json"
    if normalized in {"qasm.v3", "qasm3", "openqasm", "openqasm_3", "openqasm3"}:
        return "openqasm3"
    if normalized in {"qir", "qir.v1"}:
        return "qir"
    return normalized


def _ionq_native_features(backend: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(backend, names=("native_features", "features")))
    features.add("gate_model")
    features.add("trapped_ion")
    if _first_bool_attr(backend, names=("all_to_all", "all_to_all_connectivity")) is not False:
        features.add("all_to_all_connectivity")
    return tuple(sorted(features))


def _ionq_online_state(backend: Any) -> bool | None:
    explicit = _first_online_attr(backend)
    if explicit is not None:
        return explicit
    text_status = _first_optional_text_attr(
        backend,
        names=("availability", "target_status", "status", "state"),
    )
    if text_status is None:
        return None
    normalized = text_status.strip().lower()
    if normalized in {"available", "online", "ready", "active", "enabled"}:
        return True
    if normalized in {"unavailable", "offline", "disabled", "retired", "maintenance"}:
        return False
    return None


def _quantinuum_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    *sources: Any,
) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        *sources,
        names=(
            "supported_ir_formats",
            "ir_formats",
            "input_formats",
            "input_data_formats",
            "program_formats",
        ),
    )
    if not declared:
        return resolved.route.ir_formats
    return tuple(_quantinuum_ir_format_token(item) for item in declared)


def _quantinuum_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"tket", "pytket", "pytket_circuit", "pytket_circuit_v1"}:
        return "tket"
    if normalized in {"qasm.v3", "qasm3", "openqasm", "openqasm_3", "openqasm3"}:
        return "openqasm3"
    if normalized in {"qir", "qir.v1"}:
        return "qir"
    return normalized


def _quantinuum_native_features(*sources: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(*sources, names=("native_features", "features")))
    features.add("gate_model")
    features.add("trapped_ion")
    if _first_bool_attr(
        *sources,
        names=("supports_mid_circuit_measurement", "mid_circuit_measurement"),
    ):
        features.add("mid_circuit_measurement")
    if _first_bool_attr(*sources, names=("supports_reset", "reset")):
        features.add("conditional_reset")
    return tuple(sorted(features))


def _quantinuum_online_state(*sources: Any) -> bool | None:
    explicit = _first_online_attr(*sources)
    if explicit is not None:
        return explicit
    text_status = _first_optional_text_attr(
        *sources,
        names=("availability", "target_status", "status", "state"),
    )
    if text_status is None:
        return None
    normalized = text_status.strip().lower()
    if normalized in {"available", "online", "ready", "active", "enabled"}:
        return True
    if normalized in {"unavailable", "offline", "disabled", "retired", "maintenance"}:
        return False
    return None


def _rigetti_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    quantum_computer: Any,
) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        quantum_computer,
        names=(
            "supported_ir_formats",
            "ir_formats",
            "input_formats",
            "input_data_formats",
            "program_formats",
        ),
    )
    if not declared:
        return resolved.route.ir_formats
    return tuple(_rigetti_ir_format_token(item) for item in declared)


def _rigetti_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"quil", "rigetti.quil", "rigetti_quil", "pyquil"}:
        return "quil"
    if normalized in {"qasm.v3", "qasm3", "openqasm", "openqasm_3", "openqasm3"}:
        return "openqasm3"
    if normalized == "mlir":
        return "mlir"
    return normalized


def _rigetti_native_features(quantum_computer: Any) -> tuple[str, ...]:
    features = set(
        _first_string_tuple_attr(quantum_computer, names=("native_features", "features"))
    )
    features.add("gate_model")
    features.add("superconducting")
    if _first_bool_attr(quantum_computer, names=("lattice_connectivity", "topology")) is not False:
        features.add("lattice_connectivity")
    return tuple(sorted(features))


def _rigetti_online_state(quantum_computer: Any) -> bool | None:
    explicit = _first_online_attr(quantum_computer)
    if explicit is not None:
        return explicit
    text_status = _first_optional_text_attr(
        quantum_computer,
        names=("availability", "target_status", "status", "state"),
    )
    if text_status is None:
        return None
    normalized = text_status.strip().lower()
    if normalized in {"available", "online", "ready", "active", "enabled"}:
        return True
    if normalized in {"unavailable", "offline", "disabled", "retired", "maintenance"}:
        return False
    return None


def _braket_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    action: Any,
    *sources: Any,
) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        *sources,
        names=("supported_ir_formats", "ir_formats", "input_formats", "program_formats"),
    )
    if declared:
        return declared
    action_names = _braket_action_names(action)
    formats: list[str] = []
    for action_name in action_names:
        lowered = action_name.lower()
        if "openqasm" in lowered and "openqasm3" not in formats:
            formats.append("openqasm3")
        if "braket.ir.ahs" in lowered and "braket_ahs" not in formats:
            formats.append("braket_ahs")
        if "braket.ir.jaqcd" in lowered and "braket_ir" not in formats:
            formats.append("braket_ir")
    if formats:
        route_formats = set(resolved.route.ir_formats)
        filtered = tuple(format_name for format_name in formats if format_name in route_formats)
        return filtered or tuple(formats)
    return resolved.route.ir_formats


def _braket_action_names(action: Any) -> tuple[str, ...]:
    if isinstance(action, Mapping):
        return _string_tuple_from_value(action.keys())
    return _string_tuple_from_value(action)


def _braket_basis_gates(action: Any, *sources: Any) -> tuple[str, ...]:
    for action_entry in _braket_action_entries(action):
        basis = _first_string_tuple_attr(
            action_entry,
            names=("supportedOperations", "supported_operations", "basis_gates", "native_gates"),
        )
        if basis:
            return basis
    return _first_string_tuple_attr(*sources, names=("basis_gates", "native_gates", "gates"))


def _braket_action_entries(action: Any) -> tuple[Any, ...]:
    if isinstance(action, Mapping):
        return tuple(action.values())
    try:
        return tuple(action)
    except TypeError:
        return ()


def _braket_native_features(action: Any, *sources: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(*sources, names=("native_features", "features")))
    action_names = _braket_action_names(action)
    if any("openqasm" in action_name.lower() for action_name in action_names):
        features.add("gate_model")
    if any("braket.ir.ahs" in action_name.lower() for action_name in action_names):
        features.add("analog_hamiltonian_simulation")
    return tuple(sorted(features))


def _braket_max_shots(*sources: Any) -> int | None:
    explicit = _first_optional_int_attr(
        *sources,
        names=("max_shots", "shots_limit", "maxShots", "max_execution_shots"),
    )
    if explicit is not None:
        return explicit
    for value in _attr_candidates(*sources, names=("shotsRange", "shots_range")):
        maximum = _range_maximum(value)
        if maximum is not None:
            return maximum
    return None


def _range_maximum(value: Any) -> int | None:
    if isinstance(value, Mapping):
        return _positive_int(value.get("max") or value.get("maximum") or value.get("end"))
    if isinstance(value, tuple | list) and value:
        return _positive_int(value[-1])
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 1:
        return value
    return None


def _braket_queue_depth(device: Any) -> int | None:
    queue_depth = _optional_attr(device, "queue_depth")
    if isinstance(queue_depth, int) and queue_depth >= 0:
        return queue_depth
    if queue_depth is not None:
        return _first_optional_int_attr(
            queue_depth,
            names=("normal", "priority", "queue_depth", "queueSize", "queue_size"),
            minimum=0,
        )
    return _first_optional_int_attr(
        device,
        names=("pending_jobs", "queue_depth", "queueSize", "queue_size"),
        minimum=0,
    )


def _qiskit_native_features(
    backend: Any,
    target: Any,
    basis_gates: tuple[str, ...],
) -> tuple[str, ...]:
    features = {"cross_shot_batches"}
    basis = set(basis_gates)
    operation_names = set(_first_string_tuple_attr(target, names=("operation_names",)))
    if basis.intersection({"measure", "measurement"}) or "measure" in operation_names:
        features.add("mid_circuit_measurement")
    if "reset" in basis or "reset" in operation_names:
        features.add("conditional_reset")
    if operation_names.intersection({"if_else", "while_loop", "for_loop", "switch_case"}) or (
        _first_bool_attr(backend, names=("dynamic_circuits",)) is True
    ):
        features.add("conditional_control")
    return tuple(sorted(features))


def _qiskit_online_state(backend: Any, status: Any) -> bool | None:
    operational = _first_bool_attr(status, names=("operational", "online", "available"))
    if operational is not None:
        return operational
    text_status = _first_optional_text_attr(
        status,
        backend,
        names=("status", "status_msg", "state"),
    )
    if text_status is not None:
        return _online_state_from_text(text_status)
    return _first_online_attr(backend, status)


def _online_state_from_text(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"online", "available", "active", "ready", "operational"}:
        return True
    if normalized in {
        "offline",
        "unavailable",
        "inactive",
        "retired",
        "maintenance",
    }:
        return False
    return None


def _qiskit_calibration_timestamp(*sources: Any) -> str | None:
    for value in _attr_candidates(
        *sources,
        names=("last_update_date", "calibration_timestamp", "last_calibration", "calibrated_at"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
        isoformat = _optional_attr(value, "isoformat")
        if callable(isoformat):
            try:
                return str(isoformat())
            except Exception:
                return None
    return None


def _declared_ir_formats(
    *sources: Any, names: tuple[str, ...], field_name: str
) -> tuple[str, ...]:
    formats = _first_string_tuple_attr(*sources, names=names)
    if not formats:
        raise ValueError(f"{field_name} must declare supported IR formats")
    return formats


def _first_string_tuple_attr(*sources: Any, names: tuple[str, ...]) -> tuple[str, ...]:
    for value in _attr_candidates(*sources, names=names):
        items = _string_tuple_from_value(value)
        if items:
            return items
    return ()


def _string_tuple_from_value(value: Any) -> tuple[str, ...]:
    if value is None or isinstance(value, str):
        return (value.strip(),) if isinstance(value, str) and value.strip() else ()
    if isinstance(value, Mapping):
        return _string_tuple_from_value(value.keys())
    try:
        iterator = iter(value)
    except TypeError:
        return ()
    items: list[str] = []
    for item in iterator:
        normalized: str | None
        if isinstance(item, str) and item.strip():
            normalized = item.strip()
        else:
            normalized = _program_spec_name(item)
        if normalized and normalized not in items:
            items.append(normalized)
    return tuple(items)


def _program_spec_name(value: Any) -> str | None:
    for name in ("alias", "program_type", "package", "name", "__name__"):
        candidate = _optional_attr(value, name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


__all__ = [
    "CapabilityDecisionStatus",
    "ProviderCapabilityDecision",
    "ProviderCapabilitySnapshot",
    "ProviderMetadataProbe",
    "assess_provider_capability_snapshot",
    "probe_aggregator_provider_capability",
    "snapshot_from_azure_target",
    "snapshot_from_braket_device",
    "snapshot_from_ionq_backend",
    "snapshot_from_qiskit_runtime_backend",
    "snapshot_from_qbraid_device",
    "snapshot_from_quantinuum_backend",
    "snapshot_from_rigetti_qcs",
    "snapshot_from_strangeworks_backend",
]
