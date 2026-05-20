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
    "snapshot_from_qiskit_runtime_backend",
    "snapshot_from_qbraid_device",
    "snapshot_from_strangeworks_backend",
]
