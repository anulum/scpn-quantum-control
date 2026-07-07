# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — provider capability discovery
"""No-submit capability discovery contracts for aggregator/provider routes.

Module size note: this module is intentionally kept whole. Its top-level definitions form a single connected provider-capability-discovery cluster, so it is sized by responsibility rather than line count. See ``docs/architecture.md`` ("Module size and single-responsibility policy").
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


def snapshot_from_dwave_solver(
    resolved: ResolvedAggregatorProviderRoute,
    solver: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from direct D-Wave solver metadata."""
    properties = _first_available_attr(
        solver,
        names=("properties", "solver_properties", "metadata"),
    )
    topology = _dwave_topology_name(solver, properties)
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            solver,
            properties,
            names=("solver", "solver_id", "solver_name", "name", "id", "target"),
            field_name="D-Wave solver name",
        ),
        n_qubits=_first_positive_int_attr(
            properties,
            solver,
            names=("num_qubits", "n_qubits", "qubits", "qubit_count", "qubitCount"),
            field_name="D-Wave qubit count",
        ),
        supported_ir_formats=_dwave_supported_ir_formats(resolved, solver, properties),
        basis_gates=(),
        native_features=_dwave_native_features(topology, solver, properties),
        online=_dwave_online_state(solver, properties),
        simulator=_first_bool_attr(solver, properties, names=("simulator", "is_simulator"))
        or False,
        max_shots=_dwave_max_reads(solver, properties),
        max_circuits=_first_optional_int_attr(
            properties,
            solver,
            names=("max_problems", "max_circuits", "max_jobs", "max_experiments"),
        ),
        queue_depth=_dwave_queue_depth(solver, properties),
        calibration_timestamp=_first_optional_text_attr(
            properties,
            solver,
            names=(
                "last_update_time",
                "last_updated",
                "lastUpdated",
                "calibration_timestamp",
                "last_calibration",
            ),
        ),
        metadata={
            "adapter": "dwave_solver_no_submit",
            "topology": topology,
            "category": _first_optional_text_attr(properties, solver, names=("category",)),
        },
    )


def snapshot_from_iqm_backend(
    resolved: ResolvedAggregatorProviderRoute,
    backend: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from direct IQM backend metadata."""
    architecture = _first_available_attr(
        backend,
        names=("architecture", "quantum_architecture", "metadata"),
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            backend,
            architecture,
            names=(
                "backend",
                "backend_id",
                "quantum_computer",
                "quantum_computer_name",
                "name",
                "id",
            ),
            field_name="IQM backend name",
        ),
        n_qubits=_first_positive_int_attr(
            backend,
            architecture,
            names=("num_qubits", "n_qubits", "qubits", "qubit_count", "qubitCount"),
            field_name="IQM qubit count",
        ),
        supported_ir_formats=_iqm_supported_ir_formats(resolved, backend),
        basis_gates=_first_string_tuple_attr(
            backend,
            architecture,
            names=(
                "basis_gates",
                "native_gates",
                "gates",
                "operation_names",
                "supported_operations",
            ),
        ),
        native_features=_iqm_native_features(backend, architecture),
        online=_iqm_online_state(backend),
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
            architecture,
            names=(
                "last_calibration",
                "calibration_timestamp",
                "latest_calibration",
                "last_updated",
                "lastUpdated",
            ),
        ),
        metadata={
            "adapter": "iqm_backend_no_submit",
            "architecture": _first_optional_text_attr(
                architecture,
                backend,
                names=("architecture", "architecture_name", "name", "id"),
            ),
        },
    )


def snapshot_from_quera_bloqade(
    resolved: ResolvedAggregatorProviderRoute,
    target: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from direct QuEra/Bloqade metadata."""
    lattice = _first_available_attr(
        target,
        names=("lattice", "atom_lattice", "register", "geometry", "metadata"),
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            target,
            lattice,
            names=("target", "target_name", "backend", "backend_id", "routine", "name", "id"),
            field_name="QuEra/Bloqade target name",
        ),
        n_qubits=_first_positive_int_attr(
            lattice,
            target,
            names=(
                "n_sites",
                "num_atoms",
                "n_atoms",
                "atom_count",
                "num_qubits",
                "n_qubits",
                "qubits",
            ),
            field_name="QuEra/Bloqade atom count",
        ),
        supported_ir_formats=_quera_supported_ir_formats(resolved, target),
        basis_gates=_first_string_tuple_attr(
            target,
            lattice,
            names=("native_operations", "basis_gates", "native_gates", "gates"),
        ),
        native_features=_quera_native_features(target, lattice),
        online=_quera_online_state(target),
        simulator=_first_bool_attr(target, names=("simulator", "is_simulator")) or False,
        max_shots=_first_optional_int_attr(
            target,
            names=("max_shots", "shots_limit", "maxShots", "max_execution_shots"),
        ),
        max_circuits=_first_optional_int_attr(
            target,
            names=("max_circuits", "max_jobs", "max_experiments"),
        ),
        queue_depth=_first_optional_int_attr(
            target,
            names=("queue_depth", "pending_jobs", "queue_size", "average_queue_time"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            target,
            lattice,
            names=(
                "last_calibration",
                "calibration_timestamp",
                "latest_calibration",
                "last_updated",
                "lastUpdated",
            ),
        ),
        metadata={
            "adapter": "quera_bloqade_no_submit",
            "lattice_geometry": _first_optional_text_attr(
                lattice,
                target,
                names=("geometry", "lattice_geometry", "topology"),
            ),
        },
    )


def snapshot_from_oqc_target(
    resolved: ResolvedAggregatorProviderRoute,
    target: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from direct OQC target metadata."""
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            target,
            names=("target", "target_name", "backend", "backend_id", "name", "id"),
            field_name="OQC target name",
        ),
        n_qubits=_first_positive_int_attr(
            target,
            names=("num_qubits", "n_qubits", "qubits", "qubit_count", "qubitCount"),
            field_name="OQC qubit count",
        ),
        supported_ir_formats=_oqc_supported_ir_formats(resolved, target),
        basis_gates=_first_string_tuple_attr(
            target,
            names=("basis_gates", "native_gates", "gates", "operation_names"),
        ),
        native_features=_oqc_native_features(target),
        online=_oqc_online_state(target),
        simulator=_first_bool_attr(target, names=("simulator", "is_simulator")) or False,
        max_shots=_first_optional_int_attr(
            target,
            names=("max_shots", "shots_limit", "maxShots", "max_execution_shots"),
        ),
        max_circuits=_first_optional_int_attr(
            target,
            names=("max_circuits", "max_jobs", "max_experiments"),
        ),
        queue_depth=_first_optional_int_attr(
            target,
            names=("queue_depth", "pending_jobs", "queue_size", "average_queue_time"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            target,
            names=(
                "last_calibration",
                "calibration_timestamp",
                "latest_calibration",
                "last_updated",
                "lastUpdated",
            ),
        ),
        metadata={
            "adapter": "oqc_target_no_submit",
            "topology": _first_optional_text_attr(
                target,
                names=("topology", "coupling_map", "connectivity"),
            ),
        },
    )


def snapshot_from_pasqal_target(
    resolved: ResolvedAggregatorProviderRoute,
    target: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from direct Pasqal target metadata."""
    device_specs = _first_available_attr(
        target,
        names=("device_specs", "device", "device_capabilities", "capabilities", "metadata"),
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            target,
            device_specs,
            names=("target", "target_name", "backend", "backend_id", "name", "id"),
            field_name="Pasqal target name",
        ),
        n_qubits=_first_positive_int_attr(
            device_specs,
            target,
            names=(
                "max_atom_num",
                "max_atoms",
                "num_atoms",
                "n_atoms",
                "atom_count",
                "num_qubits",
                "n_qubits",
                "qubits",
            ),
            field_name="Pasqal atom count",
        ),
        supported_ir_formats=_pasqal_supported_ir_formats(resolved, target),
        basis_gates=_first_string_tuple_attr(
            device_specs,
            target,
            names=(
                "supported_bases",
                "basis_gates",
                "native_gates",
                "gates",
                "supported_operations",
            ),
        ),
        native_features=_pasqal_native_features(target, device_specs),
        online=_pasqal_online_state(target),
        simulator=_first_bool_attr(target, device_specs, names=("simulator", "is_simulator"))
        or False,
        max_shots=_first_optional_int_attr(
            device_specs,
            target,
            names=("max_runs", "max_shots", "shots_limit", "maxShots", "max_execution_shots"),
        ),
        max_circuits=_first_optional_int_attr(
            device_specs,
            target,
            names=("max_sequence_count", "max_circuits", "max_jobs", "max_experiments"),
        ),
        queue_depth=_first_optional_int_attr(
            target,
            device_specs,
            names=("queue_depth", "pending_jobs", "queue_size", "average_queue_time"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            target,
            device_specs,
            names=(
                "last_calibration",
                "calibration_timestamp",
                "latest_calibration",
                "last_updated",
                "lastUpdated",
            ),
        ),
        metadata={
            "adapter": "pasqal_target_no_submit",
            "channels": _first_string_tuple_attr(
                device_specs,
                target,
                names=("channels", "supported_channels", "rydberg_channels"),
            ),
            "lattice_geometry": _first_optional_text_attr(
                target,
                device_specs,
                names=("lattice_geometry", "geometry", "topology", "register_geometry"),
            ),
        },
    )


def snapshot_from_quandela_processor(
    resolved: ResolvedAggregatorProviderRoute,
    processor: Any,
) -> ProviderCapabilitySnapshot:
    """Build a no-submit capability snapshot from direct Quandela processor metadata."""
    specs = _first_available_attr(
        processor,
        names=("specs", "specification", "capabilities", "metadata"),
    )
    return ProviderCapabilitySnapshot(
        route_id=resolved.route.route_id,
        aggregator=resolved.route.aggregator,
        provider=resolved.route.provider,
        backend_id=resolved.route.backend_id,
        target_name=_first_text_attr(
            processor,
            specs,
            names=("processor", "processor_name", "target", "target_name", "name", "id"),
            field_name="Quandela processor name",
        ),
        n_qubits=_first_positive_int_attr(
            specs,
            processor,
            names=("modes", "n_modes", "num_modes", "n_qubits", "num_qubits", "qubits"),
            field_name="Quandela mode count",
        ),
        supported_ir_formats=_quandela_supported_ir_formats(resolved, processor, specs),
        basis_gates=_first_string_tuple_attr(
            specs,
            processor,
            names=("components", "supported_components", "basis_gates", "native_gates", "gates"),
        ),
        native_features=_quandela_native_features(processor, specs),
        online=_quandela_online_state(processor, specs),
        simulator=_first_bool_attr(processor, specs, names=("simulator", "is_simulator")) or False,
        max_shots=_first_optional_int_attr(
            specs,
            processor,
            names=("max_shots", "shots_limit", "maxShots", "max_execution_shots"),
        ),
        max_circuits=_first_optional_int_attr(
            specs,
            processor,
            names=("max_circuits", "max_jobs", "max_experiments", "max_programs"),
        ),
        queue_depth=_first_optional_int_attr(
            processor,
            specs,
            names=("queue_depth", "pending_jobs", "queue_size", "average_queue_time"),
            minimum=0,
        ),
        calibration_timestamp=_first_optional_text_attr(
            processor,
            specs,
            names=(
                "last_calibration",
                "calibration_timestamp",
                "latest_calibration",
                "last_updated",
                "lastUpdated",
            ),
        ),
        metadata={
            "adapter": "quandela_processor_no_submit",
            "topology": _first_optional_text_attr(
                specs,
                processor,
                names=("topology", "layout", "chip_topology"),
            ),
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
    openpulse_profile = _qiskit_openpulse_profile(configuration, target, backend)
    native_features = set(_qiskit_native_features(backend, target, basis_gates))
    if openpulse_profile["supports_pulse_control"]:
        native_features.add("pulse_control")
    if openpulse_profile["supports_drive_channel_access"]:
        native_features.add("drive_channel_access")
    if openpulse_profile["supports_measure_channel_access"]:
        native_features.add("measure_channel_access")

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
        native_features=tuple(sorted(native_features)),
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
            "openpulse_profile": openpulse_profile,
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
    supported_ir_formats = _qbraid_supported_ir_formats(profile, device)
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
        native_features=_qbraid_native_features(
            profile,
            device,
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
            "broker_route": resolved.route.route_id,
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
        names=("id", "backend_id", "device_id", "resource_id", "target_id", "name"),
        field_name="Strangeworks target name",
    )
    n_qubits = _first_positive_int_attr(
        backend,
        names=("n_qubits", "num_qubits", "qubits"),
        field_name="Strangeworks qubit count",
    )
    supported_ir_formats = _strangeworks_supported_ir_formats(
        backend,
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
        native_features=_strangeworks_native_features(backend),
        online=_strangeworks_online_state(backend),
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
        metadata={
            "adapter": "strangeworks_backend_no_submit",
            "broker_route": resolved.route.route_id,
            "provider_name": _first_optional_text_attr(
                backend,
                names=("provider_name", "provider", "vendor", "vendor_name"),
            ),
        },
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


def _dwave_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    *sources: Any,
) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        *sources,
        names=(
            "supported_problem_types",
            "supported_ir_formats",
            "ir_formats",
            "input_formats",
            "problem_types",
            "program_formats",
        ),
    )
    if not declared:
        return resolved.route.ir_formats
    return tuple(_dwave_ir_format_token(item) for item in declared)


def _dwave_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"bqm", "binary_quadratic_model"}:
        return "bqm"
    if normalized in {"ising", "ising_model"}:
        return "ising"
    if normalized in {"qubo", "quadratic_unconstrained_binary_optimisation"}:
        return "qubo"
    if normalized == "mlir":
        return "mlir"
    return normalized


def _dwave_native_features(topology: str | None, *sources: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(*sources, names=("native_features", "features")))
    features.add("quantum_annealing")
    features.add("bqm")
    if topology:
        features.add(f"{topology}_topology")
    return tuple(sorted(features))


def _dwave_online_state(*sources: Any) -> bool | None:
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


def _dwave_max_reads(*sources: Any) -> int | None:
    direct = _first_optional_int_attr(
        *sources,
        names=("max_reads", "num_reads", "max_shots", "shots_limit", "maxShots"),
    )
    if direct is not None:
        return direct
    parameters = _first_available_attr(*sources, names=("parameters", "parameter_ranges"))
    if isinstance(parameters, Mapping):
        reads_range = parameters.get("num_reads") or parameters.get("reads")
        if isinstance(reads_range, tuple | list):
            positive_ints = [
                item
                for item in reads_range
                if isinstance(item, int) and not isinstance(item, bool) and item > 0
            ]
            if positive_ints:
                return max(positive_ints)
    return None


def _dwave_queue_depth(*sources: Any) -> int | None:
    direct = _first_optional_int_attr(
        *sources,
        names=("queue_depth", "pending_jobs", "queue_size"),
        minimum=0,
    )
    if direct is not None:
        return direct
    for value in _attr_candidates(*sources, names=("avg_load", "average_load", "load")):
        if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
            return int(round(float(value) * 100))
    return None


def _dwave_topology_name(*sources: Any) -> str | None:
    text = _first_optional_text_attr(
        *sources,
        names=("topology_type", "topology", "graph_family", "family"),
    )
    if text is not None:
        return text.strip().lower().replace("-", "_")
    for value in _attr_candidates(*sources, names=("topology",)):
        if isinstance(value, Mapping):
            topology_type = value.get("type")
            if isinstance(topology_type, str) and topology_type.strip():
                return topology_type.strip().lower().replace("-", "_")
    return None


def _iqm_supported_ir_formats(
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
    return tuple(_iqm_ir_format_token(item) for item in declared)


def _iqm_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"qiskit_qpy", "qpy", "qiskit.qpy"}:
        return "qiskit_qpy"
    if normalized in {"qiskit", "qiskit_circuit", "quantum_circuit"}:
        return "qiskit"
    if normalized in {"circuit", "iqm_circuit", "iqm.circuit"}:
        return "circuit"
    if normalized in {"qasm.v3", "qasm3", "openqasm", "openqasm_3", "openqasm3"}:
        return "openqasm3"
    if normalized == "mlir":
        return "mlir"
    return normalized


def _iqm_native_features(*sources: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(*sources, names=("native_features", "features")))
    features.add("gate_model")
    features.add("superconducting")
    if _first_bool_attr(*sources, names=("lattice_connectivity", "topology")) is not False:
        features.add("lattice_connectivity")
    return tuple(sorted(features))


def _iqm_online_state(backend: Any) -> bool | None:
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


def _quera_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    target: Any,
) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        target,
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
    return tuple(_quera_ir_format_token(item) for item in declared)


def _quera_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"bloqade", "bloqade_ahs", "bloqade_ahs_plan_v1"}:
        return "bloqade"
    if normalized in {"braket.ahs", "braket_ahs", "braket.ir.ahs.program"}:
        return "braket_ahs"
    if normalized == "mlir":
        return "mlir"
    return normalized


def _quera_native_features(*sources: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(*sources, names=("native_features", "features")))
    features.add("neutral_atom")
    features.add("analog_hamiltonian")
    return tuple(sorted(features))


def _quera_online_state(target: Any) -> bool | None:
    explicit = _first_online_attr(target)
    if explicit is not None:
        return explicit
    text_status = _first_optional_text_attr(
        target,
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


def _oqc_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    target: Any,
) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        target,
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
    return tuple(_oqc_ir_format_token(item) for item in declared)


def _oqc_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"qasm.v3", "qasm3", "openqasm", "openqasm_3", "openqasm3"}:
        return "openqasm3"
    if normalized in {"qir", "qir.v1"}:
        return "qir"
    if normalized == "mlir":
        return "mlir"
    return normalized


def _oqc_native_features(target: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(target, names=("native_features", "features")))
    features.add("gate_model")
    features.add("superconducting")
    if _first_bool_attr(target, names=("lattice_connectivity", "topology")) is not False:
        features.add("lattice_connectivity")
    return tuple(sorted(features))


def _oqc_online_state(target: Any) -> bool | None:
    explicit = _first_online_attr(target)
    if explicit is not None:
        return explicit
    text_status = _first_optional_text_attr(
        target,
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


def _pasqal_supported_ir_formats(
    resolved: ResolvedAggregatorProviderRoute,
    target: Any,
) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        target,
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
    return tuple(_pasqal_ir_format_token(item) for item in declared)


def _pasqal_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"pulser", "pulser_sequence", "pulser_sequence_plan_v1"}:
        return "pulser"
    if normalized in {"pasqal_ir", "pasqal.ir"}:
        return "pasqal_ir"
    if normalized in {"qasm.v3", "qasm3", "openqasm", "openqasm_3", "openqasm3"}:
        return "openqasm3"
    if normalized == "mlir":
        return "mlir"
    return normalized


def _pasqal_native_features(*sources: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(*sources, names=("native_features", "features")))
    features.add("neutral_atom")
    features.add("analog_hamiltonian")
    features.add("rydberg")
    if _first_bool_attr(*sources, names=("digital_mode", "supports_digital")):
        features.add("digital_analog")
    return tuple(sorted(features))


def _pasqal_online_state(target: Any) -> bool | None:
    explicit = _first_online_attr(target)
    if explicit is not None:
        return explicit
    text_status = _first_optional_text_attr(
        target,
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


def _quandela_supported_ir_formats(
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
    return tuple(_quandela_ir_format_token(item) for item in declared)


def _quandela_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"perceval", "perceval_circuit", "perceval_processor"}:
        return "perceval"
    if normalized in {"qasm.v3", "qasm3", "openqasm", "openqasm_3", "openqasm3"}:
        return "openqasm3"
    if normalized == "mlir":
        return "mlir"
    return normalized


def _quandela_native_features(*sources: Any) -> tuple[str, ...]:
    features = set(_first_string_tuple_attr(*sources, names=("native_features", "features")))
    features.add("photonic")
    features.add("linear_optical")
    if _first_bool_attr(
        *sources,
        names=("photon_number_resolution", "supports_photon_number_resolution"),
    ):
        features.add("photon_number_resolution")
    return tuple(sorted(features))


def _quandela_online_state(*sources: Any) -> bool | None:
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


def _qbraid_supported_ir_formats(*sources: Any) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        *sources,
        names=(
            "supported_ir_formats",
            "ir_formats",
            "input_formats",
            "program_formats",
            "supported_program_formats",
            "program_specs",
        ),
    )
    if not declared:
        raise ValueError("qBraid IR formats must declare supported IR formats")
    return tuple(_broker_ir_format_token(item) for item in declared)


def _strangeworks_supported_ir_formats(*sources: Any) -> tuple[str, ...]:
    declared = _first_string_tuple_attr(
        *sources,
        names=(
            "input_formats",
            "supported_ir_formats",
            "ir_formats",
            "program_formats",
            "supported_program_formats",
            "available_programs",
            "program_specs",
            "programs",
            "program_types",
            "supported_programs",
        ),
    )
    if not declared:
        raise ValueError("Strangeworks IR formats must declare supported IR formats")
    return tuple(_broker_ir_format_token(item) for item in declared)


def _broker_ir_format_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "")
    if normalized in {
        "qasm3",
        "qasm.v3",
        "openqasm",
        "openqasm_3",
        "openqasm3",
        "braket.ir.openqasm.program",
    }:
        return "openqasm3"
    if normalized in {"qiskit", "qiskit.quantumcircuit", "quantum_circuit"}:
        return "qiskit"
    if normalized in {"cirq", "cirq.circuit"}:
        return "cirq"
    if normalized in {"quil", "pyquil", "pyquil.program"}:
        return "quil"
    if normalized in {"braket_ir", "braket.ir", "braket.circuit"}:
        return "braket_ir"
    if normalized in {"pennylane", "pennylane.tape", "quantum_tape"}:
        return "pennylane"
    if normalized in {"pyqubo", "qubo"}:
        return "pyqubo"
    if normalized in {"tket", "pytket", "pytket.circuit"}:
        return "tket"
    if normalized in {"qir", "qir.v1"}:
        return "qir"
    if normalized == "mlir":
        return "mlir"
    return normalized


def _qbraid_native_features(*sources: Any) -> tuple[str, ...]:
    features = set(
        _first_string_tuple_attr(*sources, names=("native_features", "features", "capabilities"))
    )
    features.add("broker_catalog_target")
    return tuple(sorted(features))


def _strangeworks_native_features(*sources: Any) -> tuple[str, ...]:
    features = set(
        _first_string_tuple_attr(*sources, names=("native_features", "features", "capabilities"))
    )
    features.add("broker_catalog_target")
    return tuple(sorted(features))


def _strangeworks_online_state(*sources: Any) -> bool | None:
    text_state = _first_optional_text_attr(
        *sources,
        names=("online", "is_online", "available", "status", "state", "availability"),
    )
    if text_state is not None:
        return _online_state_from_text(text_state)
    return _first_online_attr(*sources)


def _qiskit_calibration_timestamp(*sources: Any) -> str | None:
    for value in _attr_candidates(
        *sources,
        names=("last_update_date", "calibration_timestamp", "last_calibration", "calibrated_at"),
    ):
        normalized = normalize_calibration_timestamp(value)
        if normalized is not None:
            return normalized
    return None


def normalize_calibration_timestamp(value: Any) -> str | None:
    """Normalise provider calibration timestamp metadata to RFC3339 UTC where possible."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, datetime):
        dt = (
            value.astimezone(timezone.utc)
            if value.tzinfo is not None
            else value.replace(tzinfo=timezone.utc)
        )
        return dt.isoformat().replace("+00:00", "Z")
    isoformat = _optional_attr(value, "isoformat")
    if callable(isoformat):
        try:
            iso = isoformat()
        except Exception:
            return None
        if isinstance(iso, str):
            stripped = iso.strip()
            return stripped if stripped else None
    return None


def _qiskit_openpulse_profile(*sources: Any) -> dict[str, Any]:
    n_qubits = _first_optional_int_attr(*sources, names=("num_qubits", "n_qubits", "qubits"))
    coupling_map = _first_coupling_map(*sources)
    n_control_channels = _first_optional_int_attr(
        *sources, names=("n_uchannels", "num_u_channels", "n_control_channels")
    )
    supports_drive = bool(n_qubits and n_qubits > 0)
    supports_measure = bool(_first_optional_attr(*sources, names=("meas_map", "measure_map")))
    supports_control = bool((n_control_channels and n_control_channels > 0) or coupling_map)

    channel_map: dict[str, dict[str, Any]] = {}
    if n_qubits and n_qubits > 0:
        for qubit in range(n_qubits):
            neighbours = sorted(
                {
                    edge[1]
                    for edge in coupling_map
                    if len(edge) == 2 and edge[0] == qubit and isinstance(edge[1], int)
                }
            )
            channel_map[f"q{qubit}"] = {
                "drive": f"d{qubit}",
                "measure": f"m{qubit}" if supports_measure else None,
                "control_neighbours": neighbours,
            }

    return {
        "supports_pulse_control": supports_drive or supports_control,
        "supports_drive_channel_access": supports_drive,
        "supports_measure_channel_access": supports_measure,
        "supports_control_channel_access": supports_control,
        "n_control_channels": int(n_control_channels or 0),
        "channel_map": channel_map,
    }


def _first_optional_attr(*sources: Any, names: tuple[str, ...]) -> Any:
    for value in _attr_candidates(*sources, names=names):
        if value is not None:
            return value
    return None


def _first_coupling_map(*sources: Any) -> tuple[tuple[int, int], ...]:
    raw = _first_optional_attr(
        *sources,
        names=("coupling_map", "couplingMap", "edge_list"),
    )
    if raw is None:
        return ()
    pairs: list[tuple[int, int]] = []
    if isinstance(raw, Mapping):
        iterator = raw.keys()
    else:
        iterator = raw
    try:
        for item in iterator:
            if isinstance(item, Sequence) and len(item) == 2:
                a = item[0]
                b = item[1]
                if isinstance(a, int) and isinstance(b, int):
                    pairs.append((a, b))
    except TypeError:
        return ()
    return tuple(pairs)


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
    "OpenPulseControlReadiness",
    "ProviderCapabilityDecision",
    "ProviderCapabilitySnapshot",
    "ProviderMetadataProbe",
    "assess_provider_capability_snapshot",
    "build_openpulse_control_readiness",
    "probe_aggregator_provider_capability",
    "normalize_calibration_timestamp",
    "snapshot_from_azure_target",
    "snapshot_from_braket_device",
    "snapshot_from_dwave_solver",
    "snapshot_from_iqm_backend",
    "snapshot_from_ionq_backend",
    "snapshot_from_oqc_target",
    "snapshot_from_pasqal_target",
    "snapshot_from_quandela_processor",
    "snapshot_from_qiskit_runtime_backend",
    "snapshot_from_qbraid_device",
    "snapshot_from_quantinuum_backend",
    "snapshot_from_quera_bloqade",
    "snapshot_from_rigetti_qcs",
    "snapshot_from_strangeworks_backend",
]
