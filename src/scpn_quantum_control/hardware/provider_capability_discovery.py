# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Capability Discovery
"""No-submit provider metadata adapters and compatibility facade.

Provider-neutral capability contracts, route assessment, and OpenPulse
readiness live in :mod:`.provider_capability_core`. This module retains the
provider-specific metadata adapters and re-exports the exact core objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .aggregators import ResolvedAggregatorProviderRoute
from .provider_capability_core import (
    CapabilityDecisionStatus as CapabilityDecisionStatus,
)
from .provider_capability_core import (
    OpenPulseControlReadiness as OpenPulseControlReadiness,
)
from .provider_capability_core import (
    ProviderCapabilityDecision as ProviderCapabilityDecision,
)
from .provider_capability_core import (
    ProviderCapabilitySnapshot as ProviderCapabilitySnapshot,
)
from .provider_capability_core import (
    ProviderMetadataProbe as ProviderMetadataProbe,
)
from .provider_capability_core import (
    _require_string_tuple as _require_string_tuple,
)
from .provider_capability_core import (
    _require_text as _require_text,
)
from .provider_capability_core import (
    assess_provider_capability_snapshot as assess_provider_capability_snapshot,
)
from .provider_capability_core import (
    build_openpulse_control_readiness as build_openpulse_control_readiness,
)
from .provider_capability_core import (
    probe_aggregator_provider_capability as probe_aggregator_provider_capability,
)
from .provider_capability_gate_adapters import (
    _ionq_ir_format_token as _ionq_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _ionq_native_features as _ionq_native_features,
)
from .provider_capability_gate_adapters import (
    _ionq_online_state as _ionq_online_state,
)
from .provider_capability_gate_adapters import (
    _ionq_supported_ir_formats as _ionq_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    _iqm_ir_format_token as _iqm_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _iqm_native_features as _iqm_native_features,
)
from .provider_capability_gate_adapters import (
    _iqm_online_state as _iqm_online_state,
)
from .provider_capability_gate_adapters import (
    _iqm_supported_ir_formats as _iqm_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    _oqc_ir_format_token as _oqc_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _oqc_native_features as _oqc_native_features,
)
from .provider_capability_gate_adapters import (
    _oqc_online_state as _oqc_online_state,
)
from .provider_capability_gate_adapters import (
    _oqc_supported_ir_formats as _oqc_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    _quantinuum_ir_format_token as _quantinuum_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _quantinuum_native_features as _quantinuum_native_features,
)
from .provider_capability_gate_adapters import (
    _quantinuum_online_state as _quantinuum_online_state,
)
from .provider_capability_gate_adapters import (
    _quantinuum_supported_ir_formats as _quantinuum_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    _rigetti_ir_format_token as _rigetti_ir_format_token,
)
from .provider_capability_gate_adapters import (
    _rigetti_native_features as _rigetti_native_features,
)
from .provider_capability_gate_adapters import (
    _rigetti_online_state as _rigetti_online_state,
)
from .provider_capability_gate_adapters import (
    _rigetti_supported_ir_formats as _rigetti_supported_ir_formats,
)
from .provider_capability_gate_adapters import (
    snapshot_from_ionq_backend as snapshot_from_ionq_backend,
)
from .provider_capability_gate_adapters import (
    snapshot_from_iqm_backend as snapshot_from_iqm_backend,
)
from .provider_capability_gate_adapters import (
    snapshot_from_oqc_target as snapshot_from_oqc_target,
)
from .provider_capability_gate_adapters import (
    snapshot_from_quantinuum_backend as snapshot_from_quantinuum_backend,
)
from .provider_capability_gate_adapters import (
    snapshot_from_rigetti_qcs as snapshot_from_rigetti_qcs,
)
from .provider_capability_normalization import (
    _attr_candidates as _attr_candidates,
)
from .provider_capability_normalization import (
    _declared_ir_formats as _declared_ir_formats,
)
from .provider_capability_normalization import (
    _first_available_attr as _first_available_attr,
)
from .provider_capability_normalization import (
    _first_bool_attr as _first_bool_attr,
)
from .provider_capability_normalization import (
    _first_online_attr as _first_online_attr,
)
from .provider_capability_normalization import (
    _first_optional_int_attr as _first_optional_int_attr,
)
from .provider_capability_normalization import (
    _first_optional_text_attr as _first_optional_text_attr,
)
from .provider_capability_normalization import (
    _first_positive_int_attr as _first_positive_int_attr,
)
from .provider_capability_normalization import (
    _first_string_tuple_attr as _first_string_tuple_attr,
)
from .provider_capability_normalization import (
    _first_text_attr as _first_text_attr,
)
from .provider_capability_normalization import (
    _online_state_from_text as _online_state_from_text,
)
from .provider_capability_normalization import (
    _optional_attr as _optional_attr,
)
from .provider_capability_normalization import (
    _optional_noarg_call as _optional_noarg_call,
)
from .provider_capability_normalization import (
    _program_spec_name as _program_spec_name,
)
from .provider_capability_normalization import (
    _string_tuple_from_value as _string_tuple_from_value,
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
