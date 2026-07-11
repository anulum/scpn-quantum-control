# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Specialized Provider Capability Adapters
"""No-submit metadata adapters for specialized quantum provider routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .aggregators import ResolvedAggregatorProviderRoute
from .provider_capability_core import ProviderCapabilitySnapshot
from .provider_capability_normalization import (
    _attr_candidates,
    _first_available_attr,
    _first_bool_attr,
    _first_online_attr,
    _first_optional_int_attr,
    _first_optional_text_attr,
    _first_positive_int_attr,
    _first_string_tuple_attr,
    _first_text_attr,
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


__all__ = [
    "snapshot_from_dwave_solver",
    "snapshot_from_quera_bloqade",
    "snapshot_from_pasqal_target",
    "snapshot_from_quandela_processor",
]
