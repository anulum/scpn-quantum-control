# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Direct Gate Provider Capability Adapters
"""No-submit metadata adapters for direct gate-model provider routes."""

from __future__ import annotations

from typing import Any

from .aggregators import ResolvedAggregatorProviderRoute
from .provider_capability_core import ProviderCapabilitySnapshot
from .provider_capability_normalization import (
    _first_available_attr,
    _first_bool_attr,
    _first_online_attr,
    _first_optional_int_attr,
    _first_optional_text_attr,
    _first_positive_int_attr,
    _first_string_tuple_attr,
    _first_text_attr,
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


__all__ = [
    "snapshot_from_ionq_backend",
    "snapshot_from_iqm_backend",
    "snapshot_from_oqc_target",
    "snapshot_from_quantinuum_backend",
    "snapshot_from_rigetti_qcs",
]
