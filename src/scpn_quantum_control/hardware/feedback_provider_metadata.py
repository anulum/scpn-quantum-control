# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Feedback provider metadata adapters
"""Provider metadata adapters for S1 no-submit capability probes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .feedback_capability_probe import BackendCapabilitySnapshot


def snapshot_from_generic_metadata(metadata: Mapping[str, Any]) -> BackendCapabilitySnapshot:
    """Build an S1 capability snapshot from provider-neutral metadata."""
    provider = _required_text(metadata, "provider")
    backend_name = _required_text(metadata, "backend_name")
    n_qubits = _required_int(metadata, "n_qubits")
    return BackendCapabilitySnapshot(
        provider=provider,
        backend_name=backend_name,
        n_qubits=n_qubits,
        basis_gates=_string_tuple(metadata.get("basis_gates", ())),
        supported_features=_string_tuple(metadata.get("supported_features", ())),
        max_shots=_optional_int(metadata.get("max_shots"), "max_shots"),
        max_circuits=_optional_int(metadata.get("max_circuits"), "max_circuits"),
        simulator=bool(metadata.get("simulator", False)),
        metadata=dict(metadata.get("metadata", {}))
        if isinstance(metadata.get("metadata", {}), Mapping)
        else {},
    )


def snapshot_from_qiskit_backend(
    backend: Any,
    *,
    provider: str = "ibm",
) -> BackendCapabilitySnapshot:
    """Build an S1 snapshot from a Qiskit-style backend object without submitting jobs."""
    name = _backend_name(backend)
    n_qubits = _backend_num_qubits(backend)
    basis_gates = _backend_basis_gates(backend)
    max_shots = _backend_max_shots(backend)
    max_circuits = _backend_max_circuits(backend)
    simulator = bool(getattr(backend, "simulator", False))
    supported_features = _infer_qiskit_dynamic_features(backend, basis_gates)
    return BackendCapabilitySnapshot(
        provider=provider,
        backend_name=name,
        n_qubits=n_qubits,
        basis_gates=basis_gates,
        supported_features=supported_features,
        max_shots=max_shots,
        max_circuits=max_circuits,
        simulator=simulator,
        metadata={"adapter": "qiskit_backend_no_submit"},
    )


def _infer_qiskit_dynamic_features(backend: Any, basis_gates: tuple[str, ...]) -> tuple[str, ...]:
    features = {"cross_shot_batches"}
    basis = set(basis_gates)
    if "measure" in basis or hasattr(backend, "target"):
        features.add("mid_circuit_measurement")
    if "reset" in basis:
        features.add("conditional_reset")
    target = getattr(backend, "target", None)
    if _target_has_control_flow(target) or getattr(backend, "dynamic_circuits", False):
        features.add("conditional_control")
    return tuple(sorted(features))


def _target_has_control_flow(target: Any) -> bool:
    if target is None:
        return False
    try:
        operation_names = set(target.operation_names)
    except (AttributeError, TypeError):
        return False
    return bool(operation_names.intersection({"if_else", "while_loop", "for_loop", "switch_case"}))


def _backend_name(backend: Any) -> str:
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        value = name_attr()
    else:
        value = name_attr
    if not isinstance(value, str) or not value:
        raise ValueError("backend name must be available")
    return value


def _backend_num_qubits(backend: Any) -> int:
    value = getattr(backend, "num_qubits", None)
    if value is None:
        configuration = _optional_configuration(backend)
        value = getattr(configuration, "num_qubits", None) if configuration is not None else None
    if not isinstance(value, int) or value < 1:
        raise ValueError("backend num_qubits must be available")
    return value


def _backend_basis_gates(backend: Any) -> tuple[str, ...]:
    configuration = _optional_configuration(backend)
    if configuration is not None:
        basis = getattr(configuration, "basis_gates", None)
        if basis is not None:
            return _string_tuple(basis)
    target = getattr(backend, "target", None)
    if target is not None:
        try:
            return _string_tuple(target.operation_names)
        except (AttributeError, TypeError):
            return ()
    return ()


def _backend_max_shots(backend: Any) -> int | None:
    configuration = _optional_configuration(backend)
    if configuration is None:
        return None
    return _optional_int(getattr(configuration, "max_shots", None), "max_shots")


def _backend_max_circuits(backend: Any) -> int | None:
    configuration = _optional_configuration(backend)
    if configuration is None:
        return None
    return _optional_int(getattr(configuration, "max_experiments", None), "max_circuits")


def _optional_configuration(backend: Any) -> Any:
    configuration = getattr(backend, "configuration", None)
    if callable(configuration):
        return configuration()
    return configuration


def _required_text(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _required_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer when provided")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, tuple | list | set | frozenset):
        raise ValueError("expected a sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ValueError("string sequences must not contain empty entries")
    return result
