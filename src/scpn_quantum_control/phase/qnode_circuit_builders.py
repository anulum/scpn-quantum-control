# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Builders
"""Registered Phase-QNode vocabulary, observables, decompositions, and templates.

This one-way construction leaf depends only on the shared circuit contracts. It
contains no support-analysis, execution, gradient, measurement, framework, or
provider orchestration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .qnode_circuit_contracts import (
    _REGISTERED_DECOMPOSITIONS,
    _REGISTERED_GATES,
    _REGISTERED_NOISE_CHANNELS,
    _REGISTERED_OBSERVABLES,
    _REGISTERED_TEMPLATES,
    DenseHermitianObservable,
    OperationSpec,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeOperation,
    PhaseQNodeTemplateSpec,
    SparsePauliHamiltonian,
    _normalise_observable,
    _parse_operation,
)


def registered_phase_qnode_gates() -> tuple[str, ...]:
    """Return the local Phase-QNode gate family."""
    return _REGISTERED_GATES


def registered_phase_qnode_observables() -> tuple[str, ...]:
    """Return the local Phase-QNode observable family."""
    return _REGISTERED_OBSERVABLES


def registered_phase_qnode_templates() -> tuple[str, ...]:
    """Return the registered local multi-qubit Phase-QNode templates."""
    return _REGISTERED_TEMPLATES


def registered_phase_qnode_decompositions() -> tuple[str, ...]:
    """Return gates with exact registered operation-list decompositions."""
    return _REGISTERED_DECOMPOSITIONS


def registered_phase_qnode_noise_channels() -> tuple[str, ...]:
    """Return the registered local density-matrix noise channels."""
    return _REGISTERED_NOISE_CHANNELS


def build_sparse_ising_chain_hamiltonian(
    n_qubits: int,
    *,
    x_field: float | ArrayLike = 0.0,
    z_field: float | ArrayLike = 0.0,
    zz_coupling: float | ArrayLike = 1.0,
    periodic: bool = False,
) -> SparsePauliHamiltonian:
    """Build a sparse nearest-neighbour Ising-chain Pauli Hamiltonian.

    Coefficients are direct observable weights. Scalar values are broadcast
    across all sites or edges; arrays must match the site or edge count exactly.
    """
    width = _as_ising_chain_width(n_qubits)
    if not isinstance(periodic, bool):
        raise ValueError("periodic must be a boolean")
    edge_count = width if periodic else width - 1
    x_coefficients = _as_broadcast_coefficients("x_field", x_field, width)
    z_coefficients = _as_broadcast_coefficients("z_field", z_field, width)
    zz_coefficients = _as_broadcast_coefficients("zz_coupling", zz_coupling, edge_count)
    terms: list[PauliTerm] = []
    for qubit, coefficient in enumerate(x_coefficients):
        if coefficient != 0.0:
            terms.append(PauliTerm(coefficient, ((qubit, "x"),)))
    for qubit, coefficient in enumerate(z_coefficients):
        if coefficient != 0.0:
            terms.append(PauliTerm(coefficient, ((qubit, "z"),)))
    for edge, coefficient in enumerate(zz_coefficients):
        if coefficient != 0.0:
            left = edge
            right = (edge + 1) % width
            terms.append(PauliTerm(coefficient, ((left, "z"), (right, "z"))))
    if not terms:
        raise ValueError("sparse Ising Hamiltonian must contain at least one non-zero term")
    return SparsePauliHamiltonian(tuple(terms))


def decompose_phase_qnode_controlled_gate(
    operation: PhaseQNodeOperation | OperationSpec,
) -> tuple[PhaseQNodeOperation, ...]:
    """Return an exact registered decomposition for supported controlled gates."""
    parsed = _parse_operation(operation)
    if parsed.parameter_index is not None:
        raise ValueError("controlled-gate decompositions do not accept trainable parameters")
    if parsed.gate == "ccnot":
        _require_qubit_width(parsed, 3)
        control_a, control_b, target = parsed.qubits
        return (
            PhaseQNodeOperation("h", (target,)),
            PhaseQNodeOperation("ccz", (control_a, control_b, target)),
            PhaseQNodeOperation("h", (target,)),
        )
    if parsed.gate == "cswap":
        _require_qubit_width(parsed, 3)
        control, target_a, target_b = parsed.qubits
        return (
            PhaseQNodeOperation("cnot", (target_a, target_b)),
            PhaseQNodeOperation("ccnot", (control, target_b, target_a)),
            PhaseQNodeOperation("cnot", (target_a, target_b)),
        )
    raise ValueError(
        "no registered operation-list decomposition for gate "
        f"{parsed.gate!r}; use registered_phase_qnode_decompositions()"
    )


def build_phase_qnode_template(
    name: str,
    n_qubits: int,
    *,
    n_layers: int = 1,
    entangler: str = "chain",
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
        | None
    ) = None,
) -> PhaseQNodeTemplateSpec:
    """Build a registered multi-qubit template as an executable circuit spec.

    The templates are deterministic local statevector declarations. They do not
    imply hardware execution, dynamic circuits, finite-shot sampling, or native
    framework autodiff-through-simulator support.
    """
    normalized = str(name).strip().lower()
    if normalized not in _REGISTERED_TEMPLATES:
        raise ValueError(
            "unsupported Phase-QNode template; use registered_phase_qnode_templates()"
        )
    width = _as_template_width(n_qubits)
    layers = _as_template_layers(n_layers)
    topology = _as_template_entangler(entangler, width)
    if normalized == "ghz_chain" and topology != "chain":
        raise ValueError("ghz_chain template only supports chain entanglement")
    parsed_observable = _normalise_template_observable(observable, width)
    if normalized == "ghz_chain":
        operations = _ghz_chain_operations(width)
        parameter_count = 0
        effective_layers = 1
    else:
        rotation_gates = ("ry",) if normalized == "hardware_efficient_ry" else ("ry", "rz")
        operations, parameter_count = _hardware_efficient_operations(
            width,
            layers,
            topology,
            rotation_gates,
        )
        effective_layers = layers
    return PhaseQNodeTemplateSpec(
        name=normalized,
        n_qubits=width,
        n_layers=effective_layers,
        entangler=topology,
        parameter_count=parameter_count,
        operations=operations,
        observable=parsed_observable,
        claim_boundary=(
            "registered local multi-qubit Phase-QNode template over the bounded "
            "statevector gate family; no dynamic-circuit, provider, finite-shot, "
            "hardware, or native framework autodiff-through-simulator claim"
        ),
    )


def _as_template_width(value: int) -> int:
    if isinstance(value, bool) or value < 2:
        raise ValueError("Phase-QNode templates require at least two qubits")
    return int(value)


def _as_ising_chain_width(value: int) -> int:
    if isinstance(value, bool) or value < 2:
        raise ValueError("sparse Ising chain Hamiltonians require at least two qubits")
    return int(value)


def _as_template_layers(value: int) -> int:
    if isinstance(value, bool) or value < 1:
        raise ValueError("n_layers must be a positive integer")
    return int(value)


def _as_template_entangler(value: str, n_qubits: int) -> str:
    topology = str(value).strip().lower()
    if topology not in {"chain", "ring"}:
        raise ValueError("entangler must be 'chain' or 'ring'")
    if topology == "ring" and n_qubits < 3:
        raise ValueError("ring entanglement requires at least three qubits")
    return topology


def _as_broadcast_coefficients(name: str, value: object, width: int) -> tuple[float, ...]:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must contain finite real numeric values")
    values = np.asarray(value, dtype=np.float64)
    if values.shape == ():
        scalar = float(values.item())
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must contain finite real numeric values")
        return tuple(scalar for _ in range(width))
    if values.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array")
    if values.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite real numeric values")
    return tuple(float(item) for item in values)


def _normalise_template_observable(
    observable: (
        str
        | PauliTerm
        | SparsePauliHamiltonian
        | PauliCovarianceObservable
        | DenseHermitianObservable
        | None
    ),
    n_qubits: int,
) -> PauliTerm | SparsePauliHamiltonian | PauliCovarianceObservable | DenseHermitianObservable:
    if observable is None:
        return _z_magnetization_observable(n_qubits)
    if isinstance(observable, str):
        normalized = observable.strip().lower()
        if normalized in {"z_magnetization", "pauli_z_magnetization"}:
            return _z_magnetization_observable(n_qubits)
        if normalized in {"z_parity", "pauli_z_parity"}:
            return PauliTerm(1.0, tuple((qubit, "z") for qubit in range(n_qubits)))
    parsed = _normalise_observable(observable, n_qubits)
    if isinstance(parsed, str):
        raise ValueError("template observable strings must be z_magnetization or z_parity")
    return parsed


def _z_magnetization_observable(n_qubits: int) -> SparsePauliHamiltonian:
    weight = 1.0 / float(n_qubits)
    return SparsePauliHamiltonian(
        tuple(PauliTerm(weight, ((qubit, "z"),)) for qubit in range(n_qubits))
    )


def _ghz_chain_operations(n_qubits: int) -> tuple[PhaseQNodeOperation, ...]:
    operations = [PhaseQNodeOperation("h", (0,))]
    operations.extend(
        PhaseQNodeOperation("cnot", (control, control + 1)) for control in range(n_qubits - 1)
    )
    return tuple(operations)


def _hardware_efficient_operations(
    n_qubits: int,
    n_layers: int,
    entangler: str,
    rotation_gates: tuple[str, ...],
) -> tuple[tuple[PhaseQNodeOperation, ...], int]:
    operations: list[PhaseQNodeOperation] = []
    parameter_index = 0
    for _layer in range(n_layers):
        for gate in rotation_gates:
            for qubit in range(n_qubits):
                operations.append(PhaseQNodeOperation(gate, (qubit,), parameter_index))
                parameter_index += 1
        operations.extend(_entangler_operations(n_qubits, entangler))
    return tuple(operations), parameter_index


def _entangler_operations(n_qubits: int, entangler: str) -> tuple[PhaseQNodeOperation, ...]:
    pairs = [(control, control + 1) for control in range(n_qubits - 1)]
    if entangler == "ring":
        pairs.append((n_qubits - 1, 0))
    return tuple(PhaseQNodeOperation("cnot", pair) for pair in pairs)


def _require_qubit_width(operation: PhaseQNodeOperation, width: int) -> None:
    if len(operation.qubits) != width:
        raise ValueError(f"{operation.gate} decomposition expects {width} qubits")


__all__ = [
    "registered_phase_qnode_gates",
    "registered_phase_qnode_observables",
    "registered_phase_qnode_templates",
    "registered_phase_qnode_decompositions",
    "registered_phase_qnode_noise_channels",
    "build_sparse_ising_chain_hamiltonian",
    "decompose_phase_qnode_controlled_gate",
    "build_phase_qnode_template",
]
