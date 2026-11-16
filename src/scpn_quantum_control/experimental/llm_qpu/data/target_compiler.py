# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU offline target compiler
"""Compile a planned experiment offline against a stored target snapshot and prove each compiled cell."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..contracts.compiled import (
    CompiledBundle,
    validate_compiled_bundle,
)
from ..contracts.experiment import PlannedExperiment
from ..contracts.plan_set import CircuitPlanSet
from ..contracts.target import TargetSnapshot
from ..contracts.wire import canonical_bytes
from .measurement_compiler import render_measured_experiment

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

from .compile_records import _bundle, _compiled_cell, _counts, _log, _payload, _sha, sdk_lock
from .compiled_cell_proof import _check_all


@dataclass(frozen=True, slots=True)
class CompiledExperiment:
    """A compiled bundle with the private bytes its digests name: payloads, compile log and SDK lock."""

    bundle: CompiledBundle
    payloads: tuple[bytes, ...]
    compile_log: bytes
    sdk_lock: bytes


def chain_layout(target: TargetSnapshot, width: int) -> tuple[int, ...]:
    """Return the first simple path of ``width`` coupled qubits, searching lowest qubits first."""
    neighbours: dict[int, set[int]] = {qubit: set() for qubit in range(target.num_qubits)}
    for source, sink in target.coupling_edges:
        neighbours[source].add(sink)
        neighbours[sink].add(source)

    def extend(path: list[int]) -> list[int] | None:
        """Depth-first extension of a path through unused neighbours in ascending order."""
        if len(path) == width:
            return path
        for qubit in sorted(neighbours[path[-1]] - set(path)):
            found = extend([*path, qubit])
            if found is not None:
                return found
        return None

    for start in range(target.num_qubits):
        found = extend([start])
        if found is not None:
            return tuple(found)
    raise ValueError("the target has no coupled chain of the planned width")


def _check_basis(target: TargetSnapshot) -> None:
    """Refuse a basis whose names Qiskit does not define; such a target needs its provider adapter."""
    from qiskit.circuit.library import get_standard_gate_name_mapping

    unknown = sorted(set(target.basis_gates) - set(get_standard_gate_name_mapping()))
    if unknown:
        raise ValueError(f"the target's basis needs its provider adapter: {', '.join(unknown)}")
    if "measure" not in target.basis_gates:
        raise ValueError("the target basis must include measure")


def _transpile(
    circuit: QuantumCircuit,
    target: TargetSnapshot,
    layout: tuple[int, ...],
    options: dict[str, int],
) -> QuantumCircuit:
    """Transpile one circuit to the target basis and coupling under a fixed initial layout."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    return transpile(
        circuit,
        basis_gates=list(target.basis_gates),
        coupling_map=CouplingMap([list(edge) for edge in target.coupling_edges]),
        initial_layout=list(layout),
        optimization_level=options["optimization_level"],
        seed_transpiler=options["seed_transpiler"],
    )


def _substitutions(
    logical: tuple[str, ...], target: TargetSnapshot
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Declare what each logical gate outside the basis may become under synthesis to this basis.

    A one-qubit gate may become any one-qubit basis gate; a two-qubit gate may become any unitary
    basis gate. The declaration rests on the target basis, not on the angles of one compile.
    """
    from qiskit.circuit.library import get_standard_gate_name_mapping

    standard = get_standard_gate_name_mapping()
    unitary = tuple(name for name in target.basis_gates if name != "measure")
    single = tuple(name for name in unitary if standard[name].num_qubits == 1)
    return tuple(
        (name, single if standard[name].num_qubits == 1 else unitary)
        for name in logical
        if name not in target.basis_gates
    )


def compile_experiment(
    experiment: PlannedExperiment,
    plan_set: CircuitPlanSet,
    target: TargetSnapshot,
    *,
    seed: int,
    revisions: tuple[str, str],
    optimization_level: int = 1,
    layout: tuple[int, ...] | None = None,
    tolerance: float = 1e-9,
) -> CompiledExperiment:
    """Compile every planned cell for the stored target, prove each one and freeze the bundle."""
    _check_basis(target)
    width = experiment.measurements[0].n_qubits
    chosen = layout if layout is not None else chain_layout(target, width)
    options = {"optimization_level": optimization_level, "seed_transpiler": seed}
    logical_circuits = render_measured_experiment(experiment, plan_set)
    compiled = tuple(_transpile(c, target, chosen, options) for c in logical_circuits)
    _check_all(compiled, experiment, plan_set, target, chosen, tolerance)
    payloads = tuple(_payload(circuit, ordinal) for ordinal, circuit in enumerate(compiled))
    target_digest = hashlib.sha256(canonical_bytes(target.to_wire())).hexdigest()
    log, lock = _log(target_digest, chosen, options, compiled, payloads), sdk_lock()
    logical = tuple(sorted({item.operation.name for c in logical_circuits for item in c.data}))
    cells = tuple(
        _compiled_cell(planned, measurement, payload)
        for planned, measurement, payload in zip(
            experiment.cells, experiment.measurements, payloads, strict=True
        )
    )
    fields: dict[str, Any] = {
        "target_snapshot_digest": target_digest,
        "provider": target.provider,
        "device_name": target.device_name,
        "sdk_lock_digest": _sha(lock),
        "cells": cells,
        "layout": chosen,
        "logical_gates": logical,
        "compiled_gate_counts": _counts(compiled),
        "substitutions": _substitutions(logical, target),
        "compiled_depth": max(circuit.depth() for circuit in compiled),
        "compile_log_digest": _sha(log),
        "provider_options": tuple(sorted(options.items())),
    }
    bundle = _bundle(fields, target, revisions)
    validate_compiled_bundle(bundle, target)
    return CompiledExperiment(bundle, payloads, log, lock)


def verify_compiled_experiment(
    compiled: CompiledExperiment,
    experiment: PlannedExperiment,
    plan_set: CircuitPlanSet,
    target: TargetSnapshot,
    *,
    tolerance: float = 1e-9,
) -> None:
    """Reload every stored payload and prove it again against the bundle, the plan and the target.

    Each payload is decoded only after its digest has matched the bundle cell, and only through the
    repository's reviewed QPY wrapper.
    """
    from scpn_quantum_control.hardware.hal_qiskit import _reviewed_qpy_load_circuits

    bundle = compiled.bundle
    validate_compiled_bundle(bundle, target)
    if (_sha(compiled.sdk_lock), _sha(compiled.compile_log)) != (
        bundle.sdk_lock_digest,
        bundle.compile_log_digest,
    ):
        raise ValueError("the SDK lock or compile log differs from the bundle")
    if len(compiled.payloads) != len(bundle.cells) or len(bundle.cells) != len(experiment.cells):
        raise ValueError("every planned cell needs exactly one compiled payload")
    for cell, payload, planned, measurement in zip(
        bundle.cells, compiled.payloads, experiment.cells, experiment.measurements, strict=True
    ):
        if cell != _compiled_cell(planned, measurement, payload):
            raise ValueError("a compiled cell does not follow its planned cell and payload")
    loaded = tuple(_reviewed_qpy_load_circuits(payload)[0] for payload in compiled.payloads)
    _check_all(loaded, experiment, plan_set, target, bundle.layout, tolerance)
