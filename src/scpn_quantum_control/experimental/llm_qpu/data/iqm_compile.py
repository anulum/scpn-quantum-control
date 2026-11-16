# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU offline IQM compile
"""Compile a planned experiment offline for a stored IQM target through the IQM adapter, and prove each payload.

The IQM SDK is imported inside each function; it installs on the isolated IQM cohort only.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from ..contracts.compiled import validate_compiled_bundle
from ..contracts.experiment import PlannedExperiment
from ..contracts.plan_set import CircuitPlanSet
from ..contracts.target import TargetSnapshot
from ..contracts.wire import canonical_bytes
from .compile_records import _bundle, _compiled_cell, _log, _sha, sdk_lock
from .compiled_cell_proof import _check_all
from .measurement_compiler import render_measured_experiment
from .target_compiler import CompiledExperiment, chain_layout

_BASIS = ("cz", "measure", "prx")
_LOCK_PACKAGES = ("iqm-client", "numpy", "qiskit", "rustworkx")
_IMPLEMENTATIONS = {"prx": "drag_gaussian", "cz": "tgss", "measure": "constant"}


def _qubit_names(target: TargetSnapshot) -> list[str]:
    """Return IQM component names QB1..QBn in physical index order."""
    return [f"QB{index + 1}" for index in range(target.num_qubits)]


def iqm_architecture(target: TargetSnapshot) -> Any:
    """Return the compile-only IQM architecture of a stored IQM target snapshot.

    The calibration set ID is derived from the snapshot digest: this architecture compiles, it never
    names a live calibration.
    """
    from iqm.iqm_client import DynamicQuantumArchitecture, GateImplementationInfo, GateInfo

    if target.provider != "iqm":
        raise ValueError("an IQM compile needs an IQM target snapshot")
    if target.basis_gates != _BASIS:
        raise ValueError("an IQM target basis is exactly cz, measure and prx")
    names = _qubit_names(target)
    pairs = sorted({tuple(sorted(edge)) for edge in target.coupling_edges})
    loci = {
        "prx": tuple((name,) for name in names),
        "cz": tuple((names[a], names[b]) for a, b in pairs),
        "measure": tuple((name,) for name in names),
    }
    gates = {
        gate: GateInfo(
            implementations={_IMPLEMENTATIONS[gate]: GateImplementationInfo(loci=loci[gate])},
            default_implementation=_IMPLEMENTATIONS[gate],
            override_default_implementation={},
        )
        for gate in _BASIS
    }
    digest = hashlib.sha256(canonical_bytes(target.to_wire())).hexdigest()
    calibration = uuid.uuid5(uuid.NAMESPACE_URL, f"scpn-llm-qpu-compile-only:{digest}")
    return DynamicQuantumArchitecture(
        calibration_set_id=calibration, qubits=names, computational_resonators=[], gates=gates
    )


def _no_options(cls: type) -> Any:
    """Return no execution options; a compile-only backend never executes."""
    from qiskit.providers import Options

    return Options()


def _refuse_run(self: object, run_input: object, **options: object) -> Any:
    """Refuse every execution request of a compile-only backend."""
    raise RuntimeError("a compile-only backend never submits")


def iqm_backend(target: TargetSnapshot) -> Any:
    """Return a compile-only IQM backend of the snapshot; it exposes the adapter's target and never submits."""
    from iqm.qiskit_iqm.iqm_backend import IQMBackendBase

    backend_type = type(
        "CompileOnlyIQMBackend",
        (IQMBackendBase,),
        {"_default_options": classmethod(_no_options), "run": _refuse_run},
    )
    return backend_type(iqm_architecture(target), name=f"compile_only_{target.device_name}")


def _circuit_adapter() -> Any:
    """Return the pydantic adapter of the circuit list an IQM run request carries."""
    from iqm.station_control.interface.models import RunRequest
    from pydantic import TypeAdapter

    circuit_list_type: Any = RunRequest.model_fields["circuits"].annotation
    return TypeAdapter(circuit_list_type)


def _payload(compiled: Any, ordinal: int, backend: Any) -> bytes:
    """Return the canonical JSON of one cell as the IQM run request serialises its circuits."""
    from iqm.pulse.circuit_operations import Circuit
    from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions

    names = {index: backend.index_to_qubit_name(index) for index in range(backend.num_qubits)}
    circuit = Circuit(
        name=f"cell_{ordinal:04d}",
        instructions=tuple(serialize_instructions(compiled, names)),
        metadata=None,
    )
    (dumped,) = _circuit_adapter().dump_python([circuit], mode="json")
    return json.dumps(dumped, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _reload(payload: bytes, backend: Any) -> tuple[Any, tuple[tuple[str, int], ...]]:
    """Return the Qiskit circuit an IQM payload deserialises to, and its native gate counts."""
    from iqm.qiskit_iqm.qiskit_to_iqm import deserialize_instructions
    from qiskit import QuantumRegister
    from qiskit.transpiler import Layout

    (circuit,) = _circuit_adapter().validate_python([json.loads(payload)])
    instructions = list(circuit.instructions)
    names = {backend.index_to_qubit_name(index): index for index in range(backend.num_qubits)}
    register = QuantumRegister(backend.num_qubits, "q")
    restored = deserialize_instructions(
        instructions, names, Layout.from_intlist(list(range(backend.num_qubits)), register)
    )
    counts: dict[str, int] = {}
    for item in instructions:
        counts[item.name] = counts.get(item.name, 0) + 1
    return restored, tuple(sorted(counts.items()))


def _substitutions(logical: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Declare what each logical gate may become in the IQM basis: prx for one qubit, cz and prx for two."""
    from qiskit.circuit.library import get_standard_gate_name_mapping

    standard = get_standard_gate_name_mapping()
    return tuple(
        (name, ("prx",) if standard[name].num_qubits == 1 else ("cz", "prx"))
        for name in logical
        if name not in _BASIS
    )


def compile_iqm_experiment(
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
    """Compile every planned cell with the IQM adapter, prove each serialised payload and freeze the bundle."""
    from iqm.qiskit_iqm import transpile_to_IQM

    backend = iqm_backend(target)
    chosen = (
        layout if layout is not None else chain_layout(target, experiment.measurements[0].n_qubits)
    )
    options = {"optimization_level": optimization_level, "seed_transpiler": seed}
    logical_circuits = render_measured_experiment(experiment, plan_set)
    compiled = tuple(
        transpile_to_IQM(c, backend, initial_layout=list(chosen), **options)
        for c in logical_circuits
    )
    payloads = tuple(
        _payload(circuit, ordinal, backend) for ordinal, circuit in enumerate(compiled)
    )
    reloaded = tuple(_reload(payload, backend) for payload in payloads)
    allowed = frozenset(backend.target.operation_names)
    _check_all(
        tuple(c for c, _ in reloaded),
        experiment,
        plan_set,
        target,
        chosen,
        tolerance,
        allowed=allowed,
    )
    totals: dict[str, int] = {}
    for _, counts in reloaded:
        for name, count in counts:
            totals[name] = totals.get(name, 0) + count
    target_digest = hashlib.sha256(canonical_bytes(target.to_wire())).hexdigest()
    log, lock = _log(target_digest, chosen, options, compiled, payloads), sdk_lock(_LOCK_PACKAGES)
    logical = tuple(sorted({item.operation.name for c in logical_circuits for item in c.data}))
    fields: dict[str, Any] = {
        "target_snapshot_digest": target_digest,
        "provider": target.provider,
        "device_name": target.device_name,
        "sdk_lock_digest": _sha(lock),
        "cells": tuple(
            _compiled_cell(p, m, b)
            for p, m, b in zip(experiment.cells, experiment.measurements, payloads, strict=True)
        ),
        "layout": chosen,
        "logical_gates": logical,
        "compiled_gate_counts": tuple(sorted(totals.items())),
        "substitutions": _substitutions(logical),
        "compiled_depth": max(circuit.depth() for circuit, _ in reloaded),
        "compile_log_digest": _sha(log),
        "provider_options": tuple(sorted(options.items())),
    }
    bundle = _bundle(fields, target, revisions)
    validate_compiled_bundle(bundle, target)
    return CompiledExperiment(bundle, payloads, log, lock)


def verify_iqm_experiment(
    compiled: CompiledExperiment,
    experiment: PlannedExperiment,
    plan_set: CircuitPlanSet,
    target: TargetSnapshot,
    *,
    tolerance: float = 1e-9,
) -> None:
    """Reload every stored IQM payload through the adapter and prove it again."""
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
    backend = iqm_backend(target)
    reloaded = tuple(_reload(payload, backend)[0] for payload in compiled.payloads)
    allowed = frozenset(backend.target.operation_names)
    _check_all(reloaded, experiment, plan_set, target, bundle.layout, tolerance, allowed=allowed)
