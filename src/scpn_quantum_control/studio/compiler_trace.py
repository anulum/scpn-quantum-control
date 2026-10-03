# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — immutable native compiler trace projection
"""Project actual native pass evidence without replacing its numerical authority.

The new envelope uses the workspace typed canonical codec. Original native pass
digests retain their existing codec. An imported envelope binds metadata, not
the execution of its declared qualification or a physical device.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any, cast

from ..compiler.circuit_pass_records import CircuitIR, CircuitPassRecord
from ..compiler.mlir_records import MLIRModule
from ..studio_workspace.canonical import canonical_digest
from ..studio_workspace.json_transport import read_json

COMPILER_TRACE_SCHEMA = "studio.compiler-trace.v1"
"""Version of the immutable compiler metadata envelope."""
MAX_TRACE_BYTES = 16 * 1024 * 1024
"""UTF-8 export ceiling; a product bound, not measured available memory."""
MAX_TRACE_PASSES = 32
"""Maximum ordered native pass slots, including explicitly missing artifacts."""


@dataclass(frozen=True, slots=True)
class CompilerTrace:
    """Immutable exact trace JSON with detached exports.

    Parameters
    ----------
    json_text
        Producer JSON carrying a canonical envelope digest. Qualification remains
        a native producer declaration, not an attestation from this container.

    """

    json_text: str

    def __post_init__(self) -> None:
        """Validate the envelope binding and bounded exact UTF-8 representation."""
        if not isinstance(self.json_text, str) or len(self.json_text) > MAX_TRACE_BYTES:
            raise ValueError("compiler trace exceeds its JSON byte ceiling")
        if len(self.json_text.encode("utf-8")) > MAX_TRACE_BYTES:
            raise ValueError("compiler trace exceeds its UTF-8 byte ceiling")
        wire = self.to_dict()
        if (
            set(wire) != {"schema", "body", "extensions", "sha256"}
            or wire["schema"] != COMPILER_TRACE_SCHEMA
        ):
            raise ValueError("compiler trace envelope schema is unsupported")
        if not isinstance(wire["body"], dict) or not isinstance(wire["extensions"], dict):
            raise ValueError("compiler trace body and extensions must be objects")
        envelope = {key: wire[key] for key in ("schema", "body", "extensions")}
        if canonical_digest(COMPILER_TRACE_SCHEMA, envelope) != wire["sha256"]:
            raise ValueError("compiler trace envelope digest differs")

    def to_dict(self) -> dict[str, Any]:
        """Return fresh nested containers without mutating retained evidence.

        Returns
        -------
        dict
            Original wire envelope and its digest, with exact native floats.

        Raises
        ------
        ValueError
            The stored JSON is not an object.

        """
        wire: object = read_json(self.json_text)
        if not isinstance(wire, dict):
            raise ValueError("compiler trace JSON object is required")
        return cast(dict[str, Any], wire)

    def to_json(self) -> str:
        """Return exact portable JSON without relabelling emitted IR as executed.

        Returns
        -------
        str
            Original UTF-8 JSON text retained by this immutable record.

        """
        return self.json_text


def compiler_backend_snapshot(*, optimisation_level: int = 2) -> dict[str, object]:
    """Describe the actual local compiler and requested lowering settings.

    Parameters
    ----------
    optimisation_level
        Exact integer zero through three, preserving native compiler admission.

    Returns
    -------
    dict
        Qiskit version, original basis policy and settings; no physical target.

    Raises
    ------
    ValueError
        The optimisation level is not an admitted integer.

    """
    if type(optimisation_level) is not int or not 0 <= optimisation_level <= 3:
        raise ValueError("optimisation level must be an integer between 0 and 3")
    return {
        "compiler": "qiskit",
        "compiler_version": version("qiskit"),
        "target": None,
        "reference_backend": "qiskit.quantum_info.Operator",
        "basis_convention": "qiskit_little_endian",
        "settings": {
            "optimisation_level": optimisation_level,
            "basis_gates": ["rx", "ry", "rz", "cx"],
            "seed_transpiler": 0,
            "qubits_initially_zero": False,
        },
    }


def _snapshot(ir: CircuitIR) -> dict[str, object]:
    """Bind complete native source, registers, operands and separate global phase."""
    body = asdict(ir)
    if len(ir.source.encode("utf-8")) > 1024 * 1024:
        raise ValueError("native pass source exceeds 1 MiB")
    return {
        "sha256": canonical_digest("studio.circuit-snapshot.v1", body),
        "source_sha256": ir.source_sha256,
        "ir": body,
    }


def _metrics(ir: CircuitIR) -> dict[str, object]:
    """Describe operand dependency depth and declared dense payloads without allocation."""
    counts: dict[str, int] = {}
    levels = [0] * (ir.num_qubits + ir.num_clbits)
    for op in ir.operations:
        operands = [*op.qubits, *(ir.num_qubits + bit for bit in op.clbits)]
        previous = max((levels[bit] for bit in operands), default=0)
        depth = previous if op.name == "barrier" else previous + 1
        for bit in operands:
            levels[bit] = depth
        if op.name not in {"measure", "reset", "barrier"}:
            counts[op.name] = counts.get(op.name, 0) + 1
    return {
        "operation_count": len(ir.operations),
        "gate_counts": counts,
        "depth": max(levels, default=0),
        "num_qubits": ir.num_qubits,
        "num_clbits": ir.num_clbits,
        "statevector_bytes": 16 * (1 << ir.num_qubits),
        "operator_bytes": 16 * (1 << (2 * ir.num_qubits)),
    }


def _pass(record: CircuitPassRecord, parameters: Mapping[str, object]) -> dict[str, object]:
    """Project a native record without recomputing or widening its qualification."""
    before, after = _metrics(record.input_ir), _metrics(record.output_ir)
    delta = {
        key: cast(int, after[key]) - cast(int, before[key])
        for key in ("operation_count", "depth", "statevector_bytes", "operator_bytes")
    }
    native = asdict(record)
    native.pop("input_ir")
    native.pop("output_ir")
    return {
        "state": "qualified",
        "record": native,
        "native_record_sha256": record.sha256,
        "parameters": dict(parameters),
        "input": _snapshot(record.input_ir),
        "output": _snapshot(record.output_ir),
        "metrics": {"input": before, "output": after, "delta": delta},
        "effects": {
            "input_measurements": record.input_ir.measurements,
            "output_measurements": record.output_ir.measurements,
            "source_mapping": "each representation retains its own source spans; cross-pass operation correspondence is unavailable",
        },
    }


def project_compiler_passes(
    records: Sequence[CircuitPassRecord | None],
    *,
    backend_snapshot: Mapping[str, object],
    pass_parameters: Sequence[Mapping[str, object]] | None = None,
    emitted_ir: MLIRModule | None = None,
) -> CompilerTrace:
    """Capture native pass snapshots, settings, layout and explicit missing slots.

    Parameters
    ----------
    records
        One through 32 ordered native records. The first supplies the original
        source. Later None slots explicitly declare missing artifacts.
    backend_snapshot
        Trusted caller's exact compiler/backend metadata; retained without fetch.
    pass_parameters
        One ordered parameter mapping per slot, or empty mappings when omitted.
    emitted_ir
        Original textual MLIR, or None for unavailable interchange output.

    Returns
    -------
    CompilerTrace
        Immutable full input/intermediate/final metadata with canonical binding.

    Raises
    ------
    ValueError
        Source/IR/layout continuity, source bounds, parameter count, scalar values
        or export size is invalid. No caller records are changed.

    """
    if not 1 <= len(records) <= MAX_TRACE_PASSES or not isinstance(records[0], CircuitPassRecord):
        raise ValueError("compiler passes need a first native source record and at most 32 slots")
    parameters = [{} for _ in records] if pass_parameters is None else pass_parameters
    if len(parameters) != len(records):
        raise ValueError("compiler pass parameter count differs")
    expected_backend = {
        "compiler",
        "compiler_version",
        "target",
        "reference_backend",
        "basis_convention",
        "settings",
    }
    if (
        set(backend_snapshot) != expected_backend
        or backend_snapshot["compiler"] != "qiskit"
        or backend_snapshot["target"] is not None
        or backend_snapshot["reference_backend"] != "qiskit.quantum_info.Operator"
        or backend_snapshot["basis_convention"] != "qiskit_little_endian"
        or not isinstance(backend_snapshot["compiler_version"], str)
        or not 1 <= len(backend_snapshot["compiler_version"]) <= 256
        or not isinstance(backend_snapshot["settings"], Mapping)
    ):
        raise ValueError("compiler backend snapshot differs from the native reference contract")
    rows: list[dict[str, object]] = []
    previous: CircuitPassRecord | None = None
    for record, parameter in zip(records, parameters, strict=True):
        if record is None:
            rows.append({"state": "missing", "reason": "Native pass artifact was not supplied."})
        else:
            if not isinstance(record, CircuitPassRecord):
                raise ValueError(
                    "compiler pass must be an original native record or an explicit missing slot"
                )
            if previous is not None and (
                _snapshot(previous.output_ir)["sha256"] != _snapshot(record.input_ir)["sha256"]
                or previous.output_layout != record.input_layout
            ):
                raise ValueError("compiler pass source/IR/layout continuity differs")
            rows.append(_pass(record, parameter))
        previous = record
    first = records[0]
    assert first is not None
    body = {
        "source": first.input_ir.source,
        "source_sha256": first.input_ir.source_sha256,
        "backend_snapshot": dict(backend_snapshot),
        "passes": rows,
        "complete": all(record is not None for record in records),
        "execution_status": "emitted_not_executed",
        "emitted_ir": None
        if emitted_ir is None
        else {
            "text": emitted_ir.text,
            "sha256": emitted_ir.sha256,
            "dialect": emitted_ir.dialect,
            "resource_counts": dict(emitted_ir.resource_counts),
            "metadata": dict(emitted_ir.metadata),
            "execution_status": "textual_ir",
        },
    }
    envelope = {"schema": COMPILER_TRACE_SCHEMA, "body": body, "extensions": {}}
    wire = {**envelope, "sha256": canonical_digest(COMPILER_TRACE_SCHEMA, envelope)}
    chunks: list[str] = []
    byte_count = 1
    for chunk in json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).iterencode(wire):
        byte_count += len(chunk.encode("utf-8"))
        if byte_count > MAX_TRACE_BYTES:
            raise ValueError("compiler trace exceeds its UTF-8 byte ceiling")
        chunks.append(chunk)
    return CompilerTrace("".join(chunks) + "\n")


def build_compiler_trace(source: str, *, optimisation_level: int = 2) -> CompilerTrace:
    """Lower a bounded static source through the actual original native compiler.

    Parameters
    ----------
    source
        Original bounded OpenQASM 2 text. Static qualification refuses reset,
        conditional or nonterminal readout rather than discarding effects.
    optimisation_level
        Original exact native optimisation setting zero through three.

    Returns
    -------
    CompilerTrace
        Actual aggregate basis-lowering pass and textual interchange artifact.
        Internal SDK passes are not invented as individually observed records.

    Raises
    ------
    CircuitPassRefused
        Original native source, resource or equivalence admission fails.
    ValueError
        Settings or metadata export cannot satisfy their bounded contracts.

    """
    from ..compiler.circuit_pass_qualification import compile_circuit_to_mlir

    backend = compiler_backend_snapshot(optimisation_level=optimisation_level)
    compiled = compile_circuit_to_mlir(source, optimisation_level=optimisation_level)
    return project_compiler_passes(
        [compiled.pass_record],
        backend_snapshot=backend,
        pass_parameters=[cast(Mapping[str, object], backend["settings"])],
        emitted_ir=compiled.mlir_module,
    )


__all__ = [
    "COMPILER_TRACE_SCHEMA",
    "MAX_TRACE_BYTES",
    "MAX_TRACE_PASSES",
    "CompilerTrace",
    "build_compiler_trace",
    "compiler_backend_snapshot",
    "project_compiler_passes",
]
