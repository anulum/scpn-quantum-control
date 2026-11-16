# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU compile records
"""The private bytes a compiled bundle names (QPY payloads, compile log, SDK lock) and the frozen bundle itself."""

from __future__ import annotations

import hashlib
import io
import platform
from importlib import metadata
from typing import TYPE_CHECKING, Any

from ..contracts.cells import PlannedCell
from ..contracts.compiled import (
    CompiledBundle,
    CompiledCell,
    _payload_fields,
)
from ..contracts.measurement import MeasurementPlan
from ..contracts.target import TargetSnapshot
from ..contracts.wire import COMPILED_BUNDLE_SCHEMA, ArtifactHeader, canonical_bytes

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


_SDK_PACKAGES = ("numpy", "qiskit", "rustworkx")
_BINDING = {
    "schema": "scpn.experimental.llm_qpu.parameter_binding.v1",
    "binding_index": 0,
    "parameters": [],
}


def _sha(payload: bytes) -> str:
    """Return the SHA-256 hex digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def sdk_lock(packages: tuple[str, ...] = _SDK_PACKAGES) -> bytes:
    """Return the canonical record of the interpreter and the installed versions of the given packages."""
    versions = [[name, metadata.version(name)] for name in packages]
    return canonical_bytes(
        {
            "schema": "scpn.experimental.llm_qpu.sdk_lock.v1",
            "python": platform.python_version(),
            "packages": versions,
        }
    )


def _payload(compiled: QuantumCircuit, ordinal: int) -> bytes:
    """Return deterministic QPY bytes of the compiled cell without transpiler layout metadata."""
    from qiskit import QuantumCircuit, qpy

    bare = QuantumCircuit(
        *compiled.qregs,
        *compiled.cregs,
        name=f"cell_{ordinal:04d}",
        global_phase=compiled.global_phase,
    )
    for item in compiled.data:
        bare.append(item)
    buffer = io.BytesIO()
    qpy.dump(bare, buffer)
    return buffer.getvalue()


def _counts(circuits: tuple[QuantumCircuit, ...]) -> tuple[tuple[str, int], ...]:
    """Return the sorted total operation counts of the circuits."""
    totals: dict[str, int] = {}
    for circuit in circuits:
        for name, count in circuit.count_ops().items():
            totals[str(name)] = totals.get(str(name), 0) + int(count)
    return tuple(sorted(totals.items()))


def _log(
    target_digest: str,
    layout: tuple[int, ...],
    options: dict[str, int],
    circuits: tuple[QuantumCircuit, ...],
    payloads: tuple[bytes, ...],
) -> bytes:
    """Return the canonical compile log: options, layout and each cell's depth, counts and payload digest."""
    cells: list[dict[str, Any]] = [
        {
            "depth": circuit.depth(),
            "gate_counts": [list(pair) for pair in _counts((circuit,))],
            "payload_sha256": _sha(payload),
            "payload_bytes": len(payload),
        }
        for circuit, payload in zip(circuits, payloads, strict=True)
    ]
    return canonical_bytes(
        {
            "schema": "scpn.experimental.llm_qpu.compile_log.v1",
            "target_snapshot_digest": target_digest,
            "layout": list(layout),
            "options": [[name, options[name]] for name in sorted(options)],
            "cells": cells,
        }
    )


def _compiled_cell(
    planned: PlannedCell, measurement: MeasurementPlan, payload: bytes
) -> CompiledCell:
    """Return the compiled-cell binding of one planned cell and its payload, unbound in v1."""
    return CompiledCell(
        hashlib.sha256(canonical_bytes(planned.to_wire())).hexdigest(),
        measurement.circuit_plan_digest,
        _sha(payload),
        _sha(canonical_bytes(_BINDING)),
        planned.shots,
    )


def _bundle(
    fields: dict[str, Any], target: TargetSnapshot, revisions: tuple[str, str]
) -> CompiledBundle:
    """Freeze the bundle with its payload digest and a header that hashes its content."""
    payload = _payload_fields(
        fields["cells"],
        fields["layout"],
        fields["logical_gates"],
        fields["compiled_gate_counts"],
        fields["substitutions"],
        fields["compiled_depth"],
        fields["provider_options"],
    )
    digest = _sha(canonical_bytes(payload))
    content = {
        "schema": COMPILED_BUNDLE_SCHEMA,
        "object_kind": "compiled_bundle",
        **{
            name: fields[name]
            for name in (
                "target_snapshot_digest",
                "provider",
                "device_name",
                "sdk_lock_digest",
                "compile_log_digest",
            )
        },
        "payload_digest": digest,
        **payload,
    }
    parents = tuple(
        sorted(
            (
                fields["target_snapshot_digest"],
                fields["sdk_lock_digest"],
                fields["compile_log_digest"],
            )
        )
    )
    header = ArtifactHeader(
        "compiled_bundle",
        _sha(canonical_bytes(content)),
        parents,
        revisions[0],
        revisions[1],
        "offline_design",
        target.header.data_origin,
        "design_only",
    )
    return CompiledBundle(**fields, payload_digest=digest, header=header)
