# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Located native circuit source import
"""Bounded static OpenQASM 2 import using the native parser and source spans."""

from __future__ import annotations

import math
import re

from qiskit import QuantumCircuit, qasm2

from .circuit_pass_records import (
    CircuitDiagnostic,
    CircuitIR,
    CircuitOperation,
    CircuitPassRefused,
    SourceSpan,
)


def import_circuit_source(source: str) -> QuantumCircuit:
    """Import the bounded static unitary subset with trailing readout.

    Parameters
    ----------
    source
        OpenQASM 2 UTF-8 text, at most 1 MiB. Only the native ``qelib1.inc``
        is admitted; custom definitions and filesystem includes are refused.
        Widths are bounded before parsing to eight qubits and 64 classical bits.

    Returns
    -------
    QuantumCircuit
        The actual native parser result, with at most 4096 operations.

    Raises
    ------
    CircuitPassRefused
        Located syntax, unsupported effect, nonfinite parameter or budget refusal.

    """
    circuit, _ = _decode(source)
    return circuit


def snapshot_circuit(circuit: QuantumCircuit, *, source: str | None = None) -> CircuitIR:
    """Bind an immutable native circuit snapshot to exact statement locations.

    Parameters
    ----------
    circuit
        Bound static circuit with trailing measurement, at most eight qubits.
    source
        Optional exact source that produced the same ordered operations. Otherwise
        the original native OpenQASM 2 exporter supplies source; phase is separate.

    Returns
    -------
    CircuitIR
        Frozen source, operands, register indices and global phase in radians.

    Raises
    ------
    CircuitPassRefused
        Unsupported source/circuit, mismatched source or admission limit.

    """
    _require_budget(circuit)
    if circuit.num_parameters:
        raise CircuitPassRefused(
            CircuitDiagnostic("unbound_parameters", "circuit parameters must be bound")
        )
    if source is None:
        try:
            source = qasm2.dumps(circuit)
        except qasm2.QASM2ExportError as exc:
            raise CircuitPassRefused(
                CircuitDiagnostic(
                    "unsupported_export", "circuit has no supported static source export"
                )
            ) from exc
        _, exported_spans = _decode(source)
        for instruction, span in reversed(tuple(zip(circuit.data, exported_spans, strict=True))):
            if instruction.operation.params:
                statement = source[span.start : span.end]
                parameters = ",".join(
                    format(float(p), ".17e") for p in instruction.operation.params
                )
                exact = (
                    statement[: statement.index("(") + 1]
                    + parameters
                    + statement[statement.rindex(")") :]
                )
                source = source[: span.start] + exact + source[span.end :]
    parsed, spans = _decode(source)
    if len(circuit.data) != len(parsed.data) or (circuit.num_qubits, circuit.num_clbits) != (
        parsed.num_qubits,
        parsed.num_clbits,
    ):
        raise CircuitPassRefused(
            CircuitDiagnostic("source_mismatch", "source does not match circuit operations")
        )
    operations = tuple(_operation(circuit, i, span) for i, span in enumerate(spans))
    native_operations = tuple(_operation(parsed, i, span) for i, span in enumerate(spans))
    if len(circuit.data) != len(parsed.data) or operations != native_operations:
        raise CircuitPassRefused(
            CircuitDiagnostic("source_mismatch", "source does not match circuit operations")
        )
    phase = float(circuit.global_phase)
    if not math.isfinite(phase):
        raise CircuitPassRefused(
            CircuitDiagnostic("nonfinite_phase", "circuit phase must be finite")
        )
    return CircuitIR(
        source,
        circuit.num_qubits,
        circuit.num_clbits,
        phase,
        operations,
        tuple((r.name, tuple(circuit.find_bit(b).index for b in r)) for r in circuit.qregs),
        tuple((r.name, tuple(circuit.find_bit(b).index for b in r)) for r in circuit.cregs),
    )


def _require_budget(circuit: QuantumCircuit) -> None:
    if not isinstance(circuit, QuantumCircuit):
        raise CircuitPassRefused(
            CircuitDiagnostic("invalid_circuit", "input must be a native quantum circuit")
        )
    if not 1 <= circuit.num_qubits <= 8 or circuit.num_clbits > 64 or len(circuit.data) > 4096:
        raise CircuitPassRefused(
            CircuitDiagnostic("circuit_budget", "circuit exceeds the bounded qualification budget")
        )


def _statements(source: str) -> tuple[tuple[str, SourceSpan], ...]:
    if not isinstance(source, str) or not source.strip():
        raise CircuitPassRefused(
            CircuitDiagnostic("invalid_source", "source must be non-empty text")
        )
    try:
        byte_count = len(source.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise CircuitPassRefused(
            CircuitDiagnostic("invalid_source", "source must be valid UTF-8 text")
        ) from exc
    if byte_count > 1024 * 1024:
        raise CircuitPassRefused(
            CircuitDiagnostic("source_budget", "source exceeds the bounded import budget")
        )
    masked = re.sub(r"//[^\n]*", lambda m: " " * len(m[0]), source)
    unsupported = re.search(r"(?:^|;)\s*(gate|opaque|reset|if)\b", masked)
    if unsupported:
        start = unsupported.start(1)
        raise CircuitPassRefused(
            CircuitDiagnostic(
                "unsupported_operation",
                "operation is outside the static unitary subset",
                source.count("\n", 0, start) + 1,
                start - source.rfind("\n", 0, start),
            )
        )
    result = []
    cursor = 0
    for match in re.finditer(r"[^;]*;", masked):
        start = match.start() + len(match[0]) - len(match[0].lstrip())
        if start == match.end() - 1:
            raise CircuitPassRefused(
                CircuitDiagnostic(
                    "invalid_source",
                    "empty statements are outside the supported subset",
                    source.count("\n", 0, start) + 1,
                    start - source.rfind("\n", 0, start),
                )
            )
        span = SourceSpan(
            start,
            match.end(),
            source.count("\n", 0, start) + 1,
            start - source.rfind("\n", 0, start),
        )
        result.append((masked[start : match.end()], span))
        cursor = match.end()
    if masked[cursor:].strip():
        start = cursor + len(masked[cursor:]) - len(masked[cursor:].lstrip())
        raise CircuitPassRefused(
            CircuitDiagnostic(
                "invalid_source",
                "source statement requires a terminating semicolon",
                source.count("\n", 0, start) + 1,
                start - source.rfind("\n", 0, start),
            )
        )
    return tuple(result)


def _native_parse(source: str) -> QuantumCircuit:
    try:
        return qasm2.loads(
            source,
            include_path=(),
            strict=True,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )
    except qasm2.QASM2ParseError as exc:
        location = re.search(r"<input>:(\d+),(\d+):", str(exc))
        line, column = (int(location[1]), int(location[2]) + 1) if location else (1, 1)
        raise CircuitPassRefused(
            CircuitDiagnostic(
                "invalid_source",
                "source is not valid in the supported OpenQASM subset",
                line,
                column,
            )
        ) from exc
    except RecursionError as exc:
        raise CircuitPassRefused(
            CircuitDiagnostic(
                "expression_budget", "source expression nesting exceeds the native parser budget"
            )
        ) from exc


def _decode(source: str) -> tuple[QuantumCircuit, tuple[SourceSpan, ...]]:
    statements = _statements(source)
    prefix = []
    body = []
    quantum_width = classical_width = 0
    for text, span in statements:
        keyword = text.split(maxsplit=1)[0].split("(", 1)[0].rstrip(";")
        if keyword == "include" and text.strip() != 'include "qelib1.inc";':
            raise CircuitPassRefused(
                CircuitDiagnostic(
                    "unsupported_include",
                    "only the native standard gate include is supported",
                    span.line,
                    span.column,
                )
            )
        if any(c in text for c in "{}"):
            raise CircuitPassRefused(
                CircuitDiagnostic(
                    "unsupported_operation",
                    "operation is outside the static unitary subset",
                    span.line,
                    span.column,
                )
            )
        if keyword in {"OPENQASM", "include", "qreg", "creg"}:
            prefix.append(text)
            size = re.search(r"\[\s*(\d+)\s*\]", text) if keyword in {"qreg", "creg"} else None
            if size:
                digits = size[1].lstrip("0") or "0"
                if len(digits) > 2:
                    raise CircuitPassRefused(
                        CircuitDiagnostic(
                            "circuit_budget",
                            "circuit exceeds the bounded qualification budget",
                            span.line,
                            span.column,
                        )
                    )
                if keyword == "qreg":
                    quantum_width += int(digits)
                else:
                    classical_width += int(digits)
            if quantum_width > 8 or classical_width > 64:
                raise CircuitPassRefused(
                    CircuitDiagnostic(
                        "circuit_budget",
                        "circuit exceeds the bounded qualification budget",
                        span.line,
                        span.column,
                    )
                )
        else:
            body.append((text, span))
    if len(body) > 4096:
        raise CircuitPassRefused(
            CircuitDiagnostic("circuit_budget", "circuit exceeds the bounded qualification budget")
        )
    circuit = _native_parse(source)
    _require_budget(circuit)
    spans = []
    for text, span in body:
        part = _native_parse("\n".join((*prefix, text)))
        spans.extend([span] * len(part.data))
    seen_readout = False
    for index, span in enumerate(spans):
        operation = _operation(circuit, index, span)
        if operation.name == "measure":
            seen_readout = True
        elif operation.name != "barrier" and (seen_readout or operation.clbits):
            raise CircuitPassRefused(
                CircuitDiagnostic(
                    "unsupported_effect",
                    "only trailing measurement and barriers are supported",
                    span.line,
                    span.column,
                )
            )
    return circuit, tuple(spans)


def _operation(circuit: QuantumCircuit, index: int, span: SourceSpan) -> CircuitOperation:
    instruction = circuit.data[index]
    parameters = tuple(float(p) for p in instruction.operation.params)
    if any(not math.isfinite(p) for p in parameters):
        raise CircuitPassRefused(
            CircuitDiagnostic(
                "nonfinite_parameters", "gate parameters must be finite", span.line, span.column
            )
        )
    return CircuitOperation(
        instruction.operation.name,
        parameters,
        tuple(circuit.find_bit(q).index for q in instruction.qubits),
        tuple(circuit.find_bit(c).index for c in instruction.clbits),
        span,
    )
