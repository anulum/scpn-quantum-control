# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — supported source through the native compiler
"""Bounded native OpenQASM2 emission preserving readout and classical conditions."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit import ClassicalRegister, Instruction
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.library import RYYGate

from .program_authoring_contracts import (
    CompiledProgram,
    ProgramCondition,
    ProgramDiagnostic,
    ProgramOperation,
    ProgramSourceRefused,
    ProgramSourceSpan,
)

_LEXEME = re.compile(
    r'//[^\n]*|[ \t\r\n\f\v]+|"[^"\r\n]*"|[A-Za-z_][A-Za-z0-9_]*'
    r"|->|==|[0-9.+-][A-Za-z0-9.+-]*|[\[\](),;]"
)
_GATES = {
    **dict.fromkeys(("h", "x", "y", "z", "s", "sdg", "t", "tdg", "id", "sx", "sxdg"), (0, 1)),
    **dict.fromkeys(("rx", "ry", "rz", "p", "u1"), (1, 1)),
    "u2": (2, 1),
    "u": (3, 1),
    "u3": (3, 1),
    **dict.fromkeys(("cx", "cz", "swap"), (0, 2)),
    **dict.fromkeys(("rxx", "ryy", "rzz"), (1, 2)),
    "measure": (0, 1),
    "reset": (0, 1),
    "barrier": (0, 0),
}
_NATIVE_INSTRUCTIONS = (
    *qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    qasm2.CustomInstruction("ryy", 1, 2, RYYGate, builtin=True),
)


def compile_program_source(source: str) -> CompiledProgram:
    """Import the supported source subset through the actual native Qiskit parser.

    Parameters
    ----------
    source
        OpenQASM2 UTF8 text, at most1MiB. Only qelib1.inc and a declared q register
        of at most8 qubits plus optional c register of at most64 bits are admitted.
        Indexed gates use finite decimal parameters in radians. Readout, reset,
        barriers and whole-register conditional gates remain explicit effects.

    Returns
    -------
    CompiledProgram
        Immutable exact source, operands, IEEE phase parameters, readout,
        classical conditions and original locations. Emission does not execute.

    Raises
    ------
    ProgramSourceRefused
        Located syntax, unsupported operation, include, width or budget refusal.
        Import never evaluates Python, performs filesystem includes or submits work.

    """
    return _compile(source)[0]


def import_program_source(source: str) -> QuantumCircuit:
    """Return the actual native circuit after supported source admission.

    Parameters
    ----------
    source
        The same bounded OpenQASM2 subset as compile_program_source.

    Returns
    -------
    QuantumCircuit
        Fresh native circuit preserving measurements and conditional blocks.
        The circuit is constructed without numerical or provider execution.

    Raises
    ------
    ProgramSourceRefused
        Located admission or native parser refusal before returning a circuit.

    """
    return _compile(source)[1]


def export_program_source(circuit: QuantumCircuit) -> str:
    """Export actual admitted native operations with exact decimal phase parameters.

    Parameters
    ----------
    circuit
        Native circuit with one q register and optional c register within the
        supported bounds. Whole-register if blocks must contain exactly one
        admitted gate with no else branch. Nonzero global phase is refused
        because OpenQASM2 cannot encode it independently of gate phases.
        Operations must use the compiler's standard native gate classes;
        a custom or altered definition cannot borrow a supported gate name.

    Returns
    -------
    str
        Reimportable source with roundtrippable decimal parameters instead of approximate
        pi aliases. Readout and original classical controls are retained exactly.

    Raises
    ------
    ProgramSourceRefused
        Unsupported circuit shape, phase, condition, operation or parameter.
        Refusal leaves the native circuit unchanged.

    """
    span = ProgramSourceSpan(0, 0, 1, 1)
    if (
        not isinstance(circuit, QuantumCircuit)
        or not 1 <= circuit.num_qubits <= 8
        or circuit.num_clbits > 64
        or len(circuit.data) > 4096
        or len(circuit.qregs) != 1
        or circuit.qregs[0].name != "q"
        or len(circuit.qregs[0]) != circuit.num_qubits
        or len(circuit.cregs) > 1
        or (circuit.cregs and circuit.cregs[0].name != "c")
        or (
            circuit.num_clbits
            and (not circuit.cregs or len(circuit.cregs[0]) != circuit.num_clbits)
        )
        or circuit.num_parameters
        or circuit.global_phase != 0
    ):
        raise _refuse(
            "unsupported_export", "Circuit is outside the exact supported source export.", span
        )
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circuit.num_qubits}];"]
    if circuit.num_clbits:
        lines.append(f"creg c[{circuit.num_clbits}];")
    native_operations: list[Instruction] = []
    for instruction in circuit.data:
        operation = instruction.operation
        gate_qubits = instruction.qubits
        prefix = ""
        if isinstance(operation, IfElseOp):
            condition = operation.condition
            if (
                not isinstance(condition, tuple)
                or not isinstance(condition[0], ClassicalRegister)
                or not circuit.cregs
                or condition[0] != circuit.cregs[0]
                or len(operation.blocks) != 1
                or len(operation.blocks[0].data) != 1
                or operation.blocks[0].global_phase != 0
            ):
                raise _refuse(
                    "unsupported_export",
                    "Conditional export requires one gate and no else branch.",
                    span,
                )
            prefix = f"if(c=={condition[1]}) "
            block = operation.blocks[0]
            inner = block.data[0]
            gate_qubits = tuple(
                instruction.qubits[block.find_bit(bit).index] for bit in inner.qubits
            )
            operation = inner.operation
        if operation.name not in _GATES:
            raise _refuse("unsupported_export", "Operation has no supported source export.", span)
        native_operations.append(operation)
        try:
            parameters = [float(value) for value in operation.params]
        except (TypeError, ValueError) as error:
            raise _refuse(
                "unsupported_export", "Source export requires bound real parameters.", span
            ) from error
        if any(not math.isfinite(value) for value in parameters):
            raise _refuse("unsupported_export", "Source export requires finite parameters.", span)
        arguments = (
            f"({','.join(format(value, '.17e') for value in parameters)})" if parameters else ""
        )
        operands = ",".join(f"q[{circuit.find_bit(bit).index}]" for bit in gate_qubits)
        if operation.name == "measure":
            if len(instruction.clbits) != 1:
                raise _refuse(
                    "unsupported_export", "Measurement requires one classical destination.", span
                )
            operands += f" -> c[{circuit.find_bit(instruction.clbits[0]).index}]"
        lines.append(f"{prefix}{operation.name}{arguments} {operands};")
    source = "\n".join(lines)
    # Reimport validates exact arities, widths, native conditions and original bits.
    restored = _compile(source)[1]
    for original, instruction in zip(native_operations, restored.data, strict=True):
        operation = instruction.operation
        if isinstance(operation, IfElseOp):
            operation = operation.blocks[0].data[0].operation
        if type(original) not in (type(operation), operation.base_class):
            raise _refuse(
                "unsupported_export",
                "Operation differs from its supported native definition.",
                span,
            )
    return source


@dataclass(frozen=True, slots=True)
class _Token:
    """An inert lexical token bound to its exact original source span."""

    text: str
    span: ProgramSourceSpan


def _refuse(code: str, message: str, span: ProgramSourceSpan) -> ProgramSourceRefused:
    """Bind a stable authored refusal without copying an interpreter message."""
    return ProgramSourceRefused(ProgramDiagnostic(code, message, span))


def _tokens(source: str) -> list[_Token]:
    """Scan bounded inert tokens with one-based Unicode scalar coordinates."""
    origin = ProgramSourceSpan(0, int(isinstance(source, str) and bool(source)), 1, 1)
    if not isinstance(source, str) or not source.strip():
        raise _refuse("invalid_source", "Source must contain a supported program.", origin)
    try:
        size = len(source.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise _refuse("invalid_source", "Source must be valid UTF8 text.", origin) from error
    if size > 1_048_576:
        raise _refuse("source_budget", "Source exceeds the1MiB import budget.", origin)
    tokens: list[_Token] = []
    cursor, line, column = 0, 1, 1
    while cursor < len(source):
        match = _LEXEME.match(source, cursor)
        if match is None:
            raise _refuse(
                "invalid_token",
                "Token is outside the supported source subset.",
                ProgramSourceSpan(cursor, cursor + 1, line, column),
            )
        text = match[0]
        span = ProgramSourceSpan(cursor, match.end(), line, column)
        if not (text.startswith("//") or text.isspace()):
            tokens.append(_Token(text, span))
            if len(tokens) > 65_536:
                raise _refuse("source_budget", "Source exceeds the bounded token budget.", span)
        lines = text.count("\n")
        column = len(text.rsplit("\n", 1)[-1]) + 1 if lines else column + len(text)
        line += lines
        cursor = match.end()
    return tokens


class _Parser:
    """Nonrecursive supported grammar used only to bound the native import."""

    def __init__(self, source: str) -> None:
        """Bind original tokens and register widths before native allocation."""
        self.source = source
        self.tokens = _tokens(source)
        self.cursor = 0
        self.num_qubits = 0
        self.num_clbits = 0

    def peek(self, text: str) -> bool:
        """Inspect one literal without changing source or parser state."""
        return self.cursor < len(self.tokens) and self.tokens[self.cursor].text == text

    def take(self) -> _Token:
        """Consume one original token or locate a missing token at EOF."""
        if self.cursor == len(self.tokens):
            span = ProgramSourceSpan(
                len(self.source),
                len(self.source),
                self.source.count("\n") + 1,
                len(self.source.rsplit("\n", 1)[-1]) + 1,
            )
            raise _refuse("invalid_source", "Source is missing a required token.", span)
        token = self.tokens[self.cursor]
        self.cursor += 1
        return token

    def expect(self, text: str) -> _Token:
        """Require a supported grammar token with its exact refusal location."""
        token = self.take()
        if token.text != text:
            raise _refuse(
                "invalid_source", "Token does not match the supported source grammar.", token.span
            )
        return token

    def integer(self, maximum: int, code: str, message: str, *, positive: bool = False) -> int:
        """Bound integer text before conversion, avoiding arbitrary-size parsing."""
        token = self.take()
        if (
            not token.text.isascii()
            or not token.text.isdigit()
            or len(token.text) > 20
            or int(token.text) > maximum
            or (len(token.text) > 1 and token.text.startswith("0"))
            or (positive and int(token.text) == 0)
        ):
            raise _refuse(code, message, token.span)
        return int(token.text)

    def operand(self, register: str) -> int:
        """Preserve one native index within its previously admitted register."""
        self.expect(register)
        self.expect("[")
        width = self.num_qubits if register == "q" else self.num_clbits
        index = self.integer(
            width - 1, "invalid_operand", "Operand is outside the declared register."
        )
        self.expect("]")
        return index

    def condition(self) -> ProgramCondition | None:
        """Retain a supported whole-register comparison as an exact decimal."""
        if not self.peek("if"):
            return None
        self.expect("if")
        self.expect("(")
        self.expect("c")
        self.expect("==")
        value = self.integer(
            (1 << self.num_clbits) - 1 if self.num_clbits else -1,
            "invalid_condition",
            "Condition value must fit the declared classical register.",
        )
        self.expect(")")
        return ProgramCondition("c", str(value))

    def operation(self) -> ProgramOperation:
        """Validate one exact gate/effect statement without executing it."""
        start = self.tokens[self.cursor].span
        condition = self.condition()
        gate = self.take()
        shape = _GATES.get(gate.text)
        if shape is None:
            raise _refuse(
                "unsupported_operation",
                "Operation is outside the supported program subset.",
                gate.span,
            )
        if condition is not None and gate.text in ("measure", "reset", "barrier"):
            raise _refuse(
                "unsupported_operation",
                "Classical conditions are supported only on gates.",
                gate.span,
            )
        parameter_count, qubit_count = shape
        parameters: list[str] = []
        if parameter_count:
            self.expect("(")
            for index in range(parameter_count):
                if index:
                    self.expect(",")
                token = self.take()
                try:
                    value = float(token.text)
                except ValueError as error:
                    raise _refuse(
                        "invalid_parameter",
                        "Gate parameters must be finite decimal numbers in radians.",
                        token.span,
                    ) from error
                if not math.isfinite(value):
                    raise _refuse(
                        "invalid_parameter",
                        "Gate parameters must be finite decimal numbers in radians.",
                        token.span,
                    )
                parameters.append(struct.pack(">d", value).hex())
            self.expect(")")
        if (
            gate.text == "barrier"
            and self.peek("q")
            and self.cursor + 1 < len(self.tokens)
            and self.tokens[self.cursor + 1].text == ";"
        ):
            self.expect("q")
            qubits = list(range(self.num_qubits))
        else:
            qubits = [self.operand("q")]
            while (not qubit_count or len(qubits) < qubit_count) and self.peek(","):
                self.expect(",")
                qubits.append(self.operand("q"))
            if qubit_count and len(qubits) != qubit_count:
                raise _refuse(
                    "invalid_operand",
                    "Gate operand count does not match its declared arity.",
                    gate.span,
                )
        if len(set(qubits)) != len(qubits):
            raise _refuse(
                "invalid_operand", "An operation cannot repeat the same qubit operand.", gate.span
            )
        clbits: tuple[int, ...] = ()
        if gate.text == "measure":
            self.expect("->")
            clbits = (self.operand("c"),)
        end = self.expect(";").span.end
        return ProgramOperation(
            gate.text,
            tuple(parameters),
            tuple(qubits),
            clbits,
            condition,
            ProgramSourceSpan(start.start, end, start.line, start.column),
        )

    def parse(self) -> tuple[ProgramOperation, ...]:
        """Admit fixed source/register headers before any operations or native parse."""
        for literal in ("OPENQASM", "2.0", ";", "include"):
            self.expect(literal)
        include = self.take()
        if include.text != '"qelib1.inc"':
            raise _refuse(
                "unsupported_include",
                "Only the native qelib1.inc include is supported.",
                include.span,
            )
        for literal in (";", "qreg", "q", "["):
            self.expect(literal)
        self.num_qubits = self.integer(
            8, "circuit_budget", "Register width is outside the supported budget.", positive=True
        )
        self.expect("]")
        self.expect(";")
        if self.peek("creg"):
            for literal in ("creg", "c", "["):
                self.expect(literal)
            self.num_clbits = self.integer(
                64,
                "circuit_budget",
                "Register width is outside the supported budget.",
                positive=True,
            )
            self.expect("]")
            self.expect(";")
        operations: list[ProgramOperation] = []
        while self.cursor < len(self.tokens):
            if len(operations) == 4096:
                raise _refuse(
                    "circuit_budget",
                    "Program exceeds4096 operations.",
                    self.tokens[self.cursor].span,
                )
            operations.append(self.operation())
        return tuple(operations)


def _compile(source: str) -> tuple[CompiledProgram, QuantumCircuit]:
    """Bind the admitted grammar to actual native instructions and parameters."""
    parser = _Parser(source)
    expected = parser.parse()
    try:
        circuit = qasm2.loads(
            source,
            include_path=(),
            strict=True,
            custom_instructions=_NATIVE_INSTRUCTIONS,
        )
    except (qasm2.QASM2ParseError, RecursionError) as error:
        raise _refuse(
            "invalid_source",
            "Source is not valid in the native supported subset.",
            ProgramSourceSpan(0, min(len(source), 1), 1, 1),
        ) from error
    actual = []
    if len(expected) != len(circuit.data):
        raise _refuse(
            "source_mismatch",
            "Native operations do not match the supported source record.",
            ProgramSourceSpan(0, min(len(source), 1), 1, 1),
        )
    for original, instruction in zip(expected, circuit.data, strict=True):
        operation = instruction.operation
        condition = None
        if isinstance(operation, IfElseOp):
            native_condition = operation.condition
            if not isinstance(native_condition, tuple) or not isinstance(
                native_condition[0], ClassicalRegister
            ):
                raise _refuse(
                    "unsupported_operation",
                    "Native condition is outside the supported subset.",
                    original.source_span,
                )
            condition = ProgramCondition(native_condition[0].name, str(native_condition[1]))
            operation = operation.blocks[0].data[0].operation
        actual.append(
            ProgramOperation(
                operation.name,
                tuple(struct.pack(">d", float(value)).hex() for value in operation.params),
                tuple(circuit.find_bit(bit).index for bit in instruction.qubits),
                tuple(circuit.find_bit(bit).index for bit in instruction.clbits)
                if condition is None
                else (),
                condition,
                original.source_span,
            )
        )
    if tuple(actual) != expected:
        raise _refuse(
            "source_mismatch",
            "Native operations do not match the supported source record.",
            ProgramSourceSpan(0, min(len(source), 1), 1, 1),
        )
    plan = CompiledProgram(
        "studio.program-source.v1",
        source,
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
        parser.num_qubits,
        parser.num_clbits,
        tuple(actual),
        tuple((op.qubits[0], op.clbits[0]) for op in actual if op.name == "measure"),
    )
    return plan, circuit
