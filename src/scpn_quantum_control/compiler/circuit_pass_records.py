# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Circuit pass provenance records
"""Immutable source, operand and semantic-reference records for circuit passes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Half-open Unicode offsets and one-based start coordinates in source text.

    Parameters
    ----------
    start, end
        Python string offsets, with ``end`` excluded.
    line, column
        One-based coordinates of ``start`` in the original source.

    """

    start: int
    end: int
    line: int
    column: int

    def __post_init__(self) -> None:
        """Refuse negative offsets, empty spans and invalid coordinates."""
        if any(type(x) is not int for x in (self.start, self.end, self.line, self.column)):
            raise ValueError("source span fields must be integers")
        if not 0 <= self.start < self.end or min(self.line, self.column) < 1:
            raise ValueError("source span offsets and coordinates are invalid")


@dataclass(frozen=True, slots=True)
class CircuitDiagnostic:
    """Authored refusal and a location in the supplied or generated source.

    Parameters
    ----------
    code, message
        Stable refusal category and caller-safe authored explanation.
    line, column
        One-based source coordinates; no interpreter text is included.

    """

    code: str
    message: str
    line: int = 1
    column: int = 1

    def __post_init__(self) -> None:
        """Validate the authored refusal identity and one-based coordinates."""
        if (
            not isinstance(self.code, str)
            or not self.code
            or not isinstance(self.message, str)
            or not self.message
        ):
            raise ValueError("diagnostic code and message must be non-empty text")
        if any(type(i) is not int or i < 1 for i in (self.line, self.column)):
            raise ValueError("diagnostic coordinates must be positive integers")


class CircuitPassRefused(ValueError):
    """Refuse unsupported or semantics-changing input with a located diagnostic.

    Parameters
    ----------
    diagnostic
        Structured authored explanation retained on the exception.

    """

    def __init__(self, diagnostic: CircuitDiagnostic) -> None:
        """Store the diagnostic and expose its authored message only."""
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


@dataclass(frozen=True, slots=True)
class CircuitOperation:
    """A bound native operation with global operand indices and source location.

    Parameters
    ----------
    name
        Native Qiskit operation name.
    parameters
        Immutable finite real parameters in radians for rotation gates.
    qubits, clbits
        Global native little-endian operand indices.
    source_span
        The actual statement that produced this operation.

    """

    name: str
    parameters: tuple[float, ...]
    qubits: tuple[int, ...]
    clbits: tuple[int, ...]
    source_span: SourceSpan

    def __post_init__(self) -> None:
        """Validate immutable parameters and operands without allocating state."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("operation name must be non-empty")
        if not isinstance(self.parameters, tuple) or any(
            type(x) is not float or not math.isfinite(x) for x in self.parameters
        ):
            raise ValueError("operation parameters must be finite float tuples")
        for operands in (self.qubits, self.clbits):
            if not isinstance(operands, tuple) or any(
                type(x) is not int or x < 0 for x in operands
            ):
                raise ValueError("operation operands must be non-negative integer tuples")
        if not isinstance(self.source_span, SourceSpan):
            raise ValueError("operation requires a source span")
        if self.name == "measure" and (
            len(self.qubits) != 1 or len(self.clbits) != 1 or self.parameters
        ):
            raise ValueError("measurement requires exactly one qubit and classical bit")


@dataclass(frozen=True, slots=True)
class CircuitIR:
    """Frozen circuit instructions, registers, phase and original source text.

    Parameters
    ----------
    source
        Supplied OpenQASM 2 text or native canonical circuit export.
    num_qubits, num_clbits
        Native circuit widths before any dense reference allocation.
    global_phase
        Finite phase in radians, stored separately because QASM 2 omits it.
    operations
        Immutable ordered native instructions; measurements are trailing.
    quantum_registers, classical_registers
        Native register names and their global bit indices, preserving order.

    """

    source: str
    num_qubits: int
    num_clbits: int
    global_phase: float
    operations: tuple[CircuitOperation, ...]
    quantum_registers: tuple[tuple[str, tuple[int, ...]], ...]
    classical_registers: tuple[tuple[str, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        """Validate source binding and immutable native operand ranges."""
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("circuit source must be non-empty")
        if type(self.num_qubits) is not int or not 1 <= self.num_qubits <= 8:
            raise ValueError("circuit width must be between 1 and 8 qubits")
        if type(self.num_clbits) is not int or not 0 <= self.num_clbits <= 64:
            raise ValueError("circuit width must be between 0 and 64 classical bits")
        if type(self.global_phase) is not float or not math.isfinite(self.global_phase):
            raise ValueError("circuit phase must be finite")
        if not isinstance(self.operations, tuple) or len(self.operations) > 4096:
            raise ValueError("circuit operations must be a bounded immutable tuple")
        for operation in self.operations:
            if not isinstance(operation, CircuitOperation):
                raise ValueError("circuit operations must contain native operation records")
            span = operation.source_span
            if span.end > len(self.source):
                raise ValueError("operation source span exceeds source")
            if (span.line, span.column) != (
                self.source.count("\n", 0, span.start) + 1,
                span.start - self.source.rfind("\n", 0, span.start),
            ):
                raise ValueError("operation coordinates do not match source")
            if any(q >= self.num_qubits for q in operation.qubits) or any(
                c >= self.num_clbits for c in operation.clbits
            ):
                raise ValueError("operation operand exceeds circuit width")
        for registers, width in (
            (self.quantum_registers, self.num_qubits),
            (self.classical_registers, self.num_clbits),
        ):
            if not isinstance(registers, tuple):
                raise ValueError("registers must be immutable tuples")
            for register in registers:
                if not isinstance(register, tuple) or len(register) != 2:
                    raise ValueError("register entries must be immutable name/bit pairs")
                name, bits = register
                if not isinstance(name, str) or not name or not isinstance(bits, tuple):
                    raise ValueError("register name and bits are invalid")
                if any(type(bit) is not int or not 0 <= bit < width for bit in bits):
                    raise ValueError("register bit exceeds circuit width")

    @property
    def source_sha256(self) -> str:
        """SHA-256 of exact UTF-8 source bytes, including whitespace and comments."""
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    @property
    def measurements(self) -> tuple[tuple[int, int], ...]:
        """Ordered measured qubit/classical-bit pairs in global native indices."""
        return tuple(
            (operation.qubits[0], operation.clbits[0])
            for operation in self.operations
            if operation.name == "measure"
        )


@dataclass(frozen=True, slots=True)
class CircuitPassRecord:
    """Source-bound admission from the bounded native unitary reference.

    Parameters
    ----------
    pass_name
        Descriptive transformation identity.
    input_ir, output_ir
        Immutable snapshots taken when the pass was qualified.
    input_layout, output_layout
        Logical-to-physical qubit bijections on each side.
    output_classical_layout
        Logical-to-physical classical-bit bijection on the output side.
    observable_map
        Ordered input qubit/clbit to output qubit/clbit correspondence.
    global_phase_delta
        Phase in radians with ``U_out = exp(i*delta) U_in`` after layout mapping.
    operator_error, tolerance
        Maximum entry error after phase alignment and absolute admission bound.
    allow_global_phase
        Explicit equivalence policy used by the producer.
    reference_backend
        Executed local reference identity, distinct from emitted MLIR text.
    basis_convention, schema
        Native index convention and version of this source-bound record.

    """

    pass_name: str
    input_ir: CircuitIR
    output_ir: CircuitIR
    input_layout: tuple[int, ...]
    output_layout: tuple[int, ...]
    output_classical_layout: tuple[int, ...]
    observable_map: tuple[tuple[int, int, int, int], ...]
    global_phase_delta: float
    operator_error: float
    tolerance: float
    allow_global_phase: bool = True
    reference_backend: str = "qiskit.quantum_info.Operator"
    basis_convention: str = "qiskit_little_endian"
    schema: str = "circuit_pass.v1"

    def __post_init__(self) -> None:
        """Refuse mutable or internally inconsistent qualification records."""
        if not isinstance(self.pass_name, str) or not self.pass_name.strip():
            raise ValueError("pass name must be non-empty")
        if not isinstance(self.input_ir, CircuitIR) or not isinstance(self.output_ir, CircuitIR):
            raise ValueError("pass requires immutable input and output circuit records")
        if (self.input_ir.num_qubits, self.input_ir.num_clbits) != (
            self.output_ir.num_qubits,
            self.output_ir.num_clbits,
        ):
            raise ValueError("pass circuit widths must match")
        for layout, width in (
            (self.input_layout, self.input_ir.num_qubits),
            (self.output_layout, self.output_ir.num_qubits),
            (self.output_classical_layout, self.output_ir.num_clbits),
        ):
            if (
                not isinstance(layout, tuple)
                or any(type(i) is not int for i in layout)
                or sorted(layout) != list(range(width))
            ):
                raise ValueError("pass layout must be an immutable bit bijection")
        if not isinstance(self.observable_map, tuple) or any(
            not isinstance(row, tuple)
            or len(row) != 4
            or any(type(i) is not int or i < 0 for i in row)
            for row in self.observable_map
        ):
            raise ValueError("observable map must contain immutable operand correspondences")
        before = {c: self.input_layout.index(q) for q, c in self.input_ir.measurements}
        after = {
            self.output_classical_layout.index(c): self.output_layout.index(q)
            for q, c in self.output_ir.measurements
        }
        expected = tuple(
            (self.input_layout[q], c, self.output_layout[q], self.output_classical_layout[c])
            for c, q in before.items()
        )
        if before != after or self.observable_map != expected:
            raise ValueError("observable correspondence does not match circuit readout")
        if (
            type(self.allow_global_phase) is not bool
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in (self.global_phase_delta, self.operator_error, self.tolerance)
            )
            or not 0 <= self.operator_error <= self.tolerance <= 1e-12
            or self.tolerance == 0
        ):
            raise ValueError("pass qualification error and phase policy are invalid")
        if (self.reference_backend, self.basis_convention, self.schema) != (
            "qiskit.quantum_info.Operator",
            "qiskit_little_endian",
            "circuit_pass.v1",
        ):
            raise ValueError("pass reference identity and schema are unsupported")

    @property
    def sha256(self) -> str:
        """Deterministic SHA-256 binding both IRs, maps, phase policy and error."""
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
