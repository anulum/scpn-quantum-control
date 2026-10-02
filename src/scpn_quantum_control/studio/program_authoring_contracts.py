# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — immutable source compilation records
"""Exact source, operands, phase parameters and located emitted-only diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProgramSourceSpan:
    """Half-open scalar source offsets and one-based original coordinates.

    Parameters
    ----------
    start, end
        Unicode scalar offsets. Equal offsets describe a missing token at EOF.
    line, column
        One-based coordinates of the span start.

    """

    start: int
    end: int
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class ProgramDiagnostic:
    """A stable authored refusal that retains the original offending token.

    Parameters
    ----------
    code, message
        Stable refusal category and caller-safe explanation.
    source_span
        Original scalar offset and coordinates, including zero-width EOF.

    """

    code: str
    message: str
    source_span: ProgramSourceSpan


class ProgramSourceRefused(ValueError):
    """Refuse a source import before producing a successful compilation.

    Parameters
    ----------
    diagnostic
        Located authored refusal; native interpreter text is never exposed.

    """

    def __init__(self, diagnostic: ProgramDiagnostic) -> None:
        """Retain the exact original diagnostic and its authored message."""
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


@dataclass(frozen=True, slots=True)
class ProgramCondition:
    """A whole-register comparison with exact unsigned integer identity.

    Parameters
    ----------
    register
        Declared classical register name, c in the supported subset.
    value
        Canonical decimal string, preserving up to 64 classical bits.

    """

    register: str
    value: str


@dataclass(frozen=True, slots=True)
class ProgramOperation:
    """One original native gate or effect without numerical execution.

    Parameters
    ----------
    name
        Original native gate or measure/reset/barrier name.
    parameters
        Ordered sixteen-digit IEEE754 float64 hex values; rotations use radians.
    qubits, clbits
        Global native operand indices, preserving exact argument order.
    condition
        Original whole-register condition, or None for an unconditional operation.
    source_span
        Original statement span, including any conditional prefix.

    """

    name: str
    parameters: tuple[str, ...]
    qubits: tuple[int, ...]
    clbits: tuple[int, ...]
    condition: ProgramCondition | None
    source_span: ProgramSourceSpan


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    """Immutable native source emission with readout and effect provenance.

    Parameters
    ----------
    schema
        Versioned studio.program-source.v1 wire identifier.
    source, source_sha256
        Exact original source and its UTF8 SHA256.
    num_qubits, num_clbits
        Declared native register widths, bounded to 8 qubits and 64 classical bits.
    operations
        Ordered original gates and effects with exact source locations.
    measurements
        Ordered qubit/classical-bit pairs, including repeated readout.
    execution_status
        Emitted-not-executed boundary; compilation is not a runtime result.

    """

    schema: str
    source: str
    source_sha256: str
    num_qubits: int
    num_clbits: int
    operations: tuple[ProgramOperation, ...]
    measurements: tuple[tuple[int, int], ...]
    execution_status: str = "emitted_not_executed"

    def to_dict(self) -> dict[str, Any]:
        """Return the exact Rust/WASM JSON wire projection.

        Returns
        -------
        dict
            Original source and ordered arrays with exact parameter/integer strings.
            This projection does not execute or mutate the compiled program.

        """
        return {
            "schema": self.schema,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "operations": [
                {
                    "name": op.name,
                    "parameters": list(op.parameters),
                    "qubits": list(op.qubits),
                    "clbits": list(op.clbits),
                    "condition": asdict(op.condition) if op.condition is not None else None,
                    "source_span": asdict(op.source_span),
                }
                for op in self.operations
            ],
            "measurements": [list(pair) for pair in self.measurements],
            "execution_status": self.execution_status,
        }
