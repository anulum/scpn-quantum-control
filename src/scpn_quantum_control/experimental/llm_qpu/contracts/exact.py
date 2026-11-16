# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Exact ideal-digital feature sets, checked against an independent pure-Python statevector."""

from __future__ import annotations

import cmath
import hashlib
import math
from dataclasses import dataclass

from .measurement import _observable_ids
from .plan_set import CircuitPlanSet
from .static import StaticCircuitPlan
from .wire import (
    EXACT_FEATURE_SET_SCHEMA,
    ArtifactHeader,
    _digest,
    _strict_json,
    _text,
    canonical_bytes,
    f64,
)

_BASES = ("X", "Y", "Z")
_SIMULATORS = ("qiskit_statevector",)


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _apply(state: list[complex], gate: str, wires: tuple[int, ...], theta: float) -> list[complex]:
    """Apply one RY, RZ, RXX or RYY gate in Qiskit's little-endian, exp(-i theta P / 2) form."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    out = list(state)
    mask = sum(1 << wire for wire in wires)
    for index, amplitude in enumerate(state):
        if gate == "rz":
            out[index] = amplitude * cmath.exp(
                -0.5j * theta if not index >> wires[0] & 1 else 0.5j * theta
            )
        elif gate == "ry":
            partner = state[index ^ mask]
            out[index] = c * amplitude + (s * partner if index >> wires[0] & 1 else -s * partner)
        elif gate == "rxx":
            out[index] = c * amplitude - 1j * s * state[index ^ mask]
        else:
            source = index ^ mask
            phase = -1 if (source >> wires[0] & 1) == (source >> wires[1] & 1) else 1
            out[index] = c * amplitude - 1j * s * phase * state[source]
    return out


def _pauli_expectation(state: list[complex], pauli: str, wires: tuple[int, ...]) -> float:
    """Return the real expectation of one same-axis Pauli string on the given wires."""
    mask = sum(1 << wire for wire in wires) if pauli != "Z" else 0
    total = 0.0 + 0.0j
    for index, amplitude in enumerate(state):
        source = index ^ mask
        factor: complex = 1.0
        for wire in wires:
            bit = source >> wire & 1
            factor *= (
                (1 - 2 * bit) if pauli == "Z" else (1j if bit == 0 else -1j) if pauli == "Y" else 1
            )
        total += amplitude.conjugate() * factor * state[source]
    return total.real


def exact_expectations(plan: StaticCircuitPlan, n_qubits: int, basis: str) -> tuple[float, ...]:
    """Return the plan's ideal single-site and adjacent same-axis expectations in one basis."""
    state = [1.0 + 0.0j] + [0.0j] * ((1 << n_qubits) - 1)
    for gate, wires, parameter in plan.operations:
        state = _apply(state, gate, wires, float.fromhex(parameter))
    singles = tuple(_pauli_expectation(state, basis, (wire,)) for wire in range(n_qubits))
    pairs = tuple(
        _pauli_expectation(state, basis, (wire, wire + 1)) for wire in range(n_qubits - 1)
    )
    return singles + pairs


@dataclass(frozen=True, slots=True)
class ExactRow:
    """One plan's exact expectations, one tuple per basis in observable order."""

    row_index: int
    sample_id: str
    arm_id: str
    plan_digest: str
    expectations: tuple[tuple[str, tuple[float, ...]], ...]

    def __post_init__(self) -> None:
        """Require named lineage and finite expectations in [-1, 1] per sorted basis."""
        if type(self.row_index) is not int or not 0 <= self.row_index < 4096:
            raise ValueError("row index must be a bounded non-negative integer")
        for name in ("sample_id", "arm_id"):
            _text(getattr(self, name), name=name)
        _digest(self.plan_digest, name="plan digest")
        pairs = self.expectations
        if (
            type(pairs) is not tuple
            or not pairs
            or any(type(p) is not tuple or len(p) != 2 for p in pairs)
        ):
            raise ValueError("expectations must be (basis, values) pairs")
        if [basis for basis, _ in pairs] != sorted({basis for basis, _ in pairs}) or any(
            b not in _BASES for b, _ in pairs
        ):
            raise ValueError("expectation bases must be sorted, unique and X, Y or Z")
        for _, values in pairs:
            if type(values) is not tuple or len(values) not in (7, 15):
                raise ValueError("an n4 or n8 basis has 7 or 15 observables")
            for value in values:
                if (
                    type(value) is not float
                    or not math.isfinite(value)
                    or not -1.0 - 1e-12 <= value <= 1.0 + 1e-12
                ):
                    raise ValueError("an ideal Pauli expectation is a finite float in [-1, 1]")

    def to_wire(self) -> dict[str, object]:
        """Return the exact row with f64-tagged expectations."""
        return {
            "row_index": self.row_index,
            "sample_id": self.sample_id,
            "arm_id": self.arm_id,
            "plan_digest": self.plan_digest,
            "expectations": [
                [basis, [f64(v) for v in values]] for basis, values in self.expectations
            ],
        }


@dataclass(frozen=True, slots=True)
class ExactFeatureSet:
    """Design-only ideal-digital expectations for every plan of one plan set."""

    plan_set_digest: str
    simulator: str
    simulator_version: str
    rows: tuple[ExactRow, ...]
    tolerance_hex: str
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse an unnamed simulator, an unbounded tolerance and unordered rows."""
        _digest(self.plan_set_digest, name="plan set digest")
        if self.simulator not in _SIMULATORS:
            raise ValueError("unknown ideal-digital simulator")
        _text(self.simulator_version, name="simulator version")
        rows = self.rows
        if type(rows) is not tuple or not rows or any(type(row) is not ExactRow for row in rows):
            raise ValueError("rows must be a nonempty tuple of ExactRow records")
        if len({(row.row_index, row.arm_id) for row in rows}) != len(rows):
            raise ValueError("rows must name each plan once")
        try:
            tolerance = float.fromhex(self.tolerance_hex)
        except (TypeError, ValueError):
            tolerance = math.nan
        if not 0.0 < tolerance <= 1e-9 or tolerance.hex() != self.tolerance_hex:
            raise ValueError("tolerance must be a canonical hex float in (0, 1e-9]")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "exact_feature_set"
        ):
            raise ValueError("ExactFeatureSet requires its exact artifact header")
        self.header.validate_content(self._scientific_wire(), parents=(self.plan_set_digest,))

    def _scientific_wire(self) -> dict[str, object]:
        """Return the feature-set content hashed by its header."""
        return {
            "schema": EXACT_FEATURE_SET_SCHEMA,
            "object_kind": "exact_feature_set",
            "plan_set_digest": self.plan_set_digest,
            "simulator": self.simulator,
            "simulator_version": self.simulator_version,
            "rows": [row.to_wire() for row in self.rows],
            "tolerance_hex": self.tolerance_hex,
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned feature set."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> ExactFeatureSet:
        """Reject unknown fields and restore rows and f64 values."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("ExactFeatureSet fields mismatch")
        if (
            value["schema"] != EXACT_FEATURE_SET_SCHEMA
            or value["object_kind"] != "exact_feature_set"
        ):
            raise ValueError("unknown ExactFeatureSet schema")
        if type(value["rows"]) is not list:
            raise ValueError("rows must be a list on wire")
        return cls(
            plan_set_digest=value["plan_set_digest"],
            simulator=value["simulator"],
            simulator_version=value["simulator_version"],
            rows=tuple(_row_from_wire(row) for row in value["rows"]),
            tolerance_hex=value["tolerance_hex"],
            header=ArtifactHeader.from_wire(value["header"]),
        )


def _row_from_wire(value: object) -> ExactRow:
    """Decode one exact row."""
    fields = {"row_index", "sample_id", "arm_id", "plan_digest", "expectations"}
    if type(value) is not dict or set(value) != fields or type(value["expectations"]) is not list:
        raise ValueError("ExactRow fields mismatch")
    pairs = []
    for pair in value["expectations"]:
        if type(pair) is not list or len(pair) != 2 or type(pair[1]) is not list:
            raise ValueError("expectations must be [basis, values] pairs on wire")
        numbers = []
        for item in pair[1]:
            if type(item) is not dict or set(item) != {"$f64"} or type(item["$f64"]) is not str:
                raise ValueError("expectations must be f64 tags on wire")
            numbers.append(float.fromhex(item["$f64"]))
        pairs.append((pair[0], tuple(numbers)))
    return ExactRow(
        value["row_index"], value["sample_id"], value["arm_id"], value["plan_digest"], tuple(pairs)
    )


def validate_exact_feature_set(
    features: ExactFeatureSet, plan_set: CircuitPlanSet, n_qubits: int
) -> None:
    """Refuse another plan set, misaligned rows or any value off the independent statevector."""
    if type(features) is not ExactFeatureSet:
        raise ValueError("exact validation needs an ExactFeatureSet")
    if features.plan_set_digest != _sha(plan_set.to_wire()):
        raise ValueError("the features name a different plan set")
    if features.header.data_origin != plan_set.header.data_origin:
        raise ValueError("feature origin differs from the plan set")
    if len(features.rows) != len(plan_set.plans):
        raise ValueError("every plan needs exactly one exact row")
    tolerance = float.fromhex(features.tolerance_hex)
    for row, plan in zip(features.rows, plan_set.plans, strict=True):
        lineage = (plan.row_index, plan.sample_id, plan.arm_id, _sha(plan.to_wire()))
        if (row.row_index, row.sample_id, row.arm_id, row.plan_digest) != lineage:
            raise ValueError("an exact row does not follow its plan")
        for basis, values in row.expectations:
            if len(values) != len(_observable_ids(basis, n_qubits)):
                raise ValueError("an exact row has the wrong observable count")
            expected = exact_expectations(plan, n_qubits, basis)
            if any(abs(a - b) > tolerance for a, b in zip(values, expected, strict=True)):
                raise ValueError("an exact expectation differs from the independent statevector")


def decode_exact_feature_set(raw: bytes) -> ExactFeatureSet:
    """Decode one exact feature set from strict canonical wire bytes."""
    return ExactFeatureSet.from_wire(_strict_json(raw))
