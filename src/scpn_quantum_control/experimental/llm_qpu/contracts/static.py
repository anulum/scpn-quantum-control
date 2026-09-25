# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts


"""Static open-chain XY recipe and elementary digital gate list."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

from .compressed import CompressedLatentBatch
from .wire import (
    RESERVOIR_SCHEMA,
    STATIC_PLAN_SCHEMA,
    _digest,
    _positive_int,
    _text,
    canonical_bytes,
)


def _bounded_hex(value: object, *, name: str, limit: float, positive: bool = False) -> float:
    if type(value) is not str:
        raise ValueError(f"{name} must be canonical hex float")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be canonical hex float") from exc
    if (
        not math.isfinite(parsed)
        or abs(parsed) > limit
        or (positive and parsed <= 0.0)
        or (0.0 if parsed == 0.0 else parsed).hex() != value
    ):
        raise ValueError(f"{name} must be bounded canonical hex float")
    return parsed


@dataclass(frozen=True, slots=True)
class ReservoirSpec:
    """Exact dimensionless open-chain XY gate recipe for the static pilot."""

    kernel_id: str
    n_qubits: int
    edges: tuple[tuple[int, int], ...]
    couplings_hex: tuple[str, ...]
    fields_hex: tuple[str, ...]
    tau_hex: str
    repetitions: int
    initial_state: str
    angle_unit: str
    gate_order: str

    def __post_init__(self) -> None:
        """Refuse non-chain or implicitly transformed Hamiltonians."""
        if (
            self.kernel_id != "xy_static_digital_v1"
            or type(self.n_qubits) is not int
            or self.n_qubits not in (4, 8)
        ):
            raise ValueError("static reservoir requires a supported n4/n8 kernel")
        expected_edges = tuple((index, index + 1) for index in range(self.n_qubits - 1))
        if (
            type(self.edges) is not tuple
            or any(
                type(edge) is not tuple
                or len(edge) != 2
                or any(type(wire) is not int for wire in edge)
                for edge in self.edges
            )
            or self.edges != expected_edges
        ):
            raise ValueError("reservoir edges must be the explicit open chain")
        if type(self.couplings_hex) is not tuple or len(self.couplings_hex) != len(self.edges):
            raise ValueError("one coupling is required per ordered edge")
        if type(self.fields_hex) is not tuple or len(self.fields_hex) != self.n_qubits:
            raise ValueError("one local field is required per qubit")
        for coupling in self.couplings_hex:
            _bounded_hex(coupling, name="coupling", limit=10.0)
        for local_field in self.fields_hex:
            _bounded_hex(local_field, name="field", limit=10.0)
        _bounded_hex(self.tau_hex, name="tau", limit=10.0, positive=True)
        _positive_int(self.repetitions, name="repetitions", maximum=32)
        if (
            self.initial_state != "zero"
            or self.angle_unit != "radian"
            or self.gate_order != "encode_then_fields_even_odd_v1"
        ):
            raise ValueError("unsupported static preparation or gate convention")

    def to_wire(self) -> dict[str, object]:
        """Return a detached, complete reservoir recipe."""
        return {
            "schema": RESERVOIR_SCHEMA,
            "object_kind": "reservoir_spec",
            "kernel_id": self.kernel_id,
            "n_qubits": self.n_qubits,
            "edges": [list(edge) for edge in self.edges],
            "couplings_hex": list(self.couplings_hex),
            "fields_hex": list(self.fields_hex),
            "tau_hex": self.tau_hex,
            "repetitions": self.repetitions,
            "initial_state": self.initial_state,
            "angle_unit": self.angle_unit,
            "gate_order": self.gate_order,
        }

    @classmethod
    def from_wire(cls, value: object) -> ReservoirSpec:
        """Reject unknown fields, noncanonical edge layout and type coercion."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("reservoir spec fields mismatch")
        if value["schema"] != RESERVOIR_SCHEMA or value["object_kind"] != "reservoir_spec":
            raise ValueError("unsupported reservoir spec schema")
        for name in ("edges", "couplings_hex", "fields_hex"):
            if type(value[name]) is not list:
                raise ValueError(f"reservoir {name} must be a list")
        if any(type(edge) is not list or len(edge) != 2 for edge in value["edges"]):
            raise ValueError("reservoir edges must be index pairs")
        return cls(
            kernel_id=value["kernel_id"],
            n_qubits=value["n_qubits"],
            edges=tuple(tuple(edge) for edge in value["edges"]),
            couplings_hex=tuple(value["couplings_hex"]),
            fields_hex=tuple(value["fields_hex"]),
            tau_hex=value["tau_hex"],
            repetitions=value["repetitions"],
            initial_state=value["initial_state"],
            angle_unit=value["angle_unit"],
            gate_order=value["gate_order"],
        )


@dataclass(frozen=True, slots=True)
class StaticCircuitPlan:
    """Ordered elementary gates bound to one compressed contextual row."""

    reservoir_digest: str
    compressed_digest: str
    row_index: int
    sample_id: str
    arm_id: str
    operations: tuple[tuple[str, tuple[int, ...], str], ...]

    def __post_init__(self) -> None:
        """Bound the complete plan and reject unsupported gate vocabulary."""
        _digest(self.reservoir_digest, name="reservoir")
        _digest(self.compressed_digest, name="compressed batch")
        if type(self.row_index) is not int or not 0 <= self.row_index < 1_000_000:
            raise ValueError("plan row index out of bounds")
        _text(self.sample_id, name="plan sample")
        if self.arm_id not in ("xy", "encoding_plus_fields_no_coupling"):
            raise ValueError("unknown static arm")
        if type(self.operations) is not tuple or not 1 <= len(self.operations) <= 1024:
            raise ValueError("static operation inventory out of bounds")
        for operation in self.operations:
            if type(operation) is not tuple or len(operation) != 3:
                raise ValueError("invalid static operation")
            gate, wires, parameter = operation
            if gate not in ("ry", "rz", "rxx", "ryy") or type(wires) is not tuple:
                raise ValueError("unsupported static gate")
            if len(wires) != (1 if gate in ("ry", "rz") else 2):
                raise ValueError("static gate arity mismatch")
            if any(type(wire) is not int or not 0 <= wire < 8 for wire in wires):
                raise ValueError("static gate wire out of bounds")
            if len(set(wires)) != len(wires):
                raise ValueError("static gate repeats a wire")
            _bounded_hex(parameter, name="gate angle", limit=640.0)

    def to_wire(self) -> dict[str, object]:
        """Return the exact ordered gate list and source lineage."""
        return {
            "schema": STATIC_PLAN_SCHEMA,
            "object_kind": "static_circuit_plan",
            "reservoir_digest": self.reservoir_digest,
            "compressed_digest": self.compressed_digest,
            "row_index": self.row_index,
            "sample_id": self.sample_id,
            "arm_id": self.arm_id,
            "operations": [
                {"gate": gate, "wires": list(wires), "parameter_hex": parameter}
                for gate, wires, parameter in self.operations
            ],
        }

    @classmethod
    def from_wire(cls, value: object) -> StaticCircuitPlan:
        """Decode only the versioned exact gate-list representation."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("static circuit plan fields mismatch")
        if value["schema"] != STATIC_PLAN_SCHEMA or value["object_kind"] != "static_circuit_plan":
            raise ValueError("unsupported static circuit plan schema")
        operations = value["operations"]
        if type(operations) is not list or any(
            type(item) is not dict
            or set(item) != {"gate", "wires", "parameter_hex"}
            or type(item["wires"]) is not list
            for item in operations
        ):
            raise ValueError("static gate list fields mismatch")
        return cls(
            reservoir_digest=value["reservoir_digest"],
            compressed_digest=value["compressed_digest"],
            row_index=value["row_index"],
            sample_id=value["sample_id"],
            arm_id=value["arm_id"],
            operations=tuple(
                (item["gate"], tuple(item["wires"]), item["parameter_hex"]) for item in operations
            ),
        )


def build_static_circuit_plan(
    spec: ReservoirSpec,
    compressed: CompressedLatentBatch,
    angles_payload: bytes,
    *,
    row_index: int,
    arm_id: str,
) -> StaticCircuitPlan:
    """Build the fixed digital gate order from exact validated angle bytes."""
    if type(spec) is not ReservoirSpec or type(compressed) is not CompressedLatentBatch:
        raise ValueError("static builder requires reviewed contracts")
    if compressed.layout != "contextual" or compressed.tensor.shape[1] != spec.n_qubits:
        raise ValueError("static builder requires matching contextual angles")
    if type(row_index) is not int or not 0 <= row_index < len(compressed.sample_ids):
        raise ValueError("static row index out of bounds")
    if arm_id not in ("xy", "encoding_plus_fields_no_coupling"):
        raise ValueError("unknown static arm")
    compressed.tensor.validate_payload(angles_payload)
    angles = tuple(item[0] for item in struct.iter_unpack("<f", angles_payload))
    row = angles[row_index * spec.n_qubits : (row_index + 1) * spec.n_qubits]
    if any(not -math.pi / 2 < angle < math.pi / 2 for angle in row):
        raise ValueError("static row has an out-of-range angle")
    operations: list[tuple[str, tuple[int, ...], str]] = [
        ("ry", (wire,), (0.0 if angle == 0.0 else float(angle)).hex())
        for wire, angle in enumerate(row)
    ]
    dt = float.fromhex(spec.tau_hex) / spec.repetitions
    for _ in range(spec.repetitions):
        for wire, local_field in enumerate(spec.fields_hex):
            value = -2.0 * float.fromhex(local_field) * dt
            operations.append(("rz", (wire,), (0.0 if value == 0.0 else value).hex()))
        if arm_id == "xy":
            for parity in (0, 1):
                for edge, coupling in zip(spec.edges, spec.couplings_hex, strict=True):
                    if edge[0] % 2 == parity:
                        value = -2.0 * float.fromhex(coupling) * dt
                        parameter = (0.0 if value == 0.0 else value).hex()
                        operations.extend((("rxx", edge, parameter), ("ryy", edge, parameter)))
    return StaticCircuitPlan(
        reservoir_digest=hashlib.sha256(canonical_bytes(spec.to_wire())).hexdigest(),
        compressed_digest=hashlib.sha256(canonical_bytes(compressed.to_wire())).hexdigest(),
        row_index=row_index,
        sample_id=compressed.sample_ids[row_index],
        arm_id=arm_id,
        operations=tuple(operations),
    )


def validate_static_circuit_plan(
    spec: ReservoirSpec,
    compressed: CompressedLatentBatch,
    angles_payload: bytes,
    plan: StaticCircuitPlan,
) -> None:
    """Refuse any gate, source-row or arm drift from the exact digital recipe."""
    if type(plan) is not StaticCircuitPlan:
        raise ValueError("static plan must be a frozen contract")
    expected = build_static_circuit_plan(
        spec,
        compressed,
        angles_payload,
        row_index=plan.row_index,
        arm_id=plan.arm_id,
    )
    if plan != expected:
        raise ValueError("static plan differs from its frozen source and recipe")
