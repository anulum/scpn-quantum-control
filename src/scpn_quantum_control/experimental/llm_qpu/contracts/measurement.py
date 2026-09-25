# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU measurement contract
"""Exact common-axis readout plan with explicit classical-bit custody."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from .static import ReservoirSpec, StaticCircuitPlan
from .wire import MEASUREMENT_PLAN_SCHEMA, _digest, _positive_int, _text, canonical_bytes


def _observable_ids(basis_id: str, n_qubits: int) -> tuple[str, ...]:
    """Return the frozen single-site and neighboring same-axis feature order."""
    return tuple(f"{basis_id}{wire}" for wire in range(n_qubits)) + tuple(
        f"{basis_id}{wire}{basis_id}{wire + 1}" for wire in range(n_qubits - 1)
    )


@dataclass(frozen=True, slots=True)
class MeasurementPlan:
    """One basis cell and its complete logical/classical display convention.

    ``physical_qubits=None`` is a typed planning unknown. A provider compiler
    must bind it before hardware execution; it is never interpreted as an
    identity layout. The raw bit string is indexed through
    ``display_order_clbits`` rather than reversed or sliced implicitly.
    """

    circuit_plan_digest: str
    n_qubits: int
    basis_id: str
    shots: int
    observable_ids: tuple[str, ...]
    logical_to_clbit: tuple[int, ...]
    display_order_clbits: tuple[int, ...]
    register_name: str
    physical_qubits: tuple[int, ...] | None
    pauli_label_convention: str

    def __post_init__(self) -> None:
        """Refuse wrong axes, absent bit custody, and implicit physical maps."""
        _digest(self.circuit_plan_digest, name="circuit plan")
        if type(self.n_qubits) is not int or self.n_qubits not in (4, 8):
            raise ValueError("measurement plan requires n4/n8")
        if self.basis_id not in ("X", "Y", "Z"):
            raise ValueError("measurement basis must be X, Y or Z")
        _positive_int(self.shots, name="measurement shots", maximum=1_000_000)
        if type(self.observable_ids) is not tuple or self.observable_ids != _observable_ids(
            self.basis_id, self.n_qubits
        ):
            raise ValueError("measurement observable order or basis mismatch")
        expected_clbits = set(range(self.n_qubits))
        if (
            type(self.logical_to_clbit) is not tuple
            or len(self.logical_to_clbit) != self.n_qubits
            or any(type(bit) is not int for bit in self.logical_to_clbit)
            or set(self.logical_to_clbit) != expected_clbits
        ):
            raise ValueError("logical-to-classical map must be a complete permutation")
        if (
            type(self.display_order_clbits) is not tuple
            or len(self.display_order_clbits) != self.n_qubits
            or any(type(bit) is not int for bit in self.display_order_clbits)
            or set(self.display_order_clbits) != expected_clbits
        ):
            raise ValueError("display bit order must be a complete permutation")
        if _text(self.register_name, name="measurement register") != self.register_name:
            raise ValueError("measurement register must be NFC text")
        if self.physical_qubits is not None and (
            type(self.physical_qubits) is not tuple
            or len(self.physical_qubits) != self.n_qubits
            or any(type(wire) is not int or not 0 <= wire < 128 for wire in self.physical_qubits)
            or len(set(self.physical_qubits)) != self.n_qubits
        ):
            raise ValueError("physical qubits must be explicit unique bounded indices")
        if self.pauli_label_convention != "logical_q0_low_index_v1":
            raise ValueError("unsupported Pauli label convention")

    def to_wire(self) -> dict[str, object]:
        """Return a detached exact readout plan, including unknown layout."""
        return {
            "schema": MEASUREMENT_PLAN_SCHEMA,
            "object_kind": "measurement_plan",
            "circuit_plan_digest": self.circuit_plan_digest,
            "n_qubits": self.n_qubits,
            "basis_id": self.basis_id,
            "shots": self.shots,
            "observable_ids": list(self.observable_ids),
            "logical_to_clbit": list(self.logical_to_clbit),
            "display_order_clbits": list(self.display_order_clbits),
            "register_name": self.register_name,
            "physical_qubits": None
            if self.physical_qubits is None
            else list(self.physical_qubits),
            "pauli_label_convention": self.pauli_label_convention,
        }

    @classmethod
    def from_wire(cls, value: object) -> MeasurementPlan:
        """Reject missing/extra fields and coercions in the bit map."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("measurement plan fields mismatch")
        if (
            value["schema"] != MEASUREMENT_PLAN_SCHEMA
            or value["object_kind"] != "measurement_plan"
        ):
            raise ValueError("unsupported measurement plan schema")
        for name in ("observable_ids", "logical_to_clbit", "display_order_clbits"):
            if type(value[name]) is not list:
                raise ValueError(f"measurement {name} must be a list")
        physical = value["physical_qubits"]
        if physical is not None and type(physical) is not list:
            raise ValueError("measurement physical qubits must be a list or null")
        return cls(
            circuit_plan_digest=value["circuit_plan_digest"],
            n_qubits=value["n_qubits"],
            basis_id=value["basis_id"],
            shots=value["shots"],
            observable_ids=tuple(value["observable_ids"]),
            logical_to_clbit=tuple(value["logical_to_clbit"]),
            display_order_clbits=tuple(value["display_order_clbits"]),
            register_name=value["register_name"],
            physical_qubits=None if physical is None else tuple(physical),
            pauli_label_convention=value["pauli_label_convention"],
        )

    @property
    def scientific_digest(self) -> str:
        """Identify the logical observables, independent of shots and layout."""
        return hashlib.sha256(
            canonical_bytes(
                {
                    "schema": MEASUREMENT_PLAN_SCHEMA,
                    "circuit_plan_digest": self.circuit_plan_digest,
                    "n_qubits": self.n_qubits,
                    "basis_id": self.basis_id,
                    "observable_ids": list(self.observable_ids),
                    "pauli_label_convention": self.pauli_label_convention,
                }
            )
        ).hexdigest()

    @property
    def request_digest(self) -> str:
        """Bind shots, register, bit order, and eventual physical layout."""
        return hashlib.sha256(canonical_bytes(self.to_wire())).hexdigest()

    def readout_rotations(self) -> tuple[tuple[str, int], ...]:
        """Name logical premeasurement gates in execution order."""
        if self.basis_id == "Z":
            return ()
        gates = ("h",) if self.basis_id == "X" else ("sdg", "h")
        return tuple((gate, wire) for wire in range(self.n_qubits) for gate in gates)

    def logical_bits(self, raw_key: str) -> tuple[int, ...]:
        """Decode one compiled raw count key without guessing bit orientation."""
        if self.physical_qubits is None:
            raise ValueError("raw readout requires a compiled physical layout")
        if type(raw_key) is not str or len(raw_key) != self.n_qubits or set(raw_key) - {"0", "1"}:
            raise ValueError("raw key does not match the declared register")
        by_clbit = dict(zip(self.display_order_clbits, raw_key, strict=True))
        return tuple(int(by_clbit[clbit]) for clbit in self.logical_to_clbit)


@dataclass(frozen=True, slots=True)
class SampledBasisEstimate:
    """Sparse-count expectations and sampling covariance for one measured basis."""

    measurement_request_digest: str
    origin: str
    shots: int
    observable_ids: tuple[str, ...]
    expectations: tuple[float, ...]
    covariance_of_mean: tuple[tuple[float, ...], ...] | None


def estimate_sampled_basis(
    plan: MeasurementPlan,
    counts: Mapping[str, int],
    *,
    origin: str,
) -> SampledBasisEstimate:
    """Recover common-axis features from sparse raw counts and exact bit custody."""
    if type(plan) is not MeasurementPlan or plan.physical_qubits is None:
        raise ValueError("sampled estimate requires a compiled measurement layout")
    if origin not in ("hardware_raw", "digital_sampled"):
        raise ValueError("sampled estimate origin must be explicit")
    if not isinstance(counts, Mapping) or not 0 < len(counts) <= 1 << plan.n_qubits:
        raise ValueError("sampled estimate requires bounded nonempty counts")
    width = len(plan.observable_ids)
    totals = [0] * width
    joint = [[0] * width for _ in range(width)]
    shots = 0
    for raw_key, count in counts.items():
        if type(count) is not int or count <= 0:
            raise ValueError("raw count must be a positive integer")
        bits = plan.logical_bits(raw_key)
        single = tuple(1 - 2 * bit for bit in bits)
        values = single + tuple(
            single[index] * single[index + 1] for index in range(plan.n_qubits - 1)
        )
        shots += count
        if shots > plan.shots:
            raise ValueError("raw counts exceed requested shots")
        for index, value in enumerate(values):
            totals[index] += count * value
            for other_index, other in enumerate(values):
                joint[index][other_index] += count * value * other
    if shots != plan.shots:
        raise ValueError("raw counts do not match requested shots")
    expectations = tuple(total / shots for total in totals)
    covariance = None
    if shots > 1:
        denominator = shots * shots * (shots - 1)
        covariance = tuple(
            tuple(
                (shots * joint[index][other] - totals[index] * totals[other]) / denominator
                for other in range(width)
            )
            for index in range(width)
        )
    return SampledBasisEstimate(
        measurement_request_digest=plan.request_digest,
        origin=origin,
        shots=shots,
        observable_ids=plan.observable_ids,
        expectations=expectations,
        covariance_of_mean=covariance,
    )


def build_measurement_plan(
    circuit: StaticCircuitPlan,
    reservoir: ReservoirSpec,
    *,
    basis_id: str,
    shots: int,
    logical_to_clbit: tuple[int, ...],
    display_order_clbits: tuple[int, ...],
    register_name: str,
    physical_qubits: tuple[int, ...] | None,
) -> MeasurementPlan:
    """Bind a static circuit to one frozen common-axis readout request."""
    if type(circuit) is not StaticCircuitPlan or type(reservoir) is not ReservoirSpec:
        raise ValueError("measurement requires a static circuit and reservoir")
    expected_reservoir = hashlib.sha256(canonical_bytes(reservoir.to_wire())).hexdigest()
    if circuit.reservoir_digest != expected_reservoir:
        raise ValueError("measurement reservoir digest mismatch")
    if any(wire >= reservoir.n_qubits for _, wires, _ in circuit.operations for wire in wires):
        raise ValueError("circuit operation exceeds measurement width")
    return MeasurementPlan(
        circuit_plan_digest=hashlib.sha256(canonical_bytes(circuit.to_wire())).hexdigest(),
        n_qubits=reservoir.n_qubits,
        basis_id=basis_id,
        shots=shots,
        observable_ids=_observable_ids(basis_id, reservoir.n_qubits),
        logical_to_clbit=logical_to_clbit,
        display_order_clbits=display_order_clbits,
        register_name=register_name,
        physical_qubits=physical_qubits,
        pauli_label_convention="logical_q0_low_index_v1",
    )
