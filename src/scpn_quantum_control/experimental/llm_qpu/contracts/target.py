# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Frozen provider target metadata whose unknowns stay unknown."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .wire import (
    TARGET_SNAPSHOT_SCHEMA,
    ArtifactHeader,
    _digest,
    _positive_int,
    _strict_json,
    _text,
)

_PROVIDERS = ("ibm", "iqm")
_ORIGIN_BY_PROVENANCE = {
    "provider_target_query": "provider_metadata",
    "synthetic_fixture": "synthetic_classical",
}
_CAPABILITIES = ("calibration_id", "calibration_time_utc", "max_shots", "max_circuits_per_job")
_COUNTED = frozenset({"max_shots", "max_circuits_per_job"})
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


@dataclass(frozen=True, slots=True)
class ReportedValue:
    """One capability the provider reported, or an explicit unknown and what it blocks."""

    state: str
    value: int | str | None
    blocker: str | None

    def __post_init__(self) -> None:
        """Refuse an unknown dressed as a value and a value without a known state."""
        if self.state == "known":
            if self.blocker is not None:
                raise ValueError("a known capability cannot carry a blocker")
            if type(self.value) is int:
                _positive_int(self.value, name="reported capability", maximum=10**9)
            else:
                object.__setattr__(self, "value", _text(self.value, name="reported capability"))
        elif self.state == "unknown":
            if self.value is not None:
                raise ValueError("an unknown capability cannot carry a value")
            object.__setattr__(self, "blocker", _text(self.blocker, name="capability blocker"))
        else:
            raise ValueError("capability state must be known or unknown")

    def require(self, *, name: str) -> int | str:
        """Return the reported value, or refuse with the blocker an unknown causes."""
        if self.value is None:
            raise ValueError(f"{name} is unknown: {self.blocker}")
        return self.value

    def to_wire(self) -> dict[str, object]:
        """Return the exact state, value and blocker."""
        return {"state": self.state, "value": self.value, "blocker": self.blocker}

    @classmethod
    def from_wire(cls, value: object) -> ReportedValue:
        """Reject a capability record with missing or extra fields."""
        if type(value) is not dict or set(value) != {"state", "value", "blocker"}:
            raise ValueError("ReportedValue fields mismatch")
        return cls(state=value["state"], value=value["value"], blocker=value["blocker"])


@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    """Design-only identity of one concrete provider device as its SDK reported it."""

    provider: str
    account_route: str
    device_name: str
    sdk_name: str
    sdk_version: str
    sdk_target_digest: str
    capability_provenance: str
    num_qubits: int
    basis_gates: tuple[str, ...]
    coupling_edges: tuple[tuple[int, int], ...]
    calibration_id: ReportedValue
    calibration_time_utc: ReportedValue
    max_shots: ReportedValue
    max_circuits_per_job: ReportedValue
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse catalog profiles, guessed capabilities and inconsistent topology."""
        if self.provider not in _PROVIDERS:
            raise ValueError("unsupported target provider")
        for name in ("account_route", "device_name", "sdk_name", "sdk_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        _digest(self.sdk_target_digest, name="SDK target digest")
        if self.capability_provenance == "catalog_profile":
            raise ValueError("a catalog profile is not a live target capability")
        if self.capability_provenance not in _ORIGIN_BY_PROVENANCE:
            raise ValueError("unknown capability provenance")
        _positive_int(self.num_qubits, name="target qubit count", maximum=10_000)
        self._check_gates_and_edges()
        self._check_capabilities()
        if type(self.header) is not ArtifactHeader or self.header.object_kind != "target_snapshot":
            raise ValueError("TargetSnapshot requires its exact artifact header")
        if self.header.data_origin != _ORIGIN_BY_PROVENANCE[self.capability_provenance]:
            raise ValueError("artifact data origin does not match capability provenance")
        self.header.validate_content(self._scientific_wire(), parents=(self.sdk_target_digest,))

    def _check_gates_and_edges(self) -> None:
        """Require sorted unique gate names and in-range, sorted unique qubit pairs."""
        if type(self.basis_gates) is not tuple or not self.basis_gates:
            raise ValueError("basis gates must be a nonempty tuple")
        gates = tuple(_text(gate, name="basis gate") for gate in self.basis_gates)
        if gates != tuple(sorted(set(gates))):
            raise ValueError("basis gates must be sorted and unique")
        if type(self.coupling_edges) is not tuple or len(self.coupling_edges) > 100_000:
            raise ValueError("coupling edges must be a bounded tuple")
        for edge in self.coupling_edges:
            if type(edge) is not tuple or len(edge) != 2:
                raise ValueError("a coupling edge must be a qubit pair")
            source, target = edge
            for qubit in edge:
                if type(qubit) is not int or not 0 <= qubit < self.num_qubits:
                    raise ValueError("coupling edge qubit is outside the target")
            if source == target:
                raise ValueError("a coupling edge must join two different qubits")
        if self.coupling_edges != tuple(sorted(set(self.coupling_edges))):
            raise ValueError("coupling edges must be sorted and unique")

    def _check_capabilities(self) -> None:
        """Require every optional capability as a typed value, counts as integers."""
        for name in _CAPABILITIES:
            reported = getattr(self, name)
            if type(reported) is not ReportedValue:
                raise ValueError(f"{name} must be a ReportedValue")
            if reported.value is not None and (type(reported.value) is int) != (name in _COUNTED):
                raise ValueError(f"{name} has the wrong value type")
        time = self.calibration_time_utc.value
        if type(time) is str and _UTC.fullmatch(time) is None:
            raise ValueError("calibration time must be an ISO-8601 UTC second")

    def unknown_capabilities(self) -> tuple[tuple[str, str], ...]:
        """Return each unknown capability with the blocker it places on later planning."""
        return tuple(
            (name, str(getattr(self, name).blocker))
            for name in _CAPABILITIES
            if getattr(self, name).state == "unknown"
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the target identity and capabilities hashed as content."""
        return {
            "schema": TARGET_SNAPSHOT_SCHEMA,
            "object_kind": "target_snapshot",
            "provider": self.provider,
            "account_route": self.account_route,
            "device_name": self.device_name,
            "sdk_name": self.sdk_name,
            "sdk_version": self.sdk_version,
            "sdk_target_digest": self.sdk_target_digest,
            "capability_provenance": self.capability_provenance,
            "num_qubits": self.num_qubits,
            "basis_gates": list(self.basis_gates),
            "coupling_edges": [list(edge) for edge in self.coupling_edges],
            **{name: getattr(self, name).to_wire() for name in _CAPABILITIES},
        }

    def to_wire(self) -> dict[str, object]:
        """Return detached versioned target metadata, with no credentials or tokens."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> TargetSnapshot:
        """Reject unknown fields and restore the frozen target snapshot."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("TargetSnapshot fields mismatch")
        if value["schema"] != TARGET_SNAPSHOT_SCHEMA or value["object_kind"] != "target_snapshot":
            raise ValueError("unknown TargetSnapshot schema")
        gates, edges = value["basis_gates"], value["coupling_edges"]
        if type(gates) is not list or type(edges) is not list:
            raise ValueError("basis gates and coupling edges must be lists on wire")
        if any(type(edge) is not list or len(edge) != 2 for edge in edges):
            raise ValueError("a coupling edge must be a two-item list on wire")
        return cls(
            provider=value["provider"],
            account_route=value["account_route"],
            device_name=value["device_name"],
            sdk_name=value["sdk_name"],
            sdk_version=value["sdk_version"],
            sdk_target_digest=value["sdk_target_digest"],
            capability_provenance=value["capability_provenance"],
            num_qubits=value["num_qubits"],
            basis_gates=tuple(gates),
            coupling_edges=tuple((edge[0], edge[1]) for edge in edges),
            calibration_id=ReportedValue.from_wire(value["calibration_id"]),
            calibration_time_utc=ReportedValue.from_wire(value["calibration_time_utc"]),
            max_shots=ReportedValue.from_wire(value["max_shots"]),
            max_circuits_per_job=ReportedValue.from_wire(value["max_circuits_per_job"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def decode_target_snapshot(raw: bytes) -> TargetSnapshot:
    """Decode one target snapshot from strict canonical wire bytes.

    The package-wide ``decode_contract`` is frozen by the public export identity
    record, so later contract objects decode through their own entry point.
    """
    return TargetSnapshot.from_wire(_strict_json(raw))
