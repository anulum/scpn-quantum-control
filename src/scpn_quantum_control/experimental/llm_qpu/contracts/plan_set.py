# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Circuit plan sets: one static plan per compressed row and arm, rebuilt exactly."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .compressed import CompressedLatentBatch
from .static import (
    ReservoirSpec,
    StaticCircuitPlan,
    build_static_circuit_plan,
    validate_static_circuit_plan,
)
from .wire import (
    CIRCUIT_PLAN_SET_SCHEMA,
    ArtifactHeader,
    _digest,
    _positive_int,
    _strict_json,
    canonical_bytes,
)

_ARMS = ("encoding_plus_fields_no_coupling", "xy")


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CircuitPlanSet:
    """Design-only static plans for every row of one compressed batch in every arm."""

    reservoir_digest: str
    compressed_digest: str
    row_count: int
    arms: tuple[str, ...]
    plans: tuple[StaticCircuitPlan, ...]
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse a missing, repeated or foreign plan and an unordered inventory."""
        for name in ("reservoir_digest", "compressed_digest"):
            _digest(getattr(self, name), name=name)
        _positive_int(self.row_count, name="plan row count", maximum=4096)
        if (
            type(self.arms) is not tuple
            or not self.arms
            or self.arms != tuple(sorted(set(self.arms)))
        ):
            raise ValueError("arms must be a sorted unique tuple")
        if any(arm not in _ARMS for arm in self.arms):
            raise ValueError("unknown static arm")
        plans = self.plans
        if type(plans) is not tuple or any(type(plan) is not StaticCircuitPlan for plan in plans):
            raise ValueError("plans must be a tuple of StaticCircuitPlan records")
        keys = [(plan.row_index, plan.arm_id) for plan in plans]
        if keys != [(row, arm) for row in range(self.row_count) for arm in self.arms]:
            raise ValueError("plans must cover every row in every arm once, in row then arm order")
        if any(
            (plan.reservoir_digest, plan.compressed_digest)
            != (self.reservoir_digest, self.compressed_digest)
            for plan in plans
        ):
            raise ValueError("every plan names this reservoir and compressed batch")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "circuit_plan_set"
        ):
            raise ValueError("CircuitPlanSet requires its exact artifact header")
        self.header.validate_content(
            self._scientific_wire(), parents=(self.reservoir_digest, self.compressed_digest)
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the plan-set content hashed by its header."""
        return {
            "schema": CIRCUIT_PLAN_SET_SCHEMA,
            "object_kind": "circuit_plan_set",
            "reservoir_digest": self.reservoir_digest,
            "compressed_digest": self.compressed_digest,
            "row_count": self.row_count,
            "arms": list(self.arms),
            "plans": [plan.to_wire() for plan in self.plans],
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned plan set."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> CircuitPlanSet:
        """Reject unknown fields and restore the frozen plan set."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("CircuitPlanSet fields mismatch")
        if (
            value["schema"] != CIRCUIT_PLAN_SET_SCHEMA
            or value["object_kind"] != "circuit_plan_set"
        ):
            raise ValueError("unknown CircuitPlanSet schema")
        if type(value["arms"]) is not list or type(value["plans"]) is not list:
            raise ValueError("arms and plans must be lists on wire")
        return cls(
            reservoir_digest=value["reservoir_digest"],
            compressed_digest=value["compressed_digest"],
            row_count=value["row_count"],
            arms=tuple(value["arms"]),
            plans=tuple(StaticCircuitPlan.from_wire(plan) for plan in value["plans"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def build_circuit_plan_set(
    spec: ReservoirSpec,
    compressed: CompressedLatentBatch,
    angles_payload: bytes,
    *,
    arms: tuple[str, ...],
    revisions: tuple[str, str],
) -> CircuitPlanSet:
    """Build one static plan per row and arm; ``revisions`` are base and implementation."""
    plans = tuple(
        build_static_circuit_plan(spec, compressed, angles_payload, row_index=row, arm_id=arm)
        for row in range(len(compressed.sample_ids))
        for arm in arms
    )
    fields: dict[str, Any] = {
        "reservoir_digest": _sha(spec.to_wire()),
        "compressed_digest": _sha(compressed.to_wire()),
        "row_count": len(compressed.sample_ids),
        "arms": arms,
    }
    content = {
        "schema": CIRCUIT_PLAN_SET_SCHEMA,
        "object_kind": "circuit_plan_set",
        **fields,
        "arms": list(arms),
        "plans": [plan.to_wire() for plan in plans],
    }
    header = ArtifactHeader(
        object_kind="circuit_plan_set",
        content_digest=_sha(content),
        parents=tuple(sorted((fields["reservoir_digest"], fields["compressed_digest"]))),
        base_repo_commit=revisions[0],
        implementation_revision=revisions[1],
        execution_origin="offline_design",
        data_origin=compressed.header.data_origin,
        claim_scope="design_only",
    )
    return CircuitPlanSet(
        reservoir_digest=fields["reservoir_digest"],
        compressed_digest=fields["compressed_digest"],
        row_count=fields["row_count"],
        arms=arms,
        plans=plans,
        header=header,
    )


def validate_circuit_plan_set(
    plan_set: CircuitPlanSet,
    spec: ReservoirSpec,
    compressed: CompressedLatentBatch,
    angles_payload: bytes,
) -> None:
    """Refuse a set for another reservoir or batch, or any plan off its exact recipe."""
    if type(plan_set) is not CircuitPlanSet:
        raise ValueError("plan-set validation needs a CircuitPlanSet")
    if plan_set.reservoir_digest != _sha(spec.to_wire()):
        raise ValueError("the plan set names a different reservoir")
    if plan_set.compressed_digest != _sha(compressed.to_wire()):
        raise ValueError("the plan set names a different compressed batch")
    if plan_set.row_count != len(compressed.sample_ids):
        raise ValueError("the plan set does not cover every compressed row")
    if plan_set.header.data_origin != compressed.header.data_origin:
        raise ValueError("plan-set origin differs from the compressed batch")
    for plan in plan_set.plans:
        validate_static_circuit_plan(spec, compressed, angles_payload, plan)


def decode_circuit_plan_set(raw: bytes) -> CircuitPlanSet:
    """Decode one circuit plan set from strict canonical wire bytes."""
    return CircuitPlanSet.from_wire(_strict_json(raw))
