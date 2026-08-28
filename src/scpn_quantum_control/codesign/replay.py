# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — deterministic co-design replay
"""Versioned trace recording and bit-stable replay verification for co-design."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .contracts import (
    CODESIGN_CLAIM_BOUNDARY,
    CODESIGN_SCHEMA,
    CoDesignMode,
    LoopStepInput,
    LoopStepOutput,
    ObserverInputs,
)
from .loop import CoDesignLoop

REPLAY_SCHEMA = "quantum_classical_codesign.replay.v1"


@dataclass(frozen=True, slots=True)
class ReplayTrace:
    """Inputs, observers, expected output bytes, and integrity digest."""

    inputs: tuple[LoopStepInput, ...]
    observers: tuple[ObserverInputs, ...]
    output_json: tuple[str, ...]
    digest: str
    schema: str = REPLAY_SCHEMA
    product_schema: str = CODESIGN_SCHEMA
    claim_boundary: str = CODESIGN_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate trace cardinality and hexadecimal digest shape."""
        if self.schema != REPLAY_SCHEMA:
            raise ValueError(f"replay schema must equal {REPLAY_SCHEMA}")
        if self.product_schema != CODESIGN_SCHEMA:
            raise ValueError(f"replay product_schema must equal {CODESIGN_SCHEMA}")
        if self.claim_boundary != CODESIGN_CLAIM_BOUNDARY:
            raise ValueError("replay claim_boundary must match the co-design claim boundary")
        if not self.inputs or len(self.inputs) != len(self.observers):
            raise ValueError("replay inputs and observers must be non-empty and aligned")
        if not self.output_json or len(self.output_json) > len(self.inputs):
            raise ValueError("replay outputs must be non-empty and no longer than inputs")
        if len(self.digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.digest
        ):
            raise ValueError("replay digest must be a lowercase SHA-256 hexadecimal value")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready replay envelope."""
        return _payload(
            self.inputs,
            self.observers,
            self.output_json,
            schema=self.schema,
            product_schema=self.product_schema,
            claim_boundary=self.claim_boundary,
        ) | {"digest": self.digest}

    def to_json(self) -> str:
        """Return canonical JSON bytes as text."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> ReplayTrace:
        """Parse and integrity-check a canonical replay envelope."""
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("replay envelope must be a JSON object")
        data = cast(Mapping[str, object], raw)
        digest = _required_string(data, "digest")
        inputs = tuple(_input_from_mapping(row) for row in _mapping_rows(data, "inputs"))
        observers = tuple(_observer_from_mapping(row) for row in _mapping_rows(data, "observers"))
        output_rows = data.get("outputs")
        if not isinstance(output_rows, list) or not output_rows:
            raise ValueError("replay outputs must be a non-empty array")
        output_json = tuple(_canonical_json(row) for row in output_rows)
        payload = _payload(
            inputs,
            observers,
            output_json,
            schema=_required_string(data, "schema"),
            product_schema=_required_string(data, "product_schema"),
            claim_boundary=_required_string(data, "claim_boundary"),
        )
        expected = _digest(payload)
        if digest != expected:
            raise ValueError("replay digest mismatch")
        return cls(
            inputs=inputs,
            observers=observers,
            output_json=output_json,
            digest=digest,
            schema=cast(str, payload["schema"]),
            product_schema=cast(str, payload["product_schema"]),
            claim_boundary=cast(str, payload["claim_boundary"]),
        )


def record_replay_trace(
    loop: CoDesignLoop,
    inputs: Sequence[LoopStepInput],
    *,
    observers: Sequence[ObserverInputs] | None = None,
) -> tuple[ReplayTrace, tuple[LoopStepOutput, ...]]:
    """Run a loop and record the exact deterministic output representation."""
    observer_rows = (
        tuple(ObserverInputs() for _ in inputs) if observers is None else tuple(observers)
    )
    input_rows = tuple(inputs)
    outputs = loop.run(input_rows, observers=observer_rows)
    output_json = tuple(_canonical_json(output.to_dict()) for output in outputs)
    payload = _payload(
        input_rows,
        observer_rows,
        output_json,
        schema=REPLAY_SCHEMA,
        product_schema=CODESIGN_SCHEMA,
        claim_boundary=CODESIGN_CLAIM_BOUNDARY,
    )
    return (
        ReplayTrace(
            inputs=input_rows,
            observers=observer_rows,
            output_json=output_json,
            digest=_digest(payload),
        ),
        outputs,
    )


def verify_replay_trace(
    loop: CoDesignLoop,
    trace: ReplayTrace,
) -> tuple[LoopStepOutput, ...]:
    """Replay a trace through a fresh loop and require bit-identical outputs."""
    outputs = loop.run(trace.inputs, observers=trace.observers)
    replayed = tuple(_canonical_json(output.to_dict()) for output in outputs)
    if replayed != trace.output_json:
        raise ValueError("replayed controller trajectory does not match the recorded trace")
    return outputs


def _payload(
    inputs: tuple[LoopStepInput, ...],
    observers: tuple[ObserverInputs, ...],
    output_json: tuple[str, ...],
    *,
    schema: str,
    product_schema: str,
    claim_boundary: str,
) -> dict[str, object]:
    return {
        "schema": schema,
        "product_schema": product_schema,
        "inputs": [row.to_dict() for row in inputs],
        "observers": [row.to_dict() for row in observers],
        "outputs": [json.loads(row) for row in output_json],
        "claim_boundary": claim_boundary,
    }


def _input_from_mapping(row: Mapping[str, object]) -> LoopStepInput:
    parameters = row.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError("replay input parameters must be an array")
    return LoopStepInput(
        step=_required_int(row, "step"),
        observed_at_ms=_required_float(row, "observed_at_ms"),
        apply_at_ms=_required_float(row, "apply_at_ms"),
        parameters=tuple(float(value) for value in parameters),
        measurement=_required_float(row, "measurement"),
        target_order_parameter=_required_float(row, "target_order_parameter"),
        mode=CoDesignMode(_required_string(row, "mode")),
    )


def _observer_from_mapping(row: Mapping[str, object]) -> ObserverInputs:
    return ObserverInputs(
        active_sensing_id=_optional_string(row, "active_sensing_id"),
        identity_action=_optional_string(row, "identity_action"),
        identity_reason=_optional_string(row, "identity_reason"),
        geometry_gradient_norm=_optional_float(row, "geometry_gradient_norm"),
    )


def _mapping_rows(data: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    rows = data.get(key)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"replay {key} must be a non-empty object array")
    return tuple(cast(Mapping[str, object], row) for row in rows)


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"replay {key} must be a non-empty string")
    return value


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"replay {key} must be a string or null")
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"replay {key} must be an integer")
    return value


def _required_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"replay {key} must be numeric")
    return float(value)


def _optional_float(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"replay {key} must be numeric or null")
    return float(value)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "REPLAY_SCHEMA",
    "ReplayTrace",
    "record_replay_trace",
    "verify_replay_trace",
]
