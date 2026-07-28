# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design replay tests
"""File-format and deterministic trajectory tests for BL-33 replay."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from scpn_quantum_control.codesign import (
    CoDesignMode,
    ObserverInputs,
    ReplayTrace,
    build_demo_loop,
    demo_inputs,
    record_replay_trace,
    verify_replay_trace,
)


def test_replay_round_trip_is_byte_stable() -> None:
    """Round-trip a recorded trace and reproduce exact output bytes."""
    trace, original = record_replay_trace(build_demo_loop(), demo_inputs())
    encoded = trace.to_json()
    parsed = ReplayTrace.from_json(encoded)
    replayed = verify_replay_trace(build_demo_loop(), parsed)

    assert parsed == trace
    assert parsed.to_json() == encoded
    assert [row.safety.applied_parameters for row in replayed] == [
        row.safety.applied_parameters for row in original
    ]
    assert len(trace.digest) == 64
    assert trace.to_dict()["schema"] == "quantum_classical_codesign.replay.v1"


def test_replay_records_explicit_observers() -> None:
    """Persist and replay explicit observer telemetry and decisions."""
    observers = (
        ObserverInputs(active_sensing_id="candidate-1", geometry_gradient_norm=0.2),
        ObserverInputs(identity_action="hold", identity_reason="bounded test"),
    )
    trace, outputs = record_replay_trace(build_demo_loop(), demo_inputs(), observers=observers)
    parsed = ReplayTrace.from_json(trace.to_json())

    assert parsed.observers == observers
    assert outputs[-1].safety.action.value == "hold"
    assert len(verify_replay_trace(build_demo_loop(), trace)) == 2


def test_tampered_replay_digest_is_rejected() -> None:
    """Reject a measurement changed without recomputing the trace digest."""
    trace, _outputs = record_replay_trace(build_demo_loop(), demo_inputs())
    payload = json.loads(trace.to_json())
    payload["inputs"][0]["measurement"] = 0.1

    with pytest.raises(ValueError, match="digest mismatch"):
        ReplayTrace.from_json(json.dumps(payload))


def test_replay_detects_controller_trajectory_drift() -> None:
    """Reject a mode change that alters the controller trajectory."""
    trace, _outputs = record_replay_trace(build_demo_loop(), demo_inputs())
    changed = list(demo_inputs())
    changed[0] = replace(changed[0], mode=CoDesignMode.QUANTUM_TO_CLASSICAL)
    changed_trace = replace(trace, inputs=tuple(changed))

    with pytest.raises(ValueError, match="does not match"):
        verify_replay_trace(build_demo_loop(), changed_trace)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "JSON object"),
        ({}, "digest"),
        ({"digest": "x"}, "inputs"),
    ],
)
def test_replay_parser_rejects_malformed_envelopes(payload: object, message: str) -> None:
    """Reject malformed top-level replay envelopes."""
    with pytest.raises(ValueError, match=message):
        ReplayTrace.from_json(json.dumps(payload))


def test_replay_trace_validates_cardinality_and_digest() -> None:
    """Validate trace schemas, alignment, output cardinality, and digest shape."""
    trace, _outputs = record_replay_trace(build_demo_loop(), demo_inputs())

    with pytest.raises(ValueError, match="schema"):
        replace(trace, schema="quantum_classical_codesign.replay.v2")
    with pytest.raises(ValueError, match="product_schema"):
        replace(trace, product_schema="quantum_classical_codesign.v2")
    with pytest.raises(ValueError, match="claim_boundary"):
        replace(trace, claim_boundary="expanded claim")
    with pytest.raises(ValueError, match="aligned"):
        replace(trace, observers=())
    with pytest.raises(ValueError, match="outputs"):
        replace(trace, output_json=())
    with pytest.raises(ValueError, match="outputs"):
        replace(trace, output_json=trace.output_json + (trace.output_json[0],))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(trace, digest="X" * 64)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("outputs",), [], "outputs"),
        (("inputs", 0, "parameters"), "bad", "parameters"),
        (("inputs", 0, "step"), True, "integer"),
        (("inputs", 0, "observed_at_ms"), "bad", "numeric"),
        (("observers", 0, "active_sensing_id"), 1, "string or null"),
        (("observers", 0, "geometry_gradient_norm"), "bad", "numeric or null"),
        (("schema",), "", "schema"),
        (("observers",), [], "observers"),
    ],
)
def test_replay_parser_rejects_field_type_drift(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    """Reject malformed fields through the public replay parser."""
    trace, _outputs = record_replay_trace(build_demo_loop(), demo_inputs())
    payload = deepcopy(trace.to_dict())
    target: object = payload
    for component in path[:-1]:
        if isinstance(component, int):
            assert isinstance(target, list)
            target = target[component]
        else:
            assert isinstance(target, dict)
            target = target[component]
    final = path[-1]
    if isinstance(final, int):
        assert isinstance(target, list)
        target[final] = value
    else:
        assert isinstance(target, dict)
        target[final] = value

    with pytest.raises(ValueError, match=message):
        ReplayTrace.from_json(json.dumps(payload))
