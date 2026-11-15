# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental target snapshot contract
"""Exercise the provider-free target snapshot wire through its standalone worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    TARGET_SNAPSHOT_SCHEMA,
    ArtifactHeader,
    ReportedValue,
    TargetSnapshot,
    canonical_bytes,
    decode_contract,
    decode_target_snapshot,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_ORIGIN = {
    "synthetic_fixture": "synthetic_classical",
    "provider_target_query": "provider_metadata",
}


def _known(value: int | str) -> ReportedValue:
    """Return a capability the provider reported."""
    return ReportedValue(state="known", value=value, blocker=None)


def _unknown(blocker: str) -> ReportedValue:
    """Return an explicit unknown capability with the blocker it causes."""
    return ReportedValue(state="unknown", value=None, blocker=blocker)


def _fields(provenance: str) -> dict[str, Any]:
    """Return the scientific fields of a three-qubit line target, as a design fixture."""
    return {
        "provider": "iqm",
        "account_route": "iqm-resonance/owner-project",
        "device_name": "garnet",
        "sdk_name": "iqm-client",
        "sdk_version": "35.0.3",
        "sdk_target_digest": "a" * 64,
        "capability_provenance": provenance,
        "num_qubits": 3,
        "basis_gates": ("cz", "measure", "prx"),
        "coupling_edges": ((0, 1), (1, 2)),
        "calibration_id": _known("cal-2026-11-15-0800"),
        "calibration_time_utc": _known("2026-11-15T08:00:00Z"),
        "max_shots": _known(100_000),
        "max_circuits_per_job": _unknown("the SDK does not report a circuit batch limit"),
    }


def _target(provenance: str = "synthetic_fixture", **changes: Any) -> TargetSnapshot:
    """Build a target snapshot whose header hashes exactly its scientific fields."""
    fields = {**_fields(provenance), **changes}
    scientific = {
        "schema": TARGET_SNAPSHOT_SCHEMA,
        "object_kind": "target_snapshot",
        **{
            name: [list(edge) for edge in value]
            if name == "coupling_edges"
            else list(value)
            if name == "basis_gates"
            else value.to_wire()
            if type(value) is ReportedValue
            else value
            for name, value in fields.items()
        },
    }
    header = ArtifactHeader(
        object_kind="target_snapshot",
        content_digest=hashlib.sha256(canonical_bytes(scientific)).hexdigest(),
        parents=(fields["sdk_target_digest"],),
        base_repo_commit=_BASE,
        implementation_revision=_REVISION,
        execution_origin="offline_design",
        data_origin=_ORIGIN.get(provenance, "synthetic_classical"),
        claim_scope="design_only",
    )
    return TargetSnapshot(**fields, header=header)


def _worker(request: dict[str, object], tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the standalone protocol worker on one request with poisoned provider tokens."""
    return subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"},
        capture_output=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("provenance", ["synthetic_fixture", "provider_target_query"])
def test_target_snapshot_standalone_roundtrip(provenance: str, tmp_path: Path) -> None:
    """A fixture and a provider-query snapshot round-trip through the worker unchanged."""
    target = _target(provenance)
    response = _worker({"op": "roundtrip_target_snapshot", "target": target.to_wire()}, tmp_path)

    assert response.returncode == 0, response.stdout.decode()
    payload = json.loads(response.stdout)
    assert payload["status"] == "validated_roundtrip_no_compute"
    assert payload["hardware_submission_enabled"] is False
    assert (
        payload["target_sha256"] == hashlib.sha256(canonical_bytes(target.to_wire())).hexdigest()
    )
    assert decode_target_snapshot(canonical_bytes(payload["target"])) == target
    assert payload["target"]["header"]["data_origin"] == _ORIGIN[provenance]


def test_target_snapshot_keeps_unknowns_unknown_and_names_their_blockers() -> None:
    """A missing capability stays unknown, refuses use and reports what it blocks."""
    target = _target()

    assert target.max_shots.require(name="max_shots") == 100_000
    assert target.unknown_capabilities() == (
        ("max_circuits_per_job", "the SDK does not report a circuit batch limit"),
    )
    with pytest.raises(
        ValueError,
        match="^max_circuits_per_job is unknown: the SDK does not report a circuit batch limit$",
    ):
        target.max_circuits_per_job.require(name="max_circuits_per_job")


def test_target_snapshot_refuses_catalog_profiles_and_inconsistent_targets() -> None:
    """Catalog capabilities, unknown providers and malformed topology are refused."""
    target = _target()
    for changes, reason in (
        ({"capability_provenance": "catalog_profile"}, "^a catalog profile is not a live"),
        ({"capability_provenance": "guess"}, "^unknown capability provenance$"),
        ({"provider": "rigetti"}, "^unsupported target provider$"),
        ({"num_qubits": True}, "target qubit count"),
        ({"basis_gates": ()}, "^basis gates must be a nonempty tuple$"),
        ({"basis_gates": ("prx", "cz")}, "^basis gates must be sorted and unique$"),
        ({"coupling_edges": ((0, 3),)}, "^coupling edge qubit is outside the target$"),
        ({"coupling_edges": ((1, 1),)}, "^a coupling edge must join two different qubits$"),
        ({"coupling_edges": ((1, 2), (0, 1))}, "^coupling edges must be sorted and unique$"),
        ({"coupling_edges": ((0, 1, 2),)}, "^a coupling edge must be a qubit pair$"),
        ({"sdk_target_digest": "A" * 64}, "SDK target digest"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(target, **changes)
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(target, device_name="sirius")
    with pytest.raises(ValueError, match="^artifact data origin does not match capability"):
        replace(target, header=replace(target.header, data_origin="provider_metadata"))
    with pytest.raises(ValueError, match="^TargetSnapshot requires its exact artifact header$"):
        replace(target, header=replace(target.header, object_kind="model_descriptor"))


def test_reported_values_refuse_an_unknown_dressed_as_a_value() -> None:
    """Unknowns never carry values, knowns never carry blockers, counts stay integers."""
    for state, value, blocker, reason in (
        ("known", 5, "stale", "^a known capability cannot carry a blocker$"),
        ("unknown", 0, "not reported", "^an unknown capability cannot carry a value$"),
        ("unknown", None, "", "capability blocker"),
        ("maybe", None, None, "^capability state must be known or unknown$"),
        ("known", None, None, "reported capability must be text"),
        ("known", 0, None, "reported capability must be a bounded positive integer"),
        ("known", True, None, "reported capability must be text"),
    ):
        with pytest.raises(ValueError, match=reason):
            ReportedValue(state=state, value=value, blocker=blocker)
    with pytest.raises(ValueError, match="^max_shots has the wrong value type$"):
        _target(max_shots=_known("100000"))
    with pytest.raises(ValueError, match="^calibration_id has the wrong value type$"):
        _target(calibration_id=_known(7))
    with pytest.raises(ValueError, match="^calibration time must be an ISO-8601 UTC second$"):
        _target(calibration_time_utc=_known("2026-11-15 08:00"))
    with pytest.raises(ValueError, match="^max_shots must be a ReportedValue$"):
        _target(max_shots=100_000)


def test_target_snapshot_wire_is_exact_and_the_general_decoder_stays_closed(
    tmp_path: Path,
) -> None:
    """Edited wires are refused here and by the worker; decode_contract does not admit it."""
    target = _target()
    extra: dict[str, Any] = target.to_wire()
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="^TargetSnapshot fields mismatch$"):
        decode_target_snapshot(canonical_bytes(extra))
    tuple_edge: dict[str, Any] = target.to_wire()
    tuple_edge["coupling_edges"] = [[0, 1, 2]]
    with pytest.raises(ValueError, match="^a coupling edge must be a two-item list on wire$"):
        decode_target_snapshot(canonical_bytes(tuple_edge))
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(target.to_wire()))
    other_schema: dict[str, Any] = target.to_wire()
    other_schema["schema"] = "scpn.experimental.llm_qpu.target_snapshot.v2"
    with pytest.raises(ValueError, match="^unknown TargetSnapshot schema$"):
        decode_target_snapshot(canonical_bytes(other_schema))
    gate_text: dict[str, Any] = target.to_wire()
    gate_text["basis_gates"] = "cz"
    with pytest.raises(ValueError, match="^basis gates and coupling edges must be lists on wire$"):
        decode_target_snapshot(canonical_bytes(gate_text))
    capability: dict[str, Any] = target.to_wire()
    capability["max_shots"]["source"] = "catalog"
    with pytest.raises(ValueError, match="^ReportedValue fields mismatch$"):
        decode_target_snapshot(canonical_bytes(capability))
    edge_list: Any = [(0, 1)]
    with pytest.raises(ValueError, match="^coupling edges must be a bounded tuple$"):
        replace(target, coupling_edges=edge_list)
    for path, value, reason in (
        (("capability_provenance",), "catalog_profile", "catalog or unknown capability"),
        (("max_shots",), {"state": "known", "value": 0, "blocker": None}, "max_shots must be"),
        (("max_circuits_per_job", "value"), 0, "unknown capability cannot carry a value"),
        (("header", "content_digest"), "f" * 64, "content digest mismatch"),
    ):
        forged: dict[str, Any] = target.to_wire()
        holder = forged
        for key in path[:-1]:
            holder = holder[key]
        holder[path[-1]] = value
        response = _worker({"op": "roundtrip_target_snapshot", "target": forged}, tmp_path)
        assert response.returncode == 2, path
        assert reason in json.loads(response.stdout)["reason"], path
