# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental compiled bundle contract
"""Exercise the provider-free compiled bundle wire against its target snapshot."""

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
    COMPILED_BUNDLE_SCHEMA,
    TARGET_SNAPSHOT_SCHEMA,
    ArtifactHeader,
    CompiledBundle,
    CompiledCell,
    ReportedValue,
    TargetSnapshot,
    canonical_bytes,
    compiled_payload_digest,
    decode_compiled_bundle,
    decode_contract,
    validate_compiled_bundle,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_ORIGIN = {
    "synthetic_fixture": "synthetic_classical",
    "provider_target_query": "provider_metadata",
}


def _header(
    kind: str, content: dict[str, object], parents: tuple[str, ...], origin: str
) -> ArtifactHeader:
    """Return a design-only header that hashes exactly the given content."""
    return ArtifactHeader(
        object_kind=kind,
        content_digest=hashlib.sha256(canonical_bytes(content)).hexdigest(),
        parents=tuple(sorted(parents)),
        base_repo_commit=_BASE,
        implementation_revision=_REVISION,
        execution_origin="offline_design",
        data_origin=origin,
        claim_scope="design_only",
    )


def _target(provenance: str = "synthetic_fixture", device: str = "garnet") -> TargetSnapshot:
    """Build a three-qubit line target whose capabilities are all reported."""
    known = {
        "calibration_id": ReportedValue("known", "cal-2026-11-15-0800", None),
        "calibration_time_utc": ReportedValue("known", "2026-11-15T08:00:00Z", None),
        "max_shots": ReportedValue("known", 100_000, None),
        "max_circuits_per_job": ReportedValue("known", 200, None),
    }
    plain: dict[str, Any] = {
        "provider": "iqm",
        "account_route": "iqm-resonance/owner-project",
        "device_name": device,
        "sdk_name": "iqm-client",
        "sdk_version": "35.0.3",
        "sdk_target_digest": "a" * 64,
        "capability_provenance": provenance,
        "num_qubits": 3,
    }
    content: dict[str, object] = {
        "schema": TARGET_SNAPSHOT_SCHEMA,
        "object_kind": "target_snapshot",
        **plain,
        "basis_gates": ["cz", "measure", "prx"],
        "coupling_edges": [[0, 1], [1, 2]],
        **{name: value.to_wire() for name, value in known.items()},
    }
    header = _header("target_snapshot", content, ("a" * 64,), _ORIGIN[provenance])
    return TargetSnapshot(
        **plain,
        basis_gates=("cz", "measure", "prx"),
        coupling_edges=((0, 1), (1, 2)),
        calibration_id=known["calibration_id"],
        calibration_time_utc=known["calibration_time_utc"],
        max_shots=known["max_shots"],
        max_circuits_per_job=known["max_circuits_per_job"],
        header=header,
    )


def _payload(fields: dict[str, Any]) -> dict[str, object]:
    """Return the approvable payload as an independent oracle would serialise it."""
    return {
        "cells": [
            cell.to_wire() if type(cell) is CompiledCell else cell for cell in fields["cells"]
        ],
        "layout": list(fields["layout"]),
        "logical_gates": list(fields["logical_gates"]),
        "compiled_gate_counts": [list(pair) for pair in fields["compiled_gate_counts"]],
        "substitutions": [[source, list(targets)] for source, targets in fields["substitutions"]],
        "compiled_depth": fields["compiled_depth"],
        "provider_options": [list(pair) for pair in fields["provider_options"]],
    }


def _bundle(
    target: TargetSnapshot | None = None, *, device: str | None = None, **changes: Any
) -> CompiledBundle:
    """Build a bundle that compiles cx and rz into the target's cz and prx basis."""
    target = target or _target()
    fields: dict[str, Any] = {
        "cells": (
            CompiledCell("1" * 64, "2" * 64, "3" * 64, "4" * 64, 1000),
            CompiledCell("5" * 64, "6" * 64, "7" * 64, "8" * 64, 2000),
        ),
        "layout": (2, 0, 1),
        "logical_gates": ("cx", "measure", "rz"),
        "compiled_gate_counts": (("cz", 4), ("measure", 3), ("prx", 10)),
        "substitutions": (("cx", ("cz", "prx")), ("rz", ("prx",))),
        "compiled_depth": 12,
        "provider_options": (
            ("layout_method", "sabre"),
            ("optimization_level", 1),
            ("readout_mitigation", False),
        ),
        **changes,
    }
    payload_digest = hashlib.sha256(canonical_bytes(_payload(fields))).hexdigest()
    identity: dict[str, Any] = {
        "target_snapshot_digest": hashlib.sha256(canonical_bytes(target.to_wire())).hexdigest(),
        "provider": target.provider,
        "device_name": device or target.device_name,
        "sdk_lock_digest": "b" * 64,
        "compile_log_digest": "c" * 64,
        "payload_digest": payload_digest,
    }
    content = {
        "schema": COMPILED_BUNDLE_SCHEMA,
        "object_kind": "compiled_bundle",
        **identity,
        **_payload(fields),
    }
    parents = (identity["target_snapshot_digest"], "b" * 64, "c" * 64)
    header = _header("compiled_bundle", content, parents, target.header.data_origin)
    return CompiledBundle(**identity, **fields, header=header)


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
def test_compiled_bundle_roundtrips_and_validates_against_its_target(
    provenance: str, tmp_path: Path
) -> None:
    """A bundle compiled for a target round-trips through the worker and validates."""
    target = _target(provenance)
    bundle = _bundle(target)
    request: dict[str, object] = {
        "op": "roundtrip_compiled_bundle",
        "target": target.to_wire(),
        "bundle": bundle.to_wire(),
    }

    response = _worker(request, tmp_path)

    assert response.returncode == 0, response.stdout.decode()
    payload = json.loads(response.stdout)
    assert payload["status"] == "validated_roundtrip_no_compute"
    assert payload["hardware_submission_enabled"] is False
    assert (
        payload["bundle_sha256"] == hashlib.sha256(canonical_bytes(bundle.to_wire())).hexdigest()
    )
    assert decode_compiled_bundle(canonical_bytes(payload["bundle"])) == bundle
    validate_compiled_bundle(bundle, target)
    assert compiled_payload_digest(bundle) == bundle.payload_digest


def test_compiled_bundle_refuses_unreported_substitutions_and_edited_payloads() -> None:
    """A compiled gate must be logical or declared, and the payload digest binds every field."""
    for changes, reason in (
        (
            {"compiled_gate_counts": (("cz", 4), ("measure", 3), ("prx", 10), ("rzz", 1))},
            "^unreported gate substitution: rzz$",
        ),
        ({"substitutions": (("cx", ("cz",)),)}, "^unreported gate substitution: prx$"),
        (
            {
                "substitutions": (("cx", ("cz", "prx")),),
                "logical_gates": ("cx", "measure", "prx", "rz"),
            },
            "^logical gate vanished without substitution: rz$",
        ),
        (
            {"substitutions": (("cx", ("cz", "prx")), ("h", ("prx",)), ("rz", ("prx",)))},
            "^a substitution must replace a logical gate$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            _bundle(**changes)
    bundle = _bundle()
    edited = (replace(bundle.cells[0], shots=999), bundle.cells[1])
    with pytest.raises(ValueError, match="^compiled payload digest does not match the bundle$"):
        replace(bundle, cells=edited)
    with pytest.raises(ValueError, match="^compiled payload digest does not match the bundle$"):
        replace(bundle, compiled_depth=13)


def test_compiled_bundle_refuses_another_target_or_its_limits() -> None:
    """Another device, an absent qubit, a foreign gate, excess shots and a mixed origin fail."""
    target = _target()
    with pytest.raises(
        ValueError, match="^bundle was compiled against a different target snapshot$"
    ):
        validate_compiled_bundle(_bundle(_target(device="sirius")), target)
    with pytest.raises(ValueError, match="^layout uses a qubit the target does not have$"):
        validate_compiled_bundle(_bundle(target, layout=(0, 1, 3)), target)
    foreign = _bundle(
        target,
        logical_gates=("cx", "measure", "rz", "swap"),
        compiled_gate_counts=(("cz", 4), ("measure", 3), ("prx", 10), ("swap", 1)),
    )
    with pytest.raises(ValueError, match="^compiled gates outside the target basis: swap$"):
        validate_compiled_bundle(foreign, target)
    at_limit = (CompiledCell("1" * 64, "2" * 64, "3" * 64, "4" * 64, 100_000),)
    validate_compiled_bundle(_bundle(target, cells=at_limit), target)
    heavy = (CompiledCell("1" * 64, "2" * 64, "3" * 64, "4" * 64, 100_001),)
    with pytest.raises(
        ValueError, match="^a compiled cell requests more shots than the target allows$"
    ):
        validate_compiled_bundle(_bundle(target, cells=heavy), target)
    provider_target = _target("provider_target_query")
    mixed = _bundle(provider_target)
    with pytest.raises(ValueError, match="^bundle and target data origins differ$"):
        validate_compiled_bundle(
            replace(mixed, header=replace(mixed.header, data_origin="synthetic_classical")),
            provider_target,
        )
    not_a_target: Any = "garnet"
    with pytest.raises(
        ValueError, match="^bundle validation needs a CompiledBundle and its TargetSnapshot$"
    ):
        validate_compiled_bundle(_bundle(), not_a_target)


def test_compiled_bundle_refuses_malformed_cells_layouts_options_and_headers() -> None:
    """Repeated cells, a non-injective layout, shot options and foreign headers fail closed."""
    cell = CompiledCell("1" * 64, "2" * 64, "3" * 64, "4" * 64, 1000)
    malformed: Any = [("cz", 4)]
    for changes, reason in (
        ({"cells": ()}, "^compiled cells must be a bounded nonempty tuple$"),
        (
            {"cells": (cell, replace(cell, shots=5))},
            "^a planned cell appears twice in the bundle$",
        ),
        ({"cells": ("cell",)}, "^compiled cells must be CompiledCell records$"),
        ({"layout": ()}, "^layout must be a nonempty tuple of physical qubits$"),
        ({"layout": (0, -1, 2)}, "^layout entries must be non-negative physical qubits$"),
        ({"layout": (0, 0, 1)}, "^layout must map logical qubits to distinct physical qubits$"),
        ({"compiled_gate_counts": malformed}, "^compiled_gate_counts must be a tuple of pairs$"),
        ({"compiled_gate_counts": ()}, "^compiled gate counts must be a nonempty tuple$"),
        (
            {"compiled_gate_counts": (("cz", 0), ("measure", 3), ("prx", 10))},
            "compiled gate count",
        ),
        ({"logical_gates": ("rz", "cx", "measure")}, "^logical gates must be sorted and unique$"),
        (
            {"provider_options": (("shots", 10),)},
            "^shots belong to compiled cells, not provider options$",
        ),
        ({"logical_gates": ()}, "^logical gates must be a nonempty tuple$"),
        (
            {"provider_options": (("b", 1), ("a", 2))},
            "^provider option names must be sorted and unique$",
        ),
        ({"compiled_depth": 0}, "compiled depth"),
    ):
        with pytest.raises(ValueError, match=reason):
            _bundle(**changes)
    valid = _bundle()
    float_option: Any = (("noise", 0.5),)
    not_tuple: Any = [("cx", ("cz", "prx"))]
    too_many: Any = tuple((f"option_{index:02d}", index) for index in range(65))
    for changes, reason in (
        (
            {"provider_options": float_option},
            "^provider option values must be integers, text or booleans$",
        ),
        ({"substitutions": not_tuple}, "^substitutions must be a tuple of pairs$"),
        ({"provider_options": too_many}, "^provider options must be a bounded tuple$"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    with pytest.raises(ValueError, match="^bundle provider or device differs from its target$"):
        validate_compiled_bundle(_bundle(device="sirius"), _target())
    with pytest.raises(ValueError, match="compiled cell shots"):
        CompiledCell("1" * 64, "2" * 64, "3" * 64, "4" * 64, True)
    bundle = _bundle()
    with pytest.raises(ValueError, match="^CompiledBundle requires its exact artifact header$"):
        replace(bundle, header=replace(bundle.header, object_kind="target_snapshot"))
    with pytest.raises(
        ValueError, match="^a compiled bundle derives from provider metadata or a fixture$"
    ):
        replace(bundle, header=replace(bundle.header, data_origin="owner_dataset"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(bundle, device_name="sirius")


def test_compiled_bundle_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    target, bundle = _target(), _bundle()
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(bundle.to_wire()))
    wire_cases: tuple[tuple[str, object, str], ...] = (
        ("unexpected", True, "^CompiledBundle fields mismatch$"),
        (
            "schema",
            "scpn.experimental.llm_qpu.compiled_bundle.v2",
            "^unknown CompiledBundle schema$",
        ),
        ("layout", "0,1,2", "^bundle sequences must be lists on wire$"),
        ("substitutions", [["cx", "cz"]], "^bundle pairs must be two-item lists on wire$"),
        ("cells", [{"shots": 1}], "^CompiledCell fields mismatch$"),
    )
    for name, value, reason in wire_cases:
        wire: dict[str, Any] = bundle.to_wire()
        wire[name] = value
        with pytest.raises(ValueError, match=reason):
            decode_compiled_bundle(canonical_bytes(wire))
    worker_cases: tuple[tuple[tuple[str, ...], object, str], ...] = (
        (
            ("compiled_gate_counts",),
            [["cz", 4], ["measure", 3], ["prx", 10], ["rzz", 1]],
            "unreported gate",
        ),
        (("compiled_depth",), 13, "payload digest does not match"),
        (("header", "data_origin"), "owner_dataset", "provider metadata or a fixture"),
        (("provider_options",), [["shots", 10]], "never carry shots"),
        (("layout",), [0, 0, 1], "distinct physical qubits"),
    )
    for path, value, reason in worker_cases:
        forged: dict[str, Any] = bundle.to_wire()
        holder = forged
        for key in path[:-1]:
            holder = holder[key]
        holder[path[-1]] = value
        request: dict[str, object] = {
            "op": "roundtrip_compiled_bundle",
            "target": target.to_wire(),
            "bundle": forged,
        }
        response = _worker(request, tmp_path)
        assert response.returncode == 2, path
        assert reason in json.loads(response.stdout)["reason"], path
    other: dict[str, object] = {
        "op": "roundtrip_compiled_bundle",
        "target": target.to_wire(),
        "bundle": _bundle(_target(device="sirius")).to_wire(),
    }
    response = _worker(other, tmp_path)
    assert response.returncode == 2
    assert "different target snapshot" in json.loads(response.stdout)["reason"]
