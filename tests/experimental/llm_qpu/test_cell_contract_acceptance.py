# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contract acceptance
"""Exercise strict offline cell identities and the standalone wire boundary."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    ArrayDescriptor,
    CellKey,
    PlannedCell,
    attempt_id,
    canonical_bytes,
    content_id,
    decode_contract,
    f64,
    request_id,
    run_id,
    validate_completion,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_DIGEST = "a" * 64
_KERNEL = "xy_static_digital_v1"


def _cell(*, shots: int = 8, metadata: dict[str, object] | None = None) -> PlannedCell:
    return PlannedCell.create(
        key=CellKey("experiment", "sample", "sequence", "arm", "basis", "r1", "e1", "b1"),
        circuit_plan_digest=_DIGEST,
        shots=shots,
        measurement_width=2,
        role="data",
        metadata={} if metadata is None else metadata,
    )


def _result(cell: PlannedCell) -> dict[str, object]:
    return {
        "key": cell.key.to_wire(),
        "provider_job_id": "job-1",
        "pub_index": 0,
        "binding_index": 0,
        "counts": {"00": cell.shots},
        "actual_shots": cell.shots,
        "joint_shape": [cell.shots, cell.measurement_width],
    }


@pytest.mark.parametrize("shots", [True, 1.5, "1024", 0, -1])
def test_w02_01_refuses_noninteger_shots_before_sdk(shots: object) -> None:
    with pytest.raises(ValueError, match="shots"):
        _cell(shots=shots)  # type: ignore[arg-type]


def test_w02_01_refuses_nonfinite_or_untagged_metadata() -> None:
    for value in (float("nan"), float("inf"), 0.5):
        with pytest.raises(ValueError, match="floats"):
            _cell(metadata={"angle": value})
    with pytest.raises(ValueError, match="finite"):
        f64(float("nan"))
    assert f64(-0.0) == {"$f64": "0x0.0p+0"}


def test_w02_02_nested_input_mutation_cannot_change_approved_bytes() -> None:
    source: dict[str, object] = {"config": {"values": [1, 2], "angle": f64(0.25)}}
    cell = _cell(metadata=source)
    approved = canonical_bytes(cell.to_wire())
    config = source["config"]
    assert isinstance(config, dict)
    values = config["values"]
    assert isinstance(values, list)
    values.append(99)
    config["angle"] = f64(0.5)
    source["new_field"] = True
    assert canonical_bytes(cell.to_wire()) == approved
    assert cell.to_wire()["metadata"] != source


def test_w02_03_json_and_binary_roundtrip_in_standalone_worker(tmp_path: Path) -> None:
    cell = _cell(metadata={"angle": f64(0.25), "labels": ["e\u0301", "x"]})
    payload = struct.pack("<dd", 0.25, -0.0)
    descriptor = ArrayDescriptor("<f8", (2,), len(payload), hashlib.sha256(payload).hexdigest())
    descriptor.validate_payload(payload)
    env = {
        "PYTHONPATH": str(tmp_path),
        "IQM_TOKEN": "poison",
        "IBM_QUANTUM_TOKEN": "poison",
    }
    request = {
        "op": "roundtrip_contract",
        "cell": cell.to_wire(),
        "array": descriptor.to_wire(),
        "array_base64": base64.b64encode(payload).decode("ascii"),
    }
    result = subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode("utf-8"),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    response = json.loads(result.stdout)
    assert response["status"] == "validated_roundtrip_no_compute"
    assert response["hardware_submission_enabled"] is False
    assert response["cell_sha256"] == hashlib.sha256(canonical_bytes(cell.to_wire())).hexdigest()
    assert (
        response["array_sha256"]
        == hashlib.sha256(canonical_bytes(descriptor.to_wire())).hexdigest()
    )
    assert decode_contract(canonical_bytes(response["cell"])) == cell
    assert decode_contract(canonical_bytes(response["array"])) == descriptor
    with pytest.raises(ValueError, match="digest"):
        descriptor.validate_payload(payload[:-1] + b"x")


def test_w02_04_content_run_and_attempt_identity_split() -> None:
    kwargs = {
        "kernel_id": _KERNEL,
        "map_version": "angle_tanh_v1",
        "protocol_digest": _DIGEST,
        "scientific_fields": {"angle": f64(0.25)},
    }
    first = content_id(**kwargs, output_dir="/tmp/first")
    assert content_id(**kwargs, output_dir="/tmp/second") == first
    assert content_id(**{**kwargs, "kernel_id": "xy_sequence_digital_v1"}) != first
    request = request_id(first, _cell(), kernel_id=_KERNEL)
    assert request_id(first, _cell(shots=16), kernel_id=_KERNEL) != request
    assert request_id(first, _cell(), kernel_id="xy_sequence_digital_v1") != request
    run = run_id(request, repetition=1, calibration_epoch_id="cal-1")
    assert run_id(request, repetition=1, calibration_epoch_id="cal-2") != run
    nonce = str(uuid.uuid4())
    assert attempt_id(run, nonce=nonce) == attempt_id(run, nonce=nonce)
    assert attempt_id(run, nonce=str(uuid.uuid4())) != attempt_id(run, nonce=nonce)


def test_w02_05_refuses_unknown_schema_kernel_and_metadata_shadow() -> None:
    cell = _cell()
    wire = cell.to_wire()
    wire["schema"] = "unknown"
    with pytest.raises(ValueError, match="schema"):
        PlannedCell.from_wire(wire)
    wire = cell.to_wire()
    wire["unreviewed"] = True
    with pytest.raises(ValueError, match="fields"):
        PlannedCell.from_wire(wire)
    with pytest.raises(ValueError, match="unknown kernel"):
        content_id(
            kernel_id="unreviewed",
            map_version="angle_tanh_v1",
            protocol_digest=_DIGEST,
            scientific_fields={"angle": f64(0.25)},
        )
    with pytest.raises(ValueError, match="shadows"):
        _cell(metadata={"provider": "test"})
    with pytest.raises(ValueError, match="unsupported"):
        decode_contract(canonical_bytes({"schema": "scpn.experimental.llm_qpu.owner_approval.v1"}))
    with pytest.raises(ValueError, match="noncanonical"):
        decode_contract(b'{"schema":"scpn.experimental.llm_qpu.owner_approval.v1", "x":1}')


def test_w02_06_raw_completion_refuses_coercion_missing_and_shape() -> None:
    cell = _cell()
    validate_completion([cell], [_result(cell)])
    with pytest.raises(ValueError, match="incomplete"):
        validate_completion([cell], [])
    for count in (1.0, True):
        result = _result(cell)
        result["counts"] = {"00": count}
        with pytest.raises(ValueError, match="integers"):
            validate_completion([cell], [result])
    result = _result(cell)
    result["joint_shape"] = [cell.shots, 1]
    with pytest.raises(ValueError, match="shape"):
        validate_completion([cell], [result])
    result = _result(cell)
    result["counts"] = {}
    with pytest.raises(ValueError, match="nonempty"):
        validate_completion([cell], [result])
    result = _result(cell)
    result["actual_shots"] = True
    with pytest.raises(ValueError, match="actual shots"):
        validate_completion([cell], [result])
    with pytest.raises(ValueError, match="duplicate"):
        validate_completion([cell, replace(cell, role="sentinel")], [_result(cell), _result(cell)])
