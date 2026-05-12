# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control - QPU compute-unit tests
"""Tests for the QPU compute-unit request/result infrastructure."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_quantum_control.bridge import artifact_from_arrays, write_qpu_data_artifact
from scpn_quantum_control.qpu_compute import (
    QPUComputeRequest,
    QPUComputeResult,
    QPUFusionResult,
    QPUNodeDescriptor,
    QPUStreamDelta,
    execute_simulator_request,
    fuse_compute_results,
    make_compute_request,
    read_compute_request,
    read_compute_result,
    read_fusion_result,
    read_node_descriptor,
    read_stream_delta,
    run_simulator_from_artifact,
    write_fusion_result,
    write_node_descriptor,
    write_stream_delta,
)


def _artifact(source_mode: str = "curated"):
    return artifact_from_arrays(
        domain="unit",
        source_name="unit-source",
        source_mode=source_mode,
        K_nm=np.array(
            [
                [0.0, 0.25],
                [0.25, 0.0],
            ],
            dtype=np.float64,
        ),
        omega=np.array([0.1, 0.2], dtype=np.float64),
        normalization="unit",
        extraction_method="unit-test",
        replay_id=f"{source_mode}:unit",
    )


def test_request_hash_round_trip_is_stable():
    artifact = _artifact()
    request = make_compute_request(artifact, shots=128, trotter_depth=1)
    loaded = QPUComputeRequest.from_dict(request.to_dict())

    assert loaded.request_sha256 == request.request_sha256
    assert loaded.idempotency_key == request.idempotency_key
    assert loaded.backend_policy == "simulator_statevector"


def test_request_rejects_unsupported_kernel():
    with pytest.raises(ValueError, match="kernel"):
        QPUComputeRequest(qpu_data_artifact_sha256="abc", kernel="unsupported")


def test_request_rejects_empty_hash_backend_policy_and_shots():
    with pytest.raises(ValueError, match="qpu_data_artifact_sha256"):
        QPUComputeRequest(qpu_data_artifact_sha256=" ", kernel="sync_dla")
    with pytest.raises(ValueError, match="backend_policy"):
        QPUComputeRequest(
            qpu_data_artifact_sha256="abc",
            kernel="sync_dla",
            backend_policy="live_unverified",
        )
    with pytest.raises(ValueError, match="shots"):
        QPUComputeRequest(qpu_data_artifact_sha256="abc", kernel="sync_dla", shots=0)


def test_request_rejects_tampered_request_hash():
    request = QPUComputeRequest(qpu_data_artifact_sha256="abc", kernel="sync_dla")
    payload = request.to_dict()
    payload["request_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="request_sha256"):
        QPUComputeRequest.from_dict(payload)


def test_simulator_compute_result_contains_counts_and_observables():
    artifact = _artifact()
    request = make_compute_request(
        artifact,
        kernel="sync_dla",
        shots=256,
        trotter_depth=2,
        coupling_scale=1.5,
    )

    result = execute_simulator_request(artifact, request)

    assert result.status == "DONE_SIMULATED"
    assert sum(result.counts.values()) == 256
    assert result.backend_name == "local_statevector"
    assert result.observables.keys() >= {"sync_order", "dla_asymmetry"}
    assert result.observable_classification["sync_order"] == "simulated_exact_statevector"
    assert result.circuit_metadata["num_qubits"] == 2
    assert result.circuit_metadata["coupling_scale"] == 1.5

    loaded = QPUComputeResult.from_dict(result.to_dict())
    assert loaded.result_sha256 == result.result_sha256
    assert loaded.counts_sha256 == result.counts_sha256


def test_simulator_rejects_mismatched_artifact_hash():
    artifact = _artifact()
    request = make_compute_request(artifact)
    other = _artifact(source_mode="replay")

    with pytest.raises(ValueError, match="artifact hash"):
        execute_simulator_request(other, request)


def test_run_simulator_from_artifact_writes_request_and_result(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    write_qpu_data_artifact(artifact_path, _artifact())

    result = run_simulator_from_artifact(
        artifact_path,
        request_out=request_path,
        result_out=result_path,
        shots=64,
        trotter_depth=1,
    )

    assert result_path.exists()
    assert request_path.exists()
    assert read_compute_request(request_path).request_sha256 == result.request_sha256
    assert read_compute_result(result_path).result_sha256 == result.result_sha256


def test_publication_gate_rejects_synthetic_without_explicit_opt_in(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    write_qpu_data_artifact(artifact_path, _artifact(source_mode="synthetic"))

    with pytest.raises(ValueError, match="synthetic"):
        run_simulator_from_artifact(artifact_path)

    result = run_simulator_from_artifact(
        artifact_path,
        require_publication_safe=False,
        shots=32,
    )
    assert result.status == "DONE_SIMULATED"


def test_cli_run_simulator(tmp_path):
    from scpn_quantum_control.qpu_compute import main

    artifact_path = tmp_path / "artifact.json"
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    write_qpu_data_artifact(artifact_path, _artifact(source_mode="fixture"))

    exit_code = main(
        [
            "run-simulator",
            "--artifact",
            str(artifact_path),
            "--request-out",
            str(request_path),
            "--result-out",
            str(result_path),
            "--allow-synthetic",
            "--shots",
            "16",
        ]
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["status"] == "DONE_SIMULATED"
    assert sum(payload["counts"].values()) == 16


def test_node_descriptor_round_trip_and_file_io(tmp_path):
    descriptor = QPUNodeDescriptor(
        node_id="local.statevector",
        access_route="local",
        provider="local",
        modality="simulator",
        execution_model="emulator",
        latency_class="near_real_time",
        qubit_or_variable_limit=12,
        native_features={"counts": True},
        cost_model={"unit": "none"},
        queue_model="immediate",
        kernel_capabilities=["sync_dla", "otoc_proxy"],
        calibration_snapshot={"timestamp": "2026-04-27T00:00:00Z"},
        verification_status="simulator_green",
    )
    path = tmp_path / "node.json"

    write_node_descriptor(path, descriptor)
    loaded = read_node_descriptor(path)

    assert loaded.descriptor_sha256 == descriptor.descriptor_sha256
    assert loaded.kernel_capabilities == ["sync_dla", "otoc_proxy"]


def test_node_descriptor_rejects_unknown_modality():
    with pytest.raises(ValueError, match="modality"):
        QPUNodeDescriptor(
            node_id="bad",
            access_route="local",
            provider="local",
            modality="unknown",
            execution_model="emulator",
            latency_class="batch",
            qubit_or_variable_limit=1,
            kernel_capabilities=["sync_dla"],
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("access_route", "unsupported_route", "access_route"),
        ("execution_model", "pulse_magic", "execution_model"),
        ("latency_class", "instant", "latency_class"),
    ],
)
def test_node_descriptor_rejects_unknown_routing_metadata(field, value, match):
    kwargs = {
        "node_id": "node",
        "access_route": "local",
        "provider": "local",
        "modality": "simulator",
        "execution_model": "emulator",
        "latency_class": "near_real_time",
        "qubit_or_variable_limit": 4,
        "kernel_capabilities": ["sync_dla"],
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        QPUNodeDescriptor(**kwargs)


def test_node_descriptor_rejects_empty_kernel_capabilities():
    with pytest.raises(ValueError, match="kernel_capabilities"):
        QPUNodeDescriptor(
            node_id="node",
            access_route="local",
            provider="local",
            modality="simulator",
            execution_model="emulator",
            latency_class="near_real_time",
            qubit_or_variable_limit=4,
            kernel_capabilities=[],
        )


def test_node_descriptor_rejects_tampered_descriptor_hash():
    descriptor = QPUNodeDescriptor(
        node_id="local.statevector",
        access_route="local",
        provider="local",
        modality="simulator",
        execution_model="emulator",
        latency_class="near_real_time",
        qubit_or_variable_limit=12,
        kernel_capabilities=["sync_dla"],
    )
    payload = descriptor.to_dict()
    payload["descriptor_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="descriptor_sha256"):
        QPUNodeDescriptor.from_dict(payload)


def test_stream_delta_round_trip_and_validation(tmp_path):
    artifact = _artifact()
    delta = QPUStreamDelta(
        stream_id="plasma.loop",
        sequence_id=7,
        event_time="2026-04-27T10:00:00Z",
        ingest_time="2026-04-27T10:00:01Z",
        artifact_base_sha256=artifact.to_dict()["artifact_sha256"],
        state_delta={"omega": {"3": 0.12}},
        deadline="2026-04-27T10:00:02Z",
        control_window={"start": "2026-04-27T10:00:01Z", "stop": "2026-04-27T10:00:02Z"},
        confidence=0.82,
    )
    path = tmp_path / "delta.json"

    write_stream_delta(path, delta)
    loaded = read_stream_delta(path)

    assert loaded.delta_sha256 == delta.delta_sha256
    assert loaded.sequence_id == 7
    assert loaded.confidence == 0.82

    with pytest.raises(ValueError, match="confidence"):
        QPUStreamDelta(
            stream_id="bad",
            sequence_id=0,
            event_time="now",
            ingest_time="now",
            artifact_base_sha256="abc",
            state_delta={},
            confidence=1.5,
        )


def test_stream_delta_rejects_negative_sequence():
    with pytest.raises(ValueError, match="sequence_id"):
        QPUStreamDelta(
            stream_id="stream",
            sequence_id=-1,
            event_time="2026-04-27T10:00:00Z",
            ingest_time="2026-04-27T10:00:01Z",
            artifact_base_sha256="abc",
            state_delta={},
        )


def test_fuse_compute_results_uses_shot_weighting_and_round_trips(tmp_path):
    artifact = _artifact()
    request = make_compute_request(artifact, shots=128, trotter_depth=1)
    first = execute_simulator_request(artifact, request)
    second = QPUComputeResult(
        request_sha256=first.request_sha256,
        qpu_data_artifact_sha256=first.qpu_data_artifact_sha256,
        status="DONE_SIMULATED",
        backend_name="shadow_statevector",
        backend_family="simulator",
        execution_model="exact_statevector",
        kernel=first.kernel,
        counts={"00": 32, "11": 32},
        observables={"sync_order": 0.25, "dla_asymmetry": -0.5},
        observable_classification={
            "sync_order": "simulated_exact_statevector",
            "dla_asymmetry": "simulated_exact_statevector",
        },
    )

    fusion = fuse_compute_results([first, second])

    assert fusion.node_ids == ["local_statevector", "shadow_statevector"]
    assert set(fusion.fused_observables) >= {"sync_order", "dla_asymmetry"}
    assert fusion.agreement_metrics["sync_order_max_minus_min"] >= 0.0

    path = tmp_path / "fusion.json"
    write_fusion_result(path, fusion)
    loaded = read_fusion_result(path)
    assert loaded.fusion_sha256 == fusion.fusion_sha256


def test_compute_result_rejects_negative_counts():
    with pytest.raises(ValueError, match="counts"):
        QPUComputeResult(
            request_sha256="request",
            qpu_data_artifact_sha256="artifact",
            status="DONE_SIMULATED",
            backend_name="local_statevector",
            backend_family="simulator",
            execution_model="exact_statevector",
            kernel="sync_dla",
            counts={"00": -1},
        )


def test_compute_result_rejects_empty_identity_fields_and_unsupported_kernel():
    base = {
        "request_sha256": "request",
        "qpu_data_artifact_sha256": "artifact",
        "status": "DONE_SIMULATED",
        "backend_name": "local_statevector",
        "backend_family": "simulator",
        "execution_model": "exact_statevector",
        "kernel": "sync_dla",
    }
    for field in ("request_sha256", "qpu_data_artifact_sha256", "status"):
        payload = dict(base)
        payload[field] = " "
        with pytest.raises(ValueError, match=field):
            QPUComputeResult(**payload)

    payload = dict(base)
    payload["kernel"] = "unsupported"
    with pytest.raises(ValueError, match="kernel"):
        QPUComputeResult(**payload)


def test_compute_result_rejects_tampered_count_hash():
    result = QPUComputeResult(
        request_sha256="request",
        qpu_data_artifact_sha256="artifact",
        status="DONE_SIMULATED",
        backend_name="local_statevector",
        backend_family="simulator",
        execution_model="exact_statevector",
        kernel="sync_dla",
        counts={"00": 3, "11": 5},
    )
    payload = result.to_dict()
    payload["counts_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="counts_sha256"):
        QPUComputeResult.from_dict(payload)


def test_compute_result_rejects_tampered_result_hash():
    result = QPUComputeResult(
        request_sha256="request",
        qpu_data_artifact_sha256="artifact",
        status="DONE_SIMULATED",
        backend_name="local_statevector",
        backend_family="simulator",
        execution_model="exact_statevector",
        kernel="sync_dla",
        counts={"00": 3},
    )
    payload = result.to_dict()
    payload["result_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="result_sha256"):
        QPUComputeResult.from_dict(payload)


def test_serialised_contracts_reject_wrong_schema_versions():
    request = QPUComputeRequest(qpu_data_artifact_sha256="abc", kernel="sync_dla").to_dict()
    request["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="request schema"):
        QPUComputeRequest.from_dict(request)

    result = QPUComputeResult(
        request_sha256="request",
        qpu_data_artifact_sha256="artifact",
        status="DONE_SIMULATED",
        backend_name="local_statevector",
        backend_family="simulator",
        execution_model="exact_statevector",
        kernel="sync_dla",
    ).to_dict()
    result["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="result schema"):
        QPUComputeResult.from_dict(result)


def test_fusion_result_rejects_empty_and_mismatched_contributors():
    with pytest.raises(ValueError, match="contributing_result_sha256"):
        QPUFusionResult(
            fused_observables={},
            contributing_result_sha256=[],
            node_ids=[],
            weighting_rule="shots",
        )

    with pytest.raises(ValueError, match="node_ids length"):
        QPUFusionResult(
            fused_observables={},
            contributing_result_sha256=["abc"],
            node_ids=[],
            weighting_rule="shots",
        )


def test_fusion_result_rejects_tampered_hash_and_unsupported_weighting():
    fusion = QPUFusionResult(
        fused_observables={"sync_order": 0.5},
        contributing_result_sha256=["abc"],
        node_ids=["node"],
        weighting_rule="shots",
    )
    payload = fusion.to_dict()
    payload["fusion_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="fusion_sha256"):
        QPUFusionResult.from_dict(payload)

    result = QPUComputeResult(
        request_sha256="request",
        qpu_data_artifact_sha256="artifact",
        status="DONE_SIMULATED",
        backend_name="local_statevector",
        backend_family="simulator",
        execution_model="exact_statevector",
        kernel="sync_dla",
        counts={"00": 1},
    )
    with pytest.raises(ValueError, match="only shots"):
        fuse_compute_results([result], weighting_rule="equal")
