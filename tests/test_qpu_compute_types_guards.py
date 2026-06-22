# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Guard tests for the QPU compute contract types
"""Validation tests for the QPU compute contract dataclasses and fusion helper.

Covers the non-empty string helper, the node qubit-limit guard, the schema
-version guards on the node/stream/fusion deserialisers, the stream-delta digest
mismatch check and the fusion result/node-id length guards.
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.qpu_compute_types import (
    QPUComputeResult,
    QPUFusionResult,
    QPUNodeDescriptor,
    QPUStreamDelta,
    fuse_compute_results,
    require_non_empty,
)


def _node(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "node_id": "node",
        "access_route": "local",
        "provider": "local",
        "modality": "simulator",
        "execution_model": "emulator",
        "latency_class": "near_real_time",
        "qubit_or_variable_limit": 4,
        "kernel_capabilities": ["sync_dla"],
    }
    kwargs.update(overrides)
    return kwargs


def _result() -> QPUComputeResult:
    return QPUComputeResult(
        request_sha256="a" * 64,
        qpu_data_artifact_sha256="b" * 64,
        status="DONE_SIMULATED",
        backend_name="sim",
        backend_family="simulator",
        execution_model="exact_statevector",
        kernel="sync_dla",
        counts={"00": 1},
    )


def test_require_non_empty_rejects_blank() -> None:
    """A blank string is rejected by the non-empty helper."""
    with pytest.raises(ValueError, match="field must be non-empty"):
        require_non_empty("   ", "field")


def test_node_descriptor_rejects_non_positive_limit() -> None:
    """A qubit-or-variable limit below one is rejected."""
    with pytest.raises(ValueError, match="qubit_or_variable_limit must be >= 1"):
        QPUNodeDescriptor(**_node(qubit_or_variable_limit=0))  # type: ignore[arg-type]


def test_node_descriptor_from_dict_rejects_unknown_schema() -> None:
    """An unknown node descriptor schema version is rejected."""
    with pytest.raises(ValueError, match="unsupported QPU node descriptor schema version"):
        QPUNodeDescriptor.from_dict({"schema_version": "v0"})


def test_stream_delta_from_dict_rejects_unknown_schema() -> None:
    """An unknown stream delta schema version is rejected."""
    with pytest.raises(ValueError, match="unsupported QPU stream delta schema version"):
        QPUStreamDelta.from_dict({"schema_version": "v0"})


def test_stream_delta_from_dict_rejects_digest_mismatch() -> None:
    """A stream delta whose recorded digest does not match the payload is rejected."""
    delta = QPUStreamDelta(
        stream_id="stream",
        sequence_id=0,
        event_time="2026-04-27T10:00:00Z",
        ingest_time="2026-04-27T10:00:01Z",
        artifact_base_sha256="abc",
        state_delta={},
    )
    payload = delta.to_dict()
    payload["delta_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="delta_sha256 does not match"):
        QPUStreamDelta.from_dict(payload)


def test_fusion_from_dict_rejects_unknown_schema() -> None:
    """An unknown fusion result schema version is rejected."""
    with pytest.raises(ValueError, match="unsupported QPU fusion schema version"):
        QPUFusionResult.from_dict({"schema_version": "v0"})


def test_fuse_rejects_empty_results() -> None:
    """Fusing an empty result list is rejected."""
    with pytest.raises(ValueError, match="results must not be empty"):
        fuse_compute_results([])


def test_fuse_rejects_node_id_length_mismatch() -> None:
    """A node-id list that does not match the results count is rejected."""
    with pytest.raises(ValueError, match="node_ids length must match results"):
        fuse_compute_results([_result()], node_ids=["a", "b"])
