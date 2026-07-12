# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase torch checkpoint matrix tests
# scpn-quantum-control -- PyTorch checkpoint matrix tests
"""Long-lived checkpoint matrix tests for bounded PyTorch phase-QNN modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import (
    TORCH_CHECKPOINT_MATRIX_SCHEMA,
    PhaseTorchCheckpointMatrixResult,
    run_torch_long_lived_checkpoint_matrix,
)

pytest.importorskip("torch")


def _features() -> NDArray[np.float64]:
    """Return a deterministic two-parameter bounded phase-QNN fixture."""
    return np.array(
        [
            [0.0, 1.0],
            [np.pi / 2.0, -0.4],
            [np.pi, 0.25],
            [3.0 * np.pi / 2.0, 0.75],
        ],
        dtype=np.float64,
    )


def _labels() -> NDArray[np.float64]:
    """Return deterministic labels for the checkpoint-matrix fixture."""
    return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)


def _params() -> NDArray[np.float64]:
    """Return deterministic initial parameters for the checkpoint-matrix fixture."""
    return np.array([0.25, -0.35], dtype=np.float64)


def test_torch_checkpoint_matrix_records_versioned_local_replay(
    tmp_path: Path,
) -> None:
    """The matrix should record schema, tensor metadata, and repeated CPU replay."""
    checkpoint_path = tmp_path / "bounded_phase_qnn.pt"

    result = run_torch_long_lived_checkpoint_matrix(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        checkpoint_path=checkpoint_path,
        replay_count=2,
        learning_rate=0.05,
        tolerance=1.0e-8,
    )

    assert isinstance(result, PhaseTorchCheckpointMatrixResult)
    assert result.passed
    assert result.matrix_schema == TORCH_CHECKPOINT_MATRIX_SCHEMA
    assert result.checkpoint_path == str(checkpoint_path)
    assert result.replay_count == 2
    assert result.checkpoint_size_bytes == checkpoint_path.stat().st_size
    assert len(result.checkpoint_sha256) == 64
    assert result.route_status("versioned_checkpoint_schema") == "passed"
    assert result.route_status("weights_only_local_cpu_replay") == "passed"
    assert result.route_status("repeated_local_cpu_replay") == "passed"
    assert result.route_status("tensor_metadata_manifest") == "passed"
    assert result.route_status("runtime_fingerprint_recorded") == "passed"
    assert result.route_status("cross_runtime_checkpoint_replay") == "blocked"
    assert result.route_status("cuda_checkpoint_replay") == "blocked"
    assert result.route_status("long_lived_external_checkpoint_artifact") == "blocked"
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False
    assert result.open_gaps == (
        "cross_runtime_checkpoint_replay",
        "cuda_checkpoint_replay",
        "long_lived_external_checkpoint_artifact",
    )

    tensor_shapes = {metadata.name: metadata.shape for metadata in result.tensor_metadata}
    assert tensor_shapes["module_state_dict.features"] == (4, 2)
    assert tensor_shapes["module_state_dict.labels"] == (4,)
    assert tensor_shapes["module_state_dict.params"] == (2,)
    assert all(metadata.device == "cpu" for metadata in result.tensor_metadata)

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    runtime = cast(dict[str, str], payload["runtime_fingerprint"])
    assert payload["passed"] is True
    assert routes["repeated_local_cpu_replay"]["status"] == "passed"
    assert routes["cross_runtime_checkpoint_replay"]["status"] == "blocked"
    assert "torch_version" in runtime
    assert "python_version" in runtime
    assert "no cross-runtime" in str(payload["claim_boundary"])


def test_torch_checkpoint_matrix_supports_omitted_public_checkpoint_path() -> None:
    """The matrix should support callers that do not request a retained path."""
    result = run_torch_long_lived_checkpoint_matrix(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        checkpoint_path=None,
        replay_count=1,
    )

    assert result.passed
    assert result.checkpoint_path is None
    assert result.checkpoint_size_bytes > 0
    assert result.route_status("repeated_local_cpu_replay") == "passed"


def test_torch_checkpoint_matrix_rejects_invalid_replay_count() -> None:
    """The matrix should reject non-positive repeated replay requests."""
    with pytest.raises(ValueError, match="replay_count must be positive"):
        run_torch_long_lived_checkpoint_matrix(
            features=_features(),
            labels=_labels(),
            initial_params=_params(),
            replay_count=0,
        )


def test_torch_checkpoint_matrix_rejects_unknown_route() -> None:
    """Route lookups should fail closed for unknown matrix rows."""
    result = run_torch_long_lived_checkpoint_matrix(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
    )

    with pytest.raises(KeyError, match="unknown PyTorch checkpoint matrix route"):
        result.route_status("missing")
