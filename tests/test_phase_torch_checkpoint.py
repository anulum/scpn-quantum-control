# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase torch checkpoint tests
# scpn-quantum-control -- PyTorch checkpoint audit tests
"""Checkpoint replay tests for bounded PyTorch phase-QNN modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import (
    PhaseTorchCheckpointAuditResult,
    run_torch_module_checkpoint_audit,
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
    """Return deterministic labels for the checkpoint fixture."""
    return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)


def _params() -> NDArray[np.float64]:
    """Return deterministic initial parameters for the checkpoint fixture."""
    return np.array([0.25, -0.35], dtype=np.float64)


def test_torch_module_checkpoint_audit_replays_file_checkpoint(tmp_path: Path) -> None:
    """The audit should write a real checkpoint file and replay it on CPU."""
    checkpoint_path = tmp_path / "bounded_phase_qnn.pt"

    result = run_torch_module_checkpoint_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        checkpoint_path=checkpoint_path,
        learning_rate=0.05,
        tolerance=1.0e-8,
    )

    assert isinstance(result, PhaseTorchCheckpointAuditResult)
    assert result.passed
    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size == result.checkpoint_size_bytes
    assert result.checkpoint_path == str(checkpoint_path)
    assert result.checkpoint_size_bytes > 0
    assert len(result.checkpoint_sha256) == 64
    assert result.route_status("checkpoint_file_round_trip") == "passed"
    assert result.route_status("checkpoint_weights_only_cpu_load") == "passed"
    assert result.route_status("module_state_checkpoint_replay") == "passed"
    assert result.route_status("optimizer_state_checkpoint_replay") == "passed"
    assert result.route_status("cross_runtime_checkpoint_portability") == "blocked"
    assert set(result.state_dict_keys) == {"features", "labels", "params"}
    assert result.strict_load_missing_keys == ()
    assert result.strict_load_unexpected_keys == ()
    assert result.module_loss_error <= result.tolerance
    assert result.module_gradient_error <= result.tolerance
    assert result.optimizer_replay_parameter_error <= result.tolerance
    assert result.optimizer_replay_loss_error <= result.tolerance
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert routes["checkpoint_file_round_trip"]["status"] == "passed"
    assert routes["cross_runtime_checkpoint_portability"]["status"] == "blocked"
    assert "weights_only" in str(routes["checkpoint_weights_only_cpu_load"]["reason"])
    assert "no provider" in str(payload["claim_boundary"])


def test_torch_module_checkpoint_audit_replays_in_memory_checkpoint() -> None:
    """The audit should replay a BytesIO checkpoint when no path is requested."""
    result = run_torch_module_checkpoint_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        checkpoint_path=None,
    )

    assert result.passed
    assert result.checkpoint_path is None
    assert result.checkpoint_size_bytes > 0
    assert result.route_status("checkpoint_file_round_trip") == "passed"
    assert result.route_status("checkpoint_weights_only_cpu_load") == "passed"


def test_torch_module_checkpoint_audit_rejects_missing_parent_path(tmp_path: Path) -> None:
    """Checkpoint audit should fail closed when the destination parent is absent."""
    missing_parent = tmp_path / "missing" / "bounded_phase_qnn.pt"

    with pytest.raises(ValueError, match="checkpoint_path parent"):
        run_torch_module_checkpoint_audit(
            features=_features(),
            labels=_labels(),
            initial_params=_params(),
            checkpoint_path=missing_parent,
        )


def test_torch_module_checkpoint_audit_rejects_unknown_route() -> None:
    """Route lookups should fail closed for unknown checkpoint rows."""
    result = run_torch_module_checkpoint_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
    )

    with pytest.raises(KeyError, match="unknown PyTorch checkpoint route"):
        result.route_status("missing")
