# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase torch export tests
# scpn-quantum-control -- PyTorch export audit tests
"""Export-persistence tests for bounded PyTorch phase-QNN modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import (
    PhaseTorchExportAuditResult,
    run_torch_module_export_audit,
)

torch = pytest.importorskip("torch")
if not hasattr(torch, "export"):
    pytest.skip("torch.export is unavailable", allow_module_level=True)


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
    """Return deterministic labels for the export fixture."""
    return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)


def _params() -> NDArray[np.float64]:
    """Return deterministic initial parameters for the export fixture."""
    return np.array([0.25, -0.35], dtype=np.float64)


def test_torch_module_export_audit_persists_exported_program(tmp_path: Path) -> None:
    """The audit should export, save, load, and replay the bounded module."""
    export_path = tmp_path / "bounded_phase_qnn_export.pt2"

    result = run_torch_module_export_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        export_path=export_path,
        tolerance=1.0e-8,
    )

    assert isinstance(result, PhaseTorchExportAuditResult)
    assert result.passed
    assert export_path.is_file()
    assert export_path.stat().st_size == result.export_size_bytes
    assert result.export_path == str(export_path)
    assert result.export_size_bytes > 0
    assert len(result.export_sha256) == 64
    assert result.route_status("module_exported_program") == "passed"
    assert result.route_status("exported_program_file_round_trip") == "passed"
    assert result.route_status("exported_program_loaded_cpu_replay") == "passed"
    assert result.route_status("exported_program_graph_signature") == "passed"
    assert result.route_status("aotautograd_gradient_export_persistence") == "blocked"
    assert result.route_status("dynamic_shape_export") == "blocked"
    assert set(result.state_dict_keys) == {"features", "labels", "params"}
    assert result.graph_node_count >= 1
    assert result.original_loss_error <= result.tolerance
    assert result.loaded_loss_error <= result.tolerance
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert routes["module_exported_program"]["status"] == "passed"
    assert routes["dynamic_shape_export"]["status"] == "blocked"
    assert "torch.export.save" in str(routes["exported_program_file_round_trip"]["reason"])
    assert "no provider" in str(payload["claim_boundary"])


def test_torch_module_export_audit_rejects_missing_parent_path(tmp_path: Path) -> None:
    """Export audit should fail closed when the destination parent is absent."""
    missing_parent = tmp_path / "missing" / "bounded_phase_qnn_export.pt2"

    with pytest.raises(ValueError, match="export_path parent"):
        run_torch_module_export_audit(
            features=_features(),
            labels=_labels(),
            initial_params=_params(),
            export_path=missing_parent,
        )


def test_torch_module_export_audit_rejects_unknown_route(tmp_path: Path) -> None:
    """Route lookups should fail closed for unknown export rows."""
    result = run_torch_module_export_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        export_path=tmp_path / "bounded_phase_qnn_export.pt2",
    )

    with pytest.raises(KeyError, match="unknown PyTorch export route"):
        result.route_status("missing")
