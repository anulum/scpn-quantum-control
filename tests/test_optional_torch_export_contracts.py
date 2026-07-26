# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — optional Torch export metadata contract tests
"""Dependency-free contracts for optional PyTorch export evidence metadata."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.torch_aot_autograd_export import (
    PhaseTorchAOTAutogradExportResult,
    PhaseTorchAOTAutogradExportRoute,
    PhaseTorchAOTAutogradGraphRecord,
)
from scpn_quantum_control.phase.torch_dynamic_shape_export import (
    PhaseTorchDynamicShapeExportRecord,
    PhaseTorchDynamicShapeExportRoute,
    default_torch_dynamic_shape_export_replay_cases,
)


def test_aot_export_metadata_is_json_ready_without_optional_torch() -> None:
    """AOT graph and route metadata must remain usable without PyTorch."""
    graph = PhaseTorchAOTAutogradGraphRecord(
        kind="forward",
        artifact_path="forward.pt",
        artifact_size_bytes=17,
        artifact_sha256="a" * 64,
        graph_node_count=3,
        graph_module_type="GraphModule",
        graph_code_sha256="b" * 64,
        operation_names=("aten.cos.default",),
    )
    route = PhaseTorchAOTAutogradExportRoute(
        name="local_replay",
        status="passed",
        reason="bounded metadata contract",
        requires=("torch",),
    )

    assert graph.to_dict() == {
        "kind": "forward",
        "artifact_path": "forward.pt",
        "artifact_size_bytes": 17,
        "artifact_sha256": "a" * 64,
        "graph_node_count": 3,
        "graph_module_type": "GraphModule",
        "graph_code_sha256": "b" * 64,
        "operation_names": ["aten.cos.default"],
    }
    assert route.to_dict()["requires"] == ["torch"]

    required_routes = tuple(
        PhaseTorchAOTAutogradExportRoute(name=name, status="passed", reason="verified")
        for name in (
            "aot_autograd_forward_backward_capture",
            "aot_autograd_graph_file_round_trip",
            "loaded_backward_gradient_replay",
        )
    )
    result = PhaseTorchAOTAutogradExportResult(
        matrix_schema="test.schema.v1",
        artifact_dir="artifacts",
        forward_graph=graph,
        backward_graph=graph,
        reference_loss=0.25,
        compiled_loss=0.25,
        loaded_loss=0.25,
        compiled_loss_error=0.0,
        loaded_loss_error=0.0,
        loss_error=0.0,
        reference_gradient=np.array([0.1], dtype=np.float64),
        compiled_gradient=np.array([0.1], dtype=np.float64),
        loaded_gradient=np.array([0.1], dtype=np.float64),
        compiled_gradient_error=0.0,
        loaded_gradient_error=0.0,
        gradient_shape=(1,),
        tolerance=1.0e-9,
        torch_version="optional",
        routes=required_routes,
    )
    assert result.passed
    assert result.open_gaps == ()
    assert result.route_status("loaded_backward_gradient_replay") == "passed"
    assert result.to_dict()["passed"] is True
    with pytest.raises(KeyError, match="unknown PyTorch AOTAutograd export route"):
        result.route_status("missing")


def test_dynamic_export_metadata_is_json_ready_without_optional_torch() -> None:
    """Dynamic-shape cases and replay rows must not require PyTorch to inspect."""
    case = default_torch_dynamic_shape_export_replay_cases()[0]
    assert case.feature_shape == (2, 2)
    assert case.batch_size == 2
    assert case.feature_width == 2
    assert case.feature_matrix().dtype == np.float64
    assert case.label_vector().shape == (2,)
    assert case.to_dict()["feature_shape"] == [2, 2]

    record = PhaseTorchDynamicShapeExportRecord(
        case_name=case.name,
        feature_shape=case.feature_shape,
        batch_size=case.batch_size,
        reference_loss=0.25,
        exported_loss=0.25,
        loaded_loss=0.25,
        original_loss_error=0.0,
        loaded_loss_error=0.0,
        tolerance=1.0e-9,
    )
    route = PhaseTorchDynamicShapeExportRoute(
        name="multi_batch_loaded_cpu_replay",
        status="passed",
        reason="bounded metadata contract",
    )
    assert record.passed
    assert record.to_dict()["passed"] is True
    assert route.to_dict()["status"] == "passed"
