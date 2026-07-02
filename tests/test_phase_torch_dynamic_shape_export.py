# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- PyTorch dynamic-shape export tests
"""Dynamic-shape export tests for bounded PyTorch phase-QNN modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control.phase import (
    TORCH_DYNAMIC_SHAPE_EXPORT_SCHEMA,
    PhaseTorchDynamicShapeExportReplayCase,
    PhaseTorchDynamicShapeExportResult,
    default_torch_dynamic_shape_export_replay_cases,
    run_torch_dynamic_shape_export_audit,
)
from scpn_quantum_control.phase.torch_dynamic_shape_export import (
    _dynamic_batch_dim,
    _exported_program_loss_for_case,
)

torch = pytest.importorskip("torch")
if not hasattr(torch, "export"):
    pytest.skip("torch.export is unavailable", allow_module_level=True)


def test_torch_dynamic_shape_export_replays_multiple_batch_sizes(
    tmp_path: Path,
) -> None:
    """The audit should export one constrained program and replay multiple batches."""
    export_path = tmp_path / "bounded_phase_qnn_dynamic_shape.pt2"

    result = run_torch_dynamic_shape_export_audit(
        export_path=export_path,
        tolerance=1.0e-8,
    )

    assert isinstance(result, PhaseTorchDynamicShapeExportResult)
    assert result.passed
    assert result.matrix_schema == TORCH_DYNAMIC_SHAPE_EXPORT_SCHEMA
    assert result.export_path == str(export_path)
    assert export_path.is_file()
    assert result.export_size_bytes == export_path.stat().st_size
    assert len(result.export_sha256) == 64
    assert result.dynamic_shape_claim is True
    assert result.dynamic_feature_width_claim is False
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False
    assert result.route_status("dynamic_batch_shape_constraints") == "passed"
    assert result.route_status("dynamic_exported_program_file_round_trip") == "passed"
    assert result.route_status("multi_batch_loaded_cpu_replay") == "passed"
    assert result.route_status("aotautograd_gradient_export_persistence") == "blocked"
    assert result.route_status("cuda_dynamic_shape_export_replay") == "blocked"
    assert "aotautograd_gradient_export_persistence" in result.open_gaps
    assert "cuda_dynamic_shape_export_replay" in result.open_gaps
    assert result.replay_count >= 3
    assert len(result.batch_sizes) >= 3
    assert {record.batch_size for record in result.records} == set(result.batch_sizes)
    assert all(record.passed for record in result.records)
    assert all(record.original_loss_error <= result.tolerance for record in result.records)
    assert all(record.loaded_loss_error <= result.tolerance for record in result.records)
    assert all(record.feature_shape[1] == result.feature_width for record in result.records)
    assert "USER_INPUT" in result.graph_signature
    assert "features" in result.graph_signature
    assert "labels" in result.graph_signature
    assert "VR[" in result.range_constraints

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    records = cast(list[dict[str, Any]], payload["records"])
    assert routes["dynamic_batch_shape_constraints"]["status"] == "passed"
    assert routes["aotautograd_gradient_export_persistence"]["status"] == "blocked"
    assert records[0]["loaded_loss_error"] <= result.tolerance
    assert payload["dynamic_shape_claim"] is True
    assert "dynamic batch" in str(payload["claim_boundary"])


def test_default_torch_dynamic_shape_export_replay_cases_are_distinct() -> None:
    """Default replay cases should share width and vary the batch dimension."""
    cases = default_torch_dynamic_shape_export_replay_cases()

    assert len(cases) >= 3
    assert len({case.name for case in cases}) == len(cases)
    assert len({case.batch_size for case in cases}) >= 3
    assert {case.feature_width for case in cases} == {2}
    assert all(case.to_dict()["feature_shape"] for case in cases)


def test_torch_dynamic_shape_export_rejects_duplicate_case_names(
    tmp_path: Path,
) -> None:
    """Replay case names must be unique so replay evidence is unambiguous."""
    case = default_torch_dynamic_shape_export_replay_cases()[0]

    with pytest.raises(ValueError, match="duplicate replay case"):
        run_torch_dynamic_shape_export_audit(
            export_path=tmp_path / "dynamic.pt2",
            replay_cases=(case, case),
        )


def test_torch_dynamic_shape_export_rejects_static_only_replay_set(
    tmp_path: Path,
) -> None:
    """Dynamic-shape promotion must prove more than one concrete batch size."""
    case = PhaseTorchDynamicShapeExportReplayCase(
        name="single_batch",
        features=((0.0, 1.0), (1.0, 0.0)),
        labels=(0.0, 1.0),
    )

    with pytest.raises(ValueError, match="at least two distinct batch sizes"):
        run_torch_dynamic_shape_export_audit(
            export_path=tmp_path / "dynamic.pt2",
            replay_cases=(case,),
            initial_params=(0.25, -0.35),
        )


def test_torch_dynamic_shape_export_rejects_missing_parent_path(
    tmp_path: Path,
) -> None:
    """The audit should fail closed when the export parent is absent."""
    with pytest.raises(ValueError, match="export_path parent"):
        run_torch_dynamic_shape_export_audit(
            export_path=tmp_path / "missing" / "dynamic.pt2",
        )


def test_torch_dynamic_shape_export_rejects_unknown_route(tmp_path: Path) -> None:
    """Route lookups should fail closed for unknown dynamic-shape rows."""
    result = run_torch_dynamic_shape_export_audit(export_path=tmp_path / "dynamic.pt2")

    with pytest.raises(KeyError, match="unknown PyTorch dynamic-shape export route"):
        result.route_status("missing")


def test_torch_dynamic_shape_export_replay_case_rejects_bad_shapes() -> None:
    """Replay case construction should validate rectangular features and labels."""
    with pytest.raises(ValueError, match="name must be non-empty"):
        PhaseTorchDynamicShapeExportReplayCase(
            name=" ",
            features=((0.0,),),
            labels=(0.0,),
        )

    with pytest.raises(ValueError, match="must not contain whitespace"):
        PhaseTorchDynamicShapeExportReplayCase(
            name="bad name",
            features=((0.0,),),
            labels=(0.0,),
        )

    with pytest.raises(ValueError, match="plain artifact stem"):
        PhaseTorchDynamicShapeExportReplayCase(
            name="bad/name",
            features=((0.0,),),
            labels=(0.0,),
        )

    with pytest.raises(ValueError, match="features"):
        PhaseTorchDynamicShapeExportReplayCase(
            name="bad_features",
            features=(),
            labels=(0.0,),
        )

    with pytest.raises(ValueError, match="column"):
        PhaseTorchDynamicShapeExportReplayCase(
            name="bad_empty_columns",
            features=((),),
            labels=(0.0,),
        )

    with pytest.raises(ValueError, match="rectangular"):
        PhaseTorchDynamicShapeExportReplayCase(
            name="bad_rectangular_features",
            features=((0.0,), (1.0, 2.0)),
            labels=(0.0, 1.0),
        )

    with pytest.raises(ValueError, match="labels"):
        PhaseTorchDynamicShapeExportReplayCase(
            name="bad_labels",
            features=((0.0,), (1.0,)),
            labels=(0.0,),
        )


def test_torch_dynamic_shape_export_rejects_empty_and_mixed_width_replay_sets(
    tmp_path: Path,
) -> None:
    """Replay sets must prove dynamic batches without changing feature width."""
    with pytest.raises(ValueError, match="at least two distinct batch sizes"):
        run_torch_dynamic_shape_export_audit(
            export_path=tmp_path / "dynamic.pt2",
            replay_cases=(),
        )

    case_width_1 = PhaseTorchDynamicShapeExportReplayCase(
        name="width_1",
        features=((0.0,), (1.0,)),
        labels=(0.0, 1.0),
    )
    case_width_2 = PhaseTorchDynamicShapeExportReplayCase(
        name="width_2",
        features=((0.0, 1.0), (1.0, 0.0), (0.5, 0.25)),
        labels=(0.0, 1.0, 0.0),
    )

    with pytest.raises(ValueError, match="share one feature width"):
        run_torch_dynamic_shape_export_audit(
            export_path=tmp_path / "dynamic.pt2",
            replay_cases=(case_width_1, case_width_2),
        )


def test_torch_dynamic_shape_export_defensive_helpers_fail_closed() -> None:
    """Private defensive helpers should fail closed on missing PyTorch hooks."""
    with pytest.raises(RuntimeError, match="torch.export.Dim"):
        _dynamic_batch_dim(export_module=object(), min_batch=1, max_batch=2)

    with pytest.raises(RuntimeError, match="ExportedProgram must expose module"):
        _exported_program_loss_for_case(
            torch_module=torch,
            exported_program=object(),
            case=default_torch_dynamic_shape_export_replay_cases()[0],
        )
