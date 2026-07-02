# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- PyTorch export shape matrix tests
"""Static-shape export matrix tests for bounded PyTorch phase-QNN modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from scpn_quantum_control.phase import (
    TORCH_EXPORT_SHAPE_MATRIX_SCHEMA,
    PhaseTorchExportShapeMatrixResult,
    PhaseTorchExportShapeScenario,
    default_torch_export_shape_scenarios,
    run_torch_export_shape_matrix,
)

torch = pytest.importorskip("torch")
if not hasattr(torch, "export"):
    pytest.skip("torch.export is unavailable", allow_module_level=True)


def test_torch_export_shape_matrix_records_static_shapes_and_dynamic_blockers(
    tmp_path: Path,
) -> None:
    """The matrix should prove static-shape export replay and block dynamic promotion."""
    result = run_torch_export_shape_matrix(export_dir=tmp_path, tolerance=1.0e-8)

    assert isinstance(result, PhaseTorchExportShapeMatrixResult)
    assert result.passed
    assert result.matrix_schema == TORCH_EXPORT_SHAPE_MATRIX_SCHEMA
    assert result.scenario_count >= 2
    assert result.passed_count == result.scenario_count
    assert result.route_status("static_shape_export_matrix") == "passed"
    assert result.route_status("multi_shape_local_replay") == "passed"
    assert result.route_status("dynamic_shape_constraints") == "blocked"
    assert result.route_status("dynamic_shape_replay_matrix") == "blocked"
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False
    assert result.dynamic_shape_claim is False
    assert result.open_gaps == (
        "dynamic_shape_constraints",
        "dynamic_shape_replay_matrix",
    )

    feature_shapes = {record.feature_shape for record in result.records}
    assert (2, 1) in feature_shapes
    assert (4, 2) in feature_shapes
    assert all(record.passed for record in result.records)
    assert all(record.export_size_bytes > 0 for record in result.records)
    assert all(len(record.export_sha256) == 64 for record in result.records)
    assert all(record.loaded_loss_error <= result.tolerance for record in result.records)
    assert all(Path(record.export_path).is_file() for record in result.records)

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    records = cast(list[dict[str, Any]], payload["records"])
    assert routes["dynamic_shape_constraints"]["status"] == "blocked"
    assert routes["dynamic_shape_replay_matrix"]["status"] == "blocked"
    assert records[0]["feature_shape"]
    assert "dynamic-shape" in str(payload["claim_boundary"])


def test_default_torch_export_shape_scenarios_are_distinct() -> None:
    """Default scenarios should exercise more than one bounded static shape."""
    scenarios = default_torch_export_shape_scenarios()

    assert len(scenarios) >= 2
    assert len({scenario.name for scenario in scenarios}) == len(scenarios)
    assert len({scenario.feature_shape for scenario in scenarios}) >= 2
    assert all(scenario.parameter_width == scenario.feature_shape[1] for scenario in scenarios)
    assert all(scenario.to_dict()["feature_shape"] for scenario in scenarios)


def test_torch_export_shape_matrix_rejects_empty_scenarios(tmp_path: Path) -> None:
    """The matrix should fail closed when no static export scenario is supplied."""
    with pytest.raises(ValueError, match="at least one"):
        run_torch_export_shape_matrix(export_dir=tmp_path, scenarios=())


def test_torch_export_shape_matrix_rejects_duplicate_scenario_names(
    tmp_path: Path,
) -> None:
    """Scenario names must be unique so artifact names stay unambiguous."""
    scenario = default_torch_export_shape_scenarios()[0]

    with pytest.raises(ValueError, match="duplicate scenario"):
        run_torch_export_shape_matrix(
            export_dir=tmp_path,
            scenarios=(scenario, scenario),
        )


def test_torch_export_shape_matrix_rejects_unknown_route(tmp_path: Path) -> None:
    """Route lookups should fail closed for unknown shape-matrix rows."""
    result = run_torch_export_shape_matrix(export_dir=tmp_path)

    with pytest.raises(KeyError, match="unknown PyTorch export shape matrix route"):
        result.route_status("missing")


def test_torch_export_shape_scenario_rejects_bad_shapes() -> None:
    """Scenario construction should validate rectangular features and labels."""
    with pytest.raises(ValueError, match="features"):
        PhaseTorchExportShapeScenario(
            name="bad_features",
            features=(),
            labels=(0.0,),
            initial_params=(0.1,),
        )

    with pytest.raises(ValueError, match="rectangular"):
        PhaseTorchExportShapeScenario(
            name="bad_rectangular_features",
            features=((0.0,), (1.0, 2.0)),
            labels=(0.0, 1.0),
            initial_params=(0.1,),
        )

    with pytest.raises(ValueError, match="labels"):
        PhaseTorchExportShapeScenario(
            name="bad_labels",
            features=((0.0,), (1.0,)),
            labels=(0.0,),
            initial_params=(0.1,),
        )

    with pytest.raises(ValueError, match="initial_params"):
        PhaseTorchExportShapeScenario(
            name="bad_params",
            features=((0.0,), (1.0,)),
            labels=(0.0, 1.0),
            initial_params=(0.1, 0.2),
        )
