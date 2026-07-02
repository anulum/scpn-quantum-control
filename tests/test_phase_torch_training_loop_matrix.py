# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- PyTorch training-loop matrix tests
"""Training-loop matrix tests for bounded PyTorch phase-QNN modules."""

from __future__ import annotations

from typing import Any, cast

import pytest

from scpn_quantum_control.phase import (
    TORCH_TRAINING_LOOP_MATRIX_SCHEMA,
    PhaseTorchTrainingLoopMatrixResult,
    PhaseTorchTrainingLoopScenario,
    default_torch_training_loop_scenarios,
    run_torch_training_loop_matrix,
)

pytest.importorskip("torch")


def test_torch_training_loop_matrix_records_multi_scenario_evidence() -> None:
    """The matrix should record loss descent and gradient parity per scenario."""
    scenarios = (
        PhaseTorchTrainingLoopScenario(
            name="one_parameter_fullgraph",
            features=((0.0,), (3.141592653589793,)),
            labels=(0.0, 1.0),
            initial_params=(0.45,),
            learning_rate=0.2,
            steps=3,
            fullgraph=True,
            dynamic=False,
        ),
        PhaseTorchTrainingLoopScenario(
            name="two_parameter_non_fullgraph_dynamic_request",
            features=((0.0, 1.0), (1.5707963267948966, -0.4), (3.141592653589793, 0.25)),
            labels=(0.0, 1.0, 1.0),
            initial_params=(0.25, -0.35),
            learning_rate=0.08,
            steps=2,
            fullgraph=False,
            dynamic=True,
        ),
    )

    result = run_torch_training_loop_matrix(
        scenarios=scenarios,
        tolerance=1.0e-8,
    )

    assert isinstance(result, PhaseTorchTrainingLoopMatrixResult)
    assert result.passed
    assert result.matrix_schema == TORCH_TRAINING_LOOP_MATRIX_SCHEMA
    assert result.scenario_count == 2
    assert result.passed_count == 2
    assert result.route_status("multi_scenario_training_loop") == "passed"
    assert result.route_status("training_loop_gradient_parity") == "passed"
    assert result.route_status("training_loop_loss_descent") == "passed"
    assert result.route_status("compile_mode_matrix") == "passed"
    assert result.route_status("cuda_training_loop") == "blocked"
    assert result.route_status("provider_hardware_training_loop") == "blocked"
    assert result.route_status("isolated_benchmark_training_loop") == "blocked"
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False
    assert result.open_gaps == (
        "cuda_training_loop",
        "provider_hardware_training_loop",
        "isolated_benchmark_training_loop",
        "arbitrary_architecture_training_loop",
    )

    assert {record.scenario_name for record in result.records} == {
        "one_parameter_fullgraph",
        "two_parameter_non_fullgraph_dynamic_request",
    }
    assert {record.fullgraph for record in result.records} == {False, True}
    assert {record.dynamic for record in result.records} == {False, True}
    assert all(record.loss_drop >= -result.tolerance for record in result.records)
    assert all(record.parameter_update_norm > 0.0 for record in result.records)

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert payload["passed"] is True
    assert routes["compile_mode_matrix"]["status"] == "passed"
    assert routes["cuda_training_loop"]["status"] == "blocked"
    assert "no CUDA" in str(payload["claim_boundary"])


def test_default_torch_training_loop_scenarios_are_distinct() -> None:
    """Default scenarios should cover more than one bounded training shape."""
    scenarios = default_torch_training_loop_scenarios()

    assert len(scenarios) >= 2
    assert len({scenario.name for scenario in scenarios}) == len(scenarios)
    assert {scenario.fullgraph for scenario in scenarios} == {False, True}
    assert any(scenario.dynamic for scenario in scenarios)


def test_torch_training_loop_matrix_rejects_empty_scenarios() -> None:
    """The matrix should reject an empty scenario set."""
    with pytest.raises(ValueError, match="at least one training-loop scenario"):
        run_torch_training_loop_matrix(scenarios=())


def test_torch_training_loop_matrix_rejects_duplicate_scenario_names() -> None:
    """The matrix should reject duplicate scenario names."""
    scenario = default_torch_training_loop_scenarios()[0]

    with pytest.raises(ValueError, match="duplicate training-loop scenario"):
        run_torch_training_loop_matrix(scenarios=(scenario, scenario))


def test_torch_training_loop_matrix_rejects_unknown_route() -> None:
    """Route lookups should fail closed for unknown matrix rows."""
    result = run_torch_training_loop_matrix(
        scenarios=(default_torch_training_loop_scenarios()[0],),
        tolerance=1.0e-8,
    )

    with pytest.raises(KeyError, match="unknown PyTorch training-loop matrix route"):
        result.route_status("missing")
