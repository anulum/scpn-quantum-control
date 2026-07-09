# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ground-State Optimizer Convergence Tests
"""Tests for ground-state optimizer convergence evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import (
    GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY,
    GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS,
    GroundStateOptimizerBoundaryRow,
    GroundStateOptimizerConvergenceSuiteResult,
    KnownGroundStateObjective,
    default_ground_state_optimizer_objectives,
    run_ground_state_optimizer_convergence_suite,
)
from scpn_quantum_control.phase import optimizer_convergence_suite as convergence_module
from scpn_quantum_control.phase.optimizer_convergence_suite import BoundaryStatus, OptimizerName


@dataclass(frozen=True)
class _GradientStub:
    gradient: NDArray[np.float64]
    evaluations: int


@pytest.fixture(scope="module")
def suite() -> GroundStateOptimizerConvergenceSuiteResult:
    """Run the default BL-15 optimizer convergence suite once."""
    return run_ground_state_optimizer_convergence_suite()


def test_default_ground_state_optimizer_suite_passes(
    suite: GroundStateOptimizerConvergenceSuiteResult,
) -> None:
    assert suite.passed
    assert suite.case_count == 2
    assert suite.record_count == 10
    assert suite.optimizer_names == ("natural_gradient", "adam", "lbfgs", "spsa", "cobyla")
    assert suite.evidence_class == GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS
    assert len(suite.boundary_rows) == 1

    for record in suite.records:
        assert record.passed
        assert record.best_value <= record.initial_value
        assert record.evaluations > 0
        assert record.iterations > 0
        assert record.evidence_class == GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS
        assert record.certificate.energy_within_tolerance
        assert record.certificate.parameter_within_tolerance
        assert record.certificate.reason == "target_ground_state_reached"


def test_default_ground_state_optimizer_suite_serializes_boundary(
    suite: GroundStateOptimizerConvergenceSuiteResult,
) -> None:
    payload = suite.to_dict()

    assert payload["passed"] is True
    assert payload["case_count"] == 2
    assert payload["record_count"] == 10
    assert payload["optimizer_names"] == [
        "natural_gradient",
        "adam",
        "lbfgs",
        "spsa",
        "cobyla",
    ]
    boundary_rows = cast(list[dict[str, object]], payload["boundary_rows"])
    assert boundary_rows[0]["failure_class"] == "unsupported_qjit_metric_fusion"
    assert boundary_rows[0]["status"] == "hard_gap"


def test_records_for_case_and_best_record_are_public_contracts(
    suite: GroundStateOptimizerConvergenceSuiteResult,
) -> None:
    case_id = suite.objectives[0].case_id
    records = suite.records_for_case(case_id)
    best = suite.best_record_for_case(case_id)

    assert len(records) == 5
    assert best.case_id == case_id
    assert best.best_value == min(record.best_value for record in records)
    with pytest.raises(KeyError, match="unknown ground-state optimizer case"):
        suite.best_record_for_case("missing-case")


def test_custom_optimizer_subset_deduplicates_and_omits_boundary() -> None:
    suite = run_ground_state_optimizer_convergence_suite(
        objectives=default_ground_state_optimizer_objectives()[:1],
        optimizers=("adam", "adam", "lbfgs"),
        include_qng_qjit_boundary=False,
    )

    assert suite.passed
    assert suite.case_count == 1
    assert suite.record_count == 2
    assert suite.optimizer_names == ("adam", "lbfgs")
    assert suite.boundary_rows == ()


def test_ground_state_objective_validates_and_reports_distances() -> None:
    objective = KnownGroundStateObjective(
        case_id="unit-ground",
        ground_state_label="unit ground",
        initial_params=np.array([0.1], dtype=np.float64),
        target_params=np.array([np.pi], dtype=np.float64),
        weights=np.array([2.0], dtype=np.float64),
        exact_ground_energy=-2.0,
        target_energy_tolerance=1.0e-6,
        target_parameter_tolerance=1.0e-4,
    )

    assert objective.width == 1
    assert objective.value(objective.target_params) == pytest.approx(-2.0)
    assert objective.metric_tensor(objective.initial_params).tolist() == [[2.0]]
    assert objective.wrapped_parameter_distance(np.array([-np.pi], dtype=float)) == pytest.approx(
        0.0
    )
    assert objective.to_dict()["case_id"] == "unit-ground"


def _objective(
    *,
    case_id: str = "valid",
    ground_state_label: str = "valid",
    initial_params: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None,
    target_params: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None,
    weights: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None,
    exact_ground_energy: float = -1.0,
    target_energy_tolerance: float = 1.0e-6,
    target_parameter_tolerance: float = 1.0e-4,
) -> KnownGroundStateObjective:
    """Build a typed objective fixture for validation tests."""
    return KnownGroundStateObjective(
        case_id=case_id,
        ground_state_label=ground_state_label,
        initial_params=(
            np.array([0.1], dtype=np.float64) if initial_params is None else initial_params
        ),
        target_params=(
            np.array([0.0], dtype=np.float64) if target_params is None else target_params
        ),
        weights=np.array([1.0], dtype=np.float64) if weights is None else weights,
        exact_ground_energy=exact_ground_energy,
        target_energy_tolerance=target_energy_tolerance,
        target_parameter_tolerance=target_parameter_tolerance,
    )


def test_ground_state_objective_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="case_id must be non-empty"):
        _objective(case_id="")
    with pytest.raises(ValueError, match="ground_state_label must be non-empty"):
        _objective(ground_state_label="")
    with pytest.raises(ValueError, match="must share shape"):
        _objective(target_params=np.array([0.0, 1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="strictly positive"):
        _objective(weights=np.array([0.0], dtype=np.float64))
    with pytest.raises(ValueError, match="exact_ground_energy must be finite"):
        _objective(exact_ground_energy=float("nan"))
    with pytest.raises(ValueError, match="target_energy_tolerance"):
        _objective(target_energy_tolerance=-1.0)
    with pytest.raises(ValueError, match="target_parameter_tolerance"):
        _objective(target_parameter_tolerance=-1.0)
    with pytest.raises(ValueError, match="initial_params must be a non-empty"):
        _objective(initial_params=np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="initial_params must contain only finite values"):
        _objective(initial_params=np.array([float("nan")], dtype=np.float64))


def test_ground_state_objective_rejects_wrong_evaluation_shape() -> None:
    objective = default_ground_state_optimizer_objectives()[0]
    wrong_shape = np.array([0.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="params must match"):
        objective.value(wrong_shape)
    with pytest.raises(ValueError, match="params must match"):
        objective.metric_tensor(wrong_shape)
    with pytest.raises(ValueError, match="params must match"):
        objective.wrapped_parameter_distance(wrong_shape)


def test_ground_state_optimizer_suite_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="at least one ground-state objective is required"):
        run_ground_state_optimizer_convergence_suite(objectives=())
    with pytest.raises(ValueError, match="at least one optimizer is required"):
        run_ground_state_optimizer_convergence_suite(optimizers=())
    with pytest.raises(ValueError, match="unknown optimizer"):
        run_ground_state_optimizer_convergence_suite(
            optimizers=cast(Sequence[OptimizerName], ("unknown",))
        )
    with pytest.raises(ValueError, match="learning_rate"):
        run_ground_state_optimizer_convergence_suite(learning_rate=0.0)
    with pytest.raises(ValueError, match="max_steps"):
        run_ground_state_optimizer_convergence_suite(max_steps=0)
    with pytest.raises(ValueError, match="spsa_perturbation"):
        run_ground_state_optimizer_convergence_suite(spsa_perturbation=0.0)
    with pytest.raises(ValueError, match="spsa_seed"):
        run_ground_state_optimizer_convergence_suite(spsa_seed=-1)


def test_ground_state_optimizer_suite_rejects_bad_provider_gradient_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_gradient(
        objective: object,
        params: NDArray[np.float64],
        *,
        parameters: object = None,
        rule: object = None,
    ) -> _GradientStub:
        return _GradientStub(np.array([0.0, 0.0], dtype=np.float64), 1)

    monkeypatch.setattr(convergence_module, "value_and_parameter_shift_grad", bad_gradient)
    with pytest.raises(ValueError, match="gradient shape must match"):
        run_ground_state_optimizer_convergence_suite(
            objectives=default_ground_state_optimizer_objectives()[:1],
            optimizers=("adam",),
            max_steps=1,
        )


def test_ground_state_optimizer_suite_rejects_non_finite_provider_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_gradient(
        objective: object,
        params: NDArray[np.float64],
        *,
        parameters: object = None,
        rule: object = None,
    ) -> _GradientStub:
        return _GradientStub(np.array([float("nan")], dtype=np.float64), 1)

    monkeypatch.setattr(convergence_module, "value_and_parameter_shift_grad", bad_gradient)
    with pytest.raises(ValueError, match="gradient must contain only finite values"):
        run_ground_state_optimizer_convergence_suite(
            objectives=default_ground_state_optimizer_objectives()[:1],
            optimizers=("adam",),
            max_steps=1,
        )


def _boundary_row(
    *,
    case_id: str = "boundary",
    optimizer: str = "qng_qjit_class_boundary",
    status: str = "hard_gap",
    failure_class: str = "unsupported_qjit_metric_fusion",
    setup_instructions: str = "requires executable qjit metric-fusion contract",
    evidence_class: str = GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS,
    claim_boundary: str = GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY,
) -> GroundStateOptimizerBoundaryRow:
    """Build a typed fail-closed optimizer boundary row."""
    return GroundStateOptimizerBoundaryRow(
        case_id=case_id,
        optimizer=optimizer,
        status=cast("BoundaryStatus", status),
        failure_class=failure_class,
        setup_instructions=setup_instructions,
        evidence_class=evidence_class,
        claim_boundary=claim_boundary,
    )


def test_ground_state_optimizer_boundary_row_validates_metadata() -> None:
    assert _boundary_row().to_dict()["status"] == "hard_gap"
    with pytest.raises(ValueError, match="case_id must be non-empty"):
        _boundary_row(case_id="")
    with pytest.raises(ValueError, match="optimizer must be non-empty"):
        _boundary_row(optimizer="")
    with pytest.raises(ValueError, match="boundary status must be hard_gap"):
        _boundary_row(status="soft_gap")
    with pytest.raises(ValueError, match="failure_class must be non-empty"):
        _boundary_row(failure_class="")
    with pytest.raises(ValueError, match="setup_instructions must be non-empty"):
        _boundary_row(setup_instructions="")
    with pytest.raises(ValueError, match="boundary evidence_class"):
        _boundary_row(evidence_class="isolated_affinity")
    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        _boundary_row(claim_boundary="")
