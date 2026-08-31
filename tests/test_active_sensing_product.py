# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — active-sensing active sensing product tests
"""Production-surface tests for active-sensing active sensing composition."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.active_sensing_product import (
    ACTIVE_SENSING_PRODUCT_SCHEMA,
    InformationGainCandidate,
    demo_information_gain_candidates,
    plan_active_sensing,
    score_expected_information_gain,
    sensing_surface_inventory,
)


def _problem() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k_matrix = np.array(
        [[0.0, 0.4, 0.2], [0.4, 0.0, 0.3], [0.2, 0.3, 0.0]],
        dtype=np.float64,
    )
    omega = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    return k_matrix, omega


def test_inventory_preserves_ownership_and_hardware_boundary() -> None:
    """The inventory preserves surface ownership and no-hardware posture."""
    rows = sensing_surface_inventory()

    assert {row.surface_id for row in rows} == {
        "quantum_fisher_sync_readiness",
        "analytic_candidate_design",
        "shot_budget",
        "nv_20t",
        "codesign_observer",
    }
    assert all(row.hardware_execution is False for row in rows)
    assert next(row for row in rows if row.surface_id == "nv_20t").posture == "hardware_blocked"


def test_information_gain_uses_gaussian_posterior_update() -> None:
    """Information gain follows the scalar conjugate Gaussian update."""
    candidate = InformationGainCandidate("phase", 2.0, 0.5, 0.25)
    score = score_expected_information_gain(candidate, shots=4)

    assert score.signal_to_noise == pytest.approx(8.0)
    assert score.expected_information_gain_nats == pytest.approx(0.5 * np.log(9.0))
    assert score.posterior_variance == pytest.approx(2.0 / 9.0)
    assert score.to_dict()["observable_id"] == "phase"


@pytest.mark.parametrize(
    "candidate",
    [
        InformationGainCandidate("ok", 1.0, 0.0, 1.0),
    ],
)
def test_information_gain_rejects_nonpositive_shots(candidate: InformationGainCandidate) -> None:
    """Information-gain scoring requires a positive shot count."""
    with pytest.raises(ValueError, match="shots must be positive"):
        score_expected_information_gain(candidate, shots=0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {
                "observable_id": "",
                "prior_variance": 1.0,
                "sensitivity": 1.0,
                "noise_variance": 1.0,
            },
            "non-empty",
        ),
        (
            {
                "observable_id": "x",
                "prior_variance": np.inf,
                "sensitivity": 1.0,
                "noise_variance": 1.0,
            },
            "finite",
        ),
        (
            {
                "observable_id": "x",
                "prior_variance": 0.0,
                "sensitivity": 1.0,
                "noise_variance": 1.0,
            },
            "positive",
        ),
        (
            {
                "observable_id": "x",
                "prior_variance": 1.0,
                "sensitivity": 1.0,
                "noise_variance": 0.0,
            },
            "positive",
        ),
        (
            {
                "observable_id": "x",
                "prior_variance": 1.0,
                "sensitivity": 1.0,
                "noise_variance": 1.0,
                "channel": "",
            },
            "non-empty",
        ),
    ],
)
def test_candidate_validation(kwargs: dict[str, Any], message: str) -> None:
    """Malformed observation candidates fail closed."""
    with pytest.raises(ValueError, match=message):
        InformationGainCandidate(**kwargs)


def test_plan_runs_real_budget_and_analytic_design_surfaces() -> None:
    """Allowed plans execute budget, ranking, design, and observer surfaces."""
    k_matrix, omega = _problem()
    plan = plan_active_sensing(
        demo_information_gain_candidates(),
        k_matrix,
        omega,
        policy_id="ci_dry_run_only",
        shots_per_observable=128,
    )

    assert plan.allowed is True
    assert plan.outcome == "allowed_plan"
    assert plan.budget.estimated_total_shots == 768
    assert plan.selected is plan.scores[0]
    assert plan.selected is not None
    assert plan.observer is not None
    assert plan.observer.selected_observable_id == plan.selected.observable_id
    assert plan.observer.hardware_execution is False
    assert {row.family for row in plan.analytic_design_evidence} == {"ansatz", "pulse"}
    assert {row.analytic_design_protocol_id for row in plan.analytic_design_evidence} == {
        "ml_augmented_pulse_ansatz_design_2026-05-06"
    }
    payload = plan.to_dict()
    assert payload["schema"] == ACTIVE_SENSING_PRODUCT_SCHEMA
    assert payload["hardware_execution"] is False
    assert payload["selected"] is not None
    assert payload["observer"] is not None
    assert "analytic_design_evidence" in payload
    assert "s3_evidence" not in payload
    observer_payload = payload["observer"]
    assert isinstance(observer_payload, dict)
    assert observer_payload["analytic_design_protocol_id"] == (
        "ml_augmented_pulse_ansatz_design_2026-05-06"
    )
    assert "s3_protocol_id" not in observer_payload


def test_budget_refusal_prevents_information_and_analytic_design_evaluation() -> None:
    """Budget refusal precedes information and analytic-design evaluation."""
    k_matrix, omega = _problem()
    plan = plan_active_sensing(
        demo_information_gain_candidates(),
        k_matrix,
        omega,
        policy_id="ci_dry_run_only",
        shots_per_observable=4096,
    )

    assert plan.allowed is False
    assert plan.scores == ()
    assert plan.analytic_design_evidence == ()
    assert plan.selected is None
    assert plan.observer is None
    payload = plan.to_dict()
    assert payload["selected"] is None
    assert payload["observer"] is None


def test_hardware_adaptive_path_is_fail_closed() -> None:
    """Adaptive hardware requests remain explicitly fail closed."""
    k_matrix, omega = _problem()
    plan = plan_active_sensing(
        demo_information_gain_candidates(),
        k_matrix,
        omega,
        policy_id="default_no_submit",
        shots_per_observable=64,
        request_hardware=True,
    )

    assert plan.allowed is False
    assert any("adaptive hardware sensing" in blocker for blocker in plan.blockers)
    assert plan.analytic_design_evidence == ()


def test_plan_rejects_empty_or_duplicate_candidates() -> None:
    """Plans require a non-empty set of uniquely identified candidates."""
    k_matrix, omega = _problem()
    with pytest.raises(ValueError, match="candidates must be non-empty"):
        plan_active_sensing(
            (),
            k_matrix,
            omega,
            policy_id="ci_dry_run_only",
            shots_per_observable=64,
        )
    duplicate = InformationGainCandidate("same", 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="must be unique"):
        plan_active_sensing(
            (duplicate, duplicate),
            k_matrix,
            omega,
            policy_id="ci_dry_run_only",
            shots_per_observable=64,
        )


def test_allowed_plan_propagates_analytic_design_input_validation() -> None:
    """Allowed planning propagates analytic-design input validation."""
    _, omega = _problem()
    with pytest.raises(ValueError, match="square"):
        plan_active_sensing(
            demo_information_gain_candidates(),
            np.ones((2, 3), dtype=np.float64),
            omega,
            policy_id="ci_dry_run_only",
            shots_per_observable=64,
        )


def test_plan_rejects_stale_serialized_schema() -> None:
    """Reject the superseded payload contract without a compatibility alias."""
    k_matrix, omega = _problem()
    plan = plan_active_sensing(
        demo_information_gain_candidates(),
        k_matrix,
        omega,
        policy_id="ci_dry_run_only",
        shots_per_observable=64,
    )

    with pytest.raises(ValueError, match="unexpected active-sensing product schema"):
        replace(plan, schema="active_sensing_product.v1")


def test_plan_rejects_claim_boundary_drift() -> None:
    """Reject serialized claim language that differs from the live contract."""
    k_matrix, omega = _problem()
    plan = plan_active_sensing(
        demo_information_gain_candidates(),
        k_matrix,
        omega,
        policy_id="ci_dry_run_only",
        shots_per_observable=64,
    )

    with pytest.raises(ValueError, match="claim boundary drift"):
        replace(plan, claim_boundary="legacy planning label")
