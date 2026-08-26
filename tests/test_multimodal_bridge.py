# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multimodal forecast bridge tests
"""Production-surface tests for bounded active-sensing and co-design composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from scpn_quantum_control.forecasting.multimodal_bridge import (
    ForecastControllerInitialisation,
    forecast_to_controller_initialisation,
    plan_forecast_active_sensing,
)
from scpn_quantum_control.forecasting.multimodal_forecaster import (
    fit_multimodal_ridge_forecaster,
)
from scpn_quantum_control.forecasting.synthetic_multimodal import (
    SyntheticMultimodalConfig,
    SyntheticMultimodalDataset,
    generate_synthetic_multimodal_dataset,
)
from scpn_quantum_control.forecasting.uncertainty import (
    MultimodalIntervalForecast,
    apply_residual_interval,
    fit_residual_interval_calibrator,
)


def _evidence() -> tuple[SyntheticMultimodalDataset, MultimodalIntervalForecast]:
    dataset = generate_synthetic_multimodal_dataset(
        SyntheticMultimodalConfig(
            train_samples=32,
            calibration_samples=12,
            test_samples=12,
            history_steps=8,
            horizon_steps=3,
            missing_fraction=0.2,
            seed=3705,
        )
    )
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    calibrator = fit_residual_interval_calibrator(
        model,
        model.predict(dataset.calibration),
        dataset.calibration,
        alpha=0.2,
    )
    interval = apply_residual_interval(calibrator, model.predict(dataset.test))
    return dataset, interval


def test_interval_composes_into_real_no_submit_sensing_plan() -> None:
    dataset, interval = _evidence()
    bridge = plan_forecast_active_sensing(
        interval,
        dataset.test,
        sample_index=0,
        candidate_nodes=(0, 2),
        noise_variances=(0.04, 0.06),
        policy_id="ci_dry_run_only",
        shots_per_observable=128,
    )

    assert bridge.sample_id == dataset.test.sample_ids[0]
    assert bridge.plan.allowed is True
    assert bridge.plan.observer is not None
    assert bridge.plan.observer.hardware_execution is False
    assert bridge.hardware_execution is False
    assert len(bridge.candidates) == 2
    assert bridge.candidates[0].channel == "forecast_uncertainty_observer"
    assert bridge.to_dict()["hardware_execution"] is False
    assert "not adaptive hardware" in bridge.claim_boundary


def test_hardware_request_remains_refused_by_sensing_policy() -> None:
    dataset, interval = _evidence()
    bridge = plan_forecast_active_sensing(
        interval,
        dataset.test,
        sample_index=0,
        candidate_nodes=(1,),
        noise_variances=(0.05,),
        policy_id="default_no_submit",
        shots_per_observable=64,
        request_hardware=True,
    )

    assert bridge.plan.allowed is False
    assert bridge.plan.observer is None
    assert bridge.hardware_execution is False
    assert any("hardware" in blocker for blocker in bridge.plan.blockers)


def test_terminal_forecast_creates_bounded_unapplied_controller_proposal() -> None:
    _, interval = _evidence()
    result = forecast_to_controller_initialisation(
        interval,
        sample_index=0,
        current_parameters=(0.2, -0.1),
        target_order_parameter=1.0,
        gain_scale=4.0,
        max_abs_update=0.05,
    )

    assert 0.0 <= result.predicted_order_parameter <= 1.0
    assert result.target_order_parameter == 1.0
    assert result.proposal.parameters == (0.2, -0.1)
    assert all(abs(value) <= 0.05 for value in result.proposal.update)
    assert result.applied is False
    assert result.safety_decision is False
    assert result.to_dict()["proposal"] == result.proposal.to_dict()
    assert "unapplied" in result.claim_boundary


def test_sensing_bridge_refuses_custody_and_candidate_errors() -> None:
    dataset, interval = _evidence()
    with pytest.raises(ValueError, match="custody/shape"):
        plan_forecast_active_sensing(
            replace(interval, sample_ids=tuple(reversed(interval.sample_ids))),
            dataset.test,
            sample_index=0,
            candidate_nodes=(0,),
            noise_variances=(0.05,),
            policy_id="ci_dry_run_only",
            shots_per_observable=64,
        )
    with pytest.raises(ValueError, match="out of range"):
        plan_forecast_active_sensing(
            interval,
            dataset.test,
            sample_index=99,
            candidate_nodes=(0,),
            noise_variances=(0.05,),
            policy_id="ci_dry_run_only",
            shots_per_observable=64,
        )
    with pytest.raises(ValueError, match="non-empty and aligned"):
        plan_forecast_active_sensing(
            interval,
            dataset.test,
            sample_index=0,
            candidate_nodes=(0,),
            noise_variances=(),
            policy_id="ci_dry_run_only",
            shots_per_observable=64,
        )
    with pytest.raises(ValueError, match="must be unique"):
        plan_forecast_active_sensing(
            interval,
            dataset.test,
            sample_index=0,
            candidate_nodes=(0, 0),
            noise_variances=(0.1, 0.1),
            policy_id="ci_dry_run_only",
            shots_per_observable=64,
        )
    with pytest.raises(ValueError, match="out-of-range"):
        plan_forecast_active_sensing(
            interval,
            dataset.test,
            sample_index=0,
            candidate_nodes=(4,),
            noise_variances=(0.05,),
            policy_id="ci_dry_run_only",
            shots_per_observable=64,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        plan_forecast_active_sensing(
            interval,
            dataset.test,
            sample_index=0,
            candidate_nodes=(0,),
            noise_variances=(0.0,),
            policy_id="ci_dry_run_only",
            shots_per_observable=64,
        )


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (
            lambda interval: forecast_to_controller_initialisation(
                interval,
                sample_index=99,
                current_parameters=(0.1,),
                target_order_parameter=0.8,
                gain_scale=0.2,
                max_abs_update=0.1,
            ),
            "out of range",
        ),
        (
            lambda interval: forecast_to_controller_initialisation(
                interval,
                sample_index=0,
                current_parameters=(),
                target_order_parameter=0.8,
                gain_scale=0.2,
                max_abs_update=0.1,
            ),
            "must be non-empty",
        ),
        (
            lambda interval: forecast_to_controller_initialisation(
                interval,
                sample_index=0,
                current_parameters=(0.1,),
                target_order_parameter=1.1,
                gain_scale=0.2,
                max_abs_update=0.1,
            ),
            "finite in",
        ),
        (
            lambda interval: forecast_to_controller_initialisation(
                interval,
                sample_index=0,
                current_parameters=(0.1,),
                target_order_parameter=0.8,
                gain_scale=0.0,
                max_abs_update=0.1,
            ),
            "gain_scale",
        ),
        (
            lambda interval: forecast_to_controller_initialisation(
                interval,
                sample_index=0,
                current_parameters=(0.1,),
                target_order_parameter=0.8,
                gain_scale=0.2,
                max_abs_update=0.0,
            ),
            "max_abs_update",
        ),
    ],
)
def test_controller_bridge_refuses_invalid_controls(
    builder: Callable[[MultimodalIntervalForecast], ForecastControllerInitialisation],
    message: str,
) -> None:
    _, interval = _evidence()
    with pytest.raises(ValueError, match=message):
        builder(interval)


def test_bridge_records_refuse_promotion_and_empty_custody() -> None:
    dataset, interval = _evidence()
    sensing = plan_forecast_active_sensing(
        interval,
        dataset.test,
        sample_index=0,
        candidate_nodes=(0,),
        noise_variances=(0.05,),
        policy_id="ci_dry_run_only",
        shots_per_observable=64,
    )
    control = forecast_to_controller_initialisation(
        interval,
        sample_index=0,
        current_parameters=(0.1,),
        target_order_parameter=0.8,
        gain_scale=0.2,
        max_abs_update=0.1,
    )
    with pytest.raises(ValueError, match="cannot execute hardware"):
        replace(sensing, hardware_execution=True)
    with pytest.raises(ValueError, match="must be non-empty"):
        replace(sensing, candidates=())
    with pytest.raises(ValueError, match="must be non-empty"):
        replace(sensing, sample_id="")
    with pytest.raises(ValueError, match="digests must be non-empty"):
        replace(sensing, calibrator_digest="")
    with pytest.raises(ValueError, match="finite in"):
        replace(control, predicted_order_parameter=float("nan"))
    with pytest.raises(ValueError, match="custody must be non-empty"):
        replace(control, model_digest="")
    with pytest.raises(ValueError, match="must remain unapplied"):
        replace(control, applied=True)
