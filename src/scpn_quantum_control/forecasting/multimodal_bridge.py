# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multimodal forecast bridge ports
"""Bounded multimodal-forecasting composition into active-sensing sensing and co-design proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..active_sensing_product import (
    ActiveSensingPlan,
    InformationGainCandidate,
    plan_active_sensing,
)
from ..codesign.contracts import ControllerProposal
from .multimodal_schema import MultimodalObservationBatch
from .uncertainty import MultimodalIntervalForecast


@dataclass(frozen=True, slots=True)
class ForecastActiveSensingBridge:
    """No-submit sensing plan composed from one synthetic interval forecast."""

    sample_id: str
    candidates: tuple[InformationGainCandidate, ...]
    plan: ActiveSensingPlan
    model_digest: str
    calibrator_digest: str
    hardware_execution: bool = False
    claim_boundary: str = (
        "interval-width proxy candidates composed into the existing no-submit sensing plan; "
        "not adaptive hardware sensing, optimal sensor placement, or domain deployment"
    )

    def __post_init__(self) -> None:
        """Refuse empty candidate custody or hardware promotion."""
        if not self.sample_id.strip() or not self.candidates:
            raise ValueError("sample_id and candidates must be non-empty")
        if self.hardware_execution:
            raise ValueError("forecast active-sensing bridge cannot execute hardware")
        if not self.model_digest.strip() or not self.calibrator_digest.strip():
            raise ValueError("model and calibrator digests must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready bridge record."""
        return {
            "sample_id": self.sample_id,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "plan": self.plan.to_dict(),
            "model_digest": self.model_digest,
            "calibrator_digest": self.calibrator_digest,
            "hardware_execution": self.hardware_execution,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ForecastControllerInitialisation:
    """Unapplied controller proposal initialised from a terminal forecast."""

    sample_id: str
    predicted_order_parameter: float
    target_order_parameter: float
    proposal: ControllerProposal
    model_digest: str
    calibrator_digest: str
    applied: bool = False
    safety_decision: bool = False
    claim_boundary: str = (
        "synthetic forecast-derived controller initialisation proposal only; unapplied and "
        "not a safety, stability, closed-loop, realtime, or operational control decision"
    )

    def __post_init__(self) -> None:
        """Validate unit-interval order parameters and fail-closed state."""
        values = (self.predicted_order_parameter, self.target_order_parameter)
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("predicted and target order parameters must be finite in [0, 1]")
        if (
            not self.sample_id.strip()
            or not self.model_digest.strip()
            or not self.calibrator_digest.strip()
        ):
            raise ValueError("sample and digest custody must be non-empty")
        if self.applied or self.safety_decision:
            raise ValueError("forecast initialisation must remain unapplied and non-safety")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready initialisation record."""
        return {
            "sample_id": self.sample_id,
            "predicted_order_parameter": self.predicted_order_parameter,
            "target_order_parameter": self.target_order_parameter,
            "proposal": self.proposal.to_dict(),
            "model_digest": self.model_digest,
            "calibrator_digest": self.calibrator_digest,
            "applied": self.applied,
            "safety_decision": self.safety_decision,
            "claim_boundary": self.claim_boundary,
        }


def _validate_interval_batch(
    interval: MultimodalIntervalForecast,
    batch: MultimodalObservationBatch,
) -> None:
    if interval.sample_ids != batch.sample_ids or interval.point.shape != batch.targets.shape:
        raise ValueError("interval and batch custody/shape must match exactly")


def plan_forecast_active_sensing(
    interval: MultimodalIntervalForecast,
    batch: MultimodalObservationBatch,
    *,
    sample_index: int,
    candidate_nodes: tuple[int, ...],
    noise_variances: tuple[float, ...],
    policy_id: str,
    shots_per_observable: int,
    request_hardware: bool = False,
) -> ForecastActiveSensingBridge:
    """Compose interval-width proxies into the existing no-submit sensing path."""
    _validate_interval_batch(interval, batch)
    if not 0 <= sample_index < batch.n_samples:
        raise ValueError("sample_index is out of range")
    if not candidate_nodes or len(candidate_nodes) != len(noise_variances):
        raise ValueError("candidate_nodes and noise_variances must be non-empty and aligned")
    if len(set(candidate_nodes)) != len(candidate_nodes):
        raise ValueError("candidate_nodes must be unique")
    if any(node < 0 or node >= batch.n_nodes for node in candidate_nodes):
        raise ValueError("candidate_nodes contain an out-of-range node")
    if any(not np.isfinite(noise) or noise <= 0.0 for noise in noise_variances):
        raise ValueError("noise_variances must be finite and positive")
    half_width = 0.5 * (interval.upper[sample_index] - interval.lower[sample_index])
    candidates = tuple(
        InformationGainCandidate(
            observable_id=f"forecast_phase_node_{node}",
            prior_variance=max(float(np.mean(np.square(half_width[:, node]))), 1.0e-12),
            sensitivity=1.0 + float(np.sum(np.abs(batch.graphs[sample_index, node]))),
            noise_variance=noise,
            channel="forecast_uncertainty_observer",
        )
        for node, noise in zip(candidate_nodes, noise_variances, strict=True)
    )
    plan = plan_active_sensing(
        candidates,
        batch.graphs[sample_index],
        batch.frequencies[sample_index],
        policy_id=policy_id,
        shots_per_observable=shots_per_observable,
        request_hardware=request_hardware,
    )
    return ForecastActiveSensingBridge(
        sample_id=batch.sample_ids[sample_index],
        candidates=candidates,
        plan=plan,
        model_digest=interval.model_digest,
        calibrator_digest=interval.calibrator_digest,
    )


def forecast_to_controller_initialisation(
    interval: MultimodalIntervalForecast,
    *,
    sample_index: int,
    current_parameters: tuple[float, ...],
    target_order_parameter: float,
    gain_scale: float,
    max_abs_update: float,
) -> ForecastControllerInitialisation:
    """Map a terminal phase forecast to an unapplied controller proposal."""
    if not 0 <= sample_index < len(interval.sample_ids):
        raise ValueError("sample_index is out of range")
    if not current_parameters:
        raise ValueError("current_parameters must be non-empty")
    if not np.isfinite(target_order_parameter) or not 0.0 <= target_order_parameter <= 1.0:
        raise ValueError("target_order_parameter must be finite in [0, 1]")
    if not np.isfinite(gain_scale) or gain_scale <= 0.0:
        raise ValueError("gain_scale must be finite and positive")
    if not np.isfinite(max_abs_update) or max_abs_update <= 0.0:
        raise ValueError("max_abs_update must be finite and positive")
    terminal_phases = interval.point[sample_index, -1]
    predicted_order = float(np.abs(np.mean(np.exp(1j * terminal_phases))))
    raw_update = gain_scale * (target_order_parameter - predicted_order)
    bounded_update = float(np.clip(raw_update, -max_abs_update, max_abs_update))
    proposal = ControllerProposal(
        parameters=current_parameters,
        update=(bounded_update,) * len(current_parameters),
        gain_scale=gain_scale,
    )
    return ForecastControllerInitialisation(
        sample_id=interval.sample_ids[sample_index],
        predicted_order_parameter=predicted_order,
        target_order_parameter=target_order_parameter,
        proposal=proposal,
        model_digest=interval.model_digest,
        calibrator_digest=interval.calibrator_digest,
    )


__all__ = [
    "ForecastActiveSensingBridge",
    "ForecastControllerInitialisation",
    "forecast_to_controller_initialisation",
    "plan_forecast_active_sensing",
]
