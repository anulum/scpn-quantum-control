# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 NV quantum sensing fixtures
"""Protocol-design fixtures for Paper 0 NV-center quantum sensing validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .spec_loader import load_nv_quantum_sensing_validation_spec

CLAIM_BOUNDARY = "source-bounded NV-center quantum sensing protocol design; not empirical evidence"
HARDWARE_STATUS = "protocol_design_no_lab_execution"
SOURCE_LEDGER_SPAN = ("P0R06677", "P0R06729")


@dataclass(frozen=True, slots=True)
class NVQuantumSensingConfig:
    """Finite protocol settings for the NV quantum sensing fixture."""

    ramsey_sequences_per_condition: int = 1000
    trial_duration_minutes: int = 60
    trials_per_culture: int = 5
    days_per_trial: int = 6
    culture_count: int = 5
    beta_2_p_value: float = 0.01
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_int_at_least(
            "ramsey_sequences_per_condition", self.ramsey_sequences_per_condition, 1
        )
        _require_int_at_least("trial_duration_minutes", self.trial_duration_minutes, 1)
        _require_int_at_least("trials_per_culture", self.trials_per_culture, 1)
        _require_int_at_least("days_per_trial", self.days_per_trial, 1)
        _require_int_at_least("culture_count", self.culture_count, 1)
        _require_unit_probability("beta_2_p_value", self.beta_2_p_value)


@dataclass(frozen=True, slots=True)
class DecoherenceRegressionFit:
    """Least-squares fit for Gamma = beta_0 + beta_1 B_classical + beta_2 FIM."""

    beta_0: float
    beta_1: float
    beta_2: float
    residual_norm: float


@dataclass(frozen=True, slots=True)
class NVQuantumSensingFixtureResult:
    """Combined NV quantum sensing protocol fixture result."""

    spec_keys: tuple[str, str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    gamma_baseline: float
    gamma_spontaneous: float
    gamma_replay: float
    delta_gamma: float
    effect_size_ratio: float
    regression_beta_0: float
    regression_beta_1: float
    regression_beta_2: float
    regression_residual_norm: float
    beta_2_p_value: float
    falsification_rejected: bool
    total_protocol_days: int
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float | int]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def decoherence_excess(*, gamma_spontaneous: float, gamma_replay: float) -> float:
    """Return Delta Gamma = Gamma_spontaneous - Gamma_replay."""
    _require_finite("gamma_spontaneous", gamma_spontaneous)
    _require_finite("gamma_replay", gamma_replay)
    return gamma_spontaneous - gamma_replay


def expected_effect_size_ratio(*, delta_gamma: float, gamma_baseline: float) -> float:
    """Return Delta Gamma / Gamma_baseline."""
    _require_finite("delta_gamma", delta_gamma)
    if not isfinite(gamma_baseline) or gamma_baseline <= 0.0:
        raise ValueError("gamma_baseline must be finite and positive")
    return delta_gamma / gamma_baseline


def fit_decoherence_regression(
    *,
    gamma: NDArray[np.float64],
    b_classical: NDArray[np.float64],
    fim_proxy: NDArray[np.float64],
) -> DecoherenceRegressionFit:
    """Fit Gamma = beta_0 + beta_1 B_classical + beta_2 FIM_proxy + epsilon."""
    gamma_values = _finite_vector("gamma", gamma)
    b_values = _finite_vector("b_classical", b_classical)
    fim_values = _finite_vector("fim_proxy", fim_proxy)
    if gamma_values.shape != b_values.shape or gamma_values.shape != fim_values.shape:
        raise ValueError("all regression inputs must have the same shape")
    if gamma_values.size < 3:
        raise ValueError("at least three observations are required")
    design = np.column_stack([np.ones(gamma_values.size), b_values, fim_values])
    coefficients, residuals, _rank, _singular = np.linalg.lstsq(
        design,
        gamma_values,
        rcond=None,
    )
    residual_norm = (
        float(np.sqrt(float(residuals[0])))
        if residuals.size
        else float(np.linalg.norm(gamma_values - design @ coefficients))
    )
    return DecoherenceRegressionFit(
        beta_0=float(coefficients[0]),
        beta_1=float(coefficients[1]),
        beta_2=float(coefficients[2]),
        residual_norm=residual_norm,
    )


def falsification_decision(
    *,
    delta_gamma: float,
    beta_2: float,
    beta_2_p_value: float,
) -> bool:
    """Return True when the source falsification rule rejects the hypothesis."""
    _require_finite("delta_gamma", delta_gamma)
    _require_finite("beta_2", beta_2)
    _require_unit_probability("beta_2_p_value", beta_2_p_value)
    return delta_gamma <= 0.0 or beta_2 <= 0.0 or beta_2_p_value > 0.05


def validate_nv_quantum_sensing_fixture(
    config: NVQuantumSensingConfig | None = None,
) -> NVQuantumSensingFixtureResult:
    """Run the combined NV quantum sensing protocol fixture."""
    cfg = config or NVQuantumSensingConfig()
    keys = (
        "nv_quantum_sensing.block_framing",
        "nv_quantum_sensing.apparatus",
        "nv_quantum_sensing.protocol_steps",
        "nv_quantum_sensing.isomorphic_replay_control",
        "nv_quantum_sensing.analysis_and_falsification",
        "nv_quantum_sensing.controls_effect_size_timeline",
    )
    specs = tuple(
        load_nv_quantum_sensing_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    gamma_baseline = 1.0
    gamma_replay = 1.0
    gamma_spontaneous = 1.12
    delta = decoherence_excess(
        gamma_spontaneous=gamma_spontaneous,
        gamma_replay=gamma_replay,
    )
    ratio = expected_effect_size_ratio(delta_gamma=delta, gamma_baseline=gamma_baseline)
    b_classical = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    fim_proxy = np.array([0.0, 0.7, 0.2, 1.0, 0.5], dtype=np.float64)
    gamma = 0.9 + 0.2 * b_classical + 0.45 * fim_proxy
    fit = fit_decoherence_regression(
        gamma=gamma,
        b_classical=b_classical,
        fim_proxy=fim_proxy,
    )
    rejected = falsification_decision(
        delta_gamma=delta,
        beta_2=fit.beta_2,
        beta_2_p_value=cfg.beta_2_p_value,
    )
    controls = {
        "shape_mismatch_rejection_label": _shape_mismatch_rejection_label(),
        "invalid_baseline_rejection_label": _invalid_baseline_rejection_label(),
        "unsupported_empirical_protocol_claim_rejection_label": 1.0,
    }
    return NVQuantumSensingFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        gamma_baseline=gamma_baseline,
        gamma_spontaneous=gamma_spontaneous,
        gamma_replay=gamma_replay,
        delta_gamma=delta,
        effect_size_ratio=ratio,
        regression_beta_0=fit.beta_0,
        regression_beta_1=fit.beta_1,
        regression_beta_2=fit.beta_2,
        regression_residual_norm=fit.residual_norm,
        beta_2_p_value=cfg.beta_2_p_value,
        falsification_rejected=rejected,
        total_protocol_days=cfg.days_per_trial * cfg.culture_count,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "ramsey_sequences_per_condition": cfg.ramsey_sequences_per_condition,
                "trial_duration_minutes": cfg.trial_duration_minutes,
                "trials_per_culture": cfg.trials_per_culture,
                "days_per_trial": cfg.days_per_trial,
                "culture_count": cfg.culture_count,
                "effect_size_lower": 0.05,
                "effect_size_upper": 0.15,
            }
        ),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "design_only_no_lab_execution",
            }
        ),
    )


def _shape_mismatch_rejection_label() -> float:
    try:
        fit_decoherence_regression(
            gamma=np.array([1.0, 1.1], dtype=np.float64),
            b_classical=np.array([0.1], dtype=np.float64),
            fim_proxy=np.array([0.2, 0.3], dtype=np.float64),
        )
    except ValueError as exc:
        return float("same shape" in str(exc))
    return 0.0


def _invalid_baseline_rejection_label() -> float:
    try:
        expected_effect_size_ratio(delta_gamma=0.1, gamma_baseline=0.0)
    except ValueError as exc:
        return float("gamma_baseline must be finite and positive" in str(exc))
    return 0.0


def _finite_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(NDArray[np.float64], array)


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_unit_probability(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_int_at_least(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


__all__ = [
    "CLAIM_BOUNDARY",
    "DecoherenceRegressionFit",
    "NVQuantumSensingConfig",
    "NVQuantumSensingFixtureResult",
    "decoherence_excess",
    "expected_effect_size_ratio",
    "falsification_decision",
    "fit_decoherence_regression",
    "validate_nv_quantum_sensing_fixture",
]
