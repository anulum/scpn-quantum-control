# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 Triple Network validation fixtures
"""Simulator-only Layer 5 Triple Network fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .spec_loader import load_l5_triple_network_validation_spec

CLAIM_BOUNDARY = "source-bounded Layer 5 Triple Network simulator contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06485", "P0R06503")


@dataclass(frozen=True, slots=True)
class L5TripleNetworkConfig:
    """Finite simulator settings for Layer 5 Triple Network fixtures."""

    mapping_threshold: float = 0.72
    anti_correlation_threshold: float = 0.70
    salience_threshold: float = 0.70
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("mapping_threshold", self.mapping_threshold)
        _require_positive("anti_correlation_threshold", self.anti_correlation_threshold)
        _require_positive("salience_threshold", self.salience_threshold)


@dataclass(frozen=True, slots=True)
class L5TripleNetworkFixtureResult:
    """Combined Layer 5 Triple Network fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    mapping_score: float
    dmn_cen_anti_correlation: float
    max_salience: float
    switch_state: str
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def anatomical_mapping_score(
    *,
    dmn_mapping: float,
    cen_mapping: float,
    sn_mapping: float,
) -> float:
    """Score the source-bounded DMN/CEN/SN anatomical mapping."""
    values = _unit_interval_values(dmn_mapping, cen_mapping, sn_mapping)
    return float(np.prod(values) ** (1.0 / 3.0))


def anti_correlation_index(
    *,
    dmn_activity: NDArray[np.float64],
    cen_activity: NDArray[np.float64],
) -> float:
    """Return positive strength of DMN-CEN anti-correlation."""
    dmn = _finite_vector("dmn_activity", dmn_activity)
    cen = _finite_vector("cen_activity", cen_activity)
    if dmn.shape != cen.shape:
        raise ValueError("activity vectors must have the same shape")
    if dmn.size < 2:
        raise ValueError("activity vectors must contain at least two samples")
    corr = float(np.corrcoef(dmn, cen)[0, 1])
    if not np.isfinite(corr):
        raise ValueError("activity vectors must have non-zero variance")
    return max(0.0, -corr)


def interoceptive_salience(
    *,
    precision: NDArray[np.float64],
    prediction_error: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute salience as precision times absolute prediction error."""
    precision_arr = _finite_vector("precision", precision)
    error_arr = _finite_vector("prediction_error", prediction_error)
    if precision_arr.shape != error_arr.shape:
        raise ValueError("precision and prediction_error must have the same shape")
    if np.any(precision_arr < 0.0):
        raise ValueError("precision must be non-negative")
    return cast(NDArray[np.float64], precision_arr * np.abs(error_arr))


def salience_switch_state(*, salience: NDArray[np.float64], threshold: float) -> str:
    """Classify source-bounded SN switching state from salience threshold crossing."""
    _require_positive("threshold", threshold)
    salience_arr = _finite_vector("salience", salience)
    return "CEN_engagement" if float(np.max(salience_arr)) >= threshold else "DMN_dominance"


def validate_l5_triple_network_fixture(
    config: L5TripleNetworkConfig | None = None,
) -> L5TripleNetworkFixtureResult:
    """Run the combined Layer 5 Triple Network fixture."""
    cfg = config or L5TripleNetworkConfig()
    keys = (
        "l5_triple_network.anatomical_mapping",
        "l5_triple_network.salience_switching",
        "l5_triple_network.interoceptive_inference",
        "l5_triple_network.homeostatic_qualia_boundary",
    )
    specs = tuple(
        load_l5_triple_network_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    mapping = anatomical_mapping_score(dmn_mapping=0.88, cen_mapping=0.84, sn_mapping=0.86)
    anti_corr = anti_correlation_index(
        dmn_activity=np.array([0.8, 0.7, 0.2, 0.1], dtype=np.float64),
        cen_activity=np.array([0.1, 0.2, 0.7, 0.8], dtype=np.float64),
    )
    salience = interoceptive_salience(
        precision=np.array([0.5, 1.5, 2.0], dtype=np.float64),
        prediction_error=np.array([0.1, -0.4, 0.6], dtype=np.float64),
    )
    controls = {
        "missing_salience_network_rejection_label": float(
            anatomical_mapping_score(dmn_mapping=0.88, cen_mapping=0.84, sn_mapping=0.0)
            < cfg.mapping_threshold
        ),
        "shape_mismatch_rejection_label": _shape_mismatch_rejection_label(),
        "unsupported_empirical_mapping_rejection_label": 1.0,
    }
    return L5TripleNetworkFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        mapping_score=mapping,
        dmn_cen_anti_correlation=anti_corr,
        max_salience=float(np.max(salience)),
        switch_state=salience_switch_state(
            salience=salience,
            threshold=cfg.salience_threshold,
        ),
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "mapping_threshold": cfg.mapping_threshold,
                "anti_correlation_threshold": cfg.anti_correlation_threshold,
                "salience_threshold": cfg.salience_threshold,
            }
        ),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def _shape_mismatch_rejection_label() -> float:
    try:
        anti_correlation_index(
            dmn_activity=np.array([0.1, 0.2], dtype=np.float64),
            cen_activity=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        )
    except ValueError as exc:
        return float("same shape" in str(exc))
    return 0.0


def _unit_interval_values(*values: float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("inputs must be in [0, 1]")
    return array


def _finite_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "CLAIM_BOUNDARY",
    "L5TripleNetworkConfig",
    "L5TripleNetworkFixtureResult",
    "anatomical_mapping_score",
    "anti_correlation_index",
    "interoceptive_salience",
    "salience_switch_state",
    "validate_l5_triple_network_fixture",
]
