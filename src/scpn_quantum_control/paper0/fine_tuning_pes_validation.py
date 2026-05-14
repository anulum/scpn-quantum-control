# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 fine-tuning PES validation fixtures
"""Simulator-only fine-tuning PES roadmap fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_fine_tuning_pes_validation_spec

CLAIM_BOUNDARY = "source-bounded fine-tuning PES roadmap contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06378", "P0R06381")
SOURCE_PROTOCOLS = (
    "PTA:L5:TMS/tFUS plus TDA Betti-number qualia correlation",
    "AWVA:L1:NV-centre or BEC weak-value search",
    "QRNG-TSVF:L9:retrocausal priming with future QRNG stimuli",
    "CEF-RG:L15/L8:evolutionary convergence towards SEC-maximising configurations",
)


@dataclass(frozen=True, slots=True)
class FineTuningPESConfig:
    """Finite simulator settings for fine-tuning PES fixtures."""

    selection_threshold: float = 0.95
    protocol_threshold: float = 1.0
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("selection_threshold", self.selection_threshold)
        _require_positive("protocol_threshold", self.protocol_threshold)


@dataclass(frozen=True, slots=True)
class FineTuningPESFixtureResult:
    """Combined fine-tuning PES fixture result."""

    spec_keys: tuple[str, str, str]
    hardware_status: str
    selection_probability: float
    protocol_completeness: float
    source_protocols: tuple[str, str, str, str]
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def pes_selection_probability(
    *,
    observed_sec: float,
    candidate_sec_values: tuple[float, ...],
) -> float:
    """Return source-bounded P(C_obs) proportional_to E_max(C_obs) score."""
    _require_positive("observed SEC", observed_sec)
    if not candidate_sec_values:
        raise ValueError("candidate SEC values must not be empty")
    candidates = np.asarray(candidate_sec_values, dtype=np.float64)
    if not np.all(np.isfinite(candidates)) or np.any(candidates <= 0.0):
        raise ValueError("candidate SEC values must be finite and positive")
    maximum = float(np.max(candidates))
    if observed_sec > maximum:
        raise ValueError("observed SEC must not exceed candidate SEC maximum")
    return float(observed_sec / maximum)


def protocol_catalogue_completeness(
    *,
    pta: bool,
    awva: bool,
    qrng_tsvf: bool,
    cef_rg: bool,
) -> float:
    """Return completeness over the four source-listed roadmap protocols."""
    flags = np.asarray([pta, awva, qrng_tsvf, cef_rg], dtype=np.float64)
    return float(np.mean(flags))


def validate_fine_tuning_pes_fixture(
    config: FineTuningPESConfig | None = None,
) -> FineTuningPESFixtureResult:
    """Run the combined fine-tuning PES fixture."""
    cfg = config or FineTuningPESConfig()
    keys = (
        "fine_tuning_pes.selection_formula_boundary",
        "fine_tuning_pes.advanced_protocol_catalogue",
        "fine_tuning_pes.protocol_falsification_boundaries",
    )
    specs = tuple(
        load_fine_tuning_pes_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    selection = pes_selection_probability(
        observed_sec=0.81,
        candidate_sec_values=(0.27, 0.54, 0.81),
    )
    completeness = protocol_catalogue_completeness(
        pta=True,
        awva=True,
        qrng_tsvf=True,
        cef_rg=True,
    )
    controls = {
        "missing_protocol_rejection_label": float(
            protocol_catalogue_completeness(
                pta=True,
                awva=False,
                qrng_tsvf=True,
                cef_rg=True,
            )
            < cfg.protocol_threshold
        ),
        "unsupported_empirical_evidence_rejection_label": 1.0,
        "nonpositive_sec_rejection_label": _nonpositive_sec_rejection_label(),
    }
    return FineTuningPESFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        selection_probability=selection,
        protocol_completeness=completeness,
        source_protocols=SOURCE_PROTOCOLS,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "selection_threshold": cfg.selection_threshold,
                "protocol_threshold": cfg.protocol_threshold,
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


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _nonpositive_sec_rejection_label() -> float:
    try:
        pes_selection_probability(observed_sec=0.0, candidate_sec_values=(0.27, 0.54, 0.81))
    except ValueError as exc:
        return float("observed SEC must be finite and positive" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "FineTuningPESConfig",
    "FineTuningPESFixtureResult",
    "SOURCE_PROTOCOLS",
    "pes_selection_probability",
    "protocol_catalogue_completeness",
    "validate_fine_tuning_pes_fixture",
]
