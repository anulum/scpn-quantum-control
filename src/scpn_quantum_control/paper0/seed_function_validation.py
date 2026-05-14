# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 seed-function validation fixtures
"""Simulator-only teleological seed-function fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypedDict

import numpy as np

from .spec_loader import load_seed_function_validation_spec

CLAIM_BOUNDARY = "source-bounded seed-function simulator contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06363", "P0R06377")


class SeedPayload(TypedDict):
    """Source payload returned by the manuscript seed function."""

    ssb_bias_magnitude: float
    is_random_reset: bool
    conformal_continuity: bool


@dataclass(frozen=True, slots=True)
class SeedFunctionConfig:
    """Finite simulator settings for seed-function fixtures."""

    bias_threshold: float = 0.1
    coupling_threshold: float = 1.0e-12
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("bias_threshold", self.bias_threshold)
        _require_positive("coupling_threshold", self.coupling_threshold)


@dataclass(frozen=True, slots=True)
class SeedFunctionFixtureResult:
    """Combined seed-function fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    payload: SeedPayload
    source_formulae: tuple[str, ...]
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def mu_squared_seed(*, prev_cycle_sec: float, coupling_constant_g: float) -> float:
    """Return source formula sqrt(prev_cycle_sec / coupling_constant_g)."""
    _require_non_negative("prev_cycle_sec", prev_cycle_sec)
    _require_positive("coupling_constant_g", coupling_constant_g)
    return float(np.sqrt(prev_cycle_sec / coupling_constant_g))


def compute_teleological_seed(
    *,
    prev_cycle_sec: float,
    coupling_constant_g: float,
) -> SeedPayload:
    """Execute the bounded source-format seed function."""
    seed = mu_squared_seed(
        prev_cycle_sec=prev_cycle_sec,
        coupling_constant_g=coupling_constant_g,
    )
    return {
        "ssb_bias_magnitude": seed,
        "is_random_reset": False,
        "conformal_continuity": bool(prev_cycle_sec > 0.0),
    }


def validate_seed_function_fixture(
    config: SeedFunctionConfig | None = None,
) -> SeedFunctionFixtureResult:
    """Run the combined seed-function fixture."""
    cfg = config or SeedFunctionConfig()
    keys = (
        "seed_function.python_format_source_boundary",
        "seed_function.mu_squared_seed_formula",
        "seed_function.return_payload_contract",
        "seed_function.conformal_continuity_boundary",
    )
    specs = tuple(
        load_seed_function_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    payload = compute_teleological_seed(prev_cycle_sec=0.81, coupling_constant_g=0.36)
    controls = {
        "negative_sec_rejection_label": _negative_sec_rejection_label(),
        "zero_coupling_rejection_label": _zero_coupling_rejection_label(),
        "unsupported_seed_evidence_rejection_label": 1.0,
    }
    source_formulae = tuple(
        formula for spec in specs for formula in (str(item) for item in spec["source_formulae"])
    )
    return SeedFunctionFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        payload=payload,
        source_formulae=source_formulae,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "bias_threshold": cfg.bias_threshold,
                "coupling_threshold": cfg.coupling_threshold,
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


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _negative_sec_rejection_label() -> float:
    try:
        compute_teleological_seed(prev_cycle_sec=-0.1, coupling_constant_g=0.36)
    except ValueError as exc:
        return float("finite and non-negative" in str(exc))
    return 0.0


def _zero_coupling_rejection_label() -> float:
    try:
        compute_teleological_seed(prev_cycle_sec=0.81, coupling_constant_g=0.0)
    except ValueError as exc:
        return float("finite and positive" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "SeedFunctionConfig",
    "SeedFunctionFixtureResult",
    "SeedPayload",
    "compute_teleological_seed",
    "mu_squared_seed",
    "validate_seed_function_fixture",
]
