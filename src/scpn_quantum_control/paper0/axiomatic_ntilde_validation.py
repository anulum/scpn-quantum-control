# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 axiomatic Ntilde validation
"""Source-accounting checks for formal Logos/Ntilde records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

CLAIM_BOUNDARY = "source-bounded formal Logos and Ntilde invariant; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00578", "P0R00609")


@dataclass(frozen=True, slots=True)
class AxiomaticNtildeConfig:
    """Configuration for the formal Logos/Ntilde fixture."""

    expected_axiom_count: int = 3
    expected_ntilde_formula_count: int = 5
    next_source_boundary: str = "P0R00610"

    def __post_init__(self) -> None:
        if self.expected_axiom_count != 3:
            raise ValueError("expected_axiom_count must equal 3")
        if self.expected_ntilde_formula_count != 5:
            raise ValueError("expected_ntilde_formula_count must equal 5")
        if self.next_source_boundary != "P0R00610":
            raise ValueError("next_source_boundary must equal P0R00610")


@dataclass(frozen=True, slots=True)
class AxiomaticNtildeFixtureResult:
    """Result for the Paper 0 formal Logos/Ntilde fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    axiomatic_statuses: dict[str, str]
    ntilde_formula_count: int
    axiom_count: int
    reference_ratio: float
    reference_delta_irr: float
    reference_regime: str
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def _require_positive_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def ntilde_ratio(
    *,
    power: float,
    reversible_cost_per_bit: float,
    information_rate: float,
) -> float:
    """Compute the source-defined dimensionless Ntilde ratio."""
    p = _require_positive_finite("power", power)
    epsilon_b = _require_positive_finite("reversible_cost_per_bit", reversible_cost_per_bit)
    i_dot = _require_positive_finite("information_rate", information_rate)
    return p / (epsilon_b * i_dot)


def irreversibility_delta(ntilde: float) -> float:
    """Return delta_irr from the source tight formulation Ntilde = 1 + delta_irr."""
    numeric = float(ntilde)
    if not isfinite(numeric):
        raise ValueError("ntilde must be finite")
    return numeric - 1.0


def classify_ntilde_regime(ntilde: float, *, tolerance: float = 1.0e-9) -> str:
    """Classify an Ntilde value without treating the source target as observed data."""
    numeric = float(ntilde)
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError("ntilde must be finite and positive")
    if abs(numeric - 1.0) <= tolerance:
        return "quasicritical_reversible_threshold"
    if numeric > 1.0:
        return "irreversible_overhead"
    return "underpowered_or_efficiency_claim_rejected"


def classify_axiomatic_status(key: str) -> str:
    """Classify the formal Logos axiom status labels from source records."""
    mapping = {
        "axiom_1": "metaphysical_assumption_generative",
        "axiom_2": "falsifiable_information_geometry_hypothesis",
        "axiom_3": "normative_teleology_with_proposed_falsifiable_ntilde_invariant",
        "axiom_3_status_tension": "preserve_normative_to_physical_claim_transition",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError("unknown axiomatic status key") from exc


def validate_axiomatic_ntilde_fixture(
    config: AxiomaticNtildeConfig | None = None,
) -> AxiomaticNtildeFixtureResult:
    """Validate source accounting for the formal Logos/Ntilde slice."""
    cfg = config or AxiomaticNtildeConfig()
    status_keys = ("axiom_1", "axiom_2", "axiom_3", "axiom_3_status_tension")
    statuses = {key: classify_axiomatic_status(key) for key in status_keys}
    reference = ntilde_ratio(
        power=1.0,
        reversible_cost_per_bit=1.0,
        information_rate=1.0,
    )

    return AxiomaticNtildeFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        axiomatic_statuses=statuses,
        ntilde_formula_count=cfg.expected_ntilde_formula_count,
        axiom_count=cfg.expected_axiom_count,
        reference_ratio=reference,
        reference_delta_irr=irreversibility_delta(reference),
        reference_regime=classify_ntilde_regime(reference),
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "figure_caption_is_not_validation_evidence": 1.0,
            "status_transition_is_not_empirical_confirmation": 1.0,
            "ntilde_unity_is_target_not_observed_result": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(578, 610)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_invariant_map_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomaticNtildeConfig",
    "AxiomaticNtildeFixtureResult",
    "classify_axiomatic_status",
    "classify_ntilde_regime",
    "irreversibility_delta",
    "ntilde_ratio",
    "validate_axiomatic_ntilde_fixture",
]
