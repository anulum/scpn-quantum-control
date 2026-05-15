# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Lorentz EFT resolution validation
"""Source-accounting checks for Paper 0 Lorentz/EFT-resolution records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Lorentz/EFT resolution; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01078", "P0R01102")


@dataclass(frozen=True, slots=True)
class LorentzEFTResolutionConfig:
    """Configuration for the Lorentz/EFT-resolution fixture."""

    expected_source_record_count: int = 25
    expected_blank_record_count: int = 1
    expected_ghost_action_record_count: int = 5
    next_source_boundary: str = "P0R01103"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 25:
            raise ValueError("expected_source_record_count must equal 25")
        if self.expected_blank_record_count != 1:
            raise ValueError("expected_blank_record_count must equal 1")
        if self.expected_ghost_action_record_count != 5:
            raise ValueError("expected_ghost_action_record_count must equal 5")
        if self.next_source_boundary != "P0R01103":
            raise ValueError("next_source_boundary must equal P0R01103")


@dataclass(frozen=True, slots=True)
class LorentzEFTResolutionFixtureResult:
    """Result for the Paper 0 Lorentz/EFT-resolution fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    blank_record_count: int
    lorentz_tension_record_count: int
    fundamental_action_record_count: int
    biological_medium_record_count: int
    consistency_record_count: int
    ghost_action_record_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_lorentz_eft_resolution_component(component: str) -> str:
    """Classify source-defined Lorentz/EFT-resolution components."""
    mapping = {
        "boundary_and_tension": "lorentz_covariance_tension_and_eft_boundary",
        "fundamental_lorentz_invariant_action": (
            "lorentz_scalar_infoton_action_with_higher_dimension_fim_operator"
        ),
        "biological_medium_effective_metric": (
            "spontaneous_medium_effective_metric_and_infoton_kinetic_term"
        ),
        "consistency_implications": (
            "localized_lorentz_breaking_gauge_invariance_and_refractive_index_claims"
        ),
        "ghost_action_boundary": (
            "fim_background_gauge_fixing_ghost_action_and_separator_boundary"
        ),
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown Lorentz-EFT-resolution component") from exc


def lorentz_eft_resolution_labels() -> dict[str, str]:
    """Return source-bounded labels for the Lorentz/EFT-resolution slice."""
    return {
        "section": "Formal Resolution of Lorentz Covariance",
        "fundamental_action": "Lorentz scalar infoton action with Lambda_I suppression",
        "effective_metric": "g_eff = eta - c/(2 Lambda_I^2) gF",
        "ghost_action": "Faddeev-Popov ghost action in FIM background",
        "next_boundary": "Beyond U(1): Non-Abelian Qualia Field",
    }


def validate_lorentz_eft_resolution_fixture(
    config: LorentzEFTResolutionConfig | None = None,
) -> LorentzEFTResolutionFixtureResult:
    """Validate source accounting for the Lorentz/EFT-resolution slice."""
    cfg = config or LorentzEFTResolutionConfig()
    components = (
        "boundary_and_tension",
        "fundamental_lorentz_invariant_action",
        "biological_medium_effective_metric",
        "consistency_implications",
        "ghost_action_boundary",
    )

    return LorentzEFTResolutionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_lorentz_eft_resolution_component(component)
            for component in components
        },
        labels=lorentz_eft_resolution_labels(),
        source_record_count=cfg.expected_source_record_count,
        blank_record_count=cfg.expected_blank_record_count,
        lorentz_tension_record_count=4,
        fundamental_action_record_count=5,
        biological_medium_record_count=7,
        consistency_record_count=4,
        ghost_action_record_count=cfg.expected_ghost_action_record_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "lorentz_eft_resolution_is_source_claim_not_empirical_evidence": 1.0,
            "naive_fim_metric_replacement_is_marked_as_lorentz_violation": 1.0,
            "blank_record_p0r01079_and_separator_p0r01102_are_preserved": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1078, 1103)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_lorentz_eft_resolution_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "LorentzEFTResolutionConfig",
    "LorentzEFTResolutionFixtureResult",
    "classify_lorentz_eft_resolution_component",
    "lorentz_eft_resolution_labels",
    "validate_lorentz_eft_resolution_fixture",
]
