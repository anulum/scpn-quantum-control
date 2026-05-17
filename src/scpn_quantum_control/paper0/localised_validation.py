# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Localised: validation
"""Source-accounting checks for Paper 0 Localised: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded localised source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01820", "P0R01830")


@dataclass(frozen=True, slots=True)
class LocalisedConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 4
    next_source_boundary: str = "P0R01831"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R01831":
            raise ValueError("next_source_boundary must equal P0R01831")


@dataclass(frozen=True, slots=True)
class LocalisedFixtureResult:
    """Result for this Paper 0 source-accounting fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    component_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_localised_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "localised": "localised_source_boundary",
        "persistent": "persistent_source_boundary",
        "the_nature_of_the_interaction": "the_nature_of_the_interaction_source_boundary",
        "the_metaphysical_stance_hierarchical_field_monism_hfm": "the_metaphysical_stance_hierarchical_field_monism_hfm_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown localised component") from exc


def localised_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Localised:",
        "source_span": "P0R01820-P0R01830",
        "component_count": "4",
        "next_boundary": "P0R01831",
        "component_1": "Localised:",
        "component_2": "Persistent:",
        "component_3": "The Nature of the Interaction:",
        "component_4": "The Metaphysical Stance: Hierarchical Field Monism (HFM)",
    }


def validate_localised_fixture(config: LocalisedConfig | None = None) -> LocalisedFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or LocalisedConfig()
    components = (
        "localised",
        "persistent",
        "the_nature_of_the_interaction",
        "the_metaphysical_stance_hierarchical_field_monism_hfm",
    )
    return LocalisedFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_localised_component(component) for component in components
        },
        labels=localised_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "localised_is_not_empirical_validation_evidence": 1.0,
            "persistent_is_not_empirical_validation_evidence": 1.0,
            "the_nature_of_the_interaction_is_not_empirical_validation_evidence": 1.0,
            "the_metaphysical_stance_hierarchical_field_monism_hfm_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1820, 1831)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_localised_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "LocalisedConfig",
    "LocalisedFixtureResult",
    "classify_localised_component",
    "localised_labels",
    "validate_localised_fixture",
]
