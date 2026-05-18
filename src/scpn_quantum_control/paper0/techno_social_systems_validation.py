# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Techno-Social Systems validation
"""Source-accounting checks for Paper 0  Techno-Social Systems records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded techno social systems source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05868", "P0R05875")


@dataclass(frozen=True, slots=True)
class TechnoSocialSystemsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05876"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05876":
            raise ValueError("next_source_boundary must equal P0R05876")


@dataclass(frozen=True, slots=True)
class TechnoSocialSystemsFixtureResult:
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


def classify_techno_social_systems_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "techno_social_systems": "techno_social_systems_source_boundary",
        "neurophenomenology_first_person_science": "neurophenomenology_first_person_science_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown techno_social_systems component") from exc


def techno_social_systems_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Techno-Social Systems",
        "source_span": "P0R05868-P0R05875",
        "component_count": "2",
        "next_boundary": "P0R05876",
        "component_1": "Techno-Social Systems",
        "component_2": "Neurophenomenology & First-Person Science",
    }


def validate_techno_social_systems_fixture(
    config: TechnoSocialSystemsConfig | None = None,
) -> TechnoSocialSystemsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TechnoSocialSystemsConfig()
    components = ("techno_social_systems", "neurophenomenology_first_person_science")
    return TechnoSocialSystemsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_techno_social_systems_component(component)
            for component in components
        },
        labels=techno_social_systems_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "techno_social_systems_is_not_empirical_validation_evidence": 1.0,
            "neurophenomenology_first_person_science_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5868, 5876)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_techno_social_systems_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TechnoSocialSystemsConfig",
    "TechnoSocialSystemsFixtureResult",
    "classify_techno_social_systems_component",
    "techno_social_systems_labels",
    "validate_techno_social_systems_fixture",
]
