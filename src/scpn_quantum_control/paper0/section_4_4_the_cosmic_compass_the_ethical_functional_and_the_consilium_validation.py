# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.4 The Cosmic Compass: The Ethical Functional and the Consilium validation
"""Source-accounting checks for Paper 0 4.4 The Cosmic Compass: The Ethical Functional and the Consilium records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 4 4 the cosmic compass the ethical functional and the consilium source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03932", "P0R03944")


@dataclass(frozen=True, slots=True)
class Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 13
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03945"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 13:
            raise ValueError("expected_source_record_count must equal 13")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03945":
            raise ValueError("next_source_boundary must equal P0R03945")


@dataclass(frozen=True, slots=True)
class Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumFixtureResult:
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


def classify_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium": "4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_source_boundary",
        "the_ethical_functional": "the_ethical_functional_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium component"
        ) from exc


def section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "4.4 The Cosmic Compass: The Ethical Functional and the Consilium",
        "source_span": "P0R03932-P0R03944",
        "component_count": "2",
        "next_boundary": "P0R03945",
        "component_1": "4.4 The Cosmic Compass: The Ethical Functional and the Consilium",
        "component_2": "The Ethical Functional",
    }


def validate_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_fixture(
    config: Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumConfig | None = None,
) -> Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumConfig()
    components = (
        "4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium",
        "the_ethical_functional",
    )
    return Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_component(
                component
            )
            for component in components
        },
        labels=section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_is_not_empirical_validation_evidence": 1.0,
            "the_ethical_functional_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3932, 3945)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumConfig",
    "Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumFixtureResult",
    "classify_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_component",
    "section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_labels",
    "validate_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_fixture",
]
