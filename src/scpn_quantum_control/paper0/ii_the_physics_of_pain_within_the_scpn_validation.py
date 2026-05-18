# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Physics of Pain within the SCPN validation
"""Source-accounting checks for Paper 0 II. The Physics of Pain within the SCPN records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ii the physics of pain within the scpn source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05075", "P0R05082")


@dataclass(frozen=True, slots=True)
class IiThePhysicsOfPainWithinTheScpnConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05083"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05083":
            raise ValueError("next_source_boundary must equal P0R05083")


@dataclass(frozen=True, slots=True)
class IiThePhysicsOfPainWithinTheScpnFixtureResult:
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


def classify_ii_the_physics_of_pain_within_the_scpn_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "ii_the_physics_of_pain_within_the_scpn": "ii_the_physics_of_pain_within_the_scpn_source_boundary",
        "iii_intervention_intravenous_morphine_opioid_agonism": "iii_intervention_intravenous_morphine_opioid_agonism_source_boundary",
        "1_l2_modulation_the_molecular_brake_and_iet_interface": "1_l2_modulation_the_molecular_brake_and_iet_interface_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown ii_the_physics_of_pain_within_the_scpn component") from exc


def ii_the_physics_of_pain_within_the_scpn_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "II. The Physics of Pain within the SCPN",
        "source_span": "P0R05075-P0R05082",
        "component_count": "3",
        "next_boundary": "P0R05083",
        "component_1": "II. The Physics of Pain within the SCPN",
        "component_2": "III. Intervention: Intravenous Morphine (Opioid Agonism)",
        "component_3": "1. L2 Modulation (The Molecular Brake and IET Interface):",
    }


def validate_ii_the_physics_of_pain_within_the_scpn_fixture(
    config: IiThePhysicsOfPainWithinTheScpnConfig | None = None,
) -> IiThePhysicsOfPainWithinTheScpnFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiThePhysicsOfPainWithinTheScpnConfig()
    components = (
        "ii_the_physics_of_pain_within_the_scpn",
        "iii_intervention_intravenous_morphine_opioid_agonism",
        "1_l2_modulation_the_molecular_brake_and_iet_interface",
    )
    return IiThePhysicsOfPainWithinTheScpnFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ii_the_physics_of_pain_within_the_scpn_component(component)
            for component in components
        },
        labels=ii_the_physics_of_pain_within_the_scpn_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ii_the_physics_of_pain_within_the_scpn_is_not_empirical_validation_evidence": 1.0,
            "iii_intervention_intravenous_morphine_opioid_agonism_is_not_empirical_validation_evidence": 1.0,
            "1_l2_modulation_the_molecular_brake_and_iet_interface_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5075, 5083)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ii_the_physics_of_pain_within_the_scpn_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiThePhysicsOfPainWithinTheScpnConfig",
    "IiThePhysicsOfPainWithinTheScpnFixtureResult",
    "classify_ii_the_physics_of_pain_within_the_scpn_component",
    "ii_the_physics_of_pain_within_the_scpn_labels",
    "validate_ii_the_physics_of_pain_within_the_scpn_fixture",
]
