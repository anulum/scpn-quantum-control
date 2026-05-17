# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 LPsi Sets the Properties of Psis: validation
"""Source-accounting checks for Paper 0 LPsi Sets the Properties of Psis: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded lpsi sets the properties of psis source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01771", "P0R01778")


@dataclass(frozen=True, slots=True)
class LpsiSetsThePropertiesOfPsisConfig:
    """Configuration for the LΨ Sets the Properties of Ψs: fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R01779"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R01779":
            raise ValueError("next_source_boundary must equal P0R01779")


@dataclass(frozen=True, slots=True)
class LpsiSetsThePropertiesOfPsisFixtureResult:
    """Result for the Paper 0 LΨ Sets the Properties of Ψs: fixture."""

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


def classify_lpsi_sets_the_properties_of_psis_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "lpsi_sets_the_properties_of_psis": "lpsi_sets_the_properties_of_psis_source_boundary",
        "a_stable_vacuum_for_a_stable_interaction": "a_stable_vacuum_for_a_stable_interaction_source_boundary",
        "the_intrinsic_dynamics_of_the_psi_field_lpsi": "the_intrinsic_dynamics_of_the_psi_field_lpsi_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown lpsi_sets_the_properties_of_psis component") from exc


def lpsi_sets_the_properties_of_psis_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "LPsi Sets the Properties of Psis:",
        "source_span": "P0R01771-P0R01778",
        "component_count": "3",
        "next_boundary": "P0R01779",
        "component_1": "LPsi Sets the Properties of Psis:",
        "component_2": "A Stable Vacuum for a Stable Interaction:",
        "component_3": "The Intrinsic Dynamics of the Psi-Field (LPsi)",
    }


def validate_lpsi_sets_the_properties_of_psis_fixture(
    config: LpsiSetsThePropertiesOfPsisConfig | None = None,
) -> LpsiSetsThePropertiesOfPsisFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or LpsiSetsThePropertiesOfPsisConfig()
    components = (
        "lpsi_sets_the_properties_of_psis",
        "a_stable_vacuum_for_a_stable_interaction",
        "the_intrinsic_dynamics_of_the_psi_field_lpsi",
    )
    return LpsiSetsThePropertiesOfPsisFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_lpsi_sets_the_properties_of_psis_component(component)
            for component in components
        },
        labels=lpsi_sets_the_properties_of_psis_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "lpsi_sets_the_properties_of_psis_is_not_empirical_validation_evidence": 1.0,
            "a_stable_vacuum_for_a_stable_interaction_is_not_empirical_validation_evidence": 1.0,
            "the_intrinsic_dynamics_of_the_psi_field_lpsi_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1771, 1779)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_lpsi_sets_the_properties_of_psis_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "LpsiSetsThePropertiesOfPsisConfig",
    "LpsiSetsThePropertiesOfPsisFixtureResult",
    "classify_lpsi_sets_the_properties_of_psis_component",
    "lpsi_sets_the_properties_of_psis_labels",
    "validate_lpsi_sets_the_properties_of_psis_fixture",
]
