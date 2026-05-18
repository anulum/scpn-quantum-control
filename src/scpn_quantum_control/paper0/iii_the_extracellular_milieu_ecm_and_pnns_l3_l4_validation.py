# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 III. The Extracellular Milieu: ECM and PNNs (L3/L4) validation
"""Source-accounting checks for Paper 0 III. The Extracellular Milieu: ECM and PNNs (L3/L4) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iii the extracellular milieu ecm and pnns l3 l4 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04778", "P0R04785")


@dataclass(frozen=True, slots=True)
class IiiTheExtracellularMilieuEcmAndPnnsL3L4Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04786"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04786":
            raise ValueError("next_source_boundary must equal P0R04786")


@dataclass(frozen=True, slots=True)
class IiiTheExtracellularMilieuEcmAndPnnsL3L4FixtureResult:
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


def classify_iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "iii_the_extracellular_milieu_ecm_and_pnns_l3_l4": "iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_source_boundary",
        "1_composition_and_function": "1_composition_and_function_source_boundary",
        "2_perineuronal_nets_pnns_the_guardians_of_criticality": "2_perineuronal_nets_pnns_the_guardians_of_criticality_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iii_the_extracellular_milieu_ecm_and_pnns_l3_l4 component"
        ) from exc


def iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "III. The Extracellular Milieu: ECM and PNNs (L3/L4)",
        "source_span": "P0R04778-P0R04785",
        "component_count": "3",
        "next_boundary": "P0R04786",
        "component_1": "III. The Extracellular Milieu: ECM and PNNs (L3/L4)",
        "component_2": "1. Composition and Function:",
        "component_3": "2. Perineuronal Nets (PNNs): The Guardians of Criticality",
    }


def validate_iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_fixture(
    config: IiiTheExtracellularMilieuEcmAndPnnsL3L4Config | None = None,
) -> IiiTheExtracellularMilieuEcmAndPnnsL3L4FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiiTheExtracellularMilieuEcmAndPnnsL3L4Config()
    components = (
        "iii_the_extracellular_milieu_ecm_and_pnns_l3_l4",
        "1_composition_and_function",
        "2_perineuronal_nets_pnns_the_guardians_of_criticality",
    )
    return IiiTheExtracellularMilieuEcmAndPnnsL3L4FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_component(
                component
            )
            for component in components
        },
        labels=iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_is_not_empirical_validation_evidence": 1.0,
            "1_composition_and_function_is_not_empirical_validation_evidence": 1.0,
            "2_perineuronal_nets_pnns_the_guardians_of_criticality_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4778, 4786)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiiTheExtracellularMilieuEcmAndPnnsL3L4Config",
    "IiiTheExtracellularMilieuEcmAndPnnsL3L4FixtureResult",
    "classify_iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_component",
    "iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_labels",
    "validate_iii_the_extracellular_milieu_ecm_and_pnns_l3_l4_fixture",
]
