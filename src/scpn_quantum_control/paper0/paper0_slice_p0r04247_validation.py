# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  validation
"""Source-accounting checks for Paper 0  records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded paper0 slice p0r04247 source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04247", "P0R04256")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04247Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04257"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04257":
            raise ValueError("next_source_boundary must equal P0R04257")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04247FixtureResult:
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


def classify_paper0_slice_p0r04247_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "p0r04247": "p0r04247_source_boundary",
        "5_1_the_em_interface_an_alp_mediated_bridge": "5_1_the_em_interface_an_alp_mediated_bridge_source_boundary",
        "an_alp_mediated_bridge": "an_alp_mediated_bridge_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown paper0_slice_p0r04247 component") from exc


def paper0_slice_p0r04247_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "",
        "source_span": "P0R04247-P0R04256",
        "component_count": "3",
        "next_boundary": "P0R04257",
        "component_1": "P0R04247",
        "component_2": "5.1 The EM Interface: An ALP-Mediated Bridge",
        "component_3": "An ALP-Mediated Bridge",
    }


def validate_paper0_slice_p0r04247_fixture(
    config: Paper0SliceP0r04247Config | None = None,
) -> Paper0SliceP0r04247FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Paper0SliceP0r04247Config()
    components = (
        "p0r04247",
        "5_1_the_em_interface_an_alp_mediated_bridge",
        "an_alp_mediated_bridge",
    )
    return Paper0SliceP0r04247FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_paper0_slice_p0r04247_component(component)
            for component in components
        },
        labels=paper0_slice_p0r04247_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "p0r04247_is_not_empirical_validation_evidence": 1.0,
            "5_1_the_em_interface_an_alp_mediated_bridge_is_not_empirical_validation_evidence": 1.0,
            "an_alp_mediated_bridge_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4247, 4257)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_paper0_slice_p0r04247_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Paper0SliceP0r04247Config",
    "Paper0SliceP0r04247FixtureResult",
    "classify_paper0_slice_p0r04247_component",
    "paper0_slice_p0r04247_labels",
    "validate_paper0_slice_p0r04247_fixture",
]
