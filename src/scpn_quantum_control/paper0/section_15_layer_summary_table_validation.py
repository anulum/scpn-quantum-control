# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 15-Layer Summary Table validation
"""Source-accounting checks for Paper 0 15-Layer Summary Table records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 15 layer summary table source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02061", "P0R02087")


@dataclass(frozen=True, slots=True)
class Section15LayerSummaryTableConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 27
    expected_component_count: int = 1
    next_source_boundary: str = "P0R02088"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 27:
            raise ValueError("expected_source_record_count must equal 27")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R02088":
            raise ValueError("next_source_boundary must equal P0R02088")


@dataclass(frozen=True, slots=True)
class Section15LayerSummaryTableFixtureResult:
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


def classify_section_15_layer_summary_table_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {"15_layer_summary_table": "15_layer_summary_table_source_boundary"}
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_15_layer_summary_table component") from exc


def section_15_layer_summary_table_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "15-Layer Summary Table",
        "source_span": "P0R02061-P0R02087",
        "component_count": "1",
        "next_boundary": "P0R02088",
        "component_1": "15-Layer Summary Table",
    }


def validate_section_15_layer_summary_table_fixture(
    config: Section15LayerSummaryTableConfig | None = None,
) -> Section15LayerSummaryTableFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section15LayerSummaryTableConfig()
    components = ("15_layer_summary_table",)
    return Section15LayerSummaryTableFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_15_layer_summary_table_component(component)
            for component in components
        },
        labels=section_15_layer_summary_table_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={"15_layer_summary_table_is_not_empirical_validation_evidence": 1.0},
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2061, 2088)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_15_layer_summary_table_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section15LayerSummaryTableConfig",
    "Section15LayerSummaryTableFixtureResult",
    "classify_section_15_layer_summary_table_component",
    "section_15_layer_summary_table_labels",
    "validate_section_15_layer_summary_table_fixture",
]
