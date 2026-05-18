# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 XI. The Mathematics of Hierarchy and Scale Invariance validation
"""Source-accounting checks for Paper 0 XI. The Mathematics of Hierarchy and Scale Invariance records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded xi the mathematics of hierarchy and scale invariance source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06057", "P0R06065")


@dataclass(frozen=True, slots=True)
class XiTheMathematicsOfHierarchyAndScaleInvarianceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R06066"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R06066":
            raise ValueError("next_source_boundary must equal P0R06066")


@dataclass(frozen=True, slots=True)
class XiTheMathematicsOfHierarchyAndScaleInvarianceFixtureResult:
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


def classify_xi_the_mathematics_of_hierarchy_and_scale_invariance_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "xi_the_mathematics_of_hierarchy_and_scale_invariance": "xi_the_mathematics_of_hierarchy_and_scale_invariance_source_boundary",
        "xii_the_principle_of_fractal_self_similarity_pfss": "xii_the_principle_of_fractal_self_similarity_pfss_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown xi_the_mathematics_of_hierarchy_and_scale_invariance component"
        ) from exc


def xi_the_mathematics_of_hierarchy_and_scale_invariance_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "XI. The Mathematics of Hierarchy and Scale Invariance",
        "source_span": "P0R06057-P0R06065",
        "component_count": "2",
        "next_boundary": "P0R06066",
        "component_1": "XI. The Mathematics of Hierarchy and Scale Invariance",
        "component_2": "XII. The Principle of Fractal Self-Similarity (PFSS)",
    }


def validate_xi_the_mathematics_of_hierarchy_and_scale_invariance_fixture(
    config: XiTheMathematicsOfHierarchyAndScaleInvarianceConfig | None = None,
) -> XiTheMathematicsOfHierarchyAndScaleInvarianceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or XiTheMathematicsOfHierarchyAndScaleInvarianceConfig()
    components = (
        "xi_the_mathematics_of_hierarchy_and_scale_invariance",
        "xii_the_principle_of_fractal_self_similarity_pfss",
    )
    return XiTheMathematicsOfHierarchyAndScaleInvarianceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_xi_the_mathematics_of_hierarchy_and_scale_invariance_component(
                component
            )
            for component in components
        },
        labels=xi_the_mathematics_of_hierarchy_and_scale_invariance_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "xi_the_mathematics_of_hierarchy_and_scale_invariance_is_not_empirical_validation_evidence": 1.0,
            "xii_the_principle_of_fractal_self_similarity_pfss_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6057, 6066)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_xi_the_mathematics_of_hierarchy_and_scale_invariance_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "XiTheMathematicsOfHierarchyAndScaleInvarianceConfig",
    "XiTheMathematicsOfHierarchyAndScaleInvarianceFixtureResult",
    "classify_xi_the_mathematics_of_hierarchy_and_scale_invariance_component",
    "xi_the_mathematics_of_hierarchy_and_scale_invariance_labels",
    "validate_xi_the_mathematics_of_hierarchy_and_scale_invariance_fixture",
]
