# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VI. The Unified Geometric Principle (UGP) validation
"""Source-accounting checks for Paper 0 VI. The Unified Geometric Principle (UGP) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded vi the unified geometric principle ugp source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06039", "P0R06046")


@dataclass(frozen=True, slots=True)
class ViTheUnifiedGeometricPrincipleUgpConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R06047"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R06047":
            raise ValueError("next_source_boundary must equal P0R06047")


@dataclass(frozen=True, slots=True)
class ViTheUnifiedGeometricPrincipleUgpFixtureResult:
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


def classify_vi_the_unified_geometric_principle_ugp_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "vi_the_unified_geometric_principle_ugp": "vi_the_unified_geometric_principle_ugp_source_boundary",
        "vii_symmetry_principles_preservation_and_breaking": "vii_symmetry_principles_preservation_and_breaking_source_boundary",
        "viii_energetics_and_metabolism_of_the_scpn": "viii_energetics_and_metabolism_of_the_scpn_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown vi_the_unified_geometric_principle_ugp component") from exc


def vi_the_unified_geometric_principle_ugp_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "VI. The Unified Geometric Principle (UGP)",
        "source_span": "P0R06039-P0R06046",
        "component_count": "3",
        "next_boundary": "P0R06047",
        "component_1": "VI. The Unified Geometric Principle (UGP)",
        "component_2": "VII. Symmetry Principles: Preservation and Breaking",
        "component_3": "VIII. Energetics and Metabolism of the SCPN",
    }


def validate_vi_the_unified_geometric_principle_ugp_fixture(
    config: ViTheUnifiedGeometricPrincipleUgpConfig | None = None,
) -> ViTheUnifiedGeometricPrincipleUgpFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ViTheUnifiedGeometricPrincipleUgpConfig()
    components = (
        "vi_the_unified_geometric_principle_ugp",
        "vii_symmetry_principles_preservation_and_breaking",
        "viii_energetics_and_metabolism_of_the_scpn",
    )
    return ViTheUnifiedGeometricPrincipleUgpFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_vi_the_unified_geometric_principle_ugp_component(component)
            for component in components
        },
        labels=vi_the_unified_geometric_principle_ugp_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "vi_the_unified_geometric_principle_ugp_is_not_empirical_validation_evidence": 1.0,
            "vii_symmetry_principles_preservation_and_breaking_is_not_empirical_validation_evidence": 1.0,
            "viii_energetics_and_metabolism_of_the_scpn_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6039, 6047)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_vi_the_unified_geometric_principle_ugp_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ViTheUnifiedGeometricPrincipleUgpConfig",
    "ViTheUnifiedGeometricPrincipleUgpFixtureResult",
    "classify_vi_the_unified_geometric_principle_ugp_component",
    "vi_the_unified_geometric_principle_ugp_labels",
    "validate_vi_the_unified_geometric_principle_ugp_fixture",
]
