# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 symmetry cascade validation
"""Source-accounting checks for Paper 0 symmetry-cascade records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded symmetry-cascade bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01582", "P0R01596")


@dataclass(frozen=True, slots=True)
class SymmetryCascadeConfig:
    """Configuration for the symmetry-cascade fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 4
    next_source_boundary: str = "P0R01597"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R01597":
            raise ValueError("next_source_boundary must equal P0R01597")


@dataclass(frozen=True, slots=True)
class SymmetryCascadeFixtureResult:
    """Result for the Paper 0 symmetry-cascade fixture."""

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


def classify_symmetry_cascade_component(component: str) -> str:
    """Classify source-defined symmetry-cascade components."""
    mapping = {
        "cascade_opening": "source_field_symmetry_cascade_opening_boundary",
        "three_breaks_architecture": "three_breaks_architecture_claim_boundary",
        "psi_field_potential_stability": "psi_field_potential_stable_vacuum_boundary",
        "world_interface_summary": "geometric_informational_world_interface_summary_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown symmetry-cascade component") from exc


def symmetry_cascade_labels() -> dict[str, str]:
    """Return source-bounded labels for the symmetry-cascade slice."""
    return {
        "section": "How Reality Gets Its Structure: A Cascade of Broken Symmetries",
        "three_breaks": "laws, selves, actualisation",
        "potential": "Mexican-hat stable valley",
        "interfaces": "geometric and informational interfaces",
        "next_boundary": "Predicted Particles: The Infoton and the Psi-Higgs Boson",
    }


def validate_symmetry_cascade_fixture(
    config: SymmetryCascadeConfig | None = None,
) -> SymmetryCascadeFixtureResult:
    """Validate source accounting for the symmetry-cascade slice."""
    cfg = config or SymmetryCascadeConfig()
    components = (
        "cascade_opening",
        "three_breaks_architecture",
        "psi_field_potential_stability",
        "world_interface_summary",
    )

    return SymmetryCascadeFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_symmetry_cascade_component(component) for component in components
        },
        labels=symmetry_cascade_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "balanced_pen_analogy_is_not_physical_derivation": 1.0,
            "prime_directive_law_selection_remains_source_claim_not_validation": 1.0,
            "direct_new_force_claim_rejected_for_world_interface_summary": 1.0,
            "stable_vacuum_language_is_not_measured_potential": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1582, 1597)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_symmetry_cascade_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "SymmetryCascadeConfig",
    "SymmetryCascadeFixtureResult",
    "classify_symmetry_cascade_component",
    "symmetry_cascade_labels",
    "validate_symmetry_cascade_fixture",
]
