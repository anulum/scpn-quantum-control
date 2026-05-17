# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential validation
"""Source-accounting checks for Paper 0 The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the intrinsic dynamics of the ψ field and the stabilising potential source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01755", "P0R01762")


@dataclass(frozen=True, slots=True)
class TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialConfig:
    """Configuration for the The Intrinsic Dynamics of the Ψ-Field and the Stabilising Potential fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R01763"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R01763":
            raise ValueError("next_source_boundary must equal P0R01763")


@dataclass(frozen=True, slots=True)
class TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialFixtureResult:
    """Result for the Paper 0 The Intrinsic Dynamics of the Ψ-Field and the Stabilising Potential fixture."""

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


def classify_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential": "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_source_boundary",
        "meta_framework_integrations": "meta_framework_integrations_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential component"
        ) from exc


def the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential",
        "source_span": "P0R01755-P0R01762",
        "component_count": "2",
        "next_boundary": "P0R01763",
        "component_1": "The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential",
        "component_2": "Meta-Framework Integrations",
    }


def validate_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_fixture(
    config: TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialConfig | None = None,
) -> TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialConfig()
    components = (
        "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
        "meta_framework_integrations",
    )
    return TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_component(
                component
            )
            for component in components
        },
        labels=the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_is_not_empirical_validation_evidence": 1.0,
            "meta_framework_integrations_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1755, 1763)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialConfig",
    "TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialFixtureResult",
    "classify_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_component",
    "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_labels",
    "validate_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_fixture",
]
