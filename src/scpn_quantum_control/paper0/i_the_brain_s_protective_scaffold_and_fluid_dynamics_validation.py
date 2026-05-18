# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. The Brain's Protective Scaffold and Fluid Dynamics validation
"""Source-accounting checks for Paper 0 I. The Brain's Protective Scaffold and Fluid Dynamics records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded i the brain s protective scaffold and fluid dynamics source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04858", "P0R04870")


@dataclass(frozen=True, slots=True)
class ITheBrainSProtectiveScaffoldAndFluidDynamicsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 13
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04871"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 13:
            raise ValueError("expected_source_record_count must equal 13")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04871":
            raise ValueError("next_source_boundary must equal P0R04871")


@dataclass(frozen=True, slots=True)
class ITheBrainSProtectiveScaffoldAndFluidDynamicsFixtureResult:
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


def classify_i_the_brain_s_protective_scaffold_and_fluid_dynamics_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "i_the_brain_s_protective_scaffold_and_fluid_dynamics": "i_the_brain_s_protective_scaffold_and_fluid_dynamics_source_boundary",
        "1_the_meninges_and_cranial_vault_geometric_and_immune_boundaries_l3": "1_the_meninges_and_cranial_vault_geometric_and_immune_boundaries_l3_source_boundary",
        "2_the_blood_brain_barrier_bbb_and_neurovascular_unit_nvu_l2_l3": "2_the_blood_brain_barrier_bbb_and_neurovascular_unit_nvu_l2_l3_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown i_the_brain_s_protective_scaffold_and_fluid_dynamics component"
        ) from exc


def i_the_brain_s_protective_scaffold_and_fluid_dynamics_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "I. The Brain's Protective Scaffold and Fluid Dynamics",
        "source_span": "P0R04858-P0R04870",
        "component_count": "3",
        "next_boundary": "P0R04871",
        "component_1": "I. The Brain's Protective Scaffold and Fluid Dynamics",
        "component_2": "1. The Meninges and Cranial Vault: Geometric and Immune Boundaries (L3)",
        "component_3": "2. The Blood-Brain Barrier (BBB) and Neurovascular Unit (NVU) (L2/L3)",
    }


def validate_i_the_brain_s_protective_scaffold_and_fluid_dynamics_fixture(
    config: ITheBrainSProtectiveScaffoldAndFluidDynamicsConfig | None = None,
) -> ITheBrainSProtectiveScaffoldAndFluidDynamicsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ITheBrainSProtectiveScaffoldAndFluidDynamicsConfig()
    components = (
        "i_the_brain_s_protective_scaffold_and_fluid_dynamics",
        "1_the_meninges_and_cranial_vault_geometric_and_immune_boundaries_l3",
        "2_the_blood_brain_barrier_bbb_and_neurovascular_unit_nvu_l2_l3",
    )
    return ITheBrainSProtectiveScaffoldAndFluidDynamicsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_i_the_brain_s_protective_scaffold_and_fluid_dynamics_component(
                component
            )
            for component in components
        },
        labels=i_the_brain_s_protective_scaffold_and_fluid_dynamics_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "i_the_brain_s_protective_scaffold_and_fluid_dynamics_is_not_empirical_validation_evidence": 1.0,
            "1_the_meninges_and_cranial_vault_geometric_and_immune_boundaries_l3_is_not_empirical_validation_evidence": 1.0,
            "2_the_blood_brain_barrier_bbb_and_neurovascular_unit_nvu_l2_l3_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4858, 4871)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_i_the_brain_s_protective_scaffold_and_fluid_dynamics_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ITheBrainSProtectiveScaffoldAndFluidDynamicsConfig",
    "ITheBrainSProtectiveScaffoldAndFluidDynamicsFixtureResult",
    "classify_i_the_brain_s_protective_scaffold_and_fluid_dynamics_component",
    "i_the_brain_s_protective_scaffold_and_fluid_dynamics_labels",
    "validate_i_the_brain_s_protective_scaffold_and_fluid_dynamics_fixture",
]
