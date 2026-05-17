# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 mass eigenstates mixing-angle validation
"""Source-accounting checks for Paper 0 mass-eigenstates and mixing-angle records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded mass-eigenstates mixing-angle bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01669", "P0R01683")


@dataclass(frozen=True, slots=True)
class MassEigenstatesMixingAngleConfig:
    """Configuration for the mass-eigenstates mixing-angle fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 3
    next_source_boundary: str = "P0R01684"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R01684":
            raise ValueError("next_source_boundary must equal P0R01684")


@dataclass(frozen=True, slots=True)
class MassEigenstatesMixingAngleFixtureResult:
    """Result for the Paper 0 mass-eigenstates mixing-angle fixture."""

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


def classify_mass_eigenstates_mixing_angle_component(component: str) -> str:
    """Classify source-defined mass-eigenstates mixing-angle components."""
    mapping = {
        "mass_eigenstate_rotation": "orthogonal_mass_eigenstate_rotation_boundary",
        "lhc_invisible_decay_bound": "lhc_invisible_higgs_branching_bound_boundary",
        "perturbative_target_boundary": "perturbative_lambda_mix_search_target_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown mass-eigenstates mixing-angle component") from exc


def mass_eigenstates_mixing_angle_labels() -> dict[str, str]:
    """Return source-bounded labels for the mass-eigenstates mixing-angle slice."""
    return {
        "section": "Mass Eigenstates and the Mixing Angle (theta)",
        "rotation": "[h_SM, h_Psi]^T = R(theta) [h_bare, h_Psi,bare]^T",
        "mixing_angle": "tan(2 theta) = 2 lambda_mix v_h v_psi / (m_h_bare^2 - m_Psi_bare^2)",
        "working_bound": "sin theta lesssim 0.31",
        "next_boundary": "Phenomenology and Search Strategies at the LHC",
    }


def validate_mass_eigenstates_mixing_angle_fixture(
    config: MassEigenstatesMixingAngleConfig | None = None,
) -> MassEigenstatesMixingAngleFixtureResult:
    """Validate source accounting for the mass-eigenstates mixing-angle slice."""
    cfg = config or MassEigenstatesMixingAngleConfig()
    components = (
        "mass_eigenstate_rotation",
        "lhc_invisible_decay_bound",
        "perturbative_target_boundary",
    )

    return MassEigenstatesMixingAngleFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_mass_eigenstates_mixing_angle_component(component)
            for component in components
        },
        labels=mass_eigenstates_mixing_angle_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "rotation_formalism_is_not_measured_higgs_mixing": 1.0,
            "lhc_invisible_bound_is_constraint_not_psi_sector_detection": 1.0,
            "working_bound_is_not_model_confirmation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1669, 1684)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_mass_eigenstates_mixing_angle_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MassEigenstatesMixingAngleConfig",
    "MassEigenstatesMixingAngleFixtureResult",
    "classify_mass_eigenstates_mixing_angle_component",
    "mass_eigenstates_mixing_angle_labels",
    "validate_mass_eigenstates_mixing_angle_fixture",
]
