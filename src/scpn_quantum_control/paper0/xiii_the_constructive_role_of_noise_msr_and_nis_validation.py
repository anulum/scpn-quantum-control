# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 XIII. The Constructive Role of Noise (MSR and NIS) validation
"""Source-accounting checks for Paper 0 XIII. The Constructive Role of Noise (MSR and NIS) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded xiii the constructive role of noise msr and nis source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06066", "P0R06087")


@dataclass(frozen=True, slots=True)
class XiiiTheConstructiveRoleOfNoiseMsrAndNisConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 22
    expected_component_count: int = 3
    next_source_boundary: str = "P0R06088"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 22:
            raise ValueError("expected_source_record_count must equal 22")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R06088":
            raise ValueError("next_source_boundary must equal P0R06088")


@dataclass(frozen=True, slots=True)
class XiiiTheConstructiveRoleOfNoiseMsrAndNisFixtureResult:
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


def classify_xiii_the_constructive_role_of_noise_msr_and_nis_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "xiii_the_constructive_role_of_noise_msr_and_nis": "xiii_the_constructive_role_of_noise_msr_and_nis_source_boundary",
        "xiv_the_physics_of_information_energy_transduction_iet": "xiv_the_physics_of_information_energy_transduction_iet_source_boundary",
        "resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst": "resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown xiii_the_constructive_role_of_noise_msr_and_nis component"
        ) from exc


def xiii_the_constructive_role_of_noise_msr_and_nis_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "XIII. The Constructive Role of Noise (MSR and NIS)",
        "source_span": "P0R06066-P0R06087",
        "component_count": "3",
        "next_boundary": "P0R06088",
        "component_1": "XIII. The Constructive Role of Noise (MSR and NIS)",
        "component_2": "XIV. The Physics of Information-Energy Transduction (IET)",
        "component_3": "Resolving the First Law Paradox: The $\\Psi$-Field as an Information Catalyst",
    }


def validate_xiii_the_constructive_role_of_noise_msr_and_nis_fixture(
    config: XiiiTheConstructiveRoleOfNoiseMsrAndNisConfig | None = None,
) -> XiiiTheConstructiveRoleOfNoiseMsrAndNisFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or XiiiTheConstructiveRoleOfNoiseMsrAndNisConfig()
    components = (
        "xiii_the_constructive_role_of_noise_msr_and_nis",
        "xiv_the_physics_of_information_energy_transduction_iet",
        "resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst",
    )
    return XiiiTheConstructiveRoleOfNoiseMsrAndNisFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_xiii_the_constructive_role_of_noise_msr_and_nis_component(
                component
            )
            for component in components
        },
        labels=xiii_the_constructive_role_of_noise_msr_and_nis_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "xiii_the_constructive_role_of_noise_msr_and_nis_is_not_empirical_validation_evidence": 1.0,
            "xiv_the_physics_of_information_energy_transduction_iet_is_not_empirical_validation_evidence": 1.0,
            "resolving_the_first_law_paradox_the_psi_field_as_an_information_catalyst_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6066, 6088)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_xiii_the_constructive_role_of_noise_msr_and_nis_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "XiiiTheConstructiveRoleOfNoiseMsrAndNisConfig",
    "XiiiTheConstructiveRoleOfNoiseMsrAndNisFixtureResult",
    "classify_xiii_the_constructive_role_of_noise_msr_and_nis_component",
    "xiii_the_constructive_role_of_noise_msr_and_nis_labels",
    "validate_xiii_the_constructive_role_of_noise_msr_and_nis_fixture",
]
