# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. Propofol (GABA-A Potentiation): validation
"""Source-accounting checks for Paper 0 1. Propofol (GABA-A Potentiation): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 propofol gaba a potentiation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05102", "P0R05112")


@dataclass(frozen=True, slots=True)
class Section1PropofolGabaAPotentiationConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 4
    next_source_boundary: str = "P0R05113"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R05113":
            raise ValueError("next_source_boundary must equal P0R05113")


@dataclass(frozen=True, slots=True)
class Section1PropofolGabaAPotentiationFixtureResult:
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


def classify_section_1_propofol_gaba_a_potentiation_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_propofol_gaba_a_potentiation": "1_propofol_gaba_a_potentiation_source_boundary",
        "2_ketamine_nmda_antagonism": "2_ketamine_nmda_antagonism_source_boundary",
        "3_nsaids_acetaminophen": "3_nsaids_acetaminophen_source_boundary",
        "vi_synthesis_and_ethical_considerations_l15": "vi_synthesis_and_ethical_considerations_l15_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_1_propofol_gaba_a_potentiation component") from exc


def section_1_propofol_gaba_a_potentiation_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. Propofol (GABA-A Potentiation):",
        "source_span": "P0R05102-P0R05112",
        "component_count": "4",
        "next_boundary": "P0R05113",
        "component_1": "1. Propofol (GABA-A Potentiation):",
        "component_2": "2. Ketamine (NMDA Antagonism):",
        "component_3": "3. NSAIDs/Acetaminophen:",
        "component_4": "VI. Synthesis and Ethical Considerations (L15)",
    }


def validate_section_1_propofol_gaba_a_potentiation_fixture(
    config: Section1PropofolGabaAPotentiationConfig | None = None,
) -> Section1PropofolGabaAPotentiationFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1PropofolGabaAPotentiationConfig()
    components = (
        "1_propofol_gaba_a_potentiation",
        "2_ketamine_nmda_antagonism",
        "3_nsaids_acetaminophen",
        "vi_synthesis_and_ethical_considerations_l15",
    )
    return Section1PropofolGabaAPotentiationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_propofol_gaba_a_potentiation_component(component)
            for component in components
        },
        labels=section_1_propofol_gaba_a_potentiation_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_propofol_gaba_a_potentiation_is_not_empirical_validation_evidence": 1.0,
            "2_ketamine_nmda_antagonism_is_not_empirical_validation_evidence": 1.0,
            "3_nsaids_acetaminophen_is_not_empirical_validation_evidence": 1.0,
            "vi_synthesis_and_ethical_considerations_l15_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5102, 5113)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_propofol_gaba_a_potentiation_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1PropofolGabaAPotentiationConfig",
    "Section1PropofolGabaAPotentiationFixtureResult",
    "classify_section_1_propofol_gaba_a_potentiation_component",
    "section_1_propofol_gaba_a_potentiation_labels",
    "validate_section_1_propofol_gaba_a_potentiation_fixture",
]
