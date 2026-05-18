# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia validation
"""Source-accounting checks for Paper 0 The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the physical basis of the ethical functional causal entropy and computab source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04115", "P0R04122")


@dataclass(frozen=True, slots=True)
class ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04123"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04123":
            raise ValueError("next_source_boundary must equal P0R04123")


@dataclass(frozen=True, slots=True)
class ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabFixtureResult:
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


def classify_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab": "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_source_boundary",
        "meta_framework_integrations": "meta_framework_integrations_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab component"
        ) from exc


def the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia",
        "source_span": "P0R04115-P0R04122",
        "component_count": "2",
        "next_boundary": "P0R04123",
        "component_1": "The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia",
        "component_2": "Meta-Framework Integrations",
    }


def validate_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_fixture(
    config: ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabConfig | None = None,
) -> ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabConfig()
    components = (
        "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
        "meta_framework_integrations",
    )
    return ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_component(
                component
            )
            for component in components
        },
        labels=the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_is_not_empirical_validation_evidence": 1.0,
            "meta_framework_integrations_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4115, 4123)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabConfig",
    "ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabFixtureResult",
    "classify_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_component",
    "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_labels",
    "validate_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_fixture",
]
