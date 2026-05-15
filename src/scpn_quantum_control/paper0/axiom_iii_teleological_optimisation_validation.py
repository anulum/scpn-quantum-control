# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom III teleological optimisation validation
"""Source-accounting checks for Paper 0 Axiom III teleological-optimisation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom III teleological-optimisation map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00791", "P0R00799")


@dataclass(frozen=True, slots=True)
class AxiomIIITeleologicalOptimisationConfig:
    """Configuration for the Axiom III teleological-optimisation fixture."""

    expected_source_record_count: int = 9
    expected_sec_maximisation_count: int = 2
    next_source_boundary: str = "P0R00800"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_sec_maximisation_count != 2:
            raise ValueError("expected_sec_maximisation_count must equal 2")
        if self.next_source_boundary != "P0R00800":
            raise ValueError("next_source_boundary must equal P0R00800")


@dataclass(frozen=True, slots=True)
class AxiomIIITeleologicalOptimisationFixtureResult:
    """Result for the Paper 0 Axiom III teleological-optimisation fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    axiom_heading_count: int
    sec_maximisation_count: int
    layer15_guidance_count: int
    directionality_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_teleological_optimisation_component(component: str) -> str:
    """Classify source-defined Axiom III teleological-optimisation components."""
    mapping = {
        "opening_context": "axiom_iii_headings_and_formal_law_pointers",
        "source_material_telos": "source_telos_maximal_sustainable_ethical_coherence",
        "directional_purpose": "axiom_iii_directional_purpose_and_sec_maximisation",
        "ethical_functional_guidance": "layer15_ethical_functionals_bias_temporal_evolution",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown teleological-optimisation component") from exc


def axiom_iii_teleological_optimisation_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom III teleological-optimisation slice."""
    return {
        "section": "Axiom III: The Drive of Teleological Optimisation",
        "telos": "maximal Sustainable Ethical Coherence",
        "architecture_layer": "Layer 15 ethical functionals",
        "next_boundary": "Formal Physical Definition: The tilde_N_t Invariance Law",
    }


def validate_axiom_iii_teleological_optimisation_fixture(
    config: AxiomIIITeleologicalOptimisationConfig | None = None,
) -> AxiomIIITeleologicalOptimisationFixtureResult:
    """Validate source accounting for the Axiom III teleological-optimisation slice."""
    cfg = config or AxiomIIITeleologicalOptimisationConfig()
    components = (
        "opening_context",
        "source_material_telos",
        "directional_purpose",
        "ethical_functional_guidance",
    )

    return AxiomIIITeleologicalOptimisationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_teleological_optimisation_component(component)
            for component in components
        },
        labels=axiom_iii_teleological_optimisation_labels(),
        source_record_count=cfg.expected_source_record_count,
        axiom_heading_count=3,
        sec_maximisation_count=cfg.expected_sec_maximisation_count,
        layer15_guidance_count=1,
        directionality_count=2,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "teleological_postulate_is_source_claim_not_empirical_evidence": 1.0,
            "ntilde_law_heading_is_pointer_not_equation_in_this_slice": 1.0,
            "layer15_ethical_functional_guidance_requires_downstream_operationalisation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(791, 800)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axiom_iii_teleological_optimisation_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIIITeleologicalOptimisationConfig",
    "AxiomIIITeleologicalOptimisationFixtureResult",
    "axiom_iii_teleological_optimisation_labels",
    "classify_teleological_optimisation_component",
    "validate_axiom_iii_teleological_optimisation_fixture",
]
