# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 layer monograph suite validation
"""Executable source-accounting checks for the Paper 0 layer suite map."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded layer monograph suite map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00436", "P0R00463")


@dataclass(frozen=True, slots=True)
class LayerMonographSuiteConfig:
    """Configuration for the layer monograph suite fixture."""

    expected_blank_separator_count: int = 1
    next_source_boundary: str = "P0R00464"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count != 1:
            raise ValueError("expected_blank_separator_count must equal 1")
        if self.next_source_boundary != "P0R00464":
            raise ValueError("next_source_boundary must equal P0R00464")


@dataclass(frozen=True, slots=True)
class LayerMonographSuiteFixtureResult:
    """Result for the Paper 0 layer monograph suite fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    layer_domains: dict[int, str]
    layer_publications: dict[int, str]
    validation_suite_roles: dict[int, str]
    layer_monograph_count: int
    domain_series_count: int
    validation_suite_paper_count: int
    blank_separator_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_layer_domain(layer: int) -> str:
    """Classify a layer number into the source series/domain boundary."""
    if not 1 <= layer <= 16:
        raise ValueError("layer must be in the closed interval 1..16")
    if 1 <= layer <= 4:
        return "domain_i_biological_substrate"
    if 5 <= layer <= 8:
        return "domain_ii_organismal_planetary"
    if 9 <= layer <= 12:
        return "domain_iii_iv_memory_control_collective"
    if 13 <= layer <= 15:
        return "domain_v_meta_universal"
    return "domain_vi_cybernetic_closure"


def classify_validation_suite_paper(paper: int) -> str:
    """Classify a Part III paper into its validation-suite role."""
    mapping = {
        17: "methodological_experimental_blueprint",
        18: "unified_simulation_architecture",
        19: "critical_dialogue_falsifiability_roadmap",
        20: "philosophical_capstone",
    }
    try:
        return mapping[paper]
    except KeyError as exc:
        raise ValueError("validation-suite paper must be in the closed interval 17..20") from exc


def layer_publication_catalogue() -> dict[int, str]:
    """Return the source-bounded layer publication catalogue."""
    return {
        1: "Quantum Biological",
        2: "Neurochemical-Neurological",
        3: "Genomic-Epigenomic-Morphogenetic",
        4: "Cellular-Tissue Synchronisation",
        5: "Organismal-Psychoemotional Feedback",
        6: "Planetary-Biospheric",
        7: "Geometrical-Symbolic",
        8: "Cosmic Phase-Locking",
        9: "Memory Imprint-Existential Holograph",
        10: "Projective Field Boundary Control",
        11: "Noospheric-Cultural-Informational",
        12: "Ecological-Gaian Synchrony",
        13: "Source-Field / Meta-Universal",
        14: "Transdimensional Resonance",
        15: "Consilium / Oversoul Integrator",
        16: "Meta-Layer 16",
    }


def validate_layer_monograph_suite_fixture(
    config: LayerMonographSuiteConfig | None = None,
) -> LayerMonographSuiteFixtureResult:
    """Validate source accounting for the layer monograph suite run."""
    cfg = config or LayerMonographSuiteConfig()
    layer_publications = layer_publication_catalogue()
    layer_domains = {layer: classify_layer_domain(layer) for layer in layer_publications}
    validation_suite_roles = {
        paper: classify_validation_suite_paper(paper) for paper in range(17, 21)
    }

    return LayerMonographSuiteFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        layer_domains=layer_domains,
        layer_publications=layer_publications,
        validation_suite_roles=validation_suite_roles,
        layer_monograph_count=len(layer_publications),
        domain_series_count=len(set(layer_domains.values())),
        validation_suite_paper_count=len(validation_suite_roles),
        blank_separator_count=cfg.expected_blank_separator_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "publication_map_is_not_validation_evidence": 1.0,
            "unmapped_layer_rejection_label": 1.0,
            "unmapped_validation_suite_role_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(436, 464)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_publication_map_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "LayerMonographSuiteConfig",
    "LayerMonographSuiteFixtureResult",
    "classify_layer_domain",
    "classify_validation_suite_paper",
    "layer_publication_catalogue",
    "validate_layer_monograph_suite_fixture",
]
