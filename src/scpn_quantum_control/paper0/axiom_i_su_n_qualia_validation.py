# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I SU(N) qualia validation
"""Source-accounting checks for Paper 0 Axiom I SU(N) qualia records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom I SU(N) qualia-confinement map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00757", "P0R00760")


@dataclass(frozen=True, slots=True)
class AxiomISUNQualiaConfig:
    """Configuration for the Axiom I SU(N) qualia fixture."""

    expected_source_record_count: int = 4
    expected_blank_separator_count: int = 1
    next_source_boundary: str = "P0R00761"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 4:
            raise ValueError("expected_source_record_count must equal 4")
        if self.expected_blank_separator_count != 1:
            raise ValueError("expected_blank_separator_count must equal 1")
        if self.next_source_boundary != "P0R00761":
            raise ValueError("next_source_boundary must equal P0R00761")


@dataclass(frozen=True, slots=True)
class AxiomISUNQualiaFixtureResult:
    """Result for the Paper 0 Axiom I SU(N) qualia fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    blank_separator_count: int
    example_info_gluon_counts: dict[int, int]
    example_linear_potential: float
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def info_gluon_count(n_primary_qualic_dimensions: int) -> int:
    """Return the SU(N) adjoint gauge-boson count N^2 - 1."""
    if n_primary_qualic_dimensions < 2:
        raise ValueError("n_primary_qualic_dimensions must be at least 2")
    return n_primary_qualic_dimensions * n_primary_qualic_dimensions - 1


def linear_confinement_potential(*, distance: float, string_tension: float) -> float:
    """Return the source-bounded linear potential V(r) = sigma r."""
    if distance < 0.0:
        raise ValueError("distance must be non-negative")
    if string_tension < 0.0:
        raise ValueError("string_tension must be non-negative")
    return string_tension * distance


def classify_su_n_qualia_component(component: str) -> str:
    """Classify source-defined SU(N) qualia-confinement components."""
    mapping = {
        "group_extension": "su_n_primary_qualic_dimensions",
        "info_gluons": "n_squared_minus_one_self_interacting_bosons",
        "confinement": "linear_potential_qualia_confinement",
        "colored_self": "confined_macroscopic_colored_state",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown SU(N) qualia component") from exc


def axiom_i_su_n_qualia_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom I SU(N) qualia slice."""
    return {
        "section": "Extension to SU(N) Qualia Confinement",
        "confinement_formula": "V(r) approx sigma r",
        "next_boundary": "Axiom II: The Language of Information Geometry",
    }


def validate_axiom_i_su_n_qualia_fixture(
    config: AxiomISUNQualiaConfig | None = None,
) -> AxiomISUNQualiaFixtureResult:
    """Validate source accounting for the Axiom I SU(N) qualia slice."""
    cfg = config or AxiomISUNQualiaConfig()
    components = ("group_extension", "info_gluons", "confinement", "colored_self")

    return AxiomISUNQualiaFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_su_n_qualia_component(component) for component in components
        },
        labels=axiom_i_su_n_qualia_labels(),
        source_record_count=cfg.expected_source_record_count,
        blank_separator_count=cfg.expected_blank_separator_count,
        example_info_gluon_counts={
            n_primary: info_gluon_count(n_primary) for n_primary in (2, 3, 4)
        },
        example_linear_potential=linear_confinement_potential(
            distance=2.5,
            string_tension=0.4,
        ),
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "su_n_extension_is_not_empirical_confinement_evidence": 1.0,
            "linear_potential_is_source_formula_not_fitted_string_tension": 1.0,
            "macroscopic_colored_state_is_not_topology_measurement": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(757, 761)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_su_n_qualia_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomISUNQualiaConfig",
    "AxiomISUNQualiaFixtureResult",
    "axiom_i_su_n_qualia_labels",
    "classify_su_n_qualia_component",
    "info_gluon_count",
    "linear_confinement_potential",
    "validate_axiom_i_su_n_qualia_fixture",
]
