# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 U1 FIM multiscale dynamics validation
"""Source-accounting checks for U(1)/FIM and multiscale dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded U(1)/FIM multiscale dynamics; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00506", "P0R00544")


@dataclass(frozen=True, slots=True)
class U1FIMMultiscaleDynamicsConfig:
    """Configuration for the U(1)/FIM multiscale-dynamics fixture."""

    expected_blank_separator_count: int = 1
    next_source_boundary: str = "P0R00545"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count != 1:
            raise ValueError("expected_blank_separator_count must equal 1")
        if self.next_source_boundary != "P0R00545":
            raise ValueError("next_source_boundary must equal P0R00545")


@dataclass(frozen=True, slots=True)
class U1FIMMultiscaleDynamicsFixtureResult:
    """Result for the Paper 0 U(1)/FIM multiscale-dynamics fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    covariant_derivative: str
    upde_components: dict[str, str]
    validation_boundaries: dict[str, str]
    upde_component_count: int
    validation_boundary_count: int
    blank_separator_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def covariant_derivative_formula() -> str:
    """Return the source-bounded covariant derivative expression."""
    return "D_mu = partial_mu - i g A_mu"


def classify_upde_component(component: str) -> str:
    """Classify source UPDE components into their role labels."""
    mapping = {
        "intrinsic_dynamics": "omega_i_layer_timescale",
        "intra_layer_coupling": "kij_layer_synchronisation",
        "inter_layer_coupling": "cross_layer_causal_flow",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown UPDE component") from exc


def classify_validation_boundary(boundary: str) -> str:
    """Classify source claims into validation-boundary roles."""
    mapping = {
        "fim_geometry": "information_geometry_claim",
        "ms_qec": "requires_quantitative_biophysical_validation",
        "sfh_analogue": "conceptual_convergence_not_independent_proof",
    }
    try:
        return mapping[boundary]
    except KeyError as exc:
        raise ValueError("unknown validation boundary") from exc


def validate_u1_fim_multiscale_dynamics_fixture(
    config: U1FIMMultiscaleDynamicsConfig | None = None,
) -> U1FIMMultiscaleDynamicsFixtureResult:
    """Validate source accounting for the U(1)/FIM multiscale-dynamics run."""
    cfg = config or U1FIMMultiscaleDynamicsConfig()
    upde_keys = ("intrinsic_dynamics", "intra_layer_coupling", "inter_layer_coupling")
    boundary_keys = ("fim_geometry", "ms_qec", "sfh_analogue")
    upde_components = {key: classify_upde_component(key) for key in upde_keys}
    validation_boundaries = {key: classify_validation_boundary(key) for key in boundary_keys}

    return U1FIMMultiscaleDynamicsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        covariant_derivative=covariant_derivative_formula(),
        upde_components=upde_components,
        validation_boundaries=validation_boundaries,
        upde_component_count=len(upde_components),
        validation_boundary_count=len(validation_boundaries),
        blank_separator_count=cfg.expected_blank_separator_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "informational_lagrangian_is_not_empirical_validation": 1.0,
            "sentience_field_convergence_is_not_independent_proof": 1.0,
            "ms_qec_energy_gap_requires_external_validation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(506, 545)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_equation_and_dynamics_map_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "U1FIMMultiscaleDynamicsConfig",
    "U1FIMMultiscaleDynamicsFixtureResult",
    "classify_upde_component",
    "classify_validation_boundary",
    "covariant_derivative_formula",
    "validate_u1_fim_multiscale_dynamics_fixture",
]
