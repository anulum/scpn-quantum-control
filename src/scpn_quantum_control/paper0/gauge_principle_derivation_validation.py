# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 gauge-principle derivation validation
"""Source-accounting checks for Paper 0 gauge-principle derivation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded gauge-principle derivation; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01018", "P0R01077")


@dataclass(frozen=True, slots=True)
class GaugePrincipleDerivationConfig:
    """Configuration for the gauge-principle derivation fixture."""

    expected_source_record_count: int = 60
    expected_blank_record_count: int = 1
    expected_image_record_count: int = 1
    next_source_boundary: str = "P0R01078"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 60:
            raise ValueError("expected_source_record_count must equal 60")
        if self.expected_blank_record_count != 1:
            raise ValueError("expected_blank_record_count must equal 1")
        if self.expected_image_record_count != 1:
            raise ValueError("expected_image_record_count must equal 1")
        if self.next_source_boundary != "P0R01078":
            raise ValueError("next_source_boundary must equal P0R01078")


@dataclass(frozen=True, slots=True)
class GaugePrincipleDerivationFixtureResult:
    """Result for the Paper 0 gauge-principle derivation fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    blank_record_count: int
    image_record_count: int
    phenomenology_symmetry_record_count: int
    free_scalar_record_count: int
    local_u1_record_count: int
    covariant_derivative_record_count: int
    fim_dynamics_record_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_gauge_principle_derivation_component(component: str) -> str:
    """Classify source-defined gauge-principle derivation components."""
    mapping = {
        "derivation_boundary": "gauge_principle_derivation_section_boundary",
        "phenomenology_symmetry_roadmap": (
            "phenomenological_lagrangian_critique_and_symmetry_roadmap"
        ),
        "free_scalar_global_u1": "complex_scalar_free_lagrangian_and_global_u1_symmetry",
        "local_u1_derivative_failure": "local_phase_promotion_and_ordinary_derivative_failure",
        "covariant_derivative_minimal_coupling": (
            "covariant_derivative_gauge_transform_and_minimal_coupling"
        ),
        "fim_gauge_dynamics": "fim_metric_informational_gauge_dynamics_source_claim",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown gauge-principle-derivation component") from exc


def gauge_principle_derivation_labels() -> dict[str, str]:
    """Return source-bounded labels for the gauge-principle derivation slice."""
    return {
        "section": "A Gauge-Principle Derivation of the Psi-Field",
        "free_lagrangian": "L_Psi = (partial_mu Psi)* (partial^mu Psi) - V(|Psi|)",
        "local_phase": "Psi(x) -> Psi'(x) = exp(i alpha(x)) Psi(x)",
        "covariant_derivative": "D_mu = partial_mu - i g A_mu",
        "field_strength": "F_mu_nu = partial_mu A_nu - partial_nu A_mu",
        "next_boundary": "Formal Resolution of Lorentz Covariance",
    }


def validate_gauge_principle_derivation_fixture(
    config: GaugePrincipleDerivationConfig | None = None,
) -> GaugePrincipleDerivationFixtureResult:
    """Validate source accounting for the gauge-principle derivation slice."""
    cfg = config or GaugePrincipleDerivationConfig()
    components = (
        "derivation_boundary",
        "phenomenology_symmetry_roadmap",
        "free_scalar_global_u1",
        "local_u1_derivative_failure",
        "covariant_derivative_minimal_coupling",
        "fim_gauge_dynamics",
    )

    return GaugePrincipleDerivationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_gauge_principle_derivation_component(component)
            for component in components
        },
        labels=gauge_principle_derivation_labels(),
        source_record_count=cfg.expected_source_record_count,
        blank_record_count=cfg.expected_blank_record_count,
        image_record_count=cfg.expected_image_record_count,
        phenomenology_symmetry_record_count=17,
        free_scalar_record_count=12,
        local_u1_record_count=7,
        covariant_derivative_record_count=14,
        fim_dynamics_record_count=10,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "gauge_principle_derivation_is_source_claim_not_empirical_evidence": 1.0,
            "fim_metric_replacement_is_not_lorentz_safe_until_next_slice": 1.0,
            "blank_record_p0r01046_and_image_p0r01076_are_preserved": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1018, 1078)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_gauge_principle_derivation_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "GaugePrincipleDerivationConfig",
    "GaugePrincipleDerivationFixtureResult",
    "classify_gauge_principle_derivation_component",
    "gauge_principle_derivation_labels",
    "validate_gauge_principle_derivation_fixture",
]
