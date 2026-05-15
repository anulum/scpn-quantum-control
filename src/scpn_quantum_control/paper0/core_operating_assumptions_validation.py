# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 core operating assumptions validation
"""Source-accounting checks for Paper 0 core-operating-assumptions records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded core operating assumptions; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00635", "P0R00669")


@dataclass(frozen=True, slots=True)
class CoreOperatingAssumptionsConfig:
    """Configuration for the core-operating-assumptions fixture."""

    expected_core_assumption_count: int = 5
    expected_blank_separator_count: int = 3
    next_source_boundary: str = "P0R00670"

    def __post_init__(self) -> None:
        if self.expected_core_assumption_count != 5:
            raise ValueError("expected_core_assumption_count must equal 5")
        if self.expected_blank_separator_count != 3:
            raise ValueError("expected_blank_separator_count must equal 3")
        if self.next_source_boundary != "P0R00670":
            raise ValueError("next_source_boundary must equal P0R00670")


@dataclass(frozen=True, slots=True)
class CoreOperatingAssumptionsFixtureResult:
    """Result for the Paper 0 core-operating-assumptions fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    assumption_roles: dict[str, str]
    hint_contexts: dict[str, str]
    labels: dict[str, str]
    core_assumption_count: int
    blank_separator_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_assumption_role(assumption: str) -> str:
    """Classify one of the five source-defined operating assumptions."""
    mapping = {
        "consciousness_fundamentality": "ontological_primitive_generative_assumption",
        "bidirectional_causality": "recursive_top_down_bottom_up_causality",
        "field_realism": "physical_measurable_engineerable_field_claim",
        "unified_phase_dynamics": "universal_phase_synchronisation_language",
        "ethical_functionals": "teleological_layer15_objective_prior",
    }
    try:
        return mapping[assumption]
    except KeyError as exc:
        raise ValueError("unknown core assumption") from exc


def classify_hint_context(context: str) -> str:
    """Classify the source role of each H_int context term."""
    mapping = {
        "psi_s": "real_physical_field_context",
        "sigma": "phase_coherence_or_synchrony_candidate",
        "causality": "reciprocal_top_down_bottom_up_interaction",
        "lambda": "ethical_functional_tunes_parameter_not_force",
    }
    try:
        return mapping[context]
    except KeyError as exc:
        raise ValueError("unknown H_int context") from exc


def core_operating_assumption_labels() -> dict[str, str]:
    """Return source-bounded labels for the core-operating-assumptions slice."""
    return {
        "section": "The SCPN: Core Operating Assumptions",
        "h_int": "H_int = -lambda * Psi_s * sigma",
        "next_boundary": "Axiom I: The Primacy of Consciousness",
    }


def validate_core_operating_assumptions_fixture(
    config: CoreOperatingAssumptionsConfig | None = None,
) -> CoreOperatingAssumptionsFixtureResult:
    """Validate source accounting for the core-operating-assumptions slice."""
    cfg = config or CoreOperatingAssumptionsConfig()
    assumptions = (
        "consciousness_fundamentality",
        "bidirectional_causality",
        "field_realism",
        "unified_phase_dynamics",
        "ethical_functionals",
    )
    hint_contexts = ("psi_s", "sigma", "causality", "lambda")

    return CoreOperatingAssumptionsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        assumption_roles={
            assumption: classify_assumption_role(assumption) for assumption in assumptions
        },
        hint_contexts={context: classify_hint_context(context) for context in hint_contexts},
        labels=core_operating_assumption_labels(),
        core_assumption_count=cfg.expected_core_assumption_count,
        blank_separator_count=cfg.expected_blank_separator_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "operating_assumptions_are_not_empirical_results": 1.0,
            "field_realism_claim_requires_downstream_validation": 1.0,
            "ethical_functional_does_not_add_force_to_h_int": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(635, 670)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_assumption_map_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "CoreOperatingAssumptionsConfig",
    "CoreOperatingAssumptionsFixtureResult",
    "classify_assumption_role",
    "classify_hint_context",
    "core_operating_assumption_labels",
    "validate_core_operating_assumptions_fixture",
]
