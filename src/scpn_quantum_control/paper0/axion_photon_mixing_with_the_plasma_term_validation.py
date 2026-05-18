# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axionphoton mixing with the plasma term. validation
"""Source-accounting checks for Paper 0 Axionphoton mixing with the plasma term. records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded axion photon mixing with the plasma term source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04348", "P0R04358")


@dataclass(frozen=True, slots=True)
class AxionPhotonMixingWithThePlasmaTermConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 1
    next_source_boundary: str = "P0R04359"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R04359":
            raise ValueError("next_source_boundary must equal P0R04359")


@dataclass(frozen=True, slots=True)
class AxionPhotonMixingWithThePlasmaTermFixtureResult:
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


def classify_axion_photon_mixing_with_the_plasma_term_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "axionphoton_mixing_with_the_plasma_term": "axionphoton_mixing_with_the_plasma_term_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown axion_photon_mixing_with_the_plasma_term component") from exc


def axion_photon_mixing_with_the_plasma_term_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Axionphoton mixing with the plasma term.",
        "source_span": "P0R04348-P0R04358",
        "component_count": "1",
        "next_boundary": "P0R04359",
        "component_1": "Axionphoton mixing with the plasma term.",
    }


def validate_axion_photon_mixing_with_the_plasma_term_fixture(
    config: AxionPhotonMixingWithThePlasmaTermConfig | None = None,
) -> AxionPhotonMixingWithThePlasmaTermFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or AxionPhotonMixingWithThePlasmaTermConfig()
    components = ("axionphoton_mixing_with_the_plasma_term",)
    return AxionPhotonMixingWithThePlasmaTermFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_axion_photon_mixing_with_the_plasma_term_component(component)
            for component in components
        },
        labels=axion_photon_mixing_with_the_plasma_term_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "axionphoton_mixing_with_the_plasma_term_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4348, 4359)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axion_photon_mixing_with_the_plasma_term_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxionPhotonMixingWithThePlasmaTermConfig",
    "AxionPhotonMixingWithThePlasmaTermFixtureResult",
    "classify_axion_photon_mixing_with_the_plasma_term_component",
    "axion_photon_mixing_with_the_plasma_term_labels",
    "validate_axion_photon_mixing_with_the_plasma_term_fixture",
]
