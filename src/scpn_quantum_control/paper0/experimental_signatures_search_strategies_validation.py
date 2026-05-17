# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 experimental signatures validation
"""Source-accounting checks for Paper 0 experimental-signatures search-strategy records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded experimental-signatures search-strategy bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01647", "P0R01654")


@dataclass(frozen=True, slots=True)
class ExperimentalSignaturesSearchStrategiesConfig:
    """Configuration for the experimental-signatures search-strategy fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 4
    next_source_boundary: str = "P0R01655"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R01655":
            raise ValueError("next_source_boundary must equal P0R01655")


@dataclass(frozen=True, slots=True)
class ExperimentalSignaturesSearchStrategiesFixtureResult:
    """Result for the Paper 0 experimental-signatures search-strategy fixture."""

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


def classify_experimental_signatures_search_strategies_component(component: str) -> str:
    """Classify source-defined experimental-signatures search-strategy components."""
    mapping = {
        "falsifiability_frame": "two_particle_falsifiability_search_frame_boundary",
        "collider_channel": "lhc_exotic_higgs_decay_search_channel_boundary",
        "cosmological_channel": "superradiance_continuous_wave_search_channel_boundary",
        "complementary_test_boundary": "complementary_falsifiable_hypothesis_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown experimental-signatures search-strategy component") from exc


def experimental_signatures_search_strategies_labels() -> dict[str, str]:
    """Return source-bounded labels for the experimental-signatures search-strategy slice."""
    return {
        "section": "Experimental Signatures and Search Strategies",
        "collider": "h_SM -> h_Psi h_Psi",
        "cosmology": "black-hole superradiance continuous gravitational waves",
        "detectors": "CMS, ATLAS, LISA, Einstein Telescope, Cosmic Explorer",
        "next_boundary": "The Psi-Higgs Boson: Phenomenology and Experimental Signatures at the LHC",
    }


def validate_experimental_signatures_search_strategies_fixture(
    config: ExperimentalSignaturesSearchStrategiesConfig | None = None,
) -> ExperimentalSignaturesSearchStrategiesFixtureResult:
    """Validate source accounting for the experimental-signatures search-strategy slice."""
    cfg = config or ExperimentalSignaturesSearchStrategiesConfig()
    components = (
        "falsifiability_frame",
        "collider_channel",
        "cosmological_channel",
        "complementary_test_boundary",
    )

    return ExperimentalSignaturesSearchStrategiesFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_experimental_signatures_search_strategies_component(component)
            for component in components
        },
        labels=experimental_signatures_search_strategies_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "testability_framing_is_not_detection_evidence": 1.0,
            "lhc_search_channel_is_not_observed_excess": 1.0,
            "continuous_wave_search_channel_is_not_detected_boson_cloud": 1.0,
            "complementary_hypothesis_language_is_not_confirmation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1647, 1655)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_experimental_signatures_search_strategies_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ExperimentalSignaturesSearchStrategiesConfig",
    "ExperimentalSignaturesSearchStrategiesFixtureResult",
    "classify_experimental_signatures_search_strategies_component",
    "experimental_signatures_search_strategies_labels",
    "validate_experimental_signatures_search_strategies_fixture",
]
