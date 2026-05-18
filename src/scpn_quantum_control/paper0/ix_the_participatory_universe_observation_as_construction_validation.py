# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IX. The Participatory Universe: Observation as Construction validation
"""Source-accounting checks for Paper 0 IX. The Participatory Universe: Observation as Construction records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ix the participatory universe observation as construction source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06047", "P0R06056")


@dataclass(frozen=True, slots=True)
class IxTheParticipatoryUniverseObservationAsConstructionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 2
    next_source_boundary: str = "P0R06057"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R06057":
            raise ValueError("next_source_boundary must equal P0R06057")


@dataclass(frozen=True, slots=True)
class IxTheParticipatoryUniverseObservationAsConstructionFixtureResult:
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


def classify_ix_the_participatory_universe_observation_as_construction_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "ix_the_participatory_universe_observation_as_construction": "ix_the_participatory_universe_observation_as_construction_source_boundary",
        "x_symmetry_conservation_laws_and_the_coherence_current": "x_symmetry_conservation_laws_and_the_coherence_current_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown ix_the_participatory_universe_observation_as_construction component"
        ) from exc


def ix_the_participatory_universe_observation_as_construction_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "IX. The Participatory Universe: Observation as Construction",
        "source_span": "P0R06047-P0R06056",
        "component_count": "2",
        "next_boundary": "P0R06057",
        "component_1": "IX. The Participatory Universe: Observation as Construction",
        "component_2": "X. Symmetry, Conservation Laws, and the Coherence Current",
    }


def validate_ix_the_participatory_universe_observation_as_construction_fixture(
    config: IxTheParticipatoryUniverseObservationAsConstructionConfig | None = None,
) -> IxTheParticipatoryUniverseObservationAsConstructionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IxTheParticipatoryUniverseObservationAsConstructionConfig()
    components = (
        "ix_the_participatory_universe_observation_as_construction",
        "x_symmetry_conservation_laws_and_the_coherence_current",
    )
    return IxTheParticipatoryUniverseObservationAsConstructionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ix_the_participatory_universe_observation_as_construction_component(
                component
            )
            for component in components
        },
        labels=ix_the_participatory_universe_observation_as_construction_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ix_the_participatory_universe_observation_as_construction_is_not_empirical_validation_evidence": 1.0,
            "x_symmetry_conservation_laws_and_the_coherence_current_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6047, 6057)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ix_the_participatory_universe_observation_as_construction_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IxTheParticipatoryUniverseObservationAsConstructionConfig",
    "IxTheParticipatoryUniverseObservationAsConstructionFixtureResult",
    "classify_ix_the_participatory_universe_observation_as_construction_component",
    "ix_the_participatory_universe_observation_as_construction_labels",
    "validate_ix_the_participatory_universe_observation_as_construction_fixture",
]
