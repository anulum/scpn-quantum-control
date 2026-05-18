# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Neuroendocrine Regulation and the Hypothalamic-Pituitary-Adrenal (HPA) Axis validation
"""Source-accounting checks for Paper 0 Neuroendocrine Regulation and the Hypothalamic-Pituitary-Adrenal (HPA) Axis records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded neuroendocrine regulation and the hypothalamic pituitary adrenal hpa axi source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05445", "P0R05454")


@dataclass(frozen=True, slots=True)
class NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 1
    next_source_boundary: str = "P0R05455"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R05455":
            raise ValueError("next_source_boundary must equal P0R05455")


@dataclass(frozen=True, slots=True)
class NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiFixtureResult:
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


def classify_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi": "neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi component"
        ) from exc


def neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Neuroendocrine Regulation and the Hypothalamic-Pituitary-Adrenal (HPA) Axis",
        "source_span": "P0R05445-P0R05454",
        "component_count": "1",
        "next_boundary": "P0R05455",
        "component_1": "Neuroendocrine Regulation and the Hypothalamic-Pituitary-Adrenal (HPA) Axis",
    }


def validate_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_fixture(
    config: NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiConfig | None = None,
) -> NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiConfig()
    components = ("neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi",)
    return NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_component(
                component
            )
            for component in components
        },
        labels=neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5445, 5455)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiConfig",
    "NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiFixtureResult",
    "classify_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_component",
    "neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_labels",
    "validate_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_fixture",
]
