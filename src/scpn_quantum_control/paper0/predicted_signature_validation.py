# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Predicted Signature validation
"""Source-accounting checks for Paper 0 Predicted Signature records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded predicted signature source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05171", "P0R05181")


@dataclass(frozen=True, slots=True)
class PredictedSignatureConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 1
    next_source_boundary: str = "P0R05182"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R05182":
            raise ValueError("next_source_boundary must equal P0R05182")


@dataclass(frozen=True, slots=True)
class PredictedSignatureFixtureResult:
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


def classify_predicted_signature_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {"predicted_signature": "predicted_signature_source_boundary"}
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown predicted_signature component") from exc


def predicted_signature_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Predicted Signature",
        "source_span": "P0R05171-P0R05181",
        "component_count": "1",
        "next_boundary": "P0R05182",
        "component_1": "Predicted Signature",
    }


def validate_predicted_signature_fixture(
    config: PredictedSignatureConfig | None = None,
) -> PredictedSignatureFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or PredictedSignatureConfig()
    components = ("predicted_signature",)
    return PredictedSignatureFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_predicted_signature_component(component)
            for component in components
        },
        labels=predicted_signature_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={"predicted_signature_is_not_empirical_validation_evidence": 1.0},
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5171, 5182)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_predicted_signature_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PredictedSignatureConfig",
    "PredictedSignatureFixtureResult",
    "classify_predicted_signature_component",
    "predicted_signature_labels",
    "validate_predicted_signature_fixture",
]
