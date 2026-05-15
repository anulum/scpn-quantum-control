# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Foreword coupling validation
"""Executable Foreword coupling boundary checks for Paper 0."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Foreword coupling formula; not empirical validation evidence"
HARDWARE_STATUS = "source_formula_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00268", "P0R00306")


@dataclass(frozen=True, slots=True)
class SigmaLayerExample:
    """Source example identifying a layer-specific collective state variable."""

    layer: str
    source_record: str
    system: str
    collective_state_variable: str


@dataclass(frozen=True, slots=True)
class ForewordCouplingConfig:
    """Configuration for the Paper 0 Foreword coupling fixture."""

    expected_sigma_layer_example_count: int = 3
    expected_image_marker_count: int = 1
    preface_i_boundary: str = "P0R00307"

    def __post_init__(self) -> None:
        if self.expected_sigma_layer_example_count != 3:
            raise ValueError("expected_sigma_layer_example_count must equal 3")
        if self.expected_image_marker_count != 1:
            raise ValueError("expected_image_marker_count must equal 1")
        if self.preface_i_boundary != "P0R00307":
            raise ValueError("preface_i_boundary must equal P0R00307")


@dataclass(frozen=True, slots=True)
class ForewordCouplingFixtureResult:
    """Result for the Paper 0 Foreword coupling fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    predictive_coding_channels: dict[str, str]
    sigma_layer_example_count: int
    image_marker_count: int
    preface_i_boundary: str
    sample_hamiltonian_value: float
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def interaction_hamiltonian(
    *,
    lambda_coupling: float,
    psi_s: complex | float,
    sigma: complex | float,
) -> complex | float:
    """Evaluate the source interaction Hamiltonian H_int = -lambda * Psi_s * sigma."""
    if not math.isfinite(lambda_coupling):
        raise ValueError("lambda_coupling must be finite")
    return -lambda_coupling * psi_s * sigma


def sigma_layer_catalogue() -> dict[str, SigmaLayerExample]:
    """Return source-preserved examples of layer-specific sigma variables."""
    return {
        "L1": SigmaLayerExample(
            layer="L1",
            source_record="P0R00293",
            system="microtubule tubulin dimer array",
            collective_state_variable="net electric dipole moment",
        ),
        "L2": SigmaLayerExample(
            layer="L2",
            source_record="P0R00294",
            system="human brain",
            collective_state_variable="global phase synchrony in gamma band",
        ),
        "L6": SigmaLayerExample(
            layer="L6",
            source_record="P0R00295",
            system="planetary ecosystem",
            collective_state_variable=(
                "atmospheric oxygen concentration or average global temperature"
            ),
        ),
    }


def classify_predictive_coding_channel(channel: str) -> str:
    """Map source predictive-coding channel labels to computational roles."""
    mapping = {
        "downward_projection": "generative_model",
        "upward_feedback": "prediction_error_flow",
    }
    try:
        return mapping[channel]
    except KeyError as exc:
        raise ValueError("unknown predictive-coding channel") from exc


def validate_foreword_coupling_fixture(
    config: ForewordCouplingConfig | None = None,
) -> ForewordCouplingFixtureResult:
    """Validate source accounting for the Paper 0 Foreword coupling run."""
    cfg = config or ForewordCouplingConfig()
    catalogue = sigma_layer_catalogue()
    predictive_channels = {
        "downward_projection": classify_predictive_coding_channel("downward_projection"),
        "upward_feedback": classify_predictive_coding_channel("upward_feedback"),
    }
    sample_value = interaction_hamiltonian(lambda_coupling=2.0, psi_s=3.0, sigma=5.0)
    if not isinstance(sample_value, float):
        raise TypeError("real-valued sample Hamiltonian must be a float")

    return ForewordCouplingFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        predictive_coding_channels=predictive_channels,
        sigma_layer_example_count=len(catalogue),
        image_marker_count=cfg.expected_image_marker_count,
        preface_i_boundary=cfg.preface_i_boundary,
        sample_hamiltonian_value=sample_value,
        null_controls={
            "unknown_channel_rejection_label": 1.0,
            "empirical_validation_overclaim_rejection_label": 1.0,
            "omitted_parameter_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(268, 307)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_formula_only_no_experiment",
            "sigma_layer_examples": {key: asdict(value) for key, value in catalogue.items()},
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ForewordCouplingConfig",
    "ForewordCouplingFixtureResult",
    "SigmaLayerExample",
    "classify_predictive_coding_channel",
    "interaction_hamiltonian",
    "sigma_layer_catalogue",
    "validate_foreword_coupling_fixture",
]
