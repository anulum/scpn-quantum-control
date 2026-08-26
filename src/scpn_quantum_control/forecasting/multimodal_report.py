# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — multimodal-forecasting multimodal forecasting evidence reporting
"""Deterministic JSON and Markdown evidence for the bounded multimodal-forecasting product."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from .multimodal_bridge import (
    ForecastActiveSensingBridge,
    ForecastControllerInitialisation,
)
from .multimodal_forecaster import ForecastAccuracyCertificate, MultimodalRidgeForecaster
from .partial_observation import PartialObservationBatchCertificate
from .synthetic_multimodal import SyntheticMultimodalDataset
from .uncertainty import IntervalCoverageCertificate, ResidualIntervalCalibrator

MULTIMODAL_EVIDENCE_SCHEMA = "scpn.multimodal_forecasting.v1"
MULTIMODAL_EVIDENCE_BOUNDARY = (
    "Deterministic synthetic Kuramoto trajectory evidence under explicit simulation-only "
    "domain tags. No real EEG, clinical, grid, SCADA, plasma, plant, hardware, QPU, "
    "state-estimation, control-performance, safety, deployment, or publication claim."
)
_NUMERIC_CUSTODY_DECIMALS = 12


def _canonicalise_evidence_numbers(value: object) -> object:
    """Normalise sub-precision runtime drift before evidence serialisation."""
    if isinstance(value, float):
        rounded = round(value, _NUMERIC_CUSTODY_DECIMALS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {str(key): _canonicalise_evidence_numbers(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalise_evidence_numbers(child) for child in value]
    return value


@dataclass(frozen=True, slots=True)
class MultimodalSupportRow:
    """One executable or explicitly blocked forecasting support-matrix row."""

    surface: str
    status: Literal["synthetic_supported", "bounded_supported", "blocked_dependency"]
    evidence: str
    boundary: str

    def __post_init__(self) -> None:
        """Require complete support metadata."""
        if not all(value.strip() for value in (self.surface, self.evidence, self.boundary)):
            raise ValueError("support rows require non-empty surface, evidence, and boundary")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready support row."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MultimodalForecastingEvidence:
    """Complete deterministic multimodal-forecasting evidence bundle."""

    dataset: SyntheticMultimodalDataset
    model: MultimodalRidgeForecaster
    calibration_accuracy: ForecastAccuracyCertificate
    test_accuracy: ForecastAccuracyCertificate
    partial_observation: PartialObservationBatchCertificate
    calibrator: ResidualIntervalCalibrator
    interval_coverage: IntervalCoverageCertificate
    active_sensing: ForecastActiveSensingBridge
    controller_initialisation: ForecastControllerInitialisation
    support_rows: tuple[MultimodalSupportRow, ...]
    schema: str = MULTIMODAL_EVIDENCE_SCHEMA
    claim_boundary: str = MULTIMODAL_EVIDENCE_BOUNDARY

    def __post_init__(self) -> None:
        """Require exact digest custody and the complete bounded support surface."""
        if self.model.training_batch_digest != self.dataset.train.content_digest():
            raise ValueError("model must be bound to the evidence training batch")
        if self.calibration_accuracy.batch_digest != self.dataset.calibration.content_digest():
            raise ValueError("calibration accuracy must be bound to the calibration batch")
        if self.test_accuracy.batch_digest != self.dataset.test.content_digest():
            raise ValueError("test accuracy must be bound to the test batch")
        if self.partial_observation.batch_digest != self.dataset.test.content_digest():
            raise ValueError("partial-observation certificate must use the test batch")
        model_digests = {
            self.model.model_digest,
            self.calibration_accuracy.model_digest,
            self.test_accuracy.model_digest,
            self.partial_observation.forecast_model_digest,
            self.calibrator.model_digest,
            self.interval_coverage.model_digest,
            self.active_sensing.model_digest,
            self.controller_initialisation.model_digest,
        }
        if len(model_digests) != 1:
            raise ValueError("all forecast evidence must share one model digest")
        calibrator_digests = {
            self.calibrator.calibrator_digest,
            self.interval_coverage.calibrator_digest,
            self.active_sensing.calibrator_digest,
            self.controller_initialisation.calibrator_digest,
        }
        if len(calibrator_digests) != 1:
            raise ValueError("all interval evidence must share one calibrator digest")
        required_surfaces = {
            "synthetic_multimodal_schema",
            "missingness_aware_ridge",
            "partial_observation_objective",
            "split_residual_intervals",
            "active_sensing_bridge",
            "codesign_controller_initialisation",
            "real_eeg_clinical_data",
            "real_grid_scada_data",
            "real_plasma_plant_data",
            "hardware_qpu_execution",
        }
        if {row.surface for row in self.support_rows} != required_surfaces:
            raise ValueError("support rows must cover the complete bounded forecasting surface")

    def to_dict(self) -> dict[str, object]:
        """Return a canonical digest-bound evidence mapping."""
        payload: dict[str, object] = {
            "schema": self.schema,
            "dataset": self.dataset.to_summary_dict(),
            "model": {
                "kind": "missingness_aware_linear_ridge",
                "model_digest": self.model.model_digest,
                "training_batch_digest": self.model.training_batch_digest,
                "ridge": self.model.ridge,
                "history_steps": self.model.history_steps,
                "horizon_steps": self.model.horizon_steps,
                "nodes": self.model.n_nodes,
                "event_channels": self.model.event_channels,
            },
            "calibration_accuracy": self.calibration_accuracy.to_dict(),
            "test_accuracy": self.test_accuracy.to_dict(),
            "partial_observation": self.partial_observation.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "interval_coverage": self.interval_coverage.to_dict(),
            "active_sensing": self.active_sensing.to_dict(),
            "controller_initialisation": self.controller_initialisation.to_dict(),
            "support_rows": [row.to_dict() for row in self.support_rows],
            "primary_sources": [
                {
                    "url": "https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html",
                    "role": "missing-data time-series forecasting context",
                },
                {
                    "doi": "10.1109/TPAMI.2023.3272339",
                    "role": "time-series predictive uncertainty context",
                },
                {
                    "url": "https://arxiv.org/abs/2309.03545",
                    "role": "dynamics learning from partial observations context",
                },
                {
                    "doi": "10.1073/pnas.1212134110",
                    "role": "Kuramoto network synchronization context",
                },
            ],
            "claim_boundary": self.claim_boundary,
        }
        payload = cast(dict[str, object], _canonicalise_evidence_numbers(payload))
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["content_digest"] = hashlib.sha256(canonical).hexdigest()
        return payload


def render_multimodal_forecasting_markdown(evidence: MultimodalForecastingEvidence) -> str:
    """Render a human-readable view of deterministic forecasting evidence."""
    payload = evidence.to_dict()
    test = evidence.test_accuracy
    coverage = evidence.interval_coverage
    partial = evidence.partial_observation
    lines = [
        "# Multimodal Forecasting Evidence",
        "",
        f"Schema: `{evidence.schema}`",
        f"Content digest: `{payload['content_digest']}`",
        "",
        "## Custody and held-out point forecast",
        "",
        f"- Dataset digest: `{evidence.dataset.content_digest()}`; train / calibration / test "
        f"samples: `{evidence.dataset.train.n_samples}` / "
        f"`{evidence.dataset.calibration.n_samples}` / `{evidence.dataset.test.n_samples}`.",
        f"- Model digest: `{evidence.model.model_digest}`; test wrapped MSE "
        f"`{test.wrapped_mse:.9g}` versus persistence `{test.persistence_wrapped_mse:.9g}`; "
        f"lower MSE: `{test.lower_mse_than_persistence}`.",
        "",
        "| Synthetic tag | Samples | Forecast MSE | Persistence MSE | Lower MSE |",
        "|---|---:|---:|---:|---|",
    ]
    for domain_row in test.domains:
        lines.append(
            f"| `{domain_row.domain_tag.value}` | {domain_row.samples} | "
            f"{domain_row.wrapped_mse:.9g} | "
            f"{domain_row.persistence_wrapped_mse:.9g} | "
            f"`{domain_row.lower_mse_than_persistence}` |"
        )
    lines.extend(
        [
            "",
            "## Partial observation and uncertainty",
            "",
            f"- Partial target fraction: `{partial.observed_fraction:.9g}`; observed wrapped "
            f"RMSE `{partial.mean_observed_wrapped_rmse:.9g}`; exact-simulator Kuramoto "
            f"residual RMSE `{partial.mean_kuramoto_residual_rmse:.9g}`.",
            f"- Split residual radius: `{evidence.calibrator.radius:.9g}` at target coverage "
            f"`{coverage.target_coverage:.9g}`; empirical sample coverage "
            f"`{coverage.sample_coverage:.9g}` and value coverage `{coverage.value_coverage:.9g}`.",
            "",
            "## Composition ports",
            "",
            f"- Active-sensing plan allowed: `{evidence.active_sensing.plan.allowed}`; "
            "hardware execution: "
            f"`{evidence.active_sensing.hardware_execution}`.",
            f"- Controller proposal applied: `{evidence.controller_initialisation.applied}`; "
            "safety "
            f"decision: `{evidence.controller_initialisation.safety_decision}`.",
            "",
            "## Support matrix",
            "",
            "| Surface | Status | Evidence / boundary |",
            "|---|---|---|",
        ]
    )
    for support_row in evidence.support_rows:
        lines.append(
            f"| `{support_row.surface}` | `{support_row.status}` | "
            f"{support_row.evidence} {support_row.boundary} |"
        )
    lines.extend(["", "## Claim boundary", "", evidence.claim_boundary, ""])
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 evidence file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_multimodal_forecasting_evidence(
    evidence: MultimodalForecastingEvidence,
    *,
    json_path: Path,
    markdown_path: Path,
) -> tuple[str, str]:
    """Write deterministic JSON and Markdown evidence and return file digests."""
    json_text = json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n"
    markdown_text = render_multimodal_forecasting_markdown(evidence)
    _atomic_write(json_path, json_text)
    _atomic_write(markdown_path, markdown_text)
    return (
        hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
        hashlib.sha256(markdown_text.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "MULTIMODAL_EVIDENCE_BOUNDARY",
    "MULTIMODAL_EVIDENCE_SCHEMA",
    "MultimodalForecastingEvidence",
    "MultimodalSupportRow",
    "render_multimodal_forecasting_markdown",
    "write_multimodal_forecasting_evidence",
]
