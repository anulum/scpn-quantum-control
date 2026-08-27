# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — neural-operator baseline composition product
"""Compose existing neural-operator baselines under fail-closed governance.

The product verifies the committed runner artifact, labels training and
inference costs separately, binds advantage-language's default no-advantage certificate,
admits only public or explicitly synthetic forecast datasets, and records the
honest quantum-sync-oracle/multimodal-forecasting wiring disposition. It does not train a second model or turn
an arithmetic crossover into a quantum-advantage claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

from .advantage_language_protocol import NoAdvantageCertificate, issue_no_advantage_certificate
from .forecasting.neural_operator_advantage import SCHEMA as ADVANTAGE_ARTIFACT_SCHEMA
from .forecasting.neural_operator_cost_model import build_cost_model
from .forecasting.real_data_sync import (
    SynchronisationForecastDataset,
    load_hardware_kuramoto_4osc_trace,
    load_ieee5bus_sync_forecast_case,
)

SupportPosture = Literal["supported", "research", "boundary"]
DataClassification = Literal["public_measurement", "public_replay", "synthetic", "refused"]
IntegrationStatus = Literal["wired", "descoped_fail_closed", "design_dependency"]

NEURAL_OPERATOR_BASELINE_PRODUCT_SCHEMA: Final[str] = "neural_operator_baseline_product.v2"
NEURAL_OPERATOR_BASELINE_CLAIM_BOUNDARY: Final[str] = (
    "Classical forecast-baseline composition only: verified committed artifact, separate "
    "training/inference estimates, governed no-advantage default, and public-or-synthetic "
    "data admission; no quantum advantage, private-data acceptance, hardware forecast, "
    "challenge-oracle rank, or completed multimodal forecasting claim"
)


@dataclass(frozen=True, slots=True)
class BaselineSurfaceRow:
    """One frozen ambient neural-operator or forecasting surface."""

    surface_id: str
    authority_pointer: str
    support_posture: SupportPosture
    summary: str

    def __post_init__(self) -> None:
        """Validate inventory fields."""
        if (
            not self.surface_id.strip()
            or not self.authority_pointer.strip()
            or not self.summary.strip()
        ):
            raise ValueError("surface rows require non-empty identity, authority, and summary")
        if self.support_posture not in {"supported", "research", "boundary"}:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready inventory row."""
        return {
            "surface_id": self.surface_id,
            "authority_pointer": self.authority_pointer,
            "support_posture": self.support_posture,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class ArtifactVerification:
    """Verification result for the committed neural-operator evidence artifact."""

    artifact_path: str
    valid: bool
    errors: tuple[str, ...]
    payload_sha256: str

    def __post_init__(self) -> None:
        """Validate verification consistency."""
        if self.valid == bool(self.errors):
            raise ValueError("valid must be true exactly when errors is empty")
        if self.payload_sha256 and len(self.payload_sha256) != 64:
            raise ValueError("payload_sha256 must be empty or a SHA-256 hex digest")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready verification result."""
        return {
            "artifact_path": self.artifact_path,
            "valid": self.valid,
            "errors": list(self.errors),
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class DatasetAdmission:
    """Fail-closed privacy and provenance decision for one forecast dataset."""

    dataset_name: str
    source_kind: str
    data_classification: DataClassification
    allowed: bool
    reason: str
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate admission invariants."""
        if (
            not self.dataset_name.strip()
            or not self.source_kind.strip()
            or not self.reason.strip()
        ):
            raise ValueError("dataset admission fields must be non-empty")
        if self.allowed == bool(self.blockers):
            raise ValueError("allowed admissions must have no blockers; refusals require blockers")
        if self.allowed and self.data_classification == "refused":
            raise ValueError("allowed datasets cannot use refused classification")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready admission decision."""
        return {
            "dataset_name": self.dataset_name,
            "source_kind": self.source_kind,
            "data_classification": self.data_classification,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class IntegrationDisposition:
    """Honest downstream product integration disposition."""

    target: Literal["quantum_sync_oracle", "multimodal_forecasting"]
    status: IntegrationStatus
    reason: str

    def __post_init__(self) -> None:
        """Validate integration disposition."""
        if (
            self.target not in {"quantum_sync_oracle", "multimodal_forecasting"}
            or not self.reason.strip()
        ):
            raise ValueError("integration disposition requires a known target and reason")
        if self.status not in {"wired", "descoped_fail_closed", "design_dependency"}:
            raise ValueError(f"unknown integration status: {self.status!r}")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready integration row."""
        return {"target": self.target, "status": self.status, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class NeuralOperatorBaselineProduct:
    """Complete bounded classical forecast-baseline report."""

    schema: str
    surfaces: tuple[BaselineSurfaceRow, ...]
    artifact: ArtifactVerification
    no_advantage: NoAdvantageCertificate
    cost_labels: dict[str, str]
    datasets: tuple[DatasetAdmission, ...]
    integrations: tuple[IntegrationDisposition, ...]
    claim_boundary: str = NEURAL_OPERATOR_BASELINE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate product completeness."""
        if self.schema != NEURAL_OPERATOR_BASELINE_PRODUCT_SCHEMA:
            raise ValueError(f"unknown product schema: {self.schema!r}")
        if len(self.surfaces) != 4 or len(self.datasets) < 2 or len(self.integrations) != 2:
            raise ValueError("neural-operator baseline product inventory is incomplete")
        if self.no_advantage.language_status != "no_advantage_default":
            raise ValueError("neural-operator baseline must retain no-advantage default")
        if self.claim_boundary != NEURAL_OPERATOR_BASELINE_CLAIM_BOUNDARY:
            raise ValueError("neural-operator baseline claim boundary has drifted")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready product report."""
        return {
            "schema": self.schema,
            "surfaces": [row.to_dict() for row in self.surfaces],
            "artifact": self.artifact.to_dict(),
            "no_advantage": self.no_advantage.to_dict(),
            "cost_labels": dict(self.cost_labels),
            "datasets": [row.to_dict() for row in self.datasets],
            "integrations": [row.to_dict() for row in self.integrations],
            "claim_boundary": self.claim_boundary,
        }


_SURFACES: Final[tuple[BaselineSurfaceRow, ...]] = (
    BaselineSurfaceRow(
        "deeponet_surrogate",
        "scpn_quantum_control.forecasting.kuramoto_neural_operator",
        "research",
        "Optional PyTorch DeepONet training and forecast surface.",
    ),
    BaselineSurfaceRow(
        "advantage_study",
        "scpn_quantum_control.forecasting.neural_operator_advantage",
        "research",
        "Held-out persistence comparison with production claims disabled.",
    ),
    BaselineSurfaceRow(
        "cost_model",
        "scpn_quantum_control.forecasting.neural_operator_cost_model",
        "supported",
        "Host-independent training and per-query arithmetic estimates.",
    ),
    BaselineSurfaceRow(
        "observed_sync_forecast",
        "scpn_quantum_control.forecasting.real_data_sync",
        "supported",
        "Train-window calibration with a held-out forecast window.",
    ),
)


def _digest_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_neural_operator_artifact(path: str | Path) -> ArtifactVerification:
    """Verify schema, claim posture, cost arithmetic, and digest of an artifact."""
    artifact_path = Path(path)
    errors: list[str] = []
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ArtifactVerification(
            str(artifact_path), False, (f"cannot read artifact: {exc}",), ""
        )
    if not isinstance(raw, dict):
        return ArtifactVerification(
            str(artifact_path), False, ("artifact must be a JSON object",), ""
        )
    payload = cast(dict[str, Any], raw)
    if payload.get("schema") != ADVANTAGE_ARTIFACT_SCHEMA:
        errors.append("schema mismatch")
    if payload.get("production_claim_allowed") is not False:
        errors.append("production_claim_allowed must be false")
    cost = payload.get("cost_model")
    if not isinstance(cost, dict):
        errors.append("cost_model must be an object")
        expected_digest = ""
    else:
        try:
            rebuilt = build_cost_model(
                int(cost["n_oscillators"]),
                n_steps=int(cost["n_steps"]),
                latent_dim=int(cost["latent_dim"]),
                hidden_dim=int(cost["hidden_dim"]),
                n_trajectories=int(cost["n_trajectories"]),
                epochs=int(cost["epochs"]),
            ).to_dict()
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid cost_model: {exc}")
            expected_digest = ""
        else:
            if rebuilt != cost:
                errors.append("cost_model arithmetic mismatch")
            config = {
                "n_oscillators": payload.get("n_oscillators"),
                "dt": payload.get("dt"),
                "n_steps": payload.get("n_steps"),
                "n_trajectories": cost.get("n_trajectories"),
                "latent_dim": cost.get("latent_dim"),
                "hidden_dim": cost.get("hidden_dim"),
                "epochs": cost.get("epochs"),
            }
            expected_digest = _digest_payload(
                {"schema": ADVANTAGE_ARTIFACT_SCHEMA, "config": config, "cost_model": rebuilt}
            )
            if any(value is None for value in config.values()):
                errors.append("artifact configuration is incomplete")
    recorded_digest = payload.get("payload_sha256")
    if not isinstance(recorded_digest, str) or recorded_digest != expected_digest:
        errors.append("payload_sha256 mismatch")
    return ArtifactVerification(str(artifact_path), not errors, tuple(errors), expected_digest)


def assess_forecast_dataset(dataset: SynchronisationForecastDataset) -> DatasetAdmission:
    """Admit only built-in public provenance or explicit synthetic fixtures."""
    source_path = PurePosixPath(dataset.source_path)
    if source_path.is_absolute() or ".." in source_path.parts:
        return DatasetAdmission(
            dataset.name,
            dataset.source_kind,
            "refused",
            False,
            "Absolute or parent-traversing source paths are not public evidence.",
            ("non_public_source_path",),
        )
    if dataset.source_kind == "qpu_hardware_measurement":
        required = ("backend", "job_id", "shots")
        missing = tuple(key for key in required if dataset.provenance.get(key) in {None, ""})
        return DatasetAdmission(
            dataset.name,
            dataset.source_kind,
            "public_measurement" if not missing else "refused",
            not missing,
            "Committed hardware result with backend, job, and shot provenance."
            if not missing
            else "Hardware measurement lacks required public provenance.",
            tuple(f"missing_provenance:{key}" for key in missing),
        )
    if dataset.source_kind == "public_topology_classical_replay":
        public = dataset.provenance.get("source") == "IEEE 5-bus public benchmark constants"
        return DatasetAdmission(
            dataset.name,
            dataset.source_kind,
            "public_replay" if public else "refused",
            public,
            "Source-backed public topology replay."
            if public
            else "Replay lacks public-source proof.",
            () if public else ("public_source_unverified",),
        )
    if dataset.source_kind == "synthetic" and dataset.provenance.get("synthetic") is True:
        return DatasetAdmission(
            dataset.name,
            dataset.source_kind,
            "synthetic",
            True,
            "Explicit synthetic fixture with no observed-person data.",
            (),
        )
    return DatasetAdmission(
        dataset.name,
        dataset.source_kind,
        "refused",
        False,
        "Dataset source is neither verified public evidence nor explicit synthetic data.",
        ("source_classification_not_admitted",),
    )


def build_neural_operator_baseline_product(
    artifact_path: str | Path,
) -> NeuralOperatorBaselineProduct:
    """Build the bounded baseline composition from live ambient surfaces."""
    return NeuralOperatorBaselineProduct(
        schema=NEURAL_OPERATOR_BASELINE_PRODUCT_SCHEMA,
        surfaces=_SURFACES,
        artifact=verify_neural_operator_artifact(artifact_path),
        no_advantage=issue_no_advantage_certificate(context="neural-operator forecast baseline"),
        cost_labels={
            "training_flops": "one_time_training_estimate",
            "surrogate_flops_per_query": "per_inference_estimate",
            "wall_clock_ms": "advisory_host_bounded_measurement",
        },
        datasets=(
            assess_forecast_dataset(load_hardware_kuramoto_4osc_trace()),
            assess_forecast_dataset(load_ieee5bus_sync_forecast_case()),
        ),
        integrations=(
            IntegrationDisposition(
                "quantum_sync_oracle",
                "descoped_fail_closed",
                "The current challenge oracle has no public classical-baseline registration API.",
            ),
            IntegrationDisposition(
                "multimodal_forecasting",
                "design_dependency",
                "The classical baseline is a dependency; multimodal product wiring remains separate.",
            ),
        ),
    )


__all__ = [
    "NEURAL_OPERATOR_BASELINE_CLAIM_BOUNDARY",
    "NEURAL_OPERATOR_BASELINE_PRODUCT_SCHEMA",
    "ArtifactVerification",
    "BaselineSurfaceRow",
    "DatasetAdmission",
    "IntegrationDisposition",
    "NeuralOperatorBaselineProduct",
    "assess_forecast_dataset",
    "build_neural_operator_baseline_product",
    "verify_neural_operator_artifact",
]
