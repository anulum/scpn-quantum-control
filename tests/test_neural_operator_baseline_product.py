# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — neural-operator baseline product tests
"""Contract tests for neural-operator evidence and data admission."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.forecasting.real_data_sync import SynchronisationForecastDataset
from scpn_quantum_control.neural_operator_baseline_product import (
    ArtifactVerification,
    BaselineSurfaceRow,
    DatasetAdmission,
    IntegrationDisposition,
    NeuralOperatorBaselineProduct,
    assess_forecast_dataset,
    build_neural_operator_baseline_product,
    verify_neural_operator_artifact,
)

_ARTIFACT = Path("docs/benchmarks/neural_operator_advantage.json")


def _dataset(
    *,
    source_path: str = "fixtures/synthetic.json",
    source_kind: str = "synthetic",
    provenance: dict[str, Any] | None = None,
) -> SynchronisationForecastDataset:
    """Return a typed synthetic fixture with source-policy overrides."""
    return SynchronisationForecastDataset(
        name="fixture",
        domain="test",
        source_path=source_path,
        times=np.array([0.0, 0.5, 1.0]),
        observed_order_parameter=np.array([0.2, 0.3, 0.4]),
        coupling=np.eye(2),
        omega=np.zeros(2),
        theta0=np.zeros(2),
        train_size=2,
        source_kind=source_kind,
        provenance={"synthetic": True} if provenance is None else provenance,
    )


def test_live_product_is_complete_and_no_advantage() -> None:
    """Build the complete classical baseline with no-advantage posture."""
    product = build_neural_operator_baseline_product(_ARTIFACT)
    assert isinstance(product, NeuralOperatorBaselineProduct)
    assert product.artifact.valid
    assert product.no_advantage.language_status == "no_advantage_default"
    assert all(row.allowed for row in product.datasets)
    assert [row.status for row in product.integrations] == [
        "descoped_fail_closed",
        "design_dependency",
    ]
    assert product.cost_labels["training_flops"] == "one_time_training_estimate"
    assert product.schema == "neural_operator_baseline_product.v2"
    json.dumps(product.to_dict())


def test_committed_artifact_digest_and_arithmetic_verify() -> None:
    """Verify the committed artifact digest and cost arithmetic."""
    result = verify_neural_operator_artifact(_ARTIFACT)
    assert result.valid
    assert not result.errors
    assert len(result.payload_sha256) == 64


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"schema": "bad"}, "schema mismatch"),
        ({"production_claim_allowed": True}, "production_claim_allowed"),
        ({"payload_sha256": "0" * 64}, "payload_sha256 mismatch"),
    ],
)
def test_artifact_verifier_rejects_claim_and_digest_drift(
    tmp_path: Path, mutation: dict[str, object], error: str
) -> None:
    """Reject artifact schema, production-claim, and digest drift."""
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    payload.update(mutation)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = verify_neural_operator_artifact(path)
    assert not result.valid
    assert any(error in item for item in result.errors)


def test_artifact_verifier_rejects_unreadable_and_non_object(tmp_path: Path) -> None:
    """Reject absent artifacts and non-object JSON payloads."""
    missing = verify_neural_operator_artifact(tmp_path / "missing.json")
    assert not missing.valid and "cannot read artifact" in missing.errors[0]
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    assert verify_neural_operator_artifact(path).errors == ("artifact must be a JSON object",)


def test_artifact_verifier_rejects_missing_and_invalid_cost_model(tmp_path: Path) -> None:
    """Reject absent and structurally invalid cost models."""
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    payload["cost_model"] = None
    path = tmp_path / "none.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "cost_model must be an object" in verify_neural_operator_artifact(path).errors
    payload["cost_model"] = {"n_oscillators": "bad"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert any(
        "invalid cost_model" in item for item in verify_neural_operator_artifact(path).errors
    )


def test_artifact_verifier_rejects_cost_arithmetic_and_incomplete_config(
    tmp_path: Path,
) -> None:
    """Reject altered arithmetic and incomplete artifact configuration."""
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    payload["cost_model"]["direct_flops_per_query"] += 1
    path = tmp_path / "arithmetic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = verify_neural_operator_artifact(path)
    assert "cost_model arithmetic mismatch" in result.errors

    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    del payload["dt"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = verify_neural_operator_artifact(path)
    assert "artifact configuration is incomplete" in result.errors


def test_synthetic_dataset_admitted_and_unknown_refused() -> None:
    """Admit explicit synthetic data and refuse unknown source classes."""
    admitted = assess_forecast_dataset(_dataset())
    assert admitted.allowed and admitted.data_classification == "synthetic"
    refused = assess_forecast_dataset(_dataset(source_kind="private_upload", provenance={}))
    assert not refused.allowed
    assert refused.blockers == ("source_classification_not_admitted",)


def test_unsafe_path_and_unproven_public_sources_refused() -> None:
    """Refuse unsafe paths and public sources without provenance proof."""
    unsafe = assess_forecast_dataset(_dataset(source_path="/private/input.json"))
    assert not unsafe.allowed and unsafe.blockers == ("non_public_source_path",)
    replay = assess_forecast_dataset(
        _dataset(source_kind="public_topology_classical_replay", provenance={})
    )
    assert not replay.allowed and replay.blockers == ("public_source_unverified",)
    hardware = assess_forecast_dataset(
        _dataset(source_kind="qpu_hardware_measurement", provenance={})
    )
    assert not hardware.allowed
    assert len(hardware.blockers) == 3


def test_record_invariants_fail_closed() -> None:
    """Keep record fields, schema, claim boundary, and posture fail closed."""
    surface_values = (
        ("", "pointer", "supported", "summary"),
        ("surface", "", "supported", "summary"),
        ("surface", "pointer", "supported", ""),
        ("surface", "pointer", "invalid", "summary"),
    )
    for surface_value in surface_values:
        with pytest.raises(ValueError):
            BaselineSurfaceRow(
                surface_value[0],
                surface_value[1],
                cast(Any, surface_value[2]),
                surface_value[3],
            )
    with pytest.raises(ValueError):
        ArtifactVerification("x", True, ("error",), "")
    with pytest.raises(ValueError):
        ArtifactVerification("x", True, (), "short")
    dataset_values = (
        ("", "kind", "reason"),
        ("x", "", "reason"),
        ("x", "kind", ""),
    )
    for dataset_value in dataset_values:
        with pytest.raises(ValueError):
            DatasetAdmission(
                dataset_value[0],
                dataset_value[1],
                "synthetic",
                True,
                dataset_value[2],
                (),
            )
    with pytest.raises(ValueError):
        DatasetAdmission("x", "kind", "synthetic", True, "reason", ("block",))
    with pytest.raises(ValueError):
        DatasetAdmission("x", "kind", "refused", True, "reason", ())
    with pytest.raises(ValueError):
        IntegrationDisposition("quantum_sync_oracle", "wired", "")
    with pytest.raises(ValueError):
        IntegrationDisposition(cast(Any, "unknown_target"), "wired", "reason")
    with pytest.raises(ValueError):
        IntegrationDisposition("quantum_sync_oracle", cast(Any, "unknown"), "reason")
    product = build_neural_operator_baseline_product(_ARTIFACT)
    with pytest.raises(ValueError, match="unknown product schema"):
        replace(product, schema="neural_operator_baseline_product.v1")
    with pytest.raises(ValueError):
        replace(product, surfaces=())
    with pytest.raises(ValueError):
        replace(product, datasets=())
    with pytest.raises(ValueError):
        replace(product, integrations=())
    with pytest.raises(ValueError, match="claim boundary"):
        replace(product, claim_boundary="altered")
    object.__setattr__(product.no_advantage, "language_status", "research_observation")
    with pytest.raises(ValueError):
        replace(product)
