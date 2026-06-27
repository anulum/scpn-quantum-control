# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for QPU data artifact
"""Tests for the inter-repository QPU data artifact contract."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_quantum_control.bridge.qpu_data_artifact import (
    QPUDataArtifact,
    artifact_from_arrays,
    artifact_to_kuramoto_problem,
    read_qpu_data_artifact,
    validate_qpu_data_artifact,
    write_qpu_data_artifact,
)


def _valid_knm(n: int = 4) -> np.ndarray:
    K = np.full((n, n), 0.2, dtype=np.float64)
    np.fill_diagonal(K, 0.0)
    return K


def test_real_artifact_roundtrip_and_hashes(tmp_path):
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="c_elegans_sub",
        source_mode="recorded",
        K_nm=_valid_knm(4),
        omega=[0.1, 0.2, 0.3, 0.4],
        theta0=[0.0, 0.1, 0.2, 0.3],
        layer_assignments=["n0", "n1", "n2", "n3"],
        normalization="max coupling to 1",
        extraction_method="phase-orchestrator fixture",
        source_timestamp="2026-04-27T21:00:00Z",
        metadata={"compiler": "phase-orchestrator"},
    )

    artifact.require_publication_safe()
    payload = artifact.to_dict()
    assert payload["schema_version"].endswith(".v1")
    assert "artifact_sha256" in payload
    assert set(artifact.hashes) >= {"K_nm_sha256", "omega_sha256", "theta0_sha256"}

    path = tmp_path / "artifact.json"
    write_qpu_data_artifact(path, artifact)
    loaded = read_qpu_data_artifact(path)
    np.testing.assert_allclose(loaded.K_nm, artifact.K_nm)
    np.testing.assert_allclose(loaded.omega, artifact.omega)
    assert loaded.metadata["compiler"] == "phase-orchestrator"


def test_publication_gate_rejects_synthetic_artifact():
    artifact = artifact_from_arrays(
        domain="scpn",
        source_name="datastream-fixture",
        source_mode="synthetic",
        K_nm=_valid_knm(3),
        omega=[1.0, 2.0, 3.0],
        normalization="fixture",
        extraction_method="unit-test",
        replay_id="seed:1",
    )

    with pytest.raises(ValueError, match="synthetic"):
        validate_qpu_data_artifact(artifact)

    assert validate_qpu_data_artifact(artifact, require_publication_safe=False) is artifact


def test_publication_gate_requires_timestamp_or_replay_id():
    artifact = artifact_from_arrays(
        domain="power-grid",
        source_name="grid",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=[1.0, 2.0, 3.0],
        normalization="documented",
        extraction_method="documented",
    )

    with pytest.raises(ValueError, match="source_timestamp or replay_id"):
        artifact.require_publication_safe()


def test_artifact_to_kuramoto_problem_preserves_provenance_metadata():
    artifact = artifact_from_arrays(
        domain="power-grid",
        source_name="ieee5bus_power_grid",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=[0.1, 0.2, 0.3],
        normalization="per-unit max coupling",
        extraction_method="documented-topology-loader",
        source_timestamp="2026-04-29T00:00:00Z",
        metadata={"operator": "distribution"},
    )

    problem = artifact_to_kuramoto_problem(artifact)

    assert problem.n_oscillators == artifact.n_oscillators
    np.testing.assert_allclose(problem.K_nm, artifact.K_nm)
    np.testing.assert_allclose(problem.omega, artifact.omega)
    assert problem.metadata["domain"] == "power-grid"
    assert problem.metadata["source_name"] == "ieee5bus_power_grid"
    assert problem.metadata["source_mode"] == "recorded"
    assert problem.metadata["normalization"] == "per-unit max coupling"
    assert problem.metadata["extraction_method"] == "documented-topology-loader"
    assert problem.metadata["source_timestamp"] == "2026-04-29T00:00:00Z"
    assert problem.metadata["artifact_sha256"] == artifact.to_dict()["artifact_sha256"]


def test_rejects_invalid_knm_invariants():
    with pytest.raises(ValueError, match="at least one oscillator"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=np.zeros((0, 0), dtype=np.float64),
            omega=[],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )

    with pytest.raises(ValueError, match="diagonal"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=np.eye(2),
            omega=[1.0, 2.0],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )

    K_tiny_diagonal = _valid_knm(2)
    K_tiny_diagonal[0, 0] = 5.0e-13
    with pytest.raises(ValueError, match="diagonal"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=K_tiny_diagonal,
            omega=[1.0, 2.0],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )

    K_negative = _valid_knm(2)
    K_negative[0, 1] = -0.1
    K_negative[1, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=K_negative,
            omega=[1.0, 2.0],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )

    K_tiny_negative = _valid_knm(2)
    K_tiny_negative[0, 1] = -5.0e-13
    K_tiny_negative[1, 0] = -5.0e-13
    with pytest.raises(ValueError, match="non-negative"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=K_tiny_negative,
            omega=[1.0, 2.0],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )

    K_directed = np.array([[0.0, 0.2], [0.5, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=K_directed,
            omega=[1.0, 2.0],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )

    K_directed_large_scale = np.array([[0.0, 1.0e9], [1.0e9 + 1.0e3, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=K_directed_large_scale,
            omega=[1.0, 2.0],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )


def test_rejects_shape_mismatch_and_missing_metadata():
    with pytest.raises(ValueError, match="omega shape"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=[1.0, 2.0],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )

    with pytest.raises(ValueError, match="normalization"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=_valid_knm(2),
            omega=[1.0, 2.0],
            normalization="",
            extraction_method="e",
            replay_id="r",
        )


def test_rejects_implicit_numeric_string_coercion():
    with pytest.raises(ValueError, match="K_nm entries must be numeric"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=[["0.0", "0.2"], ["0.2", "0.0"]],
            omega=[1.0, 2.0],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )


def test_rejects_ragged_knm_payloads_with_contract_error():
    with pytest.raises(ValueError, match="K_nm must be a rectangular numeric array"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=[[0.0, 0.2], [0.2]],
            omega=[1.0, 2.0],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )


def test_rejects_boolean_frequency_payloads():
    with pytest.raises(ValueError, match="omega entries must be numeric"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=_valid_knm(2),
            omega=[True, False],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )


def test_rejects_complex_initial_phases():
    with pytest.raises(ValueError, match="theta0 entries must be real numeric"):
        artifact_from_arrays(
            domain="x",
            source_name="x",
            source_mode="recorded",
            K_nm=_valid_knm(2),
            omega=[1.0, 2.0],
            theta0=[0.0, 1.0 + 0.0j],
            normalization="n",
            extraction_method="e",
            replay_id="r",
        )


def test_from_scpn_datastream_payload_defaults_to_synthetic():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 4,
        "layer_ids": ["l1", "l2", "l3", "l4"],
        "omega_rad_s": [0.1, 0.2, 0.3, 0.4],
        "knm": _valid_knm(4).tolist(),
    }

    artifact = QPUDataArtifact.from_scpn_datastream_payload(payload)
    assert artifact.is_synthetic
    assert artifact.replay_id == "seed:11"
    assert artifact.metadata["payload_sha256"]
    np.testing.assert_allclose(artifact.K_nm, _valid_knm(4))


def test_scpn_datastream_adapter_rejects_implicit_numeric_coercion():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": ["0.1", "0.2"],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }

    with pytest.raises(ValueError, match="omega entries must be numeric"):
        QPUDataArtifact.from_scpn_datastream_payload(payload)


def test_scpn_datastream_adapter_rejects_non_string_layer_ids():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", 2],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }

    with pytest.raises(ValueError, match="layer_assignments entries must be strings"):
        QPUDataArtifact.from_scpn_datastream_payload(payload)


def test_scpn_datastream_adapter_rejects_non_mapping_payloads():
    with pytest.raises(ValueError, match="SC-NeuroCore datastream payload must be a mapping"):
        QPUDataArtifact.from_scpn_datastream_payload([("schema_version", "wrong")])


def test_scpn_datastream_adapter_requires_seed_for_replay_identity():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }

    with pytest.raises(ValueError, match="seed is required"):
        QPUDataArtifact.from_scpn_datastream_payload(payload)


def test_scpn_datastream_adapter_rejects_non_integer_seed():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": "11",
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }

    with pytest.raises(ValueError, match="seed must be an integer"):
        QPUDataArtifact.from_scpn_datastream_payload(payload)


def test_scpn_datastream_adapter_rejects_n_layers_mismatch():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 3,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }

    with pytest.raises(ValueError, match="n_layers must match layer_ids length"):
        QPUDataArtifact.from_scpn_datastream_payload(payload)


def test_scpn_datastream_adapter_rejects_invalid_dt_s():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.0,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }

    with pytest.raises(ValueError, match="dt_s must be positive finite"):
        QPUDataArtifact.from_scpn_datastream_payload(payload)


def test_scpn_datastream_adapter_rejects_invalid_n_steps():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 0,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }

    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        QPUDataArtifact.from_scpn_datastream_payload(payload)


def test_scpn_datastream_adapter_rejects_blank_source_project():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "   ",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }

    with pytest.raises(ValueError, match="source_project must be non-empty"):
        QPUDataArtifact.from_scpn_datastream_payload(payload)


def test_scpn_datastream_adapter_hashes_canonical_source_project():
    base_payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }
    padded_payload = dict(base_payload)
    padded_payload["source_project"] = "  sc-neurocore  "

    baseline = QPUDataArtifact.from_scpn_datastream_payload(base_payload)
    padded = QPUDataArtifact.from_scpn_datastream_payload(padded_payload)

    assert padded.metadata["source_project"] == "sc-neurocore"
    assert padded.metadata["payload_sha256"] == baseline.metadata["payload_sha256"]


def test_scpn_datastream_adapter_hashes_canonical_layer_ids():
    base_payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }
    padded_payload = dict(base_payload)
    padded_payload["layer_ids"] = ["  l1  ", "  l2  "]

    baseline = QPUDataArtifact.from_scpn_datastream_payload(base_payload)
    padded = QPUDataArtifact.from_scpn_datastream_payload(padded_payload)

    assert padded.layer_assignments == ("l1", "l2")
    assert padded.metadata["payload_sha256"] == baseline.metadata["payload_sha256"]


def test_scpn_datastream_adapter_hashes_canonical_numeric_containers():
    base_payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [1.0, 2.0],
        "knm": [[0.0, 1.0], [1.0, 0.0]],
    }
    integer_payload = dict(base_payload)
    integer_payload["omega_rad_s"] = [1, 2]
    integer_payload["knm"] = [[0, 1], [1, 0]]

    baseline = QPUDataArtifact.from_scpn_datastream_payload(base_payload)
    integer_variant = QPUDataArtifact.from_scpn_datastream_payload(integer_payload)

    assert integer_variant.metadata["payload_sha256"] == baseline.metadata["payload_sha256"]


def test_scpn_datastream_adapter_ignores_extra_keys_in_payload_fingerprint():
    base_payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }
    annotated_payload = dict(base_payload)
    annotated_payload["operator_note"] = "not part of canonical physics payload"

    baseline = QPUDataArtifact.from_scpn_datastream_payload(base_payload)
    annotated = QPUDataArtifact.from_scpn_datastream_payload(annotated_payload)

    assert annotated.metadata["payload_sha256"] == baseline.metadata["payload_sha256"]


def test_scpn_datastream_adapter_rejects_publication_safe_source_mode():
    payload = {
        "schema_version": "sc-neurocore.scpn.datastream.v1",
        "source_project": "sc-neurocore",
        "seed": 11,
        "dt_s": 0.01,
        "n_steps": 3,
        "n_layers": 2,
        "layer_ids": ["l1", "l2"],
        "omega_rad_s": [0.1, 0.2],
        "knm": [[0.0, 0.2], [0.2, 0.0]],
    }

    with pytest.raises(
        ValueError, match="SC-NeuroCore datastream artifacts must be smoke-test modes"
    ):
        QPUDataArtifact.from_scpn_datastream_payload(payload, source_mode="recorded")


def test_scpn_datastream_adapter_rejects_invalid_identity_overrides_before_payload_work():
    with pytest.raises(ValueError, match="domain must be a string"):
        QPUDataArtifact.from_scpn_datastream_payload(
            [("schema_version", "wrong")],
            domain=123,
        )


def test_json_loader_rejects_wrong_schema():
    with pytest.raises(ValueError, match="schema"):
        QPUDataArtifact.from_json(json.dumps({"schema_version": "wrong"}))


def test_loader_rejects_non_mapping_payloads_with_contract_error():
    with pytest.raises(ValueError, match="artifact payload must be a mapping"):
        QPUDataArtifact.from_dict([("schema_version", "wrong")])


def test_loader_rejects_stale_array_hashes():
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="tamper-check",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=[0.1, 0.2, 0.3],
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
    )
    payload = artifact.to_dict()
    payload["K_nm"][0][1] = 0.9
    payload["K_nm"][1][0] = 0.9

    with pytest.raises(ValueError, match="K_nm_sha256"):
        QPUDataArtifact.from_dict(payload)


def test_loader_rejects_stale_artifact_hash_after_metadata_tamper():
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="tamper-check",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=[0.1, 0.2, 0.3],
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
        metadata={"acquisition": "locked"},
    )
    payload = artifact.to_dict()
    payload["metadata"]["acquisition"] = "changed-after-hash"

    with pytest.raises(ValueError, match="artifact_sha256"):
        QPUDataArtifact.from_dict(payload)


def test_loader_rejects_malformed_artifact_hash():
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="hash-syntax-check",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
        metadata={"acquisition": "baseline"},
    )
    payload = artifact.to_dict()
    payload["artifact_sha256"] = "not-a-sha256"

    with pytest.raises(ValueError, match="artifact_sha256 must be lowercase SHA-256 hex"):
        QPUDataArtifact.from_dict(payload)


def test_artifact_arrays_are_defensive_copies_and_read_only():
    K_nm = _valid_knm(3)
    omega = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    theta0 = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="immutability-check",
        source_mode="recorded",
        K_nm=K_nm,
        omega=omega,
        theta0=theta0,
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
    )

    K_nm[0, 1] = 0.9
    omega[0] = 99.0
    theta0[0] = 99.0

    assert artifact.K_nm[0, 1] == pytest.approx(0.2)
    assert artifact.omega[0] == pytest.approx(0.1)
    assert artifact.theta0 is not None
    assert artifact.theta0[0] == pytest.approx(0.0)
    assert artifact.K_nm.flags.writeable is False
    assert artifact.omega.flags.writeable is False
    assert artifact.theta0.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        artifact.K_nm[0, 1] = 0.7


def test_artifact_mappings_are_defensive_copies_and_read_only():
    metadata = {"acquisition": "original"}
    artifact = QPUDataArtifact(
        domain="connectome",
        source_name="mapping-immutability-check",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
        metadata=metadata,
    )
    original_k_nm_hash = artifact.hashes["K_nm_sha256"]

    metadata["acquisition"] = "mutated-after-validation"

    assert artifact.metadata["acquisition"] == "original"
    assert artifact.hashes["K_nm_sha256"] == original_k_nm_hash
    with pytest.raises(TypeError):
        artifact.metadata["acquisition"] = "blocked"
    with pytest.raises(TypeError):
        artifact.hashes["K_nm_sha256"] = "blocked"


def test_artifact_metadata_is_deep_frozen_and_serializes_as_json_lists():
    nested_metadata = {
        "calibration": {
            "operator": "source-a",
            "window": [0.0, 1.0],
        }
    }
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="nested-metadata-immutability-check",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
        metadata=nested_metadata,
    )

    nested_metadata["calibration"]["operator"] = "mutated-after-validation"
    nested_metadata["calibration"]["window"].append(2.0)

    assert artifact.metadata["calibration"]["operator"] == "source-a"
    assert artifact.metadata["calibration"]["window"] == (0.0, 1.0)
    with pytest.raises(TypeError):
        artifact.metadata["calibration"]["operator"] = "blocked"
    with pytest.raises(AttributeError):
        artifact.metadata["calibration"]["window"].append(2.0)

    serialized = artifact.to_dict()
    assert serialized["metadata"]["calibration"]["window"] == [0.0, 1.0]


def test_layer_assignments_are_defensive_copies_and_read_only():
    layer_assignments = ["cortex", "thalamus", "brainstem"]
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="layer-immutability-check",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
        layer_assignments=layer_assignments,
    )

    layer_assignments[0] = "mutated-after-validation"

    assert artifact.layer_assignments == ("cortex", "thalamus", "brainstem")
    with pytest.raises(AttributeError):
        artifact.layer_assignments.append("blocked")
    assert artifact.to_dict()["layer_assignments"] == ["cortex", "thalamus", "brainstem"]


def test_layer_assignments_reject_non_string_labels():
    with pytest.raises(ValueError, match="layer_assignments entries must be strings"):
        artifact_from_arrays(
            domain="connectome",
            source_name="layer-label-type-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
            layer_assignments=["cortex", 2, "brainstem"],
        )


def test_layer_assignments_reject_blank_labels():
    with pytest.raises(ValueError, match="layer_assignments entries must be non-empty"):
        artifact_from_arrays(
            domain="connectome",
            source_name="layer-label-blank-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
            layer_assignments=["cortex", "   ", "brainstem"],
        )


def test_metadata_rejects_non_string_keys():
    with pytest.raises(ValueError, match="metadata keys must be strings"):
        artifact_from_arrays(
            domain="connectome",
            source_name="metadata-key-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
            metadata={1: "not-a-json-object-key"},
        )


def test_metadata_rejects_non_finite_floats():
    with pytest.raises(ValueError, match="metadata floats must be finite"):
        artifact_from_arrays(
            domain="connectome",
            source_name="metadata-float-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
            metadata={"calibration_gain": float("nan")},
        )


def test_metadata_rejects_non_json_values():
    with pytest.raises(ValueError, match="metadata values must be JSON-compatible"):
        artifact_from_arrays(
            domain="connectome",
            source_name="metadata-value-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
            metadata={"opaque": object()},
        )


def test_hashes_reject_unknown_keys():
    with pytest.raises(ValueError, match="unknown hash key"):
        QPUDataArtifact(
            domain="connectome",
            source_name="hash-key-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
            hashes={"operator_note": "belongs-in-metadata"},
        )


def test_hashes_reject_malformed_sha256_values():
    with pytest.raises(ValueError, match="K_nm_sha256 must be lowercase SHA-256 hex"):
        QPUDataArtifact(
            domain="connectome",
            source_name="hash-value-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
            hashes={"K_nm_sha256": "not-a-sha256"},
        )


def test_metadata_rejects_non_mapping_containers():
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        QPUDataArtifact(
            domain="connectome",
            source_name="metadata-container-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
            metadata=[("operator", "pair-list")],
        )


def test_hashes_reject_non_mapping_containers():
    with pytest.raises(ValueError, match="hashes must be a mapping"):
        QPUDataArtifact(
            domain="connectome",
            source_name="hash-container-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
            hashes=[("K_nm_sha256", "0" * 64)],
        )


def test_source_timestamp_rejects_non_string_values():
    with pytest.raises(ValueError, match="source_timestamp must be a string"):
        QPUDataArtifact(
            domain="connectome",
            source_name="timestamp-type-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            source_timestamp=123,
        )


def test_replay_id_rejects_blank_values():
    with pytest.raises(ValueError, match="replay_id must be non-empty"):
        artifact_from_arrays(
            domain="connectome",
            source_name="replay-id-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="   ",
        )


def test_provenance_identifiers_are_trimmed_before_hashing():
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="provenance-normalization",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        normalization="documented",
        extraction_method="unit-test",
        source_timestamp="  2026-05-24T00:00:00Z  ",
        replay_id="  source-run-1  ",
    )

    assert artifact.source_timestamp == "2026-05-24T00:00:00Z"
    assert artifact.replay_id == "source-run-1"
    assert artifact.to_dict()["source_timestamp"] == "2026-05-24T00:00:00Z"
    assert artifact.to_dict()["replay_id"] == "source-run-1"


def test_required_text_fields_reject_non_string_values():
    with pytest.raises(ValueError, match="domain must be a string"):
        QPUDataArtifact(
            domain=123,
            source_name="required-text-validation",
            source_mode="recorded",
            K_nm=_valid_knm(3),
            omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            normalization="documented",
            extraction_method="unit-test",
            replay_id="source-run-1",
        )


def test_required_text_fields_are_trimmed_before_hashing():
    artifact = QPUDataArtifact(
        domain="  connectome  ",
        source_name="  required-text-normalization  ",
        source_mode="  recorded  ",
        K_nm=_valid_knm(3),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        normalization="  documented  ",
        extraction_method="  unit-test  ",
        replay_id="source-run-1",
    )

    assert artifact.domain == "connectome"
    assert artifact.source_name == "required-text-normalization"
    assert artifact.source_mode == "recorded"
    assert artifact.normalization == "documented"
    assert artifact.extraction_method == "unit-test"


def test_loader_rejects_non_string_required_identity_fields():
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="loader-required-text-validation",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
    )
    payload = artifact.to_dict()
    payload["domain"] = 123
    payload.pop("artifact_sha256")

    with pytest.raises(ValueError, match="domain must be a string"):
        QPUDataArtifact.from_dict(payload)


def test_loader_rejects_non_mapping_metadata():
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="loader-metadata-container-validation",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
    )
    payload = artifact.to_dict()
    payload["metadata"] = [("operator", "pair-list")]
    payload.pop("artifact_sha256")

    with pytest.raises(ValueError, match="metadata must be a mapping"):
        QPUDataArtifact.from_dict(payload)


def test_loader_rejects_string_layer_assignment_container():
    artifact = artifact_from_arrays(
        domain="connectome",
        source_name="loader-layer-container-validation",
        source_mode="recorded",
        K_nm=_valid_knm(3),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        normalization="documented",
        extraction_method="unit-test",
        replay_id="source-run-1",
    )
    payload = artifact.to_dict()
    payload["layer_assignments"] = "abc"
    payload.pop("artifact_sha256")

    with pytest.raises(ValueError, match="layer_assignments must be a sequence of strings"):
        QPUDataArtifact.from_dict(payload)
