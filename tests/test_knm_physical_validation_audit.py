# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for K_nm Physical Validation Audit Runner
"""Tests for K_nm physical validation audit helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_knm_physical_validation_audit.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "_run_knm_physical_validation_audit",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_module = _load_script_module()
build_audit_payload = audit_module.build_audit_payload
compare_measured_couplings = audit_module.compare_measured_couplings
evaluate_candidate_systems = audit_module.evaluate_candidate_systems
load_measured_couplings = audit_module.load_measured_couplings


def test_compare_measured_couplings_marks_missing_dataset_open():
    result = compare_measured_couplings(np.ones((2, 2)), None)

    assert result["available"] is False
    assert result["status"] == "missing_measured_system_dataset"


def test_compare_measured_couplings_validates_with_uncertainty():
    K = np.array([[0.0, 0.302], [0.302, 0.0]])
    measured = {
        "system": "unit-test",
        "unit": "dimensionless",
        "normalisation": "already normalised",
        "couplings": [{"i": 1, "j": 2, "value": 0.301, "uncertainty": 0.002}],
    }

    result = compare_measured_couplings(K, measured)

    assert result["available"] is True
    assert result["status"] == "validated_with_measured_dataset"
    assert result["matched_edges"] == 1


def test_load_measured_couplings_requires_couplings_list(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"system": "bad"}), encoding="utf-8")

    try:
        load_measured_couplings(path)
    except ValueError as exc:
        assert "couplings list" in str(exc)
    else:
        raise AssertionError("Expected invalid measured coupling schema to fail")


def test_evaluate_candidate_systems_marks_curated_topology_as_non_closing(tmp_path):
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    canonical = np.asarray(audit_module.build_knm_paper27(L=4, K_base=0.45, K_alpha=0.3))
    measured = canonical * 2.0
    np.fill_diagonal(measured, 0.0)
    artifact = {
        "K_nm": measured.tolist(),
        "domain": "unit",
        "metadata": {
            "public_reference": "unit-test reference",
            "normalisation_locked": False,
        },
        "normalisation": "curated topology matrix scaled to [0, 1]",
        "source_mode": "curated",
        "source_name": "unit_candidate",
    }
    (candidate_dir / "unit_candidate.json").write_text(json.dumps(artifact), encoding="utf-8")

    scan = evaluate_candidate_systems(candidate_dir, k_base=0.45, alpha=0.3)

    assert scan["available"] is True
    assert scan["status"] == "topology_candidates_scanned"
    assert scan["best_topology_candidate"]["source_name"] == "unit_candidate"
    assert scan["systems"][0]["topology"]["spearman_offdiag"] == 1.0
    assert scan["systems"][0]["decision"]["closes_physical_magnitude_gap"] is False
    assert scan["systems"][0]["decision"]["status"] == "does_not_close_exact_magnitude_gap"


def test_build_audit_payload_records_candidate_scan_without_closing_gap(tmp_path):
    candidate_dir = tmp_path / "empty-candidates"
    candidate_dir.mkdir()
    payload = build_audit_payload(
        codebase_path=tmp_path / "missing-codebase",
        measured_path=None,
        candidate_dir=candidate_dir,
        n_layers=4,
        k_base=0.45,
        alpha=0.3,
        command=["python", "scripts/run_knm_physical_validation_audit.py"],
    )

    assert payload["schema_version"] == 2
    assert payload["candidate_system_scan"]["status"] == "missing_candidate_artifacts"
    assert payload["decision"]["physical_validation_closed"] is False
    assert (
        payload["decision"]["current_label"] == "open_requires_measured_system_coupling_magnitudes"
    )
