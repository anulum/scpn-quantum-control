# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- readout mitigation eligibility audit tests
"""Tests for dataset-level readout-mitigation eligibility markers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "audit_readout_mitigation_eligibility.py"
)


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "audit_readout_mitigation_eligibility", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_dataset(path: Path, circuits: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"circuits": circuits}), encoding="utf-8")


def test_full_basis_marker_allows_confusion_matrix(tmp_path: Path) -> None:
    module = _load_module()
    dataset = tmp_path / "full.json"
    _write_dataset(
        dataset,
        [
            {"meta": {"experiment": "readout_baseline", "n_qubits": 2, "initial": "00"}},
            {"meta": {"experiment": "readout_baseline", "n_qubits": 2, "initial": "01"}},
            {"meta": {"experiment": "readout_baseline", "n_qubits": 2, "initial": "10"}},
            {"meta": {"experiment": "readout_baseline", "n_qubits": 2, "initial": "11"}},
        ],
    )

    marker = module.audit_dataset(dataset)  # type: ignore[attr-defined]

    assert marker["eligibility_status"] == "full_basis_confusion_matrix_available"
    assert marker["qpu_spend_required_for_full_basis"] is False
    assert marker["missing_readout_bitstrings"] == []


def test_partial_marker_lists_missing_basis_states(tmp_path: Path) -> None:
    module = _load_module()
    dataset = tmp_path / "partial.json"
    _write_dataset(
        dataset,
        [
            {"meta": {"experiment": "main", "n_qubits": 2, "initial": "00"}},
            {"meta": {"experiment": "readout_baseline", "n_qubits": 2, "initial": "00"}},
            {"meta": {"experiment": "readout_baseline", "n_qubits": 2, "initial": "11"}},
        ],
    )

    marker = module.audit_dataset(dataset)  # type: ignore[attr-defined]

    assert marker["eligibility_status"] == "partial_exact_state_baseline_only"
    assert marker["prepared_readout_bitstrings"] == ["00", "11"]
    assert marker["missing_readout_bitstrings"] == ["01", "10"]
    assert marker["allowed_mitigation"] == "exact_state_or_parity_readout_correction_only"


def test_missing_readout_marker_blocks_mitigation_claim(tmp_path: Path) -> None:
    module = _load_module()
    dataset = tmp_path / "missing.json"
    _write_dataset(dataset, [{"meta": {"experiment": "main", "n_qubits": 3, "initial": "000"}}])

    marker = module.audit_dataset(dataset)  # type: ignore[attr-defined]

    assert marker["eligibility_status"] == "missing_readout_calibration"
    assert marker["prepared_readout_count"] == 0
    assert marker["qpu_spend_required_for_full_basis"] is True
