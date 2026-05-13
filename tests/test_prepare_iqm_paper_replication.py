# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM paper-replication readiness tests
"""Tests for the no-submit IQM paper-replication preparation script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prepare_iqm_paper_replication.py"
TRANSPILER_HELPER = REPO_ROOT / "scripts" / "iqm_fake_transpile_payload.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_iqm_paper_replication", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM replication script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_helper_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("iqm_fake_transpile_payload", TRANSPILER_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM fake-transpile helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeIQMAdapter:
    def transpile_circuit(self, circuit: QuantumCircuit, config: object) -> QuantumCircuit:
        return circuit.copy()


def test_generate_manifest_contains_paper_critical_tiers(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "IQMQuantumBackend", lambda: _FakeIQMAdapter())

    manifest = module.generate(fake_backend="garnet", smoke_shots=128, replication_shots=256)

    assert manifest["submission_status"] == "not_submitted"
    assert manifest["provider"] == "iqm"
    assert manifest["real_qpu_spend_authorised"] is False
    assert manifest["transpile_status"] == "iqm_fake_transpile_passed"
    tiers = {row["tier"] for row in manifest["rows"]}
    assert {
        "smoke_account_probe",
        "dla_parity_minimal",
        "dla_parity_paper_core",
        "fim_negative_control_minimal",
        "readout_full_basis_optional",
    } <= tiers


def test_generate_manifest_blocks_cleanly_when_iqm_dependency_missing(monkeypatch) -> None:
    module = _load_module()

    class MissingIQM:
        def transpile_circuit(self, circuit: QuantumCircuit, config: object) -> QuantumCircuit:
            raise ImportError("iqm-client[qiskit] missing")

    monkeypatch.setattr(module, "IQMQuantumBackend", lambda: MissingIQM())

    manifest = module.generate(fake_backend="garnet")

    assert manifest["transpile_status"] == "blocked_missing_iqm_dependency_or_fake_backend"
    assert manifest["blocked_reason"] == "iqm-client[qiskit] missing"
    assert all(row["iqm_fake_status"] == "blocked" for row in manifest["rows"])


def test_main_writes_json_csv_and_markdown(monkeypatch, tmp_path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "IQMQuantumBackend", lambda: _FakeIQMAdapter())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_iqm_paper_replication.py",
            "--output-dir",
            str(tmp_path),
            "--fake-backend",
            "garnet",
        ],
    )

    assert module.main() == 0

    json_files = sorted(tmp_path.glob("iqm_paper_replication_readiness_*.json"))
    csv_files = sorted(tmp_path.glob("iqm_paper_replication_readiness_*.csv"))
    md_files = sorted(tmp_path.glob("iqm_paper_replication_readiness_*.md"))
    assert len(json_files) == len(csv_files) == len(md_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["total_circuits"] == len(payload["rows"])
    assert payload["recommended_first_real_run"]["tier"] == "smoke_account_probe"
    assert "No IQM service was contacted" in md_files[0].read_text(encoding="utf-8")


def test_iqm_fake_transpile_helper_backend_modules_are_consistent() -> None:
    module = _load_helper_module()

    for backend_name, (module_name, class_name) in module.FAKE_BACKENDS.items():
        assert backend_name in module_name
        assert class_name.lower().endswith(backend_name.lower())
