# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Hamiltonian index spec tests
"""Tests for Paper 0 Appendix C Hamiltonian/operator index promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_hamiltonian_index_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_hamiltonian_index_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_hamiltonian_index_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Hamiltonian index spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "New content to be allocated into the document:",
        "canonical_category": "context",
        "math_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6878, 6916)]


def test_hamiltonian_index_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_hamiltonian_index_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 38
    assert bundle.summary["consumed_source_record_count"] == 38
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06878", "P0R06915"]
    assert bundle.summary["spec_count"] == 7
    assert bundle.summary["operator_count"] == 9
    assert bundle.summary["hardware_status"] == "operator_index_no_execution"


def test_hamiltonian_index_specs_preserve_operator_boundaries() -> None:
    module = _load_module()

    bundle = module.build_hamiltonian_index_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(6878, 6916))
        for spec in specs.values()
    )
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert specs["appendix_c.hamiltonian_index.microtubule_layer1"].operator_symbols == (
        "H_MT",
        "H_PQT",
        "H_iso",
    )
    assert specs["appendix_c.hamiltonian_index.master_lagrangian"].source_equation_ids == (
        "P0R06885:master_lagrangian",
    )
    assert specs["appendix_c.hamiltonian_index.informational_operators"].operator_symbols == (
        "R_Psi",
        "O_sem",
    )


def test_hamiltonian_index_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06892"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_hamiltonian_index_specs(incomplete)


def test_hamiltonian_index_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_hamiltonian_index_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_hamiltonian_index_validation_spec(
        "appendix_c.hamiltonian_index.microtubule_layer1",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R06878"
    assert loaded["source_ledger_ids"][-1] == "P0R06915"
    assert "Paper 0 Hamiltonian Index Specs" in report
    assert "not empirical evidence" in report
