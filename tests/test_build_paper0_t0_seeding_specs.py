# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 t0-seeding spec tests
"""Tests for Paper 0 t=0 SSB seeding and spin-torsion bridge promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_t0_seeding_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_t0_seeding_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_t0_seeding_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 t0-seeding spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 t0 SSB Seeding",
        "math_ids": [],
        "image_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6339, 6363)]


def test_t0_seeding_specs_consume_complete_contiguous_source_span() -> None:
    module = _load_module()

    bundle = module.build_t0_seeding_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 24
    assert bundle.summary["consumed_source_record_count"] == 24
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06339", "P0R06362"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["spec_keys"] == [
        "t0_seeding.initial_value_problem_boundary",
        "t0_seeding.j_sec_memory_bias_boundary",
        "t0_seeding.teleological_tachyonic_potential",
        "t0_seeding.spin_torsion_bridge_equations",
        "t0_seeding.conformal_invariant_torsion_boundary",
    ]


def test_t0_seeding_specs_preserve_source_formulae_and_structural_records() -> None:
    module = _load_module()

    bundle = module.build_t0_seeding_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}
    torsion = specs["t0_seeding.spin_torsion_bridge_equations"]

    assert all(spec.anchor_math_ids == () for spec in specs.values())
    assert all(spec.validation_targets for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())
    assert specs["t0_seeding.teleological_tachyonic_potential"].source_equation_ids == (
        "P0R06344:t0_effective_potential",
    )
    assert torsion.source_equation_ids == (
        "P0R06349:torsion_spin_bridge",
        "P0R06354:psi_torsion_spin_bridge",
    )
    assert "torsion_ijk = 8 pi G s_ijk" in torsion.source_formulae
    assert bundle.summary["structural_source_ledger_ids"] == [
        "P0R06340",
        "P0R06346",
        "P0R06357",
        "P0R06358",
        "P0R06359",
        "P0R06360",
        "P0R06361",
    ]


def test_t0_seeding_builder_rejects_missing_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06354"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_t0_seeding_specs(incomplete)


def test_t0_seeding_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_t0_seeding_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_t0_seeding_validation_spec(
        "t0_seeding.spin_torsion_bridge_equations",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6339, 6363)]
    assert loaded["source_formulae"] == [
        "torsion_ijk = 8 pi G s_ijk",
        "torsion_ijk = 8 pi G s_ijk_psi",
    ]
    assert "Paper 0 t0 Seeding Specs" in report
    assert "not empirical evidence" in report
