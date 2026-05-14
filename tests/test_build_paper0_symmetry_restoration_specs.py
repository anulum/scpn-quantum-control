# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 symmetry-restoration spec tests
"""Tests for Paper 0 MMC symmetry-restoration promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_symmetry_restoration_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_symmetry_restoration_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_symmetry_restoration_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 symmetry-restoration spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 MMC Symmetry Restoration",
        "math_ids": [],
        "image_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6324, 6339)]


def test_symmetry_restoration_specs_consume_complete_contiguous_source_span() -> None:
    module = _load_module()

    bundle = module.build_symmetry_restoration_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 15
    assert bundle.summary["consumed_source_record_count"] == 15
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06324", "P0R06338"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["spec_keys"] == [
        "symmetry_restoration.mmc_conformal_geometry_boundary",
        "symmetry_restoration.conformal_boundary_masslessness_constraint",
        "symmetry_restoration.effective_potential_flip_boundary",
        "symmetry_restoration.vev_melting_massless_limit",
        "symmetry_restoration.legal_conformal_rescaling_boundary",
    ]


def test_symmetry_restoration_specs_preserve_all_source_formulae() -> None:
    module = _load_module()

    bundle = module.build_symmetry_restoration_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(spec.anchor_math_ids == () for spec in specs.values())
    assert all(spec.validation_targets for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())
    assert specs["symmetry_restoration.mmc_conformal_geometry_boundary"].source_equation_ids == (
        "P0R06326:conformal_rescaling",
    )
    assert specs["symmetry_restoration.effective_potential_flip_boundary"].source_equation_ids == (
        "P0R06333:effective_potential",
    )
    assert specs["symmetry_restoration.vev_melting_massless_limit"].source_equation_ids == (
        "P0R06336:vev_limit",
        "P0R06337:mass_limits",
    )
    assert (
        "V_eff"
        in specs["symmetry_restoration.effective_potential_flip_boundary"].source_formulae[0]
    )
    assert (
        "m_A = g v" in specs["symmetry_restoration.vev_melting_massless_limit"].source_formulae[1]
    )


def test_symmetry_restoration_builder_rejects_missing_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06333"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_symmetry_restoration_specs(incomplete)


def test_symmetry_restoration_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_symmetry_restoration_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_symmetry_restoration_validation_spec(
        "symmetry_restoration.vev_melting_massless_limit",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6324, 6339)]
    assert loaded["source_formulae"] == [
        "lim_{t -> infinity} v(t) = 0",
        "m_A = g v; m_h = sqrt(2 lambda) v",
    ]
    assert "Paper 0 Symmetry Restoration Specs" in report
    assert "not empirical evidence" in report
