# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 dark-sector spec tests
"""Tests for Paper 0 dark-energy and psi-DM promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_dark_sector_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_dark_sector_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_dark_sector_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 dark-sector spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 Dark Energy and Psi-Dark Matter",
        "math_ids": [],
        "image_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6311, 6324)]


def test_dark_sector_specs_consume_complete_contiguous_source_span() -> None:
    module = _load_module()

    bundle = module.build_dark_sector_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 13
    assert bundle.summary["consumed_source_record_count"] == 13
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06311", "P0R06323"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["spec_keys"] == [
        "dark_sector.mmc_operator_information_preservation",
        "dark_sector.dark_energy_teleological_potential_boundary",
        "dark_sector.psi_dark_matter_hypothesis_boundary",
        "dark_sector.psi_dm_interaction_mechanisms",
        "dark_sector.cosmic_coherence_reservoir_boundary",
    ]


def test_dark_sector_specs_preserve_source_formula_without_invented_ids() -> None:
    module = _load_module()

    bundle = module.build_dark_sector_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}
    interaction = specs["dark_sector.psi_dm_interaction_mechanisms"]

    assert all(spec.anchor_math_ids == () for spec in specs.values())
    assert all(spec.validation_targets for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())
    assert interaction.source_equation_ids == ("P0R06319:L_geometric",)
    assert interaction.source_formulae == ("L_Geometric proportional to -xi R Psi* Psi",)
    assert "stress-energy tensor" in interaction.canonical_statement


def test_dark_sector_builder_rejects_missing_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06319"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_dark_sector_specs(incomplete)


def test_dark_sector_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_dark_sector_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_dark_sector_validation_spec(
        "dark_sector.psi_dm_interaction_mechanisms",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6311, 6324)]
    assert loaded["source_formulae"] == ["L_Geometric proportional to -xi R Psi* Psi"]
    assert "Paper 0 Dark Sector Specs" in report
    assert "not empirical evidence" in report
