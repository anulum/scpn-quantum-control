# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 pathology/criticality spec tests
"""Tests for Paper 0 pathology/criticality validation spec promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_pathology_criticality_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_pathology_criticality_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_pathology_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 pathology/criticality spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 pathology/criticality test section",
        "math_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6197, 6206)]


def test_pathology_criticality_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_pathology_criticality_validation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06197", "P0R06205"]
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["spec_keys"] == [
        "applied.pathology.coherence_breakdown_index",
        "applied.pathology.criticality_deviation_classifier",
        "applied.pathology.therapeutic_restoration_targets",
    ]
    assert bundle.summary["hardware_status"] == "simulator_only_no_provider_submission"


def test_pathology_criticality_specs_have_controls_without_clinical_promotion() -> None:
    module = _load_module()

    bundle = module.build_pathology_criticality_validation_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(spec.source_equation_ids == () for spec in specs.values())
    assert all(spec.anchor_math_ids == () for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all(spec.executable_validation_targets for spec in specs.values())
    assert all("clinical" in spec.claim_boundary for spec in specs.values())
    assert any(
        "sigma-neutral" in control
        for control in specs["applied.pathology.criticality_deviation_classifier"].null_controls
    )


def test_pathology_criticality_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06202"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_pathology_criticality_validation_specs(incomplete)


def test_pathology_criticality_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_pathology_criticality_validation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_pathology_criticality_validation_spec(
        "applied.pathology.therapeutic_restoration_targets",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == ["P0R06204", "P0R06205"]
    assert "Paper 0 Pathology/Criticality Validation Specs" in report
    assert "not clinical evidence" in report
