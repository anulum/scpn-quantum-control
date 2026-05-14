# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 free-energy spec tests
"""Tests for Paper 0 EQ0130-EQ0131 validation spec promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_free_energy_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_free_energy_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_free_energy_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 free-energy spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(
    ledger_id: str,
    block_index: int,
    *,
    math_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 free energy test section",
        "math_ids": [] if math_ids is None else math_ids,
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [
        _record("P0R06148", 6148),
        _record("P0R06149", 6149),
        _record("P0R06150", 6150),
        _record("P0R06151", 6151),
        _record("P0R06152", 6152, math_ids=["EQ0130"]),
        _record("P0R06153", 6153, math_ids=["EQ0131"]),
        _record("P0R06154", 6154),
        _record("P0R06155", 6155),
    ]


def test_free_energy_specs_promote_eq0130_eq0131() -> None:
    module = _load_module()

    bundle = module.build_free_energy_validation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 8
    assert bundle.summary["consumed_source_record_count"] == 8
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 1
    assert bundle.summary["spec_keys"] == ["computational.variational_free_energy"]
    assert bundle.summary["hardware_status"] == "simulator_only_no_provider_submission"


def test_free_energy_specs_have_controls_and_falsifiers() -> None:
    module = _load_module()

    bundle = module.build_free_energy_validation_specs(_complete_records())
    spec = bundle.specs[0]

    assert spec.source_equation_ids == ("EQ0130", "EQ0131")
    assert set(spec.anchor_math_ids) == {"EQ0130", "EQ0131"}
    assert any("KL" in target for target in spec.executable_validation_targets)
    assert any("support-mismatch" in control for control in spec.null_controls)
    assert spec.hardware_status == "simulator_only_no_provider_submission"


def test_free_energy_builder_rejects_missing_required_anchor() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06153"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_free_energy_validation_specs(incomplete)


def test_write_outputs_records_policy_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_free_energy_validation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_free_energy_validation_spec(
        "computational.variational_free_energy",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_equation_ids"] == ["EQ0130", "EQ0131"]
    assert "Paper 0 Free-Energy Validation Specs" in report
    assert "Provider submission remains out of scope" in report
