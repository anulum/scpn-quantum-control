# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 glial-control validation spec tests
"""Tests for Paper 0 abiogenesis/cellular sigma validation spec promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_glial_control_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_glial_control_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_glial_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 glial-control spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(
    ledger_id: str,
    block_index: int,
    *,
    math_ids: list[str] | None = None,
    text: str = "source text",
) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 glial-control test section",
        "math_ids": [] if math_ids is None else math_ids,
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    return [
        _record("P0R05360", 5360),
        _record("P0R05361", 5361, math_ids=["EQ0105"]),
        _record("P0R05363", 5363),
        _record("P0R05366", 5366),
        _record("P0R05367", 5367),
        _record("P0R05368", 5368),
        _record("P0R05369", 5369, math_ids=["EQ0106"]),
        _record("P0R05370", 5370),
        _record("P0R05371", 5371),
        _record("P0R05372", 5372, math_ids=["EQ0107"]),
        _record("P0R05376", 5376, math_ids=["EQ0108"]),
        _record("P0R05377", 5377, math_ids=["EQ0109"]),
        _record("P0R05385", 5385, math_ids=["EQ0110"]),
        _record("P0R05388", 5388),
        _record("P0R05390", 5390),
        _record("P0R05391", 5391),
        _record("P0R05392", 5392),
        _record("P0R05395", 5395),
        _record("P0R05396", 5396),
        _record("P0R05397", 5397),
        _record("P0R05399", 5399, math_ids=["EQ0111"]),
        _record("P0R05400", 5400),
        _record("P0R05403", 5403, math_ids=["EQ0112"]),
        _record("P0R05404", 5404),
        _record("P0R05405", 5405),
        _record("P0R05406", 5406),
    ]


def test_glial_control_specs_promote_eq0105_to_eq0112_with_full_coverage() -> None:
    module = _load_module()

    bundle = module.build_glial_control_validation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 26
    assert bundle.summary["consumed_source_record_count"] == 26
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 2
    assert bundle.summary["spec_keys"] == [
        "embodied.quantum_immune_interface",
        "embodied.glial_sigma_control",
    ]
    assert bundle.summary["hardware_status"] == "simulator_only_no_provider_submission"


def test_glial_control_specs_are_source_anchored_and_have_falsifiers() -> None:
    module = _load_module()

    bundle = module.build_glial_control_validation_specs(_complete_records())
    by_key = {spec.key: spec for spec in bundle.specs}

    immune = by_key["embodied.quantum_immune_interface"]
    assert immune.source_equation_ids == ("EQ0105",)
    assert immune.anchor_math_ids == ("EQ0105",)
    assert "P0R05361" in immune.source_ledger_ids
    assert any("cytokine" in target for target in immune.executable_validation_targets)
    assert any("zero lambda" in control for control in immune.null_controls)

    glial = by_key["embodied.glial_sigma_control"]
    assert glial.source_equation_ids == (
        "EQ0106",
        "EQ0107",
        "EQ0108",
        "EQ0109",
        "EQ0110",
        "EQ0111",
        "EQ0112",
    )
    assert set(glial.anchor_math_ids) == {
        "EQ0106",
        "EQ0107",
        "EQ0108",
        "EQ0109",
        "EQ0110",
        "EQ0111",
        "EQ0112",
    }
    assert "P0R05406" in glial.source_ledger_ids
    assert any("gliotransmitter blockade" in target for target in glial.validation_targets)
    assert any("gamma = 0" in control for control in glial.null_controls)
    assert glial.hardware_status == "simulator_only_no_provider_submission"


def test_glial_control_builder_rejects_missing_required_anchor() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R05403"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_glial_control_validation_specs(incomplete)


def test_write_outputs_records_policy_and_report(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_glial_control_validation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["summary"]["coverage_match"] is True
    assert payload["summary"]["hardware_status"] == "simulator_only_no_provider_submission"
    assert "Paper 0 Glial-Control Validation Specs" in report
    assert "Provider submission remains out of scope" in report
    assert "embodied.glial_sigma_control" in report

    loaded = load_glial_control_validation_spec(
        "embodied.glial_sigma_control",
        spec_bundle_path=outputs["json"],
    )
    assert loaded["source_equation_ids"] == [
        "EQ0106",
        "EQ0107",
        "EQ0108",
        "EQ0109",
        "EQ0110",
        "EQ0111",
        "EQ0112",
    ]
