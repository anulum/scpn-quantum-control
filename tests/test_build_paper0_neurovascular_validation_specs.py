# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 neurovascular validation spec tests
"""Tests for Paper 0 neurovascular validation spec promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_neurovascular_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_neurovascular_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 neurovascular spec script")
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
        "section_path": "Paper 0 neurovascular test section",
        "math_ids": [] if math_ids is None else math_ids,
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    return [
        _record("P0R04877", 4877),
        _record("P0R04878", 4878),
        _record("P0R04879", 4879),
        _record("P0R04880", 4880),
        _record("P0R04882", 4882),
        _record("P0R04883", 4883),
        _record("P0R04885", 4885),
        _record("P0R04886", 4886),
        _record("P0R04887", 4887),
        _record("P0R04889", 4889),
        _record("P0R04890", 4890, math_ids=["EQ0093"]),
        _record("P0R04891", 4891),
        _record("P0R04892", 4892),
        _record("P0R04896", 4896),
        _record("P0R04897", 4897),
    ]


def test_neurovascular_spec_promotes_eq0093_with_full_source_coverage() -> None:
    module = _load_module()

    bundle = module.build_neurovascular_validation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 15
    assert bundle.summary["consumed_source_record_count"] == 15
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 1
    assert bundle.summary["spec_keys"] == ["embodied.neurovascular_phase_coupling"]
    assert bundle.summary["hardware_status"] == "simulator_only_no_provider_submission"


def test_neurovascular_spec_is_source_anchored_and_has_controls() -> None:
    module = _load_module()

    bundle = module.build_neurovascular_validation_specs(_complete_records())
    spec = bundle.specs[0]

    assert spec.key == "embodied.neurovascular_phase_coupling"
    assert spec.source_equation_ids == ("EQ0093",)
    assert "P0R04890" in spec.source_ledger_ids
    assert "P0R04887" in spec.source_ledger_ids
    assert "P0R04892" in spec.source_ledger_ids
    assert "EQ0093" in spec.anchor_math_ids
    assert any("phase locking" in target for target in spec.executable_validation_targets)
    assert any("zero K_NH" in control for control in spec.null_controls)
    assert any("Mayer" in control for control in spec.null_controls)
    assert spec.hardware_status == "simulator_only_no_provider_submission"


def test_neurovascular_builder_rejects_missing_required_anchor() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R04890"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_neurovascular_validation_specs(incomplete)


def test_write_outputs_records_policy_and_report(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_neurovascular_validation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["summary"]["coverage_match"] is True
    assert payload["summary"]["hardware_status"] == "simulator_only_no_provider_submission"
    assert "Paper 0 Neurovascular Validation Specs" in report
    assert "Provider submission remains out of scope" in report
    assert "embodied.neurovascular_phase_coupling" in report
