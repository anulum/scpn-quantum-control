# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 macro-transition validation spec tests
"""Tests for Paper 0 macro-transition validation spec promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_macro_transition_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_macro_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 macro-transition spec script")
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
        "section_path": "Paper 0 test section",
        "math_ids": [] if math_ids is None else math_ids,
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    return [
        _record("P0R00007", 7),
        _record("P0R00382", 382),
        _record("P0R00538", 538),
        _record("P0R05266", 5266),
        _record("P0R05272", 5272),
        _record("P0R05556", 5556),
        _record("P0R05557", 5557, math_ids=["EQ0113"]),
        _record("P0R05558", 5558),
        _record("P0R05636", 5636),
        _record("P0R05639", 5639, math_ids=["EQ0114"]),
    ]


def test_macro_specs_promote_spin_glass_and_rg_flow_with_full_coverage() -> None:
    module = _load_module()

    bundle = module.build_macro_transition_validation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 10
    assert bundle.summary["consumed_source_record_count"] == 10
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 2
    assert bundle.summary["spec_keys"] == [
        "nths.spin_glass_hamiltonian",
        "macro_transition.effective_coupling_rg",
    ]
    assert bundle.summary["hardware_status"] == "simulator_only_no_provider_submission"


def test_macro_specs_are_source_anchored_and_have_null_controls() -> None:
    module = _load_module()

    bundle = module.build_macro_transition_validation_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    spin_glass = specs["nths.spin_glass_hamiltonian"]
    assert spin_glass.source_equation_ids == ("EQ0113",)
    assert "P0R05557" in spin_glass.source_ledger_ids
    assert "P0R05266" in spin_glass.source_ledger_ids
    assert any("q_EA" in target for target in spin_glass.executable_validation_targets)
    assert any("shuffle" in control for control in spin_glass.null_controls)
    assert spin_glass.hardware_status == "simulator_only_no_provider_submission"

    rg_flow = specs["macro_transition.effective_coupling_rg"]
    assert rg_flow.source_equation_ids == ("EQ0114",)
    assert "P0R05639" in rg_flow.source_ledger_ids
    assert "P0R00538" in rg_flow.source_ledger_ids
    assert any("fixed point" in target for target in rg_flow.executable_validation_targets)
    assert any("constant beta" in control for control in rg_flow.null_controls)


def test_macro_spec_builder_rejects_missing_required_anchor() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R05557"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_macro_transition_validation_specs(incomplete)


def test_write_outputs_records_policy_and_report(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_macro_transition_validation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["summary"]["coverage_match"] is True
    assert payload["summary"]["hardware_status"] == "simulator_only_no_provider_submission"
    assert "Paper 0 Macro-Transition Validation Specs" in report
    assert "Provider submission remains out of scope" in report
    assert "nths.spin_glass_hamiltonian" in report
