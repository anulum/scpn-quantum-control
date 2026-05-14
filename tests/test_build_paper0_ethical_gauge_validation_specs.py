# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 ethical-gauge spec tests
"""Tests for Paper 0 EQ0123-EQ0128 validation spec promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_ethical_gauge_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_ethical_gauge_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_ethical_gauge_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 ethical-gauge spec script")
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
        "section_path": "Paper 0 ethical gauge test section",
        "math_ids": [] if math_ids is None else math_ids,
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [
        _record("P0R06088", 6088),
        _record("P0R06089", 6089),
        _record("P0R06090", 6090),
        _record("P0R06091", 6091),
        _record("P0R06092", 6092),
        _record("P0R06093", 6093, math_ids=["EQ0123"]),
        _record("P0R06094", 6094),
        _record("P0R06095", 6095),
        _record("P0R06096", 6096),
        _record("P0R06097", 6097, math_ids=["EQ0124"]),
        _record("P0R06098", 6098),
        _record("P0R06099", 6099),
        _record("P0R06100", 6100),
        _record("P0R06101", 6101, math_ids=["EQ0125"]),
        _record("P0R06102", 6102),
        _record("P0R06103", 6103, math_ids=["EQ0126"]),
        _record("P0R06104", 6104),
        _record("P0R06105", 6105, math_ids=["EQ0127"]),
        _record("P0R06106", 6106),
        _record("P0R06107", 6107),
        _record("P0R06108", 6108),
        _record("P0R06109", 6109, math_ids=["EQ0128"]),
        _record("P0R06110", 6110),
        _record("P0R06111", 6111),
        _record("P0R06112", 6112),
    ]


def test_ethical_gauge_specs_promote_eq0123_eq0128() -> None:
    module = _load_module()

    bundle = module.build_ethical_gauge_validation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 25
    assert bundle.summary["consumed_source_record_count"] == 25
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["spec_keys"] == [
        "computational.ethical_yang_mills_action",
        "computational.ethical_connection_boundary",
        "computational.causal_entropic_force",
    ]
    assert bundle.summary["hardware_status"] == "simulator_only_no_provider_submission"


def test_ethical_gauge_specs_have_controls_and_falsifiers() -> None:
    module = _load_module()

    bundle = module.build_ethical_gauge_validation_specs(_complete_records())
    by_key = {spec.key: spec for spec in bundle.specs}

    action = by_key["computational.ethical_yang_mills_action"]
    assert action.source_equation_ids == ("EQ0123", "EQ0124")
    assert set(action.anchor_math_ids) == {"EQ0123", "EQ0124"}
    assert any("gauge" in target for target in action.executable_validation_targets)
    assert any("wrong-sign" in control for control in action.null_controls)

    boundary = by_key["computational.ethical_connection_boundary"]
    assert boundary.source_equation_ids == ("EQ0125", "EQ0126", "EQ0127")
    assert set(boundary.anchor_math_ids) == {"EQ0125", "EQ0126", "EQ0127"}
    assert any("boundary-flux" in target for target in boundary.executable_validation_targets)
    assert any("wrong-sign" in control for control in boundary.null_controls)

    cef = by_key["computational.causal_entropic_force"]
    assert cef.source_equation_ids == ("EQ0128",)
    assert cef.anchor_math_ids == ("EQ0128",)
    assert any("entropy gradient" in target for target in cef.executable_validation_targets)
    assert any("flat-entropy" in control for control in cef.null_controls)
    assert cef.hardware_status == "simulator_only_no_provider_submission"


def test_ethical_gauge_builder_rejects_missing_required_anchor() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06103"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_ethical_gauge_validation_specs(incomplete)


def test_write_outputs_records_policy_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_ethical_gauge_validation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_ethical_gauge_validation_spec(
        "computational.causal_entropic_force",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert payload["summary"]["hardware_status"] == "simulator_only_no_provider_submission"
    assert loaded["source_equation_ids"] == ["EQ0128"]
    assert "Paper 0 Ethical-Gauge Validation Specs" in report
    assert "Provider submission remains out of scope" in report
    assert "computational.ethical_connection_boundary" in report
