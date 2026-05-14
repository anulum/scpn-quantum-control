# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational-threshold spec tests
"""Tests for Paper 0 EQ0119-EQ0122 validation spec promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_computational_threshold_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_computational_threshold_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_computational_threshold_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 computational-threshold spec script")
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
        "section_path": "Paper 0 computational threshold test section",
        "math_ids": [] if math_ids is None else math_ids,
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    return [
        _record("P0R05986", 5986),
        _record("P0R05987", 5987),
        _record("P0R05988", 5988),
        _record("P0R05989", 5989),
        _record("P0R05990", 5990, math_ids=["EQ0119"]),
        _record("P0R05991", 5991),
        _record("P0R06051", 6051),
        _record("P0R06052", 6052),
        _record("P0R06053", 6053),
        _record("P0R06054", 6054),
        _record("P0R06055", 6055, math_ids=["EQ0120"]),
        _record("P0R06056", 6056),
        _record("P0R06069", 6069),
        _record("P0R06070", 6070),
        _record("P0R06071", 6071, math_ids=["EQ0121", "EQ0122"]),
        _record("P0R06072", 6072),
    ]


def test_computational_threshold_specs_promote_eq0119_eq0122() -> None:
    module = _load_module()

    bundle = module.build_computational_threshold_validation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 16
    assert bundle.summary["consumed_source_record_count"] == 16
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["spec_keys"] == [
        "computational.iit_or_threshold",
        "computational.coherence_noether_current",
        "computational.information_energy_transduction",
    ]
    assert bundle.summary["hardware_status"] == "simulator_only_no_provider_submission"


def test_computational_threshold_specs_have_controls_and_falsifiers() -> None:
    module = _load_module()

    bundle = module.build_computational_threshold_validation_specs(_complete_records())
    by_key = {spec.key: spec for spec in bundle.specs}

    threshold = by_key["computational.iit_or_threshold"]
    assert threshold.source_equation_ids == ("EQ0119",)
    assert threshold.anchor_math_ids == ("EQ0119",)
    assert any("threshold" in target for target in threshold.executable_validation_targets)
    assert any("shuffle" in control for control in threshold.null_controls)

    noether = by_key["computational.coherence_noether_current"]
    assert noether.source_equation_ids == ("EQ0120",)
    assert noether.anchor_math_ids == ("EQ0120",)
    assert any("divergence" in target for target in noether.executable_validation_targets)
    assert any("phase-broken" in control for control in noether.null_controls)

    iet = by_key["computational.information_energy_transduction"]
    assert iet.source_equation_ids == ("EQ0121", "EQ0122")
    assert set(iet.anchor_math_ids) == {"EQ0121", "EQ0122"}
    assert any("Gaussian" in target for target in iet.executable_validation_targets)
    assert any("non-positive rho" in control for control in iet.null_controls)
    assert iet.hardware_status == "simulator_only_no_provider_submission"


def test_computational_threshold_builder_rejects_missing_required_anchor() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06055"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_computational_threshold_validation_specs(incomplete)


def test_write_outputs_records_policy_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_computational_threshold_validation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_computational_threshold_validation_spec(
        "computational.coherence_noether_current",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert payload["summary"]["hardware_status"] == "simulator_only_no_provider_submission"
    assert loaded["source_equation_ids"] == ["EQ0120"]
    assert "Paper 0 Computational-Threshold Validation Specs" in report
    assert "Provider submission remains out of scope" in report
    assert "computational.information_energy_transduction" in report
