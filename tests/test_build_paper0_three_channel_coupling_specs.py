# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 three-channel coupling spec tests
"""Tests for Paper 0 unified coupling parameter scan promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_three_channel_coupling_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_three_channel_coupling_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_three_channel_coupling_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 three-channel coupling spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "New content to be allocated into the document:",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(7081, 7130)]


def test_three_channel_coupling_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_three_channel_coupling_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 49
    assert bundle.summary["consumed_source_record_count"] == 49
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R07081", "P0R07129"]
    assert bundle.summary["spec_count"] == 6
    assert bundle.summary["channel_count"] == 3
    assert bundle.summary["sweet_spot_window"] == [1.0e-6, 1.0e-5]


def test_three_channel_coupling_specs_preserve_ratios_constraints_and_falsification() -> None:
    module = _load_module()

    bundle = module.build_three_channel_coupling_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(7081, 7130))
        for spec in specs.values()
    )
    assert specs["three_channel_coupling.geometry_factors"].source_equation_ids == (
        "P0R07083:canonical_warp_parameters",
        "P0R07084:single_lambda0_mapping",
        "P0R07086:c_g",
        "P0R07087:c_em",
        "P0R07088:c_q",
        "P0R07089:c_s",
    )
    assert specs["three_channel_coupling.experimental_constraints"].source_equation_ids == (
        "P0R07095:three_independent_channels",
        "P0R07099:gravitational_constraint",
        "P0R07101:em_clock_constraint",
        "P0R07103:quantum_coherence_constraint",
        "P0R07105:constraint_ranking",
    )
    assert specs["three_channel_coupling.falsification_fingerprint"].source_equation_ids == (
        "P0R07121:falsifiable_fingerprint",
        "P0R07127:three_way_correlation",
        "P0R07128:falsification_boundary",
    )
    assert all("source-bounded parameter scan" in spec.claim_boundary for spec in specs.values())


def test_three_channel_coupling_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R07128"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_three_channel_coupling_specs(incomplete)


def test_three_channel_coupling_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_three_channel_coupling_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_three_channel_coupling_validation_spec(
        "three_channel_coupling.cross_channel_propagation",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R07081"
    assert loaded["source_ledger_ids"][-1] == "P0R07129"
    assert "Paper 0 Three-Channel Coupling Specs" in report
    assert "not empirical support" in report
