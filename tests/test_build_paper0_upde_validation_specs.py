# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE validation spec tests
"""Tests for Paper 0 UPDE validation spec promotion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_upde_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_upde_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 UPDE validation spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _anchor(
    ledger_id: str,
    source_record_id: str,
    key: str,
    math_ids: list[str],
    block_index: int,
) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": source_record_id,
        "source_block_index": block_index,
        "paper0_equation_record_keys": [key],
        "math_ids": math_ids,
    }


def _complete_anchor_records() -> list[dict[str, object]]:
    return [
        _anchor("P0R00520", "P0B00520", "upde.base_phase", ["EQ0003"], 520),
        _anchor("P0R02507", "P0B02507", "upde.base_phase", ["EQ0032"], 2507),
        _anchor("P0R02510", "P0B02510", "upde.interlayer_coupling", ["EQ0033"], 2510),
        _anchor("P0R02512", "P0B02512", "upde.field_coupling", ["EQ0034"], 2512),
        _anchor("P0R02530", "P0B02530", "upde.base_phase", ["EQ0036", "EQ0037"], 2530),
        _anchor("P0R02622", "P0B02622", "upde.base_phase", ["EQ0039"], 2622),
        _anchor("P0R02630", "P0B02630", "upde.interlayer_coupling", ["EQ0040"], 2630),
        _anchor("P0R02634", "P0B02634", "upde.field_coupling", ["EQ0041"], 2634),
        _anchor("P0R02642", "P0B02642", "upde.natural_gradient", ["EQ0042"], 2642),
        _anchor("P0R02644", "P0B02644", "upde.field_coupling", ["EQ0043"], 2644),
        _anchor("P0R02910", "P0B02910", "upde.adaptive_coupling", ["EQ0045"], 2910),
        _anchor("P0R06120", "P0B06120", "upde.base_phase", ["EQ0129"], 6120),
    ]


def test_build_upde_specs_consumes_every_source_anchor() -> None:
    module = _load_module()

    bundle = module.build_upde_validation_specs(_complete_anchor_records())

    assert bundle.summary["anchor_record_count"] == 12
    assert bundle.summary["consumed_anchor_record_count"] == 12
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["unconsumed_anchor_ledger_ids"] == []


def test_specs_are_source_anchored_and_simulator_only() -> None:
    module = _load_module()

    bundle = module.build_upde_validation_specs(_complete_anchor_records())

    for spec in bundle.specs:
        assert spec.domain_review_status == "promoted_to_validation_spec"
        assert spec.hardware_status == "simulator_only_no_provider_submission"
        assert spec.source_ledger_ids
        assert spec.source_record_ids
        assert spec.source_equation_ids
        assert spec.variables
        assert spec.assumptions
        assert spec.validation_targets
        assert spec.executable_validation_targets
        assert spec.null_controls
        assert spec.implementation_links
        assert "provider_submission" not in spec.validation_protocol


def test_table_anchor_preserves_unmapped_non_upde_math_id() -> None:
    module = _load_module()

    bundle = module.build_upde_validation_specs(_complete_anchor_records())
    base_phase = next(spec for spec in bundle.specs if spec.key == "upde.base_phase")

    assert "P0R02530" in base_phase.source_ledger_ids
    assert "EQ0037" in base_phase.anchor_math_ids
    assert base_phase.unmapped_anchor_math_ids == ("EQ0036",)


def test_report_lists_protocols_and_coverage_policy() -> None:
    module = _load_module()
    bundle = module.build_upde_validation_specs(_complete_anchor_records())

    report = module.build_validation_report(bundle)

    assert "Coverage status: `match`" in report
    assert "paper0.upde.base_phase.xy_gradient_and_locking" in report
    assert "simulator_only_no_provider_submission" in report
    assert "Provider submission remains out of scope" in report
