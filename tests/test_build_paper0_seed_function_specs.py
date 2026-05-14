# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 seed-function spec tests
"""Tests for Paper 0 Python-format teleological seed promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_seed_function_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_seed_function_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_seed_function_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 seed-function spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 teleological seed function",
        "math_ids": [],
        "image_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6363, 6378)]


def test_seed_function_specs_consume_complete_contiguous_source_span() -> None:
    module = _load_module()

    bundle = module.build_seed_function_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 15
    assert bundle.summary["consumed_source_record_count"] == 15
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06363", "P0R06377"]
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["spec_keys"] == [
        "seed_function.python_format_source_boundary",
        "seed_function.mu_squared_seed_formula",
        "seed_function.return_payload_contract",
        "seed_function.conformal_continuity_boundary",
    ]


def test_seed_function_specs_preserve_code_like_source_and_structural_boundary() -> None:
    module = _load_module()

    bundle = module.build_seed_function_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(spec.anchor_math_ids == () for spec in specs.values())
    assert all(spec.validation_targets for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())
    assert specs["seed_function.mu_squared_seed_formula"].source_equation_ids == (
        "P0R06371:mu_squared_seed",
    )
    assert specs["seed_function.return_payload_contract"].source_formulae == (
        "ssb_bias_magnitude = mu_squared_seed",
        "is_random_reset = False",
    )
    assert specs["seed_function.conformal_continuity_boundary"].source_formulae == (
        "conformal_continuity = prev_cycle_sec > 0",
    )
    assert bundle.summary["structural_source_ledger_ids"] == ["P0R06377"]


def test_seed_function_builder_rejects_missing_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06371"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_seed_function_specs(incomplete)


def test_seed_function_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_seed_function_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_seed_function_validation_spec(
        "seed_function.mu_squared_seed_formula",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6363, 6378)]
    assert loaded["source_formulae"] == [
        "mu_squared_seed = sqrt(prev_cycle_sec / coupling_constant_g)"
    ]
    assert "Paper 0 Seed Function Specs" in report
    assert "not empirical evidence" in report
