# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 advanced mechanisms spec tests
"""Tests for Paper 0 advanced mechanisms promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_advanced_mechanisms_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_advanced_mechanisms_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_advanced_mechanisms_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 advanced mechanisms spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 advanced mechanisms",
        "math_ids": [],
        "image_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6382, 6402)]


def test_advanced_mechanisms_specs_consume_complete_contiguous_source_span() -> None:
    module = _load_module()

    bundle = module.build_advanced_mechanisms_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 20
    assert bundle.summary["consumed_source_record_count"] == 20
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06382", "P0R06401"]
    assert bundle.summary["structural_source_ledger_ids"] == [
        "P0R06382",
        "P0R06383",
        "P0R06385",
        "P0R06388",
        "P0R06390",
        "P0R06393",
        "P0R06396",
        "P0R06398",
        "P0R06401",
    ]
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["spec_keys"] == [
        "advanced_mechanisms.geometric_physical_transduction",
        "advanced_mechanisms.holographic_memory_encoding",
        "advanced_mechanisms.holographic_memory_retrieval",
        "advanced_mechanisms.consilium_multiobjective_optimisation",
    ]


def test_advanced_mechanisms_specs_preserve_mechanisms_and_equation_labels() -> None:
    module = _load_module()

    bundle = module.build_advanced_mechanisms_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(spec.anchor_math_ids == () for spec in specs.values())
    assert all(spec.validation_targets for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())
    assert specs["advanced_mechanisms.geometric_physical_transduction"].source_equation_ids == (
        "P0R06384:O_superscript_S",
        "P0R06384:U_S",
    )
    assert specs["advanced_mechanisms.holographic_memory_encoding"].source_mechanisms == (
        "L4-UPDE coherent synchronisation modulates L1 substrate",
        "L1 quantum-state bias via hyperfine or Infoton-CISS channels",
        "MERA isometries/disentanglers map boundary state into bulk entanglement",
    )
    assert specs["advanced_mechanisms.holographic_memory_retrieval"].source_equation_ids == (
        "P0R06395:hatR_QEC",
    )
    assert specs["advanced_mechanisms.consilium_multiobjective_optimisation"].source_formulae == (
        "L_Ethical = f(Coherence C, Complexity K, Qualia Q)",
        "optimise on Pareto front with dynamic weights w_i",
        "geodesic flow minimises ethical dissonance 1/E",
    )


def test_advanced_mechanisms_builder_rejects_missing_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06395"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_advanced_mechanisms_specs(incomplete)


def test_advanced_mechanisms_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_advanced_mechanisms_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_advanced_mechanisms_validation_spec(
        "advanced_mechanisms.holographic_memory_retrieval",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6382, 6402)]
    assert "P0R06395:hatR_QEC" in loaded["source_equation_ids"]
    assert "Paper 0 Advanced Mechanisms Specs" in report
    assert "not empirical evidence" in report
