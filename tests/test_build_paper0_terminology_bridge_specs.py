# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 terminology bridge spec tests
"""Tests for Paper 0 terminology-bridge promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_terminology_bridge_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_terminology_bridge_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_terminology_bridge_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 terminology bridge spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.3 The Logos > Terminology Bridge",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    text = {
        610: "Terminology Bridge for Domain Experts",
        611: "maps Psi-field, UPDE, Geometric Qualia to fibre bundles, coupled oscillators, TDA",
        612: "PELA Yang-Mills-like action is heuristic regulariser and optimisation prior",
        616: "Psi-field as fibre bundle",
        617: "UPDE as coupled oscillator model",
        618: "Geometric Qualia as Topological Data Analysis",
        619: "PELA analogy is a tool, not literal claim that ethics is physics",
        623: "increasing precision of deepest priors, inverse variance",
        624: "mainstream anchor increases precision weighting and falsifiable predictions",
        626: "H_int = -lambda * Psi_s * sigma",
        628: "Psi-field is field theory section of a fibre bundle",
        630: "sigma among topological invariants, Betti numbers, Ricci curvature",
        632: "PELA sets boundary conditions or tunes parameters, not new force term",
        633: "[TABLE]",
        634: "Yang-Mills-like action is analogy and regulariser, no deductive derivation",
    }
    return [
        _record(f"P0R{number:05d}", number, text.get(number, "source text"))
        for number in range(610, 635)
    ]


def test_terminology_bridge_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_terminology_bridge_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 25
    assert bundle.summary["consumed_source_record_count"] == 25
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00610", "P0R00634"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["mainstream_anchor_count"] == 4
    assert bundle.summary["analogy_boundary_count"] == 2
    assert bundle.summary["next_source_boundary"] == "P0R00635"


def test_terminology_bridge_specs_preserve_anchor_and_analogy_boundaries() -> None:
    module = _load_module()

    bundle = module.build_terminology_bridge_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["terminology_bridge.mainstream_anchor_map"].source_equation_ids == (
        "P0R00616:psi_field_fibre_bundle_anchor",
        "P0R00617:upde_coupled_oscillator_anchor",
        "P0R00618:geometric_qualia_tda_anchor",
        "P0R00619:pela_yang_mills_tool_not_literal_claim",
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["terminology_bridge.psi_field_coupling_context"].source_formulae
    )
    assert (
        "Betti numbers or Ricci curvature as sigma candidate properties"
        in specs["terminology_bridge.sigma_topology_target"].source_formulae
    )
    assert (
        "no deductive derivation of ethics from gauge theory"
        in specs["terminology_bridge.pela_yang_mills_analogy_boundary"].source_formulae
    )
    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(610, 635))
        for spec in specs.values()
    )


def test_terminology_bridge_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00634"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_terminology_bridge_specs(incomplete)


def test_terminology_bridge_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_terminology_bridge_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_terminology_bridge_validation_spec(
        "terminology_bridge.pela_yang_mills_analogy_boundary",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00610"
    assert loaded["source_ledger_ids"][-1] == "P0R00634"
    assert "Paper 0 Terminology Bridge Specs" in report
    assert "regulariser" in report
