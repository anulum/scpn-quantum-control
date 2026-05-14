# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 Triple Network spec tests
"""Tests for Paper 0 Layer 5 Triple Network spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import load_l5_triple_network_validation_spec
from scripts.build_paper0_l5_triple_network_specs import (
    build_l5_triple_network_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "docs/internal/paper0_foundational_extraction/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_l5_triple_network_builder_consumes_complete_source_span() -> None:
    bundle = build_l5_triple_network_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 19
    assert bundle.summary["consumed_source_record_count"] == 19
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06485", "P0R06503"]
    assert bundle.summary["structural_source_ledger_ids"] == [
        "P0R06485",
        "P0R06488",
        "P0R06491",
        "P0R06492",
        "P0R06493",
        "P0R06496",
        "P0R06499",
        "P0R06501",
    ]
    assert bundle.summary["caption_source_ledger_ids"] == [
        "P0R06489",
        "P0R06497",
        "P0R06500",
        "P0R06502",
    ]
    assert tuple(spec.key for spec in bundle.specs) == (
        "l5_triple_network.anatomical_mapping",
        "l5_triple_network.salience_switching",
        "l5_triple_network.interoceptive_inference",
        "l5_triple_network.homeostatic_qualia_boundary",
    )


def test_l5_triple_network_specs_preserve_mechanisms_and_validation_targets() -> None:
    bundle = build_l5_triple_network_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["l5_triple_network.anatomical_mapping"].source_mechanisms == (
        "DMN supports internally focused self-referential processing and narrative self",
        "CEN supports externally focused working memory, planning, and goal-directed control",
        "SN is anchored in anterior insula and dorsal anterior cingulate cortex",
    )
    assert specs["l5_triple_network.salience_switching"].source_mechanisms == (
        "DMN and CEN are typically anti-correlated",
        "SN detects salient prediction errors",
        "SN switches dominance between DMN and CEN when salience demands attention",
    )
    assert specs["l5_triple_network.interoceptive_inference"].source_formulae == (
        "salience approximates precision x abs(prediction_error)",
        "salience threshold crossing triggers CEN engagement",
        "below salience threshold supports DMN dominance",
    )
    assert specs["l5_triple_network.homeostatic_qualia_boundary"].source_mechanisms == (
        "emotional qualia are bounded to subjective interoceptive inference in insula",
        "psychosomatic harmonics are insula-body feedback loops via autonomic and neuroendocrine pathways",
        "organismal self-organisation is bounded to homeostasis and low-surprise body regulation",
    )
    assert (
        "not empirical evidence"
        in specs["l5_triple_network.homeostatic_qualia_boundary"].claim_boundary
    )


def test_l5_triple_network_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "l5_triple_network_specs.json"
    report_path = tmp_path / "l5_triple_network_specs.md"
    bundle = build_l5_triple_network_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_l5_triple_network_validation_spec(
        "l5_triple_network.interoceptive_inference",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6485, 6504)]
    assert loaded["hardware_status"] == "simulator_only_no_provider_submission"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 Layer 5 Triple Network Specs" in report
    assert json_path.exists()
    assert report_path.exists()
