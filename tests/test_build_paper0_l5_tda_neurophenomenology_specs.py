# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 TDA/neurophenomenology spec tests
"""Tests for Paper 0 Layer 5 TDA/neurophenomenology spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import (
    load_l5_tda_neurophenomenology_validation_spec,
)
from scripts.build_paper0_l5_tda_neurophenomenology_specs import (
    build_l5_tda_neurophenomenology_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "docs/internal/paper0_foundational_extraction/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_l5_tda_neurophenomenology_builder_consumes_complete_source_span() -> None:
    bundle = build_l5_tda_neurophenomenology_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 15
    assert bundle.summary["consumed_source_record_count"] == 15
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06504", "P0R06518"]
    assert bundle.summary["structural_source_ledger_ids"] == [
        "P0R06504",
        "P0R06508",
        "P0R06510",
        "P0R06513",
        "P0R06515",
        "P0R06517",
    ]
    assert bundle.summary["caption_source_ledger_ids"] == [
        "P0R06509",
        "P0R06511",
        "P0R06514",
    ]
    assert bundle.summary["protocol_step_ledger_ids"] == ["P0R06518"]
    assert tuple(spec.key for spec in bundle.specs) == (
        "l5_tda_neurophenomenology.geometric_qualia_hypothesis",
        "l5_tda_neurophenomenology.neurophenomenology_protocol",
        "l5_tda_neurophenomenology.persistent_homology_features",
        "l5_tda_neurophenomenology.qualia_richness_regression",
    )


def test_l5_tda_neurophenomenology_specs_preserve_formulae_and_protocol() -> None:
    bundle = build_l5_tda_neurophenomenology_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["l5_tda_neurophenomenology.geometric_qualia_hypothesis"].source_mechanisms == (
        "quality of consciousness is hypothesised to be determined by geometry of the consciousness manifold M",
        "neurophenomenology supplies structured first-person experience vectors",
        "TDA supplies neural-state manifold topology from EEG or fMRI",
    )
    assert specs["l5_tda_neurophenomenology.persistent_homology_features"].source_formulae == (
        "persistence distance from diagonal contributes to sum_k b_k(M)",
        "persistent bars represent feature lifetimes across filtration",
        "Betti features include b0 connected components, b1 loops, and bk higher-dimensional voids",
    )
    assert specs["l5_tda_neurophenomenology.qualia_richness_regression"].source_formulae == (
        "Qualia Richness proportional_to Vol(M) x sum_k b_k(M)",
        "regress richness, intensity, and structure against Vol(M) x sum_k b_k(M)",
    )
    assert specs[
        "l5_tda_neurophenomenology.neurophenomenology_protocol"
    ].source_protocol_steps == (
        "record high-density EEG while eliciting varied subjective experiences",
        "conduct structured neurophenomenological interview immediately after task",
        "score reports for richness, intensity, and structural complexity",
        "compute Betti numbers of neural manifold M with TDA",
        "test systematic correlation between reports and topological features",
    )
    assert (
        "not empirical evidence"
        in specs["l5_tda_neurophenomenology.qualia_richness_regression"].claim_boundary
    )


def test_l5_tda_neurophenomenology_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "l5_tda_neurophenomenology_specs.json"
    report_path = tmp_path / "l5_tda_neurophenomenology_specs.md"
    bundle = build_l5_tda_neurophenomenology_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_l5_tda_neurophenomenology_validation_spec(
        "l5_tda_neurophenomenology.qualia_richness_regression",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6504, 6519)]
    assert loaded["hardware_status"] == "simulator_only_no_provider_submission"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 Layer 5 TDA Neurophenomenology Specs" in report
    assert json_path.exists()
    assert report_path.exists()
