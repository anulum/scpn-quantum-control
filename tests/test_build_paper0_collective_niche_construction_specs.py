# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 collective niche construction spec tests
"""Tests for Paper 0 collective niche construction spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import (
    load_collective_niche_construction_validation_spec,
)
from scripts.build_paper0_collective_niche_construction_specs import (
    build_collective_niche_construction_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "docs/internal/paper0_foundational_extraction/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_collective_niche_builder_consumes_complete_source_span() -> None:
    bundle = build_collective_niche_construction_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 11
    assert bundle.summary["consumed_source_record_count"] == 11
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06519", "P0R06529"]
    assert bundle.summary["structural_source_ledger_ids"] == [
        "P0R06519",
        "P0R06521",
        "P0R06528",
    ]
    assert bundle.summary["caption_source_ledger_ids"] == ["P0R06529"]
    assert tuple(spec.key for spec in bundle.specs) == (
        "collective_niche.shared_generative_model",
        "collective_niche.noosphere_entrainment",
        "collective_niche.biosphere_feedback_loop",
        "collective_niche.gaian_synchrony_boundary",
    )


def test_collective_niche_specs_preserve_mechanisms_and_boundaries() -> None:
    bundle = build_collective_niche_construction_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["collective_niche.shared_generative_model"].source_mechanisms == (
        "culture is framed as agents converging on a shared generative model",
        "shared beliefs, values, language, and norms make others predictable",
        "communication, imitation, and shared artefacts actively achieve convergence",
    )
    assert specs["collective_niche.noosphere_entrainment"].source_mechanisms == (
        "cultural attractors and memes are components of the shared generative model",
        "institutions, rituals, language, and art entrain individual generative models",
        "collective free-energy reduction is linked to Noosphere emergence",
    )
    assert specs["collective_niche.biosphere_feedback_loop"].source_mechanisms == (
        "collective niche construction modifies the biosphere to fit shared predictions",
        "modified environment supplies training data to the next generation",
        "collective mind and planet form a co-evolutionary feedback loop",
    )
    assert specs["collective_niche.gaian_synchrony_boundary"].source_formulae == (
        "collective mind shapes planet and earth shapes collective mind",
        "planetary-scale active inference seeks mutual predictability, stability, and coherence",
    )
    assert (
        "not empirical evidence"
        in specs["collective_niche.gaian_synchrony_boundary"].claim_boundary
    )


def test_collective_niche_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "collective_niche_specs.json"
    report_path = tmp_path / "collective_niche_specs.md"
    bundle = build_collective_niche_construction_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_collective_niche_construction_validation_spec(
        "collective_niche.biosphere_feedback_loop",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6519, 6530)]
    assert loaded["hardware_status"] == "simulator_only_no_provider_submission"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 Collective Niche Construction Specs" in report
    assert json_path.exists()
    assert report_path.exists()
