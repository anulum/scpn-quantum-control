# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom III teleological optimisation builder tests
"""Tests for Paper 0 Axiom III teleological-optimisation spec promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_axiom_iii_teleological_optimisation_validation_spec,
)
from scripts.build_paper0_axiom_iii_teleological_optimisation_specs import (
    SOURCE_LEDGER_IDS,
    build_axiom_iii_teleological_optimisation_specs,
    build_from_ledger,
    render_report,
    write_outputs,
)


def test_teleological_optimisation_builder_preserves_contiguous_source_boundary() -> None:
    bundle = build_from_ledger()

    assert bundle.summary["source_ledger_span"] == ["P0R00791", "P0R00799"]
    assert bundle.summary["source_record_count"] == 9
    assert bundle.summary["consumed_source_record_count"] == 9
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["axiom_heading_count"] == 3
    assert bundle.summary["sec_maximisation_count"] == 2
    assert bundle.summary["layer15_guidance_count"] == 1
    assert bundle.summary["directionality_count"] == 2
    assert bundle.summary["next_source_boundary"] == "P0R00800"
    assert [spec.key for spec in bundle.specs] == [
        "axiom_iii_teleological_optimisation.opening_context",
        "axiom_iii_teleological_optimisation.source_material_telos",
        "axiom_iii_teleological_optimisation.directional_purpose",
        "axiom_iii_teleological_optimisation.ethical_functional_guidance",
    ]
    assert all(spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in bundle.specs)


def test_teleological_optimisation_builder_keeps_source_formulae_and_boundary() -> None:
    bundle = build_from_ledger()
    specs = {spec.key: spec for spec in bundle.specs}

    assert (
        "Axiom III: The Drive of Teleological Optimisation"
        in specs["axiom_iii_teleological_optimisation.opening_context"].source_formulae
    )
    assert (
        "Formal Physical Definition: the tilde_N_t Invariance Law"
        in specs["axiom_iii_teleological_optimisation.opening_context"].source_formulae
    )
    assert (
        "Equivalence of SEC and the tilde_N_t = 1 State"
        in specs["axiom_iii_teleological_optimisation.opening_context"].source_formulae
    )
    assert (
        "universe is not random but evolving towards maximal Sustainable Ethical Coherence"
        in specs["axiom_iii_teleological_optimisation.source_material_telos"].source_formulae
    )
    assert (
        "Axiom III defines purpose"
        in specs["axiom_iii_teleological_optimisation.directional_purpose"].source_formulae
    )
    assert (
        "universe evolves to maximise Sustainable Ethical Coherence SEC"
        in specs["axiom_iii_teleological_optimisation.directional_purpose"].source_formulae
    )
    assert (
        "Ethical Functionals computed at Layer 15 guide temporal evolution"
        in specs["axiom_iii_teleological_optimisation.ethical_functional_guidance"].source_formulae
    )
    assert (
        "prime directive ultimate prior for the cosmic generative model"
        in specs["axiom_iii_teleological_optimisation.ethical_functional_guidance"].source_formulae
    )
    assert (
        "increasing coherence complexity and experiential depth"
        in specs["axiom_iii_teleological_optimisation.ethical_functional_guidance"].source_formulae
    )


def test_teleological_optimisation_builder_rejects_missing_source_records() -> None:
    records = []
    for record_id in SOURCE_LEDGER_IDS:
        if record_id == "P0R00798":
            continue
        records.append(
            {
                "ledger_id": record_id,
                "source_record_id": f"{record_id}:stub",
                "source_block_index": int(record_id[3:]),
                "section_path": "Paper 0 > Axiom III > Teleological Optimisation",
            }
        )

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        build_axiom_iii_teleological_optimisation_specs(records)


def test_teleological_optimisation_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    bundle = build_from_ledger()
    outputs = write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_axiom_iii_teleological_optimisation_validation_spec(
        "axiom_iii_teleological_optimisation.ethical_functional_guidance",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert "Paper 0 Axiom III Teleological Optimisation Specs" in report
    assert loaded["key"] == "axiom_iii_teleological_optimisation.ethical_functional_guidance"
    assert (
        "Ethical Functionals computed at Layer 15 guide temporal evolution"
        in loaded["source_formulae"]
    )
    assert "Teleological Optimisation" in render_report(bundle)
