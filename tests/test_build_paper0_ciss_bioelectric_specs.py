# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 CISS-bioelectric spec tests
"""Tests for Paper 0 Layer 3 CISS-bioelectric spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import (
    load_ciss_bioelectric_validation_spec,
)
from scripts.build_paper0_ciss_bioelectric_specs import (
    build_ciss_bioelectric_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source_validation_artifacts/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_ciss_bioelectric_builder_consumes_complete_source_span() -> None:
    bundle = build_ciss_bioelectric_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 22
    assert bundle.summary["consumed_source_record_count"] == 22
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06560", "P0R06581"]
    assert bundle.summary["equation_source_ledger_ids"] == [
        "P0R06565",
        "P0R06568",
        "P0R06571",
        "P0R06574",
        "P0R06576",
    ]
    assert tuple(spec.key for spec in bundle.specs) == (
        "ciss_bioelectric.layer3_framing",
        "ciss_bioelectric.ciss_spin_filter",
        "ciss_bioelectric.radical_pair_modulation",
        "ciss_bioelectric.bioelectric_cascade_feedback",
        "ciss_bioelectric.observable_predictions",
    )


def test_ciss_bioelectric_specs_preserve_equations_mechanisms_and_boundary() -> None:
    bundle = build_ciss_bioelectric_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["ciss_bioelectric.ciss_spin_filter"].source_formulae == (
        "H_total = epsilon_0 + (Delta/2) sigma_z + (lambda / L^2)(sigma dot L) + g S dot sigma",
        "lambda is spin-orbit coupling and generates effective B_eff in the 10-100 T source range",
    )
    assert specs["ciss_bioelectric.radical_pair_modulation"].source_formulae == (
        "H_RP = sum_i [omega_i S_iz + sum_k A_ik S_i dot I_k] + J(1/2 + 2 S_1 dot S_2)",
        "singlet/triplet ratio is modulated by B_eff from CISS",
    )
    assert specs["ciss_bioelectric.bioelectric_cascade_feedback"].source_formulae == (
        "E = -grad V_target -> activates Ca_v channels -> intracellular Ca2+ spike",
        "dV_mem/dt = -I_ion(V_mem, B_eff(lambda(E))) + I_pump",
        "lambda(E) is a function of local electric field E",
        "H_RP = H_RP_base + B_local(V_mem) dot (g_1 S_1 + g_2 S_2)",
    )
    assert specs["ciss_bioelectric.observable_predictions"].source_mechanisms == (
        "bioelectric field perturbation by optogenetics predicts epigenetic changes",
        "CISS blockade by chiral molecular disruption predicts loss of field-guided morphogenesis",
        "radical-pair yield versus applied E-field is expected to show non-linear modulation",
    )
    assert (
        "not empirical evidence"
        in specs["ciss_bioelectric.bioelectric_cascade_feedback"].claim_boundary
    )


def test_ciss_bioelectric_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "ciss_bioelectric_specs.json"
    report_path = tmp_path / "ciss_bioelectric_specs.md"
    bundle = build_ciss_bioelectric_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_ciss_bioelectric_validation_spec(
        "ciss_bioelectric.bioelectric_cascade_feedback",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6560, 6582)]
    assert loaded["hardware_status"] == "simulator_only_no_provider_submission"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 CISS-Bioelectric Feedback Specs" in report
    assert json_path.exists()
    assert report_path.exists()
