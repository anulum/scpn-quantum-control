# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 L11 NTHS computational spec tests
"""Tests for Paper 0 L11 NTHS computational experiment spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import (
    load_l11_nths_computational_validation_spec,
)
from scripts.build_paper0_l11_nths_computational_specs import (
    build_l11_nths_computational_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source_validation_artifacts/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_l11_nths_builder_consumes_complete_source_span() -> None:
    bundle = build_l11_nths_computational_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 85
    assert bundle.summary["consumed_source_record_count"] == 85
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06730", "P0R06814"]
    assert "P0R06742" in bundle.summary["equation_source_ledger_ids"]
    assert "P0R06750" in bundle.summary["equation_source_ledger_ids"]
    assert "P0R06778" in bundle.summary["equation_source_ledger_ids"]
    assert "P0R06800" in bundle.summary["equation_source_ledger_ids"]
    assert tuple(spec.key for spec in bundle.specs) == (
        "l11_nths_computational.block_framing",
        "l11_nths_computational.agent_architecture",
        "l11_nths_computational.environment_spin_glass",
        "l11_nths_computational.ai_objective_conditions",
        "l11_nths_computational.simulation_protocol",
        "l11_nths_computational.order_parameters",
        "l11_nths_computational.predicted_outcomes",
        "l11_nths_computational.statistics_falsification_extensions",
    )


def test_l11_nths_specs_preserve_core_equations_and_protocol_boundary() -> None:
    bundle = build_l11_nths_computational_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["l11_nths_computational.agent_architecture"].source_formulae == (
        "A matrix: likelihood P(observation|hidden state)",
        "B matrix: transition dynamics P(s_{t+1}|s_t, action)",
        "C matrix: preferences including confirmation bias and values",
        "D matrix: priors P(s_0)",
        "Q(s_t|o_{1:t}) proportional to P(o_t|s_t) sum_{s_{t-1}} B(s_t|s_{t-1},a) Q(s_{t-1})",
        "G(pi) = E_Q[ln Q(s|pi) - ln P(o,s|pi)]",
    )
    assert specs["l11_nths_computational.environment_spin_glass"].source_formulae == (
        "N = 1000 agents",
        "initial topology: Barabasi-Albert scale-free m=3",
        "dynamic coupling J_ij based on belief similarity/influence",
        "S_i = sign(mean hidden belief state_i) in {-1,+1}",
        "J_ij = trust/influence weight dynamic",
        "H_Noosphere = -sum_{i<j} J_ij S_i S_j",
    )
    assert specs["l11_nths_computational.order_parameters"].source_formulae == (
        "m(t) = (1/N) sum_i mean S_i(t)",
        "q_EA(t) = (1/N) sum_i mean(S_i)^2",
        "ultrametricity: compute correlation distance d(i,j) for triplets",
        "check d(i,k) <= max(d(i,j), d(j,k)) frequency",
        "cluster size distribution P(s) proportional to s^(-tau) at critical point",
    )
    assert specs["l11_nths_computational.statistics_falsification_extensions"].source_formulae == (
        "N_replicas = 50 per condition",
        "ANOVA on order parameters at t=5000",
        "Cohen d expected greater than 2.0",
        "significance threshold p < 0.001 Bonferroni corrected",
        "reject if order parameters do not show statistically significant divergence between conditions",
        "timeline 3 months and computational cost less than $5K cloud compute",
    )
    assert "not empirical evidence" in specs["l11_nths_computational.block_framing"].claim_boundary


def test_l11_nths_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "l11_nths_computational_specs.json"
    report_path = tmp_path / "l11_nths_computational_specs.md"
    bundle = build_l11_nths_computational_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_l11_nths_computational_validation_spec(
        "l11_nths_computational.environment_spin_glass",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6730, 6815)]
    assert loaded["hardware_status"] == "computational_protocol_no_external_execution"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 L11 NTHS Computational Experiment Specs" in report
    assert json_path.exists()
    assert report_path.exists()
