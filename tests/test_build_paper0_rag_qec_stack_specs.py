# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 RAG QEC stack spec tests
"""Tests for Paper 0 RAG Layer 1 QEC stack spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import load_rag_qec_stack_validation_spec
from scripts.build_paper0_rag_qec_stack_specs import (
    build_rag_qec_stack_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "docs/internal/paper0_foundational_extraction/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_rag_qec_stack_builder_consumes_complete_source_span() -> None:
    bundle = build_rag_qec_stack_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 30
    assert bundle.summary["consumed_source_record_count"] == 30
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06530", "P0R06559"]
    assert bundle.summary["equation_source_ledger_ids"] == [
        "P0R06542",
        "P0R06544",
        "P0R06545",
        "P0R06546",
        "P0R06557",
    ]
    assert tuple(spec.key for spec in bundle.specs) == (
        "rag_qec_stack.insert_framing",
        "rag_qec_stack.layer1_qec_hamiltonian",
        "rag_qec_stack.gap_coherence_protection",
        "rag_qec_stack.programmability_and_observable",
    )


def test_rag_qec_stack_specs_preserve_equations_and_boundary() -> None:
    bundle = build_rag_qec_stack_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["rag_qec_stack.layer1_qec_hamiltonian"].source_formulae == (
        "H_QEC = H_MT + H_stab + H_syndrome",
        "H_MT = -J_x sum_<ij> sigma_i^x sigma_j^x - J_z sum_i sigma_i^z",
        "H_stab = -J_s sum_p S_p - J_l sum_l L_l",
        "H_syndrome = -gamma_s sum_i (sigma_i^z tensor E_i)",
    )
    assert specs["rag_qec_stack.gap_coherence_protection"].source_formulae == (
        "Delta E approximately 1.64 eV >> k_B T approximately 0.026 eV",
        "tau_coherence approximately hbar / Delta E approximately 400 fs",
        "tau_thermal approximately hbar / (k_B T) approximately 25 fs",
        "Protection Factor approximately 16x enhancement",
        "p_th = [1 - exp(-2 Delta E / k_B T)] / [1 + exp(-2 Delta E / k_B T)] approximately 10^(-14)",
    )
    assert specs["rag_qec_stack.programmability_and_observable"].source_mechanisms == (
        "tubulin conformational states act as classical control bits",
        "classical control bits select topological operations on the quantum substrate",
        "observable target is spectroscopic signature near 1.64 eV under coherent versus anaesthetic states",
    )
    assert (
        "not empirical evidence" in specs["rag_qec_stack.gap_coherence_protection"].claim_boundary
    )


def test_rag_qec_stack_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "rag_qec_stack_specs.json"
    report_path = tmp_path / "rag_qec_stack_specs.md"
    bundle = build_rag_qec_stack_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_rag_qec_stack_validation_spec(
        "rag_qec_stack.gap_coherence_protection",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6530, 6560)]
    assert loaded["hardware_status"] == "simulator_only_no_provider_submission"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 RAG QEC Stack Specs" in report
    assert json_path.exists()
    assert report_path.exists()
