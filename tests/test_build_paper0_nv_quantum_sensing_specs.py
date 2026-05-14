# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 NV quantum sensing spec tests
"""Tests for Paper 0 NV-center quantum sensing protocol spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import load_nv_quantum_sensing_validation_spec
from scripts.build_paper0_nv_quantum_sensing_specs import (
    build_nv_quantum_sensing_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "docs/internal/paper0_foundational_extraction/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_nv_quantum_sensing_builder_consumes_complete_source_span() -> None:
    bundle = build_nv_quantum_sensing_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 53
    assert bundle.summary["consumed_source_record_count"] == 53
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06677", "P0R06729"]
    assert bundle.summary["equation_source_ledger_ids"] == [
        "P0R06707",
        "P0R06711",
        "P0R06716",
        "P0R06717",
        "P0R06718",
        "P0R06720",
        "P0R06727",
        "P0R06728",
    ]
    assert tuple(spec.key for spec in bundle.specs) == (
        "nv_quantum_sensing.block_framing",
        "nv_quantum_sensing.apparatus",
        "nv_quantum_sensing.protocol_steps",
        "nv_quantum_sensing.isomorphic_replay_control",
        "nv_quantum_sensing.analysis_and_falsification",
        "nv_quantum_sensing.controls_effect_size_timeline",
    )


def test_nv_quantum_sensing_specs_preserve_protocol_equations_and_boundary() -> None:
    bundle = build_nv_quantum_sensing_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["nv_quantum_sensing.apparatus"].source_mechanisms == (
        "high-density primary cortical culture on 256-electrode MEA",
        "TTX subcritical sigma < 1 and bicuculline critical/supercritical sigma >= 1 states",
        "ensemble NV centers in diamond at 10^9 centers/mm^3",
        "less than 50 nm proximity to culture via diamond cantilever",
        "room-temperature operation without cryogenics",
        "532 nm excitation, 650-800 nm collection, 2.87 GHz microwave delivery, 30 kHz MEA sampling, and B_ambient < 10 nT shielding",
    )
    assert specs["nv_quantum_sensing.isomorphic_replay_control"].source_formulae == (
        "silence culture with TTX",
        "electrically replay exact spike train from spontaneous step via MEA",
        "identical classical B-field but FIM approximately 0",
        "measure Gamma_replay",
    )
    assert specs["nv_quantum_sensing.analysis_and_falsification"].source_formulae == (
        "Delta Gamma = Gamma_spontaneous - Gamma_replay",
        "hypothesis: Delta Gamma > 0",
        "model: Gamma = beta_0 + beta_1 B_classical + beta_2 FIM_proxy + epsilon",
        "prediction: beta_2 > 0 significant independent of beta_1",
        "reject if Delta Gamma <= 0 or beta_2 not significant with p > 0.05",
    )
    assert specs["nv_quantum_sensing.controls_effect_size_timeline"].source_formulae == (
        "temperature stability +/-0.1 C",
        "NV ensemble uniformity less than 5 percent T2* variation across diamond",
        "Delta Gamma / Gamma_baseline approximately 0.05-0.15",
        "timeline 6 days per trial, N=5 cultures, approximately 6 weeks total",
        "cost estimate approximately $150K",
    )
    assert "not empirical evidence" in specs["nv_quantum_sensing.block_framing"].claim_boundary


def test_nv_quantum_sensing_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "nv_quantum_sensing_specs.json"
    report_path = tmp_path / "nv_quantum_sensing_specs.md"
    bundle = build_nv_quantum_sensing_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_nv_quantum_sensing_validation_spec(
        "nv_quantum_sensing.analysis_and_falsification",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6677, 6730)]
    assert loaded["hardware_status"] == "protocol_design_no_lab_execution"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 NV-Center Quantum Sensing Protocol Specs" in report
    assert json_path.exists()
    assert report_path.exists()
