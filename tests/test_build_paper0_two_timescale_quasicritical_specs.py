# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 two-timescale quasicritical spec tests
"""Tests for Paper 0 two-timescale quasicritical controller spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import (
    load_two_timescale_quasicritical_validation_spec,
)
from scripts.build_paper0_two_timescale_quasicritical_specs import (
    build_two_timescale_quasicritical_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "docs/internal/paper0_foundational_extraction/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_two_timescale_builder_consumes_complete_source_span() -> None:
    bundle = build_two_timescale_quasicritical_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 31
    assert bundle.summary["consumed_source_record_count"] == 31
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06646", "P0R06676"]
    assert bundle.summary["equation_source_ledger_ids"] == [
        "P0R06660",
        "P0R06661",
        "P0R06663",
        "P0R06664",
        "P0R06667",
        "P0R06668",
        "P0R06670",
        "P0R06671",
        "P0R06673",
        "P0R06674",
    ]
    assert tuple(spec.key for spec in bundle.specs) == (
        "two_timescale_quasicritical.block_framing",
        "two_timescale_quasicritical.dual_channel_architecture",
        "two_timescale_quasicritical.affective_gain_scheduling",
        "two_timescale_quasicritical.bibo_stability_certificate",
        "two_timescale_quasicritical.operational_consequence",
    )


def test_two_timescale_specs_preserve_equations_controls_and_boundary() -> None:
    bundle = build_two_timescale_quasicritical_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["two_timescale_quasicritical.dual_channel_architecture"].source_mechanisms == (
        "fast channel tau_f provides MS-QEC error correction and local homeostatic feedback",
        "fast gain G_f(sigma,A) maintains coherence and suppresses error",
        "slow channel tau_s >> tau_f supports controlled drift in the quasicritical band",
        "slow gain G_s(sigma,A) preserves sensitivity and state-space sampling",
    )
    assert specs["two_timescale_quasicritical.affective_gain_scheduling"].source_formulae == (
        "A = -grad F (affective landscape steepness)",
        "G_f(sigma) = G_f,min + k_f |partial A / partial sigma| + k_f_prime |sigma - 1|",
        "G_s(sigma) = G_s,max * Window(|sigma - 1| <= delta) * [1 - tanh(c |partial A / partial sigma|)]",
        "flat landscape + near sigma=1 allows exploration",
    )
    assert specs["two_timescale_quasicritical.bibo_stability_certificate"].source_formulae == (
        "V_total = V_fast + V_slow",
        "V_total = (sigma - 1)^2 + beta (R - R_star)^2",
        "under tau_f / tau_s << 1: dV_total/dt <= -alpha_f V_fast - alpha_s V_slow + bounded noise",
        "all trajectories remain bounded (BIBO stable)",
    )
    assert specs["two_timescale_quasicritical.operational_consequence"].source_formulae == (
        "high surprise steep |partial A / partial sigma| drives G_f up and G_s down for exploit",
        "low surprise flat |partial A / partial sigma| near sigma=1 maintains G_f and raises G_s for explore",
        "exploration-exploitation dilemma is addressed at the level of criticality maintenance",
    )
    assert (
        "not empirical evidence"
        in specs["two_timescale_quasicritical.block_framing"].claim_boundary
    )


def test_two_timescale_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "two_timescale_quasicritical_specs.json"
    report_path = tmp_path / "two_timescale_quasicritical_specs.md"
    bundle = build_two_timescale_quasicritical_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_two_timescale_quasicritical_validation_spec(
        "two_timescale_quasicritical.affective_gain_scheduling",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6646, 6677)]
    assert loaded["hardware_status"] == "simulator_only_no_provider_submission"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 Two-Timescale Quasicritical Controller Specs" in report
    assert json_path.exists()
    assert report_path.exists()
