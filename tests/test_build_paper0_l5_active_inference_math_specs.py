# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 Active Inference math spec tests
"""Tests for Paper 0 Layer 5 Active Inference math spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import (
    load_l5_active_inference_math_validation_spec,
)
from scripts.build_paper0_l5_active_inference_math_specs import (
    build_l5_active_inference_math_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "docs/internal/paper0_foundational_extraction/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_l5_active_inference_math_builder_consumes_complete_source_span() -> None:
    bundle = build_l5_active_inference_math_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 35
    assert bundle.summary["consumed_source_record_count"] == 35
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06450", "P0R06484"]
    assert bundle.summary["equation_source_ledger_ids"] == [
        "P0R06465",
        "P0R06466",
        "P0R06469",
        "P0R06471",
        "P0R06473",
        "P0R06475",
        "P0R06476",
        "P0R06477",
        "P0R06479",
        "P0R06481",
        "P0R06482",
    ]
    assert tuple(spec.key for spec in bundle.specs) == (
        "l5_active_inference_math.generative_hierarchy",
        "l5_active_inference_math.layer_free_energy",
        "l5_active_inference_math.message_passing_update",
        "l5_active_inference_math.action_and_precision_control",
    )


def test_l5_active_inference_math_specs_preserve_all_equations() -> None:
    bundle = build_l5_active_inference_math_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["l5_active_inference_math.generative_hierarchy"].source_formulae == (
        "Layer 15: p(cosmos)",
        "Layer 14: p(dimensions|cosmos)",
        "Layer 5: p(self|body,world)",
        "Layer 1: p(quantum|classical)",
    )
    assert specs["l5_active_inference_math.layer_free_energy"].source_formulae == (
        "F_L = E_q(psi_L)[log q(psi_L) - log p(psi_L, o_L)]",
        "F_L = D_KL[q(psi_L)||p(psi_L)] - E_q[log p(o_L|psi_L)]",
    )
    assert specs["l5_active_inference_math.message_passing_update"].source_formulae == (
        "epsilon_L^up = partial F_L / partial mu_L = o_L - g(mu_L)",
        "epsilon_L^down = partial F_{L+1} / partial mu_L = mu_L - f(mu_{L+1})",
        "Delta mu_L = -kappa(epsilon_L^up + epsilon_L^down)",
    )
    assert specs["l5_active_inference_math.action_and_precision_control"].source_formulae == (
        "G(pi) = E_q[H[p(o|s,pi)]] + E_q[D_KL[q(s|pi)||p(s|C)]]",
        "G(pi) = -information_gain + divergence_from_prior",
        "a_star = argmin_pi G(pi)",
        "Delta mu = Pi^(-1) x epsilon",
        "Pi = precision matrix = inverse covariance",
        "epsilon = prediction error",
    )
    assert (
        "not empirical evidence"
        in specs["l5_active_inference_math.action_and_precision_control"].claim_boundary
    )


def test_l5_active_inference_math_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "l5_active_inference_math_specs.json"
    report_path = tmp_path / "l5_active_inference_math_specs.md"
    bundle = build_l5_active_inference_math_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_l5_active_inference_math_validation_spec(
        "l5_active_inference_math.message_passing_update",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6450, 6485)]
    assert loaded["hardware_status"] == "simulator_only_no_provider_submission"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 Layer 5 Active Inference Math Specs" in report
    assert json_path.exists()
    assert report_path.exists()
