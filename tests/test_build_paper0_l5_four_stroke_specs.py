# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 four-stroke spec tests
"""Tests for Paper 0 Layer 5 four-stroke engine spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import (
    load_l5_four_stroke_validation_spec,
)
from scripts.build_paper0_l5_four_stroke_specs import (
    build_l5_four_stroke_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "docs/internal/paper0_foundational_extraction/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_l5_four_stroke_builder_consumes_complete_source_span() -> None:
    bundle = build_l5_four_stroke_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 33
    assert bundle.summary["consumed_source_record_count"] == 33
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06582", "P0R06614"]
    assert bundle.summary["equation_source_ledger_ids"] == [
        "P0R06595",
        "P0R06596",
        "P0R06612",
    ]
    assert tuple(spec.key for spec in bundle.specs) == (
        "l5_four_stroke.engine_framing",
        "l5_four_stroke.policy_selection",
        "l5_four_stroke.prediction_generation",
        "l5_four_stroke.error_processing",
        "l5_four_stroke.model_consolidation",
        "l5_four_stroke.upde_coherence_prediction",
    )


def test_l5_four_stroke_specs_preserve_phases_equations_and_boundary() -> None:
    bundle = build_l5_four_stroke_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["l5_four_stroke.policy_selection"].source_mechanisms == (
        "basal ganglia evaluate competing policies pi based on reward predictions",
        "basal ganglia output selective disinhibition that releases one action and suppresses others",
        "policy selection implements precision weighting over policy space",
    )
    assert specs["l5_four_stroke.prediction_generation"].source_mechanisms == (
        "cerebellum acts as universal forward model receiving efference copy",
        "cerebellum computes high-fidelity sensory consequence predictions",
        "cerebellum projects top-down signal to cortex and implements generative model f(pi)",
    )
    assert specs["l5_four_stroke.error_processing"].source_formulae == (
        "Perception = Sensory input - Prediction",
        "Residual = Prediction Error epsilon = (y - y_hat)",
        "prediction error epsilon propagates up hierarchy for model updating",
        "error processing implements gradient of Free Energy F",
    )
    assert specs["l5_four_stroke.model_consolidation"].source_mechanisms == (
        "NREM uses hippocampal replay plus cortical slow oscillations",
        "NREM supports memory transfer from L5 to L9",
        "NREM synaptic homeostasis restores criticality sigma toward 1",
        "REM performs offline policy simulation, explores counterfactual trajectories, and refines generative model parameters",
    )
    assert specs["l5_four_stroke.upde_coherence_prediction"].source_formulae == (
        "theta_BG(t): Policy phase",
        "theta_CB(t): Prediction phase",
        "theta_CTX(t): Error phase",
        "eta_Sleep(t): Resetting noise during consolidation",
        "R_L5 = |mean(exp(i[theta_BG - theta_CB - theta_CTX]))|",
        "TMS disruption of a specific phase predicts selective impairment",
    )
    assert (
        "not empirical evidence"
        in specs["l5_four_stroke.upde_coherence_prediction"].claim_boundary
    )


def test_l5_four_stroke_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "l5_four_stroke_specs.json"
    report_path = tmp_path / "l5_four_stroke_specs.md"
    bundle = build_l5_four_stroke_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_l5_four_stroke_validation_spec(
        "l5_four_stroke.upde_coherence_prediction",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6582, 6615)]
    assert loaded["hardware_status"] == "simulator_only_no_provider_submission"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 Layer 5 Four-Stroke Engine Specs" in report
    assert json_path.exists()
    assert report_path.exists()
