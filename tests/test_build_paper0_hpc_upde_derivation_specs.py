# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 HPC-UPDE derivation spec tests
"""Tests for Paper 0 HPC-UPDE mathematical bridge spec promotion."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.paper0.spec_loader import (
    load_hpc_upde_derivation_validation_spec,
)
from scripts.build_paper0_hpc_upde_derivation_specs import (
    build_hpc_upde_derivation_specs,
    build_validation_report,
    load_jsonl,
    write_outputs,
)

LEDGER_PATH = Path(
    "paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source_validation_artifacts/paper0_canonical_review_ledger_2026-05-13.jsonl"
)


def test_hpc_upde_derivation_builder_consumes_complete_source_span() -> None:
    bundle = build_hpc_upde_derivation_specs(load_jsonl(LEDGER_PATH))

    assert bundle.summary["source_record_count"] == 31
    assert bundle.summary["consumed_source_record_count"] == 31
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06615", "P0R06645"]
    assert bundle.summary["equation_source_ledger_ids"] == [
        "P0R06624",
        "P0R06628",
        "P0R06630",
        "P0R06631",
        "P0R06633",
        "P0R06634",
        "P0R06642",
        "P0R06644",
    ]
    assert tuple(spec.key for spec in bundle.specs) == (
        "hpc_upde_derivation.block_framing",
        "hpc_upde_derivation.free_energy_functional",
        "hpc_upde_derivation.gradient_descent",
        "hpc_upde_derivation.upde_core_equation",
        "hpc_upde_derivation.hpc_interpretation",
        "hpc_upde_derivation.active_inference_boundary",
    )


def test_hpc_upde_derivation_specs_preserve_equations_interpretation_and_boundary() -> None:
    bundle = build_hpc_upde_derivation_specs(load_jsonl(LEDGER_PATH))
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["hpc_upde_derivation.free_energy_functional"].source_formulae == (
        "F(theta_1,...,theta_N) = -sum_{i,j} K_ij cos(theta_j - theta_i)",
        "minima occur at zero phase differences; perfect synchrony = zero prediction error",
    )
    assert specs["hpc_upde_derivation.gradient_descent"].source_formulae == (
        "d theta_i / dt = -partial F / partial theta_i",
        "partial F / partial theta_i = -partial/partial theta_i[-sum_j K_ij cos(theta_j - theta_i)]",
        "partial F / partial theta_i = -sum_j K_ij sin(theta_j - theta_i)",
    )
    assert specs["hpc_upde_derivation.upde_core_equation"].source_formulae == (
        "d theta_i / dt = omega_i - partial F / partial theta_i + eta_i(t)",
        "d theta_i / dt = omega_i + sum_j K_ij sin(theta_j - theta_i) + eta_i(t)",
    )
    assert specs["hpc_upde_derivation.hpc_interpretation"].source_mechanisms == (
        "F is variational Free Energy / surprise",
        "cos(theta_j - theta_i) is phase coherence / prediction accuracy",
        "sin(theta_j - theta_i) is phase error / prediction error epsilon",
        "K_ij is precision weighting / inverse variance of prediction",
        "eta_i is stochastic exploration / sampling",
    )
    assert specs["hpc_upde_derivation.active_inference_boundary"].source_formulae == (
        "phase-locking sin Delta theta -> 0 equals prediction error minimization and Free Energy minimization",
        "UPDE dynamics = physical substrate for Active Inference",
    )
    assert "not empirical evidence" in specs["hpc_upde_derivation.block_framing"].claim_boundary


def test_hpc_upde_derivation_outputs_and_loader_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "hpc_upde_derivation_specs.json"
    report_path = tmp_path / "hpc_upde_derivation_specs.md"
    bundle = build_hpc_upde_derivation_specs(load_jsonl(LEDGER_PATH))

    write_outputs(bundle=bundle, output_path=json_path, report_path=report_path)
    loaded = load_hpc_upde_derivation_validation_spec(
        "hpc_upde_derivation.upde_core_equation",
        spec_bundle_path=json_path,
    )
    report = build_validation_report(bundle)

    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6615, 6646)]
    assert loaded["hardware_status"] == "simulator_only_no_provider_submission"
    assert "not empirical evidence" in loaded["claim_boundary"]
    assert "# Paper 0 HPC-UPDE Mathematical Bridge Specs" in report
    assert json_path.exists()
    assert report_path.exists()
