# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S19 resource-signature scan tests
"""Tests for the S19 simulator-only resource-signature scan generator."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from scpn_quantum_control.dense_budget import DenseAllocationError

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_s19_resource_signature_scan.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_s19_resource_signature_scan", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load S19 scan script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_scan_emits_finite_simulator_only_rows() -> None:
    module = _load_module()

    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.5, 1.25),
        topologies=("ring",),
        omega_spread=0.4,
        max_dense_gib=0.25,
        krylov_t_max=2.0,
        krylov_n_times=8,
        krylov_max_lanczos=8,
    )

    assert summary["schema"] == "scpn_s19_resource_signature_scan_v1"
    assert summary["submission_status"] == "not_submitted"
    assert summary["active_lane"] == "S19_resource_signatures"
    assert summary["separation_from_submitted_papers"]["submitted_papers_modified"] is False
    assert summary["row_count"] == 2
    assert summary["alignment_summary"]["group_count"] == 1
    assert summary["claim_boundary"].startswith("Simulator-only")

    for row in summary["rows"]:
        assert row["n_qubits"] == 4
        assert row["topology"] == "ring"
        assert row["synchronization_onset_K_estimate"] > 0.0
        assert 0.0 <= row["sync_order_ground"] <= 1.0
        assert isinstance(row["krylov_n_lanczos"], int)
        assert row["krylov_n_lanczos"] <= 8
        for key in (
            "entropy",
            "schmidt_gap",
            "spectral_gap",
            "magic_sre_m2",
            "pairing_mean",
            "pairing_topology_correlation",
            "krylov_peak_complexity",
            "krylov_mean_b",
        ):
            assert math.isfinite(row[key]), key


def test_write_outputs_uses_s19_directory_and_claim_boundary(tmp_path: Path) -> None:
    module = _load_module()
    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.75,),
        topologies=("chain",),
        omega_spread=0.2,
        max_dense_gib=0.25,
        krylov_t_max=1.0,
        krylov_n_times=5,
        krylov_max_lanczos=6,
    )

    paths = module.write_outputs(summary, output_dir=tmp_path, date_tag="2026-05-13")

    assert paths["manifest"].name == "s19_scan_manifest_2026-05-13.json"
    assert paths["rows"].name == "s19_resource_rows_2026-05-13.csv"
    assert paths["claim_boundary"].name == "s19_claim_boundary_2026-05-13.md"
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert payload["row_count"] == 1
    assert "not a hardware claim" in paths["claim_boundary"].read_text(encoding="utf-8")
    assert "onset estimate" in paths["claim_boundary"].read_text(encoding="utf-8")
    assert "submitted manuscript" in paths["claim_boundary"].read_text(encoding="utf-8")


def test_run_scan_rejects_dense_overbudget_before_resource_claims() -> None:
    module = _load_module()

    with pytest.raises(DenseAllocationError):
        module.run_scan(
            n_values=(8,),
            k_values=(1.0,),
            topologies=("ring",),
            omega_spread=0.3,
            max_dense_gib=1e-12,
        )


def test_topology_and_omega_profiles_are_deterministic() -> None:
    module = _load_module()

    ring = module.build_topology(5, "ring")
    chain = module.build_topology(5, "chain")
    omega_a = module.build_omega(5, spread=0.4)
    omega_b = module.build_omega(5, spread=0.4)

    assert np.allclose(ring, ring.T)
    assert np.allclose(chain, chain.T)
    assert np.count_nonzero(chain) < np.count_nonzero(ring)
    assert np.allclose(omega_a, omega_b)
    assert abs(float(np.mean(omega_a))) < 1e-12


def test_fiedler_onset_estimator_is_topology_sensitive() -> None:
    module = _load_module()

    ring_onset = module.estimate_synchronization_onset_k(
        module.build_topology(6, "ring"), omega_spread=0.5
    )
    chain_onset = module.estimate_synchronization_onset_k(
        module.build_topology(6, "chain"), omega_spread=0.5
    )

    assert math.isfinite(ring_onset)
    assert math.isfinite(chain_onset)
    assert ring_onset > 0.0
    assert chain_onset > ring_onset


def test_run_scan_supports_multiple_topologies_and_alignment_distances() -> None:
    module = _load_module()

    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.5, 1.5),
        topologies=("ring", "chain"),
        omega_spread=0.4,
        max_dense_gib=0.25,
        krylov_t_max=1.5,
        krylov_n_times=6,
        krylov_max_lanczos=6,
    )

    assert summary["row_count"] == 4
    assert summary["parameters"]["topologies"] == ["ring", "chain"]
    assert summary["alignment_summary"]["group_count"] == 2
    for group in summary["alignment_summary"]["groups"]:
        assert group["synchronization_onset_K_estimate"] > 0.0
        assert group["mean_abs_diagnostic_distance_from_onset"] >= 0.0
        assert set(group["diagnostic_distance_from_onset"]) == {
            "entropy",
            "schmidt_gap",
            "magic",
            "pairing_mean",
            "krylov_peak",
        }


def test_run_scan_supports_reproducible_disorder_ensemble_alignment() -> None:
    module = _load_module()

    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.5, 1.0),
        topologies=("ring",),
        omega_spread=0.4,
        disorder_seeds=(11, 17),
        max_dense_gib=0.25,
        krylov_t_max=1.0,
        krylov_n_times=5,
        krylov_max_lanczos=5,
    )

    assert summary["row_count"] == 4
    assert summary["parameters"]["disorder_seeds"] == [11, 17]
    assert {row["omega_profile"] for row in summary["rows"]} == {
        "disorder_seed_11",
        "disorder_seed_17",
    }
    assert {row["disorder_seed"] for row in summary["rows"]} == {11, 17}
    assert summary["alignment_summary"]["group_count"] == 2
    assert summary["alignment_summary"]["ensemble_group_count"] == 1

    ensemble = summary["alignment_summary"]["ensemble_groups"][0]
    assert ensemble["n_qubits"] == 4
    assert ensemble["topology"] == "ring"
    assert ensemble["realisation_count"] == 2
    assert 0.0 <= ensemble["mean_alignment_score"] <= 1.0
    assert ensemble["std_alignment_score"] >= 0.0
    assert set(ensemble["jackknife_alignment_score_ci95"]) == {
        "low",
        "high",
        "standard_error",
    }
    assert ensemble["jackknife_alignment_score_ci95"]["low"] <= (ensemble["mean_alignment_score"])
    assert ensemble["jackknife_alignment_score_ci95"]["high"] >= (ensemble["mean_alignment_score"])


def test_seeded_omega_profiles_are_deterministic_and_zero_mean() -> None:
    module = _load_module()

    omega_a = module.build_omega(6, spread=0.7, disorder_seed=23)
    omega_b = module.build_omega(6, spread=0.7, disorder_seed=23)
    omega_c = module.build_omega(6, spread=0.7, disorder_seed=29)

    assert np.allclose(omega_a, omega_b)
    assert not np.allclose(omega_a, omega_c)
    assert abs(float(np.mean(omega_a))) < 1e-12
    assert float(np.max(np.abs(omega_a))) <= 0.35 + 1e-12


def test_paper27_topology_is_heterogeneous_symmetric_and_normalised() -> None:
    module = _load_module()

    topology = module.build_topology(6, "paper27")
    off_diagonal = topology[~np.eye(6, dtype=bool)]

    assert np.allclose(topology, topology.T)
    assert np.allclose(np.diag(topology), 0.0)
    assert float(np.max(off_diagonal)) == pytest.approx(1.0)
    assert len({round(float(value), 6) for value in off_diagonal if value > 0.0}) > 3


def test_run_scan_records_topology_provenance_for_paper27() -> None:
    module = _load_module()

    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.5,),
        topologies=("paper27",),
        omega_spread=0.4,
        disorder_seeds=(11,),
        max_dense_gib=0.25,
        krylov_t_max=1.0,
        krylov_n_times=5,
        krylov_max_lanczos=5,
    )

    assert summary["row_count"] == 1
    assert summary["parameters"]["topologies"] == ["paper27"]
    assert summary["topology_provenance"]["paper27"]["description"].startswith("Paper 27")
    assert summary["topology_provenance"]["paper27"]["numeric_topology_label"] == (
        "legacy.paper27_provisional_not_paper0"
    )
    assert summary["topology_provenance"]["paper27"]["paper0_topology_claim"] is False
    assert summary["rows"][0]["topology"] == "paper27"


def test_refined_k_values_include_onset_neighbourhood_and_base_grid() -> None:
    module = _load_module()

    refined = module.refine_k_values_for_onset(
        base_values=(0.5, 1.5),
        onset_k=1.0,
        half_width=0.25,
        step=0.125,
        min_k=0.0,
        max_k=2.0,
    )

    assert refined == (0.5, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5)


def test_run_scan_refines_each_topology_around_its_own_onset() -> None:
    module = _load_module()

    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.5, 1.5),
        topologies=("ring", "chain"),
        omega_spread=0.5,
        disorder_seeds=(11,),
        refine_onsets=True,
        refinement_half_width=0.25,
        refinement_step=0.25,
        max_dense_gib=0.25,
        krylov_t_max=1.0,
        krylov_n_times=5,
        krylov_max_lanczos=5,
    )

    assert summary["parameters"]["refinement"]["enabled"] is True
    by_topology = {
        topology: {
            round(float(row["K_base"]), 6)
            for row in summary["rows"]
            if row["topology"] == topology
        }
        for topology in ("ring", "chain")
    }
    assert {0.5, 1.5}.issubset(by_topology["ring"])
    assert {0.5, 1.5}.issubset(by_topology["chain"])
    assert {0.0, 0.25}.issubset(by_topology["ring"])
    assert {0.603553, 0.853553, 1.103553}.issubset(by_topology["chain"])
    assert (
        summary["parameters"]["effective_k_values_by_n_topology"]["4:ring"]
        != (summary["parameters"]["effective_k_values_by_n_topology"]["4:chain"])
    )


def test_curvature_feature_k_handles_nonuniform_grid() -> None:
    module = _load_module()

    feature_k = module.curvature_feature_k(
        k_values=(0.0, 0.25, 0.5, 1.0, 1.75),
        observable_values=(0.0, 0.2, 1.0, 0.1, 0.0),
    )

    assert feature_k == pytest.approx(0.5)


def test_alignment_summary_records_curvature_features_and_scores() -> None:
    module = _load_module()

    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.5, 1.0, 1.5, 2.0),
        topologies=("chain",),
        omega_spread=0.585786437626905,
        disorder_seeds=(11, 17),
        refine_onsets=True,
        refinement_half_width=0.25,
        refinement_step=0.125,
        max_dense_gib=0.25,
        krylov_t_max=1.0,
        krylov_n_times=5,
        krylov_max_lanczos=5,
    )

    group = summary["alignment_summary"]["groups"][0]
    assert set(group["curvature_feature_K"]) == {
        "entropy",
        "schmidt_gap",
        "magic",
        "pairing_mean",
        "krylov_peak",
    }
    assert set(group["curvature_distance_from_onset"]) == set(group["curvature_feature_K"])
    assert 0.0 <= group["curvature_alignment_score"] <= 1.0

    ensemble = summary["alignment_summary"]["ensemble_groups"][0]
    assert 0.0 <= ensemble["mean_curvature_alignment_score"] <= 1.0
    assert set(ensemble["jackknife_curvature_alignment_score_ci95"]) == {
        "low",
        "high",
        "standard_error",
    }


def test_off_onset_control_centres_are_matched_and_bounded() -> None:
    module = _load_module()

    centres = module.off_onset_control_centres(
        onset_k=1.0,
        half_width=0.25,
        min_k=0.0,
        max_k=2.0,
    )

    assert centres == (0.25, 1.75)


def test_run_scan_records_off_onset_curvature_controls() -> None:
    module = _load_module()

    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.5, 1.0, 1.5, 2.0),
        topologies=("chain",),
        omega_spread=0.585786437626905,
        disorder_seeds=(11, 17),
        refine_onsets=True,
        include_off_onset_controls=True,
        refinement_half_width=0.25,
        refinement_step=0.125,
        max_dense_gib=0.25,
        krylov_t_max=1.0,
        krylov_n_times=5,
        krylov_max_lanczos=5,
    )

    assert summary["parameters"]["refinement"]["off_onset_controls_enabled"] is True
    assert summary["parameters"]["off_onset_control_centres_by_n_topology"]["4:chain"]

    group = summary["alignment_summary"]["groups"][0]
    assert set(group["off_onset_curvature_control_scores"]) == {
        "lower",
        "upper",
    }
    assert set(group["curvature_onset_minus_best_control_by_observable"]) == {
        "entropy",
        "schmidt_gap",
        "magic",
        "pairing_mean",
        "krylov_peak",
    }
    assert group["best_off_onset_curvature_control_score"] >= 0.0
    assert "curvature_onset_minus_best_control" in group

    ensemble = summary["alignment_summary"]["ensemble_groups"][0]
    assert "mean_curvature_onset_minus_best_control" in ensemble
    assert set(ensemble["mean_curvature_onset_minus_best_control_by_observable"]) == {
        "entropy",
        "schmidt_gap",
        "magic",
        "pairing_mean",
        "krylov_peak",
    }
    assert set(ensemble["jackknife_curvature_onset_minus_best_control_ci95"]) == {
        "low",
        "high",
        "standard_error",
    }


def test_graph_topology_diagnostics_classify_connectivity_regimes() -> None:
    module = _load_module()

    chain = module.graph_topology_diagnostics(module.build_topology(6, "chain"))
    ring = module.graph_topology_diagnostics(module.build_topology(6, "ring"))
    complete = module.graph_topology_diagnostics(module.build_topology(6, "all_to_all"))
    paper27 = module.graph_topology_diagnostics(module.build_topology(6, "paper27"))

    assert chain["boundary_class"] == "open_chain"
    assert chain["edge_count"] == 5
    assert chain["edge_density"] == pytest.approx(5.0 / 15.0)
    assert chain["algebraic_connectivity"] == pytest.approx(0.2679491924311227)
    assert chain["degree_min"] == pytest.approx(1.0)
    assert chain["degree_max"] == pytest.approx(2.0)

    assert ring["boundary_class"] == "periodic_ring"
    assert ring["edge_count"] == 6
    assert ring["edge_density"] == pytest.approx(0.4)
    assert ring["algebraic_connectivity"] == pytest.approx(1.0)

    assert complete["boundary_class"] == "complete"
    assert complete["edge_count"] == 15
    assert complete["edge_density"] == pytest.approx(1.0)
    assert complete["algebraic_connectivity"] == pytest.approx(6.0)

    assert paper27["boundary_class"] == "weighted_heterogeneous"
    assert paper27["edge_density"] == pytest.approx(1.0)
    assert paper27["weighted_degree_cv"] > 0.0


def test_run_scan_records_topology_diagnostics_and_control_status() -> None:
    module = _load_module()

    summary = module.run_scan(
        n_values=(6,),
        k_values=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5),
        topologies=("chain",),
        omega_spread=0.5,
        disorder_seeds=(11, 17, 23),
        refine_onsets=True,
        include_off_onset_controls=True,
        refinement_half_width=0.25,
        refinement_step=0.125,
        max_dense_gib=0.25,
        krylov_t_max=4.0,
        krylov_n_times=24,
        krylov_max_lanczos=24,
    )

    diagnostics = summary["parameters"]["topology_diagnostics_by_n_topology"]["6:chain"]
    assert diagnostics["boundary_class"] == "open_chain"
    assert diagnostics["edge_count"] == 5

    group = summary["alignment_summary"]["groups"][0]
    assert group["topology_diagnostics"]["boundary_class"] == "open_chain"

    ensemble = summary["alignment_summary"]["ensemble_groups"][0]
    assert ensemble["topology_diagnostics"]["boundary_class"] == "open_chain"
    assert ensemble["control_status"] == "fails_off_onset_control"
    assert ensemble["failing_observable_families"] == [
        "entropy",
        "magic",
        "pairing_mean",
        "schmidt_gap",
    ]
    assert ensemble["passing_observable_families"] == ["krylov_peak"]


def test_run_scan_attaches_paper0_source_boundary_and_numeric_topology_labels() -> None:
    module = _load_module()

    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.5,),
        topologies=("ring",),
        omega_spread=0.4,
        disorder_seeds=(11,),
        max_dense_gib=0.25,
        krylov_t_max=1.0,
        krylov_n_times=5,
        krylov_max_lanczos=5,
    )

    boundary = summary["paper0_topology_source_boundary"]
    assert boundary["schema_key"] == "paper0.topology.source_boundary.v1"
    assert boundary["hardware_status"] == "source_boundary_only_no_provider_submission"
    assert boundary["provider_ready"] is False
    assert boundary["numeric_coupling_matrix"] is None
    assert summary["parameters"]["numeric_topology_labels_by_topology"] == {
        "ring": "synthetic_control.ring"
    }
    assert summary["rows"][0]["numeric_topology_label"] == "synthetic_control.ring"
    assert summary["rows"][0]["paper0_source_boundary_schema"] == (
        "paper0.topology.source_boundary.v1"
    )
    assert summary["topology_provenance"]["ring"]["numeric_topology_label"] == (
        "synthetic_control.ring"
    )
    assert summary["topology_provenance"]["ring"]["paper0_topology_claim"] is False


def test_source_boundary_only_topology_cannot_be_simulated() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="source boundary is not a numeric topology"):
        module.run_scan(
            n_values=(4,),
            k_values=(0.5,),
            topologies=("paper0_source_boundary_only",),
            omega_spread=0.4,
            max_dense_gib=0.25,
        )


def test_claim_boundary_documents_paper0_source_boundary_and_numeric_labels(
    tmp_path: Path,
) -> None:
    module = _load_module()
    summary = module.run_scan(
        n_values=(4,),
        k_values=(0.5,),
        topologies=("chain",),
        omega_spread=0.2,
        max_dense_gib=0.25,
        krylov_t_max=1.0,
        krylov_n_times=5,
        krylov_max_lanczos=6,
    )

    paths = module.write_outputs(summary, output_dir=tmp_path, date_tag="2026-05-13")
    claim_boundary = paths["claim_boundary"].read_text(encoding="utf-8")

    assert "Paper 0 topology source boundary" in claim_boundary
    assert "synthetic_control.chain" in claim_boundary
    assert "source boundary is not a numeric coupling matrix" in claim_boundary
