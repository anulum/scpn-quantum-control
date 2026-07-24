# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for open-system MCWF product (BL-51)
"""Real-surface tests for ``open_system_mcwf_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.open_system_mcwf_product as mcwf_product
from scpn_quantum_control.open_system_mcwf_product import (
    NOISE_MODEL_SCHEMA_ID,
    OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY,
    OPEN_SYSTEM_MCWF_PRODUCT_SCHEMA,
    MaterialisedMcwfEnsembleProbe,
    MaterialisedReproducibilityProbe,
    OpenSystemBoundaryRow,
    OpenSystemSurfaceRow,
    PathEligibilityDecision,
    assert_open_system_mcwf_product_integrity,
    build_open_system_mcwf_product_registry,
    decide_open_system_path,
    export_sim_noise_model,
    get_open_system_boundary,
    get_open_system_surface,
    import_sim_noise_model,
    iter_open_system_boundaries,
    iter_open_system_surfaces,
    list_ambient_objective_boundary_ids,
    list_default_objective_case_ids,
    list_open_system_boundary_ids,
    list_open_system_surface_ids,
    map_open_system_mcwf_public_surfaces,
    materialise_demo_mcwf_ensemble_probe,
    materialise_mcwf_ensemble_probe,
    materialise_reproducibility_probe,
)


def test_list_and_filters() -> None:
    ids = list_open_system_surface_ids()
    assert "mcwf_ensemble" in ids
    assert "mcwf_trajectory" in ids
    assert "lindblad_density" in ids
    assert "noise_model_io" in ids
    assert "gradient_boundary" in ids
    assert len(ids) == 5
    bounds = list_open_system_boundary_ids()
    assert "non_cp_map" in bounds
    assert "hardware_noise_fidelity" in bounds
    assert len(bounds) == 5
    ens = iter_open_system_surfaces(kind="mcwf_ensemble")
    assert len(ens) == 1
    non_cp = iter_open_system_boundaries(kind="non_cp")
    assert len(non_cp) == 1


def test_get_known_and_unknown_fail_closed() -> None:
    row = get_open_system_surface("mcwf_ensemble")
    assert row.claim_boundary == OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY
    assert row.hardware_submit_allowed is False
    boundary = get_open_system_boundary("adjoint_lindblad_gradient")
    assert boundary.fail_closed is True
    with pytest.raises(ValueError, match="non-empty"):
        get_open_system_surface("  ")
    with pytest.raises(ValueError, match="unknown surface_id"):
        get_open_system_surface("not_a_surface")
    with pytest.raises(ValueError, match="non-empty"):
        get_open_system_boundary("")
    with pytest.raises(ValueError, match="unknown boundary_id"):
        get_open_system_boundary("ghost")


def test_decide_open_system_path() -> None:
    ok = decide_open_system_path("mcwf_ensemble")
    assert ok.allowed is True
    hw = decide_open_system_path("mcwf_ensemble", invent_green_hardware_noise=True)
    assert hw.allowed is False
    assert any("hardware" in b.lower() for b in hw.blockers)
    adj = decide_open_system_path("gradient_boundary", invent_green_adjoint_lindblad=True)
    assert adj.allowed is False
    nm = decide_open_system_path("mcwf_ensemble", invent_green_non_markovian=True)
    assert nm.allowed is False
    ncp = decide_open_system_path("mcwf_ensemble", invent_green_non_cp=True)
    assert ncp.allowed is False
    seed = decide_open_system_path("mcwf_ensemble", unseeded_variance_claim=True)
    assert seed.allowed is False


def test_mcwf_ensemble_probe_real_ambient() -> None:
    probe = materialise_demo_mcwf_ensemble_probe()
    assert probe.surface_id == "mcwf_ensemble"
    assert probe.n_trajectories >= 1
    assert probe.time_steps >= 1
    assert len(probe.probe_digest) == 64
    assert probe.invent_green_hardware_noise is False
    assert probe.invent_green_adjoint_lindblad is False
    # Deterministic seed: second product probe must match digest (real ambient path).
    again = materialise_mcwf_ensemble_probe("mcwf_ensemble", seed=51, n_trajectories=4)
    assert again.probe_digest == probe.probe_digest
    assert again.total_jumps == probe.total_jumps
    assert probe.to_dict()["seed"] == 51

    traj_probe = materialise_mcwf_ensemble_probe("mcwf_trajectory", n_trajectories=1)
    assert traj_probe.n_trajectories == 1
    assert traj_probe.time_steps >= 1


def test_mcwf_probe_refuses_invent_green() -> None:
    with pytest.raises(ValueError, match="hardware"):
        materialise_mcwf_ensemble_probe("mcwf_ensemble", invent_green_hardware_noise=True)
    with pytest.raises(ValueError, match="adjoint"):
        materialise_mcwf_ensemble_probe("mcwf_ensemble", invent_green_adjoint_lindblad=True)
    with pytest.raises(ValueError, match="MCWF"):
        materialise_mcwf_ensemble_probe("noise_model_io")


def test_reproducibility_probe() -> None:
    probe = materialise_reproducibility_probe()
    assert probe.surface_id == "mcwf_ensemble"
    assert probe.certificate["passed"] is True
    assert probe.certificate["same_seed_max_abs_diff"] == 0.0 or (
        float(cast(float, probe.certificate["same_seed_max_abs_diff"])) < 1e-12
    )
    assert len(probe.probe_digest) == 64
    assert "finite" in probe.certificate
    assert probe.to_dict()["demo_label"]


def test_noise_model_io() -> None:
    exported = export_sim_noise_model(gamma_amp=0.1, gamma_deph=0.05, label="demo")
    assert exported["schema"] == NOISE_MODEL_SCHEMA_ID
    assert exported["hardware_noise_fidelity_claim"] is False
    assert exported["domain"] == "simulation_only"
    imported = import_sim_noise_model(exported)
    assert imported["gamma_amp"] == 0.1
    assert imported["gamma_deph"] == 0.05
    with pytest.raises(ValueError, match="non-negative"):
        export_sim_noise_model(gamma_amp=-0.1, gamma_deph=0.0)
    with pytest.raises(ValueError, match="schema"):
        import_sim_noise_model({"schema": "wrong", "domain": "simulation_only"})
    with pytest.raises(ValueError, match="simulation_only"):
        import_sim_noise_model(
            {
                "schema": NOISE_MODEL_SCHEMA_ID,
                "domain": "hardware",
                "hardware_noise_fidelity_claim": False,
                "gamma_amp": 0.1,
                "gamma_deph": 0.0,
                "label": "x",
            }
        )
    with pytest.raises(ValueError, match="hardware_noise_fidelity_claim"):
        import_sim_noise_model(
            {
                "schema": NOISE_MODEL_SCHEMA_ID,
                "domain": "simulation_only",
                "hardware_noise_fidelity_claim": True,
                "gamma_amp": 0.1,
                "gamma_deph": 0.0,
                "label": "x",
            }
        )


def test_public_surfaces_and_registry() -> None:
    surfaces = map_open_system_mcwf_public_surfaces()
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.open_system_mcwf_product" in paths
    assert "scpn_quantum_control.phase.tensor_jump" in paths
    ambient_bounds = list_ambient_objective_boundary_ids()
    assert "adjoint_lindblad_gradient_boundary" in ambient_bounds
    cases = list_default_objective_case_ids()
    assert cases

    registry = build_open_system_mcwf_product_registry()
    assert registry["schema"] == OPEN_SYSTEM_MCWF_PRODUCT_SCHEMA
    assert registry["hardware_submit_allowed_policy"] is False
    assert registry["adjoint_lindblad_allowed_policy"] is False
    validated = assert_open_system_mcwf_product_integrity(registry)
    assert validated["surface_count"] == 5
    assert validated["boundary_count"] == 5
    assert assert_open_system_mcwf_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    registry = build_open_system_mcwf_product_registry()
    surfaces = cast(list[dict[str, object]], registry["surfaces"])

    broken = dict(registry)
    broken["surfaces"] = surfaces + [
        {
            "surface_id": "ghost",
            "kind": "mcwf_ensemble",
            "title": "t",
            "summary": "s",
            "ambient_module": "m",
            "ambient_symbol": "x",
            "hardware_submit_allowed": False,
            "support_posture": "local_research",
            "as_of": "2026-07-24",
            "claim_boundary": OPEN_SYSTEM_MCWF_CLAIM_BOUNDARY,
        }
    ]
    broken["surface_count"] = len(cast(list[object], broken["surfaces"]))
    with pytest.raises(ValueError, match="drift"):
        assert_open_system_mcwf_product_integrity(broken)

    empty: dict[str, object] = {
        "surfaces": [],
        "boundaries": registry["boundaries"],
        "blank_entry_count": 0,
        "surface_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty surfaces"):
        assert_open_system_mcwf_product_integrity(empty)

    no_bounds = dict(registry)
    no_bounds["boundaries"] = []
    no_bounds["boundary_count"] = 0
    with pytest.raises(ValueError, match="non-empty boundaries"):
        assert_open_system_mcwf_product_integrity(no_bounds)

    for policy in (
        "hardware_submit_allowed_policy",
        "hardware_noise_fidelity_claim_policy",
        "adjoint_lindblad_allowed_policy",
        "non_markovian_process_tensor_allowed_policy",
    ):
        bad = dict(registry)
        bad[policy] = True
        with pytest.raises(ValueError, match=policy):
            assert_open_system_mcwf_product_integrity(bad)

    hw_cap = dict(registry)
    mut = [dict(row) for row in surfaces]
    mut[0]["hardware_submit_allowed"] = True
    hw_cap["surfaces"] = mut
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        assert_open_system_mcwf_product_integrity(hw_cap)

    bounds = cast(list[dict[str, object]], registry["boundaries"])
    fc = dict(registry)
    mut_b = [dict(row) for row in bounds]
    mut_b[0]["fail_closed"] = False
    fc["boundaries"] = mut_b
    with pytest.raises(ValueError, match="fail_closed"):
        assert_open_system_mcwf_product_integrity(fc)

    blank = dict(registry)
    blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_open_system_mcwf_product_integrity(blank)


def test_dataclass_invariants() -> None:
    with pytest.raises(ValueError, match="surface_id"):
        OpenSystemSurfaceRow(
            surface_id="",
            kind="mcwf_ensemble",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="unknown surface kind"):
        OpenSystemSurfaceRow(
            surface_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="title"):
        OpenSystemSurfaceRow(
            surface_id="x",
            kind="mcwf_ensemble",
            title="",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="summary"):
        OpenSystemSurfaceRow(
            surface_id="x",
            kind="mcwf_ensemble",
            title="t",
            summary="  ",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="ambient_module"):
        OpenSystemSurfaceRow(
            surface_id="x",
            kind="mcwf_ensemble",
            title="t",
            summary="s",
            ambient_module="",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="ambient_symbol"):
        OpenSystemSurfaceRow(
            surface_id="x",
            kind="mcwf_ensemble",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="",
        )
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        OpenSystemSurfaceRow(
            surface_id="x",
            kind="mcwf_ensemble",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
            hardware_submit_allowed=True,
        )
    with pytest.raises(ValueError, match="support_posture"):
        OpenSystemSurfaceRow(
            surface_id="x",
            kind="mcwf_ensemble",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
            support_posture=cast(Any, "bogus"),
        )
    with pytest.raises(ValueError, match="as_of"):
        OpenSystemSurfaceRow(
            surface_id="x",
            kind="mcwf_ensemble",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
            as_of="",
        )
    ok_s = OpenSystemSurfaceRow(
        surface_id="x",
        kind="mcwf_ensemble",
        title="t",
        summary="s",
        ambient_module="m",
        ambient_symbol="x",
    )
    assert ok_s.to_dict()["surface_id"] == "x"

    with pytest.raises(ValueError, match="boundary_id"):
        OpenSystemBoundaryRow(
            boundary_id="",
            kind="non_cp",
            title="t",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="unknown boundary kind"):
        OpenSystemBoundaryRow(
            boundary_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="title"):
        OpenSystemBoundaryRow(
            boundary_id="x",
            kind="non_cp",
            title="",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="failure_class"):
        OpenSystemBoundaryRow(
            boundary_id="x",
            kind="non_cp",
            title="t",
            failure_class="",
            summary="s",
        )
    with pytest.raises(ValueError, match="summary"):
        OpenSystemBoundaryRow(
            boundary_id="x",
            kind="non_cp",
            title="t",
            failure_class="f",
            summary="",
        )
    with pytest.raises(ValueError, match="fail_closed"):
        OpenSystemBoundaryRow(
            boundary_id="x",
            kind="non_cp",
            title="t",
            failure_class="f",
            summary="s",
            fail_closed=False,
        )
    ok_b = OpenSystemBoundaryRow(
        boundary_id="x",
        kind="non_cp",
        title="t",
        failure_class="f",
        summary="s",
    )
    assert ok_b.to_dict()["fail_closed"] is True

    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "maybe"),
            allowed=True,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="",
            blockers=(),
        )
    with pytest.raises(ValueError, match="outcome=allowed"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="outcome=refused"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=False,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("",),
        )
    ok_d = PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason="ok",
        blockers=(),
    )
    assert ok_d.to_dict()["allowed"] is True

    with pytest.raises(ValueError, match="surface_id"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="",
            n_trajectories=4,
            seed=1,
            time_steps=5,
            final_mean_order_parameter=0.5,
            final_std_order_parameter=0.1,
            total_jumps=0,
            probe_digest="a" * 64,
            invent_green_hardware_noise=False,
            invent_green_adjoint_lindblad=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="n_trajectories"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="mcwf_ensemble",
            n_trajectories=0,
            seed=1,
            time_steps=5,
            final_mean_order_parameter=0.5,
            final_std_order_parameter=0.1,
            total_jumps=0,
            probe_digest="a" * 64,
            invent_green_hardware_noise=False,
            invent_green_adjoint_lindblad=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="time_steps"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="mcwf_ensemble",
            n_trajectories=4,
            seed=1,
            time_steps=0,
            final_mean_order_parameter=0.5,
            final_std_order_parameter=0.1,
            total_jumps=0,
            probe_digest="a" * 64,
            invent_green_hardware_noise=False,
            invent_green_adjoint_lindblad=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="total_jumps"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="mcwf_ensemble",
            n_trajectories=4,
            seed=1,
            time_steps=5,
            final_mean_order_parameter=0.5,
            final_std_order_parameter=0.1,
            total_jumps=-1,
            probe_digest="a" * 64,
            invent_green_hardware_noise=False,
            invent_green_adjoint_lindblad=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="final_mean"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="mcwf_ensemble",
            n_trajectories=4,
            seed=1,
            time_steps=5,
            final_mean_order_parameter=float("nan"),
            final_std_order_parameter=0.1,
            total_jumps=0,
            probe_digest="a" * 64,
            invent_green_hardware_noise=False,
            invent_green_adjoint_lindblad=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="final_std"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="mcwf_ensemble",
            n_trajectories=4,
            seed=1,
            time_steps=5,
            final_mean_order_parameter=0.5,
            final_std_order_parameter=float("inf"),
            total_jumps=0,
            probe_digest="a" * 64,
            invent_green_hardware_noise=False,
            invent_green_adjoint_lindblad=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="mcwf_ensemble",
            n_trajectories=4,
            seed=1,
            time_steps=5,
            final_mean_order_parameter=0.5,
            final_std_order_parameter=0.1,
            total_jumps=0,
            probe_digest="short",
            invent_green_hardware_noise=False,
            invent_green_adjoint_lindblad=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_hardware_noise"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="mcwf_ensemble",
            n_trajectories=4,
            seed=1,
            time_steps=5,
            final_mean_order_parameter=0.5,
            final_std_order_parameter=0.1,
            total_jumps=0,
            probe_digest="a" * 64,
            invent_green_hardware_noise=True,
            invent_green_adjoint_lindblad=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_adjoint_lindblad"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="mcwf_ensemble",
            n_trajectories=4,
            seed=1,
            time_steps=5,
            final_mean_order_parameter=0.5,
            final_std_order_parameter=0.1,
            total_jumps=0,
            probe_digest="a" * 64,
            invent_green_hardware_noise=False,
            invent_green_adjoint_lindblad=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedMcwfEnsembleProbe(
            surface_id="mcwf_ensemble",
            n_trajectories=4,
            seed=1,
            time_steps=5,
            final_mean_order_parameter=0.5,
            final_std_order_parameter=0.1,
            total_jumps=0,
            probe_digest="a" * 64,
            invent_green_hardware_noise=False,
            invent_green_adjoint_lindblad=False,
            demo_label="",
        )
    ok_p = MaterialisedMcwfEnsembleProbe(
        surface_id="mcwf_ensemble",
        n_trajectories=4,
        seed=1,
        time_steps=5,
        final_mean_order_parameter=0.5,
        final_std_order_parameter=0.1,
        total_jumps=0,
        probe_digest="a" * 64,
        invent_green_hardware_noise=False,
        invent_green_adjoint_lindblad=False,
        demo_label="d",
    )
    assert ok_p.to_dict()["n_trajectories"] == 4

    with pytest.raises(ValueError, match="surface_id"):
        MaterialisedReproducibilityProbe(
            surface_id="",
            certificate={"passed": True},
            ambient_claim_boundary="b",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="certificate"):
        MaterialisedReproducibilityProbe(
            surface_id="mcwf_ensemble",
            certificate={},
            ambient_claim_boundary="b",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="passed"):
        MaterialisedReproducibilityProbe(
            surface_id="mcwf_ensemble",
            certificate={"passed": False},
            ambient_claim_boundary="b",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="ambient_claim_boundary"):
        MaterialisedReproducibilityProbe(
            surface_id="mcwf_ensemble",
            certificate={"passed": True},
            ambient_claim_boundary="",
            probe_digest="a" * 64,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedReproducibilityProbe(
            surface_id="mcwf_ensemble",
            certificate={"passed": True},
            ambient_claim_boundary="b",
            probe_digest="x",
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedReproducibilityProbe(
            surface_id="mcwf_ensemble",
            certificate={"passed": True},
            ambient_claim_boundary="b",
            probe_digest="a" * 64,
            demo_label="",
        )
    ok_r = MaterialisedReproducibilityProbe(
        surface_id="mcwf_ensemble",
        certificate={"passed": True},
        ambient_claim_boundary="b",
        probe_digest="a" * 64,
        demo_label="d",
    )
    assert ok_r.to_dict()["surface_id"] == "mcwf_ensemble"


def test_subprocess_and_integrity_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="n_trajectories"):
        materialise_mcwf_ensemble_probe("mcwf_ensemble", n_trajectories=0)
    with pytest.raises(ValueError, match="n_trajectories"):
        materialise_reproducibility_probe(n_trajectories=0)

    def _bad_json(*_a: object, **_k: object) -> dict[str, object]:
        return {"not": "enough"}

    monkeypatch.setattr(mcwf_product, "_run_ambient_mcwf_json", _bad_json)
    with pytest.raises(ValueError, match="missing fields"):
        materialise_mcwf_ensemble_probe("mcwf_ensemble")

    def _empty_steps(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_trajectories": 1,
            "time_steps": 0,
            "final_mean_order_parameter": 0.1,
            "final_std_order_parameter": 0.0,
            "total_jumps": 0,
        }

    monkeypatch.setattr(mcwf_product, "_run_ambient_mcwf_json", _empty_steps)
    with pytest.raises(ValueError, match="non-empty"):
        materialise_mcwf_ensemble_probe("mcwf_ensemble")

    def _nan_mean(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_trajectories": 1,
            "time_steps": 3,
            "final_mean_order_parameter": float("nan"),
            "final_std_order_parameter": 0.0,
            "total_jumps": 0,
        }

    monkeypatch.setattr(mcwf_product, "_run_ambient_mcwf_json", _nan_mean)
    with pytest.raises(ValueError, match="finite"):
        materialise_mcwf_ensemble_probe("mcwf_ensemble")

    def _neg_jumps(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_trajectories": 1,
            "time_steps": 3,
            "final_mean_order_parameter": 0.2,
            "final_std_order_parameter": 0.0,
            "total_jumps": -1,
        }

    monkeypatch.setattr(mcwf_product, "_run_ambient_mcwf_json", _neg_jumps)
    with pytest.raises(ValueError, match="total_jumps"):
        materialise_mcwf_ensemble_probe("mcwf_ensemble")

    def _no_cert(*_a: object, **_k: object) -> dict[str, object]:
        return {"certificate": "nope"}

    monkeypatch.setattr(mcwf_product, "_run_ambient_mcwf_json", _no_cert)
    with pytest.raises(ValueError, match="certificate"):
        materialise_reproducibility_probe()

    def _fail_cert(*_a: object, **_k: object) -> dict[str, object]:
        return {"certificate": {"passed": False}}

    monkeypatch.setattr(mcwf_product, "_run_ambient_mcwf_json", _fail_cert)
    with pytest.raises(ValueError, match="did not pass"):
        materialise_reproducibility_probe()

    registry = build_open_system_mcwf_product_registry()
    surfaces = cast(list[dict[str, object]], registry["surfaces"])
    not_map = dict(registry)
    not_map["surfaces"] = cast(list[dict[str, object]], ["x"])
    with pytest.raises(ValueError, match="mapping"):
        assert_open_system_mcwf_product_integrity(not_map)

    blank_id = dict(registry)
    blank_caps = [dict(row) for row in surfaces]
    blank_caps[0]["surface_id"] = "  "
    blank_id["surfaces"] = blank_caps
    with pytest.raises(ValueError, match="blank"):
        assert_open_system_mcwf_product_integrity(blank_id)

    dup = dict(registry)
    dup_caps = [dict(row) for row in surfaces]
    dup_caps[1] = dict(dup_caps[0])
    dup["surfaces"] = dup_caps
    with pytest.raises(ValueError, match="duplicate surface_id"):
        assert_open_system_mcwf_product_integrity(dup)

    no_symbol = dict(registry)
    sym = [dict(row) for row in surfaces]
    sym[0]["ambient_symbol"] = ""
    no_symbol["surfaces"] = sym
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_open_system_mcwf_product_integrity(no_symbol)

    no_mcwf = dict(registry)
    filtered = [dict(row) for row in surfaces if row["surface_id"] != "mcwf_ensemble"]
    no_mcwf["surfaces"] = filtered
    no_mcwf["surface_count"] = len(filtered)
    with pytest.raises(ValueError, match="mcwf_ensemble|drift"):
        assert_open_system_mcwf_product_integrity(no_mcwf)

    bounds = cast(list[dict[str, object]], registry["boundaries"])
    b_not_map = dict(registry)
    b_not_map["boundaries"] = cast(list[dict[str, object]], [1])
    with pytest.raises(ValueError, match="mapping"):
        assert_open_system_mcwf_product_integrity(b_not_map)

    blank_b = dict(registry)
    bb = [dict(row) for row in bounds]
    bb[0]["boundary_id"] = ""
    blank_b["boundaries"] = bb
    with pytest.raises(ValueError, match="boundary_id"):
        assert_open_system_mcwf_product_integrity(blank_b)

    dup_b = dict(registry)
    db = [dict(row) for row in bounds]
    db[1] = dict(db[0])
    dup_b["boundaries"] = db
    with pytest.raises(ValueError, match="duplicate boundary_id"):
        assert_open_system_mcwf_product_integrity(dup_b)

    count_mismatch = dict(registry)
    count_mismatch["surface_count"] = 99
    with pytest.raises(ValueError, match="surface_count"):
        assert_open_system_mcwf_product_integrity(count_mismatch)

    bcount = dict(registry)
    bcount["boundary_count"] = 99
    with pytest.raises(ValueError, match="boundary_count"):
        assert_open_system_mcwf_product_integrity(bcount)

    with pytest.raises(ValueError, match="label"):
        export_sim_noise_model(gamma_amp=0.1, gamma_deph=0.0, label="")
    with pytest.raises(ValueError, match="gamma_deph"):
        export_sim_noise_model(gamma_amp=0.1, gamma_deph=-1.0)
    with pytest.raises(ValueError, match="gamma_amp"):
        import_sim_noise_model(
            {
                "schema": NOISE_MODEL_SCHEMA_ID,
                "domain": "simulation_only",
                "hardware_noise_fidelity_claim": False,
                "gamma_amp": "x",
                "gamma_deph": 0.0,
                "label": "x",
            }
        )
    with pytest.raises(ValueError, match="gamma_deph"):
        import_sim_noise_model(
            {
                "schema": NOISE_MODEL_SCHEMA_ID,
                "domain": "simulation_only",
                "hardware_noise_fidelity_claim": False,
                "gamma_amp": 0.1,
                "gamma_deph": True,
                "label": "x",
            }
        )
    with pytest.raises(ValueError, match="label"):
        import_sim_noise_model(
            {
                "schema": NOISE_MODEL_SCHEMA_ID,
                "domain": "simulation_only",
                "hardware_noise_fidelity_claim": False,
                "gamma_amp": 0.1,
                "gamma_deph": 0.0,
                "label": 1,
            }
        )
    with pytest.raises(ValueError, match="mapping"):
        import_sim_noise_model(cast(Any, "not-a-map"))


def test_module_exports_stable() -> None:
    assert "assert_open_system_mcwf_product_integrity" in mcwf_product.__all__
    assert "materialise_demo_mcwf_ensemble_probe" in mcwf_product.__all__
    assert OPEN_SYSTEM_MCWF_PRODUCT_SCHEMA == "open_system_mcwf_product.v1"
