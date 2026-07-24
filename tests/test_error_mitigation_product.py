# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for error mitigation product (BL-59)
"""Real-surface tests for ``error_mitigation_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.error_mitigation_product as mit_product
from scpn_quantum_control.error_mitigation_product import (
    ERROR_MITIGATION_CLAIM_BOUNDARY,
    ERROR_MITIGATION_PRODUCT_SCHEMA,
    MaterialisedReadoutProbe,
    MaterialisedZneProbe,
    MitigationBoundaryRow,
    MitigatorTaxonomyRow,
    PathEligibilityDecision,
    assert_error_mitigation_product_integrity,
    build_error_mitigation_product_registry,
    decide_mitigation_path,
    get_mitigation_boundary,
    get_mitigator,
    iter_mitigation_boundaries,
    iter_mitigators,
    list_mitigation_boundary_ids,
    list_mitigator_ids,
    map_error_mitigation_public_surfaces,
    materialise_demo_zne_probe,
    materialise_readout_probe,
    materialise_zne_probe,
    studio_mitigate_claim_boundary,
)


def test_list_and_filters() -> None:
    ids = list_mitigator_ids()
    assert "zne_richardson" in ids
    assert "readout_confusion" in ids
    assert "mitiq_optional" in ids
    assert "studio_executive_mitigate" in ids
    assert len(ids) == 9
    bounds = list_mitigation_boundary_ids()
    assert "ideal_gradient_restore" in bounds
    assert "live_qpu_mitigation" in bounds
    assert len(bounds) == 5
    zne = iter_mitigators(kind="zne")
    assert len(zne) == 1
    fd = iter_mitigators(differentiability="fd_only")
    assert all(row.differentiability == "fd_only" for row in fd)
    assert iter_mitigation_boundaries(kind="mitiq_hard_dependency")


def test_get_known_and_unknown() -> None:
    row = get_mitigator("zne_richardson")
    assert row.claim_boundary == ERROR_MITIGATION_CLAIM_BOUNDARY
    assert row.hardware_submit_allowed is False
    assert row.differentiability == "fd_only"
    b = get_mitigation_boundary("ideal_gradient_restore")
    assert b.fail_closed is True
    with pytest.raises(ValueError, match="non-empty"):
        get_mitigator("  ")
    with pytest.raises(ValueError, match="unknown mitigator_id"):
        get_mitigator("ghost")
    with pytest.raises(ValueError, match="unknown boundary_id"):
        get_mitigation_boundary("ghost")


def test_decide_mitigation_path() -> None:
    ok = decide_mitigation_path("zne_richardson")
    assert ok.allowed is True
    ideal = decide_mitigation_path("zne_richardson", invent_green_ideal_gradient_restore=True)
    assert ideal.allowed is False
    assert any("ideal" in x.lower() for x in ideal.blockers)
    qpu = decide_mitigation_path("readout_confusion", invent_green_live_qpu=True)
    assert qpu.allowed is False
    mitiq = decide_mitigation_path("mitiq_optional", invent_green_mitiq_hard_dep=True)
    assert mitiq.allowed is False
    non_diff = decide_mitigation_path("pec_pauli_twirl", invent_green_non_diff_as_analytic=True)
    assert non_diff.allowed is False
    # Even FD path refuses the non_diff_as_analytic invent-green flag.
    fd_flag = decide_mitigation_path("zne_richardson", invent_green_non_diff_as_analytic=True)
    assert fd_flag.allowed is False


def test_zne_probe_real_ambient() -> None:
    probe = materialise_demo_zne_probe()
    assert probe.mitigator_id == "zne_richardson"
    assert probe.n_points == 3
    assert len(probe.probe_digest) == 64
    assert probe.invent_green_ideal_gradient_restore is False
    # Deterministic re-run of product path (ambient via clean subprocess).
    again = materialise_zne_probe("zne_richardson")
    assert again.probe_digest == probe.probe_digest
    assert again.zero_noise_estimate == pytest.approx(probe.zero_noise_estimate)

    unc = materialise_zne_probe("zne_uncertainty")
    assert unc.n_points == 3
    assert unc.zero_noise_estimate == unc.zero_noise_estimate  # finite

    studio = materialise_zne_probe("studio_executive_mitigate")
    assert studio.mitigator_id == "studio_executive_mitigate"


def test_zne_probe_refuses() -> None:
    with pytest.raises(ValueError, match="ideal"):
        materialise_zne_probe("zne_richardson", invent_green_ideal_gradient_restore=True)
    with pytest.raises(ValueError, match="QPU|qpu|live"):
        materialise_zne_probe("zne_richardson", invent_green_live_qpu=True)
    with pytest.raises(ValueError, match="ZNE"):
        materialise_zne_probe("readout_confusion")


def test_readout_probe() -> None:
    probe = materialise_readout_probe()
    assert probe.n_qubits == 1
    assert probe.n_basis == 2
    assert abs(probe.mitigated_probability_sum - 1.0) < 0.05
    assert len(probe.probe_digest) == 64
    with pytest.raises(ValueError, match="readout"):
        materialise_readout_probe("zne_richardson")
    with pytest.raises(ValueError, match="ideal"):
        materialise_readout_probe(invent_green_ideal_gradient_restore=True)


def test_studio_boundary_and_registry() -> None:
    boundary = studio_mitigate_claim_boundary()
    assert "expectation" in boundary.lower() or "extrapol" in boundary.lower()
    surfaces = map_error_mitigation_public_surfaces()
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.error_mitigation_product" in paths
    assert "scpn_quantum_control.studio.executive_mitigate" in paths

    registry = build_error_mitigation_product_registry()
    assert registry["schema"] == ERROR_MITIGATION_PRODUCT_SCHEMA
    assert registry["ideal_gradient_restore_policy"] is False
    assert registry["mitiq_hard_dependency_policy"] is False
    validated = assert_error_mitigation_product_integrity(registry)
    assert validated["mitigator_count"] == 9
    assert validated["boundary_count"] == 5
    assert assert_error_mitigation_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift() -> None:
    registry = build_error_mitigation_product_registry()
    mitigators = cast(list[dict[str, object]], registry["mitigators"])

    broken = dict(registry)
    broken["mitigators"] = mitigators + [
        {
            "mitigator_id": "ghost",
            "kind": "zne",
            "title": "t",
            "summary": "s",
            "differentiability": "fd_only",
            "ambient_module": "m",
            "ambient_symbol": "x",
            "hardware_submit_allowed": False,
            "support_posture": "local_research",
            "as_of": "2026-07-24",
            "claim_boundary": ERROR_MITIGATION_CLAIM_BOUNDARY,
        }
    ]
    broken["mitigator_count"] = len(cast(list[object], broken["mitigators"]))
    with pytest.raises(ValueError, match="drift"):
        assert_error_mitigation_product_integrity(broken)

    empty: dict[str, object] = {
        "mitigators": [],
        "boundaries": registry["boundaries"],
        "blank_entry_count": 0,
        "mitigator_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty mitigators"):
        assert_error_mitigation_product_integrity(empty)

    no_b = dict(registry)
    no_b["boundaries"] = []
    no_b["boundary_count"] = 0
    with pytest.raises(ValueError, match="non-empty boundaries"):
        assert_error_mitigation_product_integrity(no_b)

    for policy in (
        "hardware_submit_allowed_policy",
        "ideal_gradient_restore_policy",
        "mitiq_hard_dependency_policy",
    ):
        bad = dict(registry)
        bad[policy] = True
        with pytest.raises(ValueError, match=policy):
            assert_error_mitigation_product_integrity(bad)

    hw = dict(registry)
    mut = [dict(row) for row in mitigators]
    mut[0]["hardware_submit_allowed"] = True
    hw["mitigators"] = mut
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        assert_error_mitigation_product_integrity(hw)

    blank = dict(registry)
    blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_error_mitigation_product_integrity(blank)

    no_studio = dict(registry)
    no_studio["studio_mitigate_claim_boundary"] = ""
    with pytest.raises(ValueError, match="studio_mitigate_claim_boundary"):
        assert_error_mitigation_product_integrity(no_studio)

    bounds = cast(list[dict[str, object]], registry["boundaries"])
    fc = dict(registry)
    mut_b = [dict(row) for row in bounds]
    mut_b[0]["fail_closed"] = False
    fc["boundaries"] = mut_b
    with pytest.raises(ValueError, match="fail_closed"):
        assert_error_mitigation_product_integrity(fc)


def test_dataclass_invariants() -> None:
    with pytest.raises(ValueError, match="mitigator_id"):
        MitigatorTaxonomyRow(
            mitigator_id="",
            kind="zne",
            title="t",
            summary="s",
            differentiability="fd_only",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="unknown mitigator kind"):
        MitigatorTaxonomyRow(
            mitigator_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            summary="s",
            differentiability="fd_only",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="title"):
        MitigatorTaxonomyRow(
            mitigator_id="x",
            kind="zne",
            title="",
            summary="s",
            differentiability="fd_only",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="summary"):
        MitigatorTaxonomyRow(
            mitigator_id="x",
            kind="zne",
            title="t",
            summary="",
            differentiability="fd_only",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="differentiability"):
        MitigatorTaxonomyRow(
            mitigator_id="x",
            kind="zne",
            title="t",
            summary="s",
            differentiability=cast(Any, "bogus"),
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="ambient_module"):
        MitigatorTaxonomyRow(
            mitigator_id="x",
            kind="zne",
            title="t",
            summary="s",
            differentiability="fd_only",
            ambient_module="",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="ambient_symbol"):
        MitigatorTaxonomyRow(
            mitigator_id="x",
            kind="zne",
            title="t",
            summary="s",
            differentiability="fd_only",
            ambient_module="m",
            ambient_symbol="",
        )
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        MitigatorTaxonomyRow(
            mitigator_id="x",
            kind="zne",
            title="t",
            summary="s",
            differentiability="fd_only",
            ambient_module="m",
            ambient_symbol="x",
            hardware_submit_allowed=True,
        )
    with pytest.raises(ValueError, match="support_posture"):
        MitigatorTaxonomyRow(
            mitigator_id="x",
            kind="zne",
            title="t",
            summary="s",
            differentiability="fd_only",
            ambient_module="m",
            ambient_symbol="x",
            support_posture=cast(Any, "bogus"),
        )
    with pytest.raises(ValueError, match="as_of"):
        MitigatorTaxonomyRow(
            mitigator_id="x",
            kind="zne",
            title="t",
            summary="s",
            differentiability="fd_only",
            ambient_module="m",
            ambient_symbol="x",
            as_of="",
        )
    ok = MitigatorTaxonomyRow(
        mitigator_id="x",
        kind="zne",
        title="t",
        summary="s",
        differentiability="fd_only",
        ambient_module="m",
        ambient_symbol="x",
    )
    assert ok.to_dict()["mitigator_id"] == "x"

    with pytest.raises(ValueError, match="boundary_id"):
        MitigationBoundaryRow(
            boundary_id="",
            kind="ideal_gradient_restore",
            title="t",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="unknown boundary kind"):
        MitigationBoundaryRow(
            boundary_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="fail_closed"):
        MitigationBoundaryRow(
            boundary_id="x",
            kind="ideal_gradient_restore",
            title="t",
            failure_class="f",
            summary="s",
            fail_closed=False,
        )
    with pytest.raises(ValueError, match="title"):
        MitigationBoundaryRow(
            boundary_id="x",
            kind="ideal_gradient_restore",
            title="",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="failure_class"):
        MitigationBoundaryRow(
            boundary_id="x",
            kind="ideal_gradient_restore",
            title="t",
            failure_class="",
            summary="s",
        )
    with pytest.raises(ValueError, match="summary"):
        MitigationBoundaryRow(
            boundary_id="x",
            kind="ideal_gradient_restore",
            title="t",
            failure_class="f",
            summary="",
        )
    ok_b = MitigationBoundaryRow(
        boundary_id="x",
        kind="ideal_gradient_restore",
        title="t",
        failure_class="f",
        summary="s",
    )
    assert ok_b.to_dict()["fail_closed"] is True

    with pytest.raises(ValueError, match="blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )
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

    with pytest.raises(ValueError, match="mitigator_id"):
        MaterialisedZneProbe(
            mitigator_id="",
            zero_noise_estimate=1.0,
            fit_residual=0.0,
            order=1,
            n_points=3,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="zero_noise_estimate"):
        MaterialisedZneProbe(
            mitigator_id="zne_richardson",
            zero_noise_estimate=float("nan"),
            fit_residual=0.0,
            order=1,
            n_points=3,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="fit_residual"):
        MaterialisedZneProbe(
            mitigator_id="zne_richardson",
            zero_noise_estimate=1.0,
            fit_residual=-1.0,
            order=1,
            n_points=3,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="order"):
        MaterialisedZneProbe(
            mitigator_id="zne_richardson",
            zero_noise_estimate=1.0,
            fit_residual=0.0,
            order=0,
            n_points=3,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="n_points"):
        MaterialisedZneProbe(
            mitigator_id="zne_richardson",
            zero_noise_estimate=1.0,
            fit_residual=0.0,
            order=1,
            n_points=1,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedZneProbe(
            mitigator_id="zne_richardson",
            zero_noise_estimate=1.0,
            fit_residual=0.0,
            order=1,
            n_points=3,
            probe_digest="x",
            invent_green_ideal_gradient_restore=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_ideal_gradient_restore"):
        MaterialisedZneProbe(
            mitigator_id="zne_richardson",
            zero_noise_estimate=1.0,
            fit_residual=0.0,
            order=1,
            n_points=3,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=True,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_live_qpu"):
        MaterialisedZneProbe(
            mitigator_id="zne_richardson",
            zero_noise_estimate=1.0,
            fit_residual=0.0,
            order=1,
            n_points=3,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            invent_green_live_qpu=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedZneProbe(
            mitigator_id="zne_richardson",
            zero_noise_estimate=1.0,
            fit_residual=0.0,
            order=1,
            n_points=3,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            invent_green_live_qpu=False,
            demo_label="",
        )
    ok_z = MaterialisedZneProbe(
        mitigator_id="zne_richardson",
        zero_noise_estimate=1.0,
        fit_residual=0.0,
        order=1,
        n_points=3,
        probe_digest="a" * 64,
        invent_green_ideal_gradient_restore=False,
        invent_green_live_qpu=False,
        demo_label="d",
    )
    assert ok_z.to_dict()["n_points"] == 3

    with pytest.raises(ValueError, match="n_qubits"):
        MaterialisedReadoutProbe(
            mitigator_id="readout_confusion",
            n_qubits=0,
            n_basis=1,
            mitigated_probability_sum=1.0,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="n_basis"):
        MaterialisedReadoutProbe(
            mitigator_id="readout_confusion",
            n_qubits=1,
            n_basis=3,
            mitigated_probability_sum=1.0,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="mitigator_id"):
        MaterialisedReadoutProbe(
            mitigator_id="",
            n_qubits=1,
            n_basis=2,
            mitigated_probability_sum=1.0,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="mitigated_probability_sum"):
        MaterialisedReadoutProbe(
            mitigator_id="readout_confusion",
            n_qubits=1,
            n_basis=2,
            mitigated_probability_sum=float("nan"),
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedReadoutProbe(
            mitigator_id="readout_confusion",
            n_qubits=1,
            n_basis=2,
            mitigated_probability_sum=1.0,
            probe_digest="x",
            invent_green_ideal_gradient_restore=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_ideal_gradient_restore"):
        MaterialisedReadoutProbe(
            mitigator_id="readout_confusion",
            n_qubits=1,
            n_basis=2,
            mitigated_probability_sum=1.0,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedReadoutProbe(
            mitigator_id="readout_confusion",
            n_qubits=1,
            n_basis=2,
            mitigated_probability_sum=1.0,
            probe_digest="a" * 64,
            invent_green_ideal_gradient_restore=False,
            demo_label="",
        )
    ok_r = MaterialisedReadoutProbe(
        mitigator_id="readout_confusion",
        n_qubits=1,
        n_basis=2,
        mitigated_probability_sum=1.0,
        probe_digest="a" * 64,
        invent_green_ideal_gradient_restore=False,
        demo_label="d",
    )
    assert ok_r.to_dict()["n_basis"] == 2


def test_integrity_more_edges() -> None:
    registry = build_error_mitigation_product_registry()
    mitigators = cast(list[dict[str, object]], registry["mitigators"])
    not_map = dict(registry)
    not_map["mitigators"] = cast(list[dict[str, object]], ["x"])
    with pytest.raises(ValueError, match="mapping"):
        assert_error_mitigation_product_integrity(not_map)

    blank_id = dict(registry)
    bc = [dict(row) for row in mitigators]
    bc[0]["mitigator_id"] = "  "
    blank_id["mitigators"] = bc
    with pytest.raises(ValueError, match="blank"):
        assert_error_mitigation_product_integrity(blank_id)

    dup = dict(registry)
    dc = [dict(row) for row in mitigators]
    dc[1] = dict(dc[0])
    dup["mitigators"] = dc
    with pytest.raises(ValueError, match="duplicate mitigator_id"):
        assert_error_mitigation_product_integrity(dup)

    bad_diff = dict(registry)
    bd = [dict(row) for row in mitigators]
    bd[0]["differentiability"] = "magic"
    bad_diff["mitigators"] = bd
    with pytest.raises(ValueError, match="differentiability"):
        assert_error_mitigation_product_integrity(bad_diff)

    no_sym = dict(registry)
    ns = [dict(row) for row in mitigators]
    ns[0]["ambient_symbol"] = ""
    no_sym["mitigators"] = ns
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_error_mitigation_product_integrity(no_sym)

    no_zne = dict(registry)
    filtered = [dict(row) for row in mitigators if row["mitigator_id"] != "zne_richardson"]
    no_zne["mitigators"] = filtered
    no_zne["mitigator_count"] = len(filtered)
    with pytest.raises(ValueError, match="zne_richardson|drift"):
        assert_error_mitigation_product_integrity(no_zne)

    bounds = cast(list[dict[str, object]], registry["boundaries"])
    b_not = dict(registry)
    b_not["boundaries"] = cast(list[dict[str, object]], [1])
    with pytest.raises(ValueError, match="mapping"):
        assert_error_mitigation_product_integrity(b_not)

    blank_b = dict(registry)
    bb = [dict(row) for row in bounds]
    bb[0]["boundary_id"] = ""
    blank_b["boundaries"] = bb
    with pytest.raises(ValueError, match="boundary_id"):
        assert_error_mitigation_product_integrity(blank_b)

    dup_b = dict(registry)
    db = [dict(row) for row in bounds]
    db[1] = dict(db[0])
    dup_b["boundaries"] = db
    with pytest.raises(ValueError, match="duplicate boundary_id"):
        assert_error_mitigation_product_integrity(dup_b)

    count_m = dict(registry)
    count_m["mitigator_count"] = 99
    with pytest.raises(ValueError, match="mitigator_count"):
        assert_error_mitigation_product_integrity(count_m)

    count_b = dict(registry)
    count_b["boundary_count"] = 99
    with pytest.raises(ValueError, match="boundary_count"):
        assert_error_mitigation_product_integrity(count_b)


def test_subprocess_payload_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    def _bad_zne(*_a: object, **_k: object) -> dict[str, object]:
        return {"not": "enough"}

    monkeypatch.setattr(mit_product, "_run_ambient_mitigation_json", _bad_zne)
    with pytest.raises(ValueError, match="missing fields"):
        materialise_zne_probe("zne_richardson")

    def _nan_zne(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "zero_noise_estimate": float("nan"),
            "fit_residual": 0.0,
            "order": 1,
            "n_points": 3,
        }

    monkeypatch.setattr(mit_product, "_run_ambient_mitigation_json", _nan_zne)
    with pytest.raises(ValueError, match="finite"):
        materialise_zne_probe("zne_richardson")

    def _neg_res(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "zero_noise_estimate": 1.0,
            "fit_residual": -0.1,
            "order": 1,
            "n_points": 3,
        }

    monkeypatch.setattr(mit_product, "_run_ambient_mitigation_json", _neg_res)
    with pytest.raises(ValueError, match="non-negative"):
        materialise_zne_probe("zne_richardson")

    def _few_pts(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "zero_noise_estimate": 1.0,
            "fit_residual": 0.0,
            "order": 1,
            "n_points": 1,
        }

    monkeypatch.setattr(mit_product, "_run_ambient_mitigation_json", _few_pts)
    with pytest.raises(ValueError, match="n_points"):
        materialise_zne_probe("zne_richardson")

    def _bad_ro(*_a: object, **_k: object) -> dict[str, object]:
        return {"n_qubits": 1}

    monkeypatch.setattr(mit_product, "_run_ambient_mitigation_json", _bad_ro)
    with pytest.raises(ValueError, match="missing fields"):
        materialise_readout_probe()

    def _nan_ro(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_qubits": 1,
            "n_basis": 2,
            "mitigated_probability_sum": float("nan"),
        }

    monkeypatch.setattr(mit_product, "_run_ambient_mitigation_json", _nan_ro)
    with pytest.raises(ValueError, match="finite"):
        materialise_readout_probe()

    monkeypatch.setattr(mit_product, "MITIGATE_CLAIM_BOUNDARY", "")
    with pytest.raises(ValueError, match="non-empty"):
        studio_mitigate_claim_boundary()
    monkeypatch.setattr(
        mit_product,
        "MITIGATE_CLAIM_BOUNDARY",
        "promotional claim without honesty",
    )
    with pytest.raises(ValueError, match="honesty"):
        studio_mitigate_claim_boundary()


def test_module_exports_stable() -> None:
    assert "assert_error_mitigation_product_integrity" in mit_product.__all__
    assert "materialise_demo_zne_probe" in mit_product.__all__
    assert ERROR_MITIGATION_PRODUCT_SCHEMA == "error_mitigation_product.v1"
