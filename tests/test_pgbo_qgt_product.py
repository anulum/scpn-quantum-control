# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for PGBO QGT product (BL-71)
"""Real-surface tests for ``pgbo_qgt_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.pgbo_qgt_product as qgt_product
from scpn_quantum_control.pgbo_qgt_product import (
    MAX_OSCILLATORS,
    PGBO_QGT_CLAIM_BOUNDARY,
    PGBO_QGT_PRODUCT_SCHEMA,
    MaterialisedPgboTensorProbe,
    PathEligibilityDecision,
    QgtBoundaryRow,
    QgtCapabilityRow,
    assert_pgbo_qgt_product_integrity,
    build_pgbo_qgt_product_registry,
    decide_qgt_path,
    get_qgt_boundary,
    get_qgt_capability,
    iter_qgt_boundaries,
    iter_qgt_capabilities,
    list_qgt_boundary_ids,
    list_qgt_capability_ids,
    map_pgbo_qgt_public_surfaces,
    materialise_demo_pgbo_tensor_probe,
    materialise_pgbo_tensor_probe,
)


def test_list_and_filters() -> None:
    ids = list_qgt_capability_ids()
    assert "pgbo_tensor" in ids
    assert "fubini_study_metric" in ids
    assert "berry_curvature" in ids
    assert "size_cap_policy" in ids
    assert "bl50_compose" in ids
    assert len(ids) == 5
    bounds = list_qgt_boundary_ids()
    assert "experimental_geometry_claim" in bounds
    assert "unbounded_system_size" in bounds
    assert len(bounds) == 4
    assert len(iter_qgt_capabilities(kind="pgbo_tensor")) == 1
    assert iter_qgt_boundaries(kind="live_qpu_qgt")


def test_get_known_and_unknown() -> None:
    row = get_qgt_capability("pgbo_tensor")
    assert row.claim_boundary == PGBO_QGT_CLAIM_BOUNDARY
    assert row.hardware_submit_allowed is False
    b = get_qgt_boundary("experimental_geometry_claim")
    assert b.fail_closed is True
    with pytest.raises(ValueError, match="non-empty"):
        get_qgt_capability("  ")
    with pytest.raises(ValueError, match="unknown capability_id"):
        get_qgt_capability("ghost")
    with pytest.raises(ValueError, match="unknown boundary_id"):
        get_qgt_boundary("ghost")


def test_decide_qgt_path() -> None:
    ok = decide_qgt_path("pgbo_tensor")
    assert ok.allowed is True
    exp = decide_qgt_path("pgbo_tensor", invent_green_experimental_geometry=True)
    assert exp.allowed is False
    assert any("experimental" in x.lower() for x in exp.blockers)
    qpu = decide_qgt_path("pgbo_tensor", invent_green_live_qpu=True)
    assert qpu.allowed is False
    n = decide_qgt_path("size_cap_policy", invent_green_unbounded_n=True)
    assert n.allowed is False
    fd = decide_qgt_path("pgbo_tensor", invent_green_fd_as_exact=True)
    assert fd.allowed is False


def test_pgbo_tensor_probe_real_ambient() -> None:
    probe = materialise_demo_pgbo_tensor_probe()
    assert probe.capability_id == "pgbo_tensor"
    assert probe.n_oscillators == 2
    assert probe.n_parameters == 1
    assert len(probe.probe_digest) == 64
    assert probe.invent_green_experimental_geometry is False
    # Deterministic re-run (ambient via clean subprocess under pytest-cov).
    again = materialise_pgbo_tensor_probe("pgbo_tensor")
    assert again.probe_digest == probe.probe_digest
    assert again.metric_determinant == pytest.approx(probe.metric_determinant)
    assert again.total_curvature == pytest.approx(probe.total_curvature)

    metric_cap = materialise_pgbo_tensor_probe("fubini_study_metric")
    assert metric_cap.metric_frobenius >= 0.0
    curv = materialise_pgbo_tensor_probe("berry_curvature")
    assert curv.total_curvature >= 0.0


def test_pgbo_probe_size_cap_and_refuses() -> None:
    with pytest.raises(ValueError, match="exceeds product cap|max"):
        materialise_pgbo_tensor_probe(n_oscillators=MAX_OSCILLATORS + 1)
    with pytest.raises(ValueError, match="experimental"):
        materialise_pgbo_tensor_probe(invent_green_experimental_geometry=True)
    with pytest.raises(ValueError, match="QPU|qpu|live"):
        materialise_pgbo_tensor_probe(invent_green_live_qpu=True)
    with pytest.raises(ValueError, match="tensor-family"):
        materialise_pgbo_tensor_probe("size_cap_policy")
    with pytest.raises(ValueError, match="epsilon"):
        materialise_pgbo_tensor_probe(epsilon=-0.1)


def test_n3_probe() -> None:
    probe = materialise_pgbo_tensor_probe(n_oscillators=3)
    assert probe.n_parameters == 3
    assert len(probe.parameter_labels) == 3


def test_registry_and_integrity() -> None:
    surfaces = map_pgbo_qgt_public_surfaces()
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.pgbo_qgt_product" in paths
    assert "scpn_quantum_control.pgbo.quantum_bridge" in paths

    registry = build_pgbo_qgt_product_registry()
    assert registry["schema"] == PGBO_QGT_PRODUCT_SCHEMA
    assert registry["max_oscillators"] == MAX_OSCILLATORS
    assert registry["experimental_geometry_claim_policy"] is False
    assert registry["unbounded_system_size_policy"] is False
    validated = assert_pgbo_qgt_product_integrity(registry)
    assert validated["capability_count"] == 5
    assert validated["boundary_count"] == 4
    assert assert_pgbo_qgt_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift() -> None:
    registry = build_pgbo_qgt_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])

    broken = dict(registry)
    broken["capabilities"] = caps + [
        {
            "capability_id": "ghost",
            "kind": "pgbo_tensor",
            "title": "t",
            "summary": "s",
            "ambient_module": "m",
            "ambient_symbol": "x",
            "hardware_submit_allowed": False,
            "support_posture": "local_research",
            "as_of": "2026-07-24",
            "claim_boundary": PGBO_QGT_CLAIM_BOUNDARY,
        }
    ]
    broken["capability_count"] = len(cast(list[object], broken["capabilities"]))
    with pytest.raises(ValueError, match="drift"):
        assert_pgbo_qgt_product_integrity(broken)

    empty: dict[str, object] = {
        "capabilities": [],
        "boundaries": registry["boundaries"],
        "blank_entry_count": 0,
        "capability_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty capabilities"):
        assert_pgbo_qgt_product_integrity(empty)

    no_b = dict(registry)
    no_b["boundaries"] = []
    no_b["boundary_count"] = 0
    with pytest.raises(ValueError, match="non-empty boundaries"):
        assert_pgbo_qgt_product_integrity(no_b)

    for policy in (
        "hardware_submit_allowed_policy",
        "experimental_geometry_claim_policy",
        "unbounded_system_size_policy",
        "fd_derivative_as_exact_policy",
    ):
        bad = dict(registry)
        bad[policy] = True
        with pytest.raises(ValueError, match=policy):
            assert_pgbo_qgt_product_integrity(bad)

    hw = dict(registry)
    mut = [dict(row) for row in caps]
    mut[0]["hardware_submit_allowed"] = True
    hw["capabilities"] = mut
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        assert_pgbo_qgt_product_integrity(hw)

    blank = dict(registry)
    blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_pgbo_qgt_product_integrity(blank)

    bad_max = dict(registry)
    bad_max["max_oscillators"] = 99
    with pytest.raises(ValueError, match="max_oscillators"):
        assert_pgbo_qgt_product_integrity(bad_max)

    bounds = cast(list[dict[str, object]], registry["boundaries"])
    fc = dict(registry)
    mut_b = [dict(row) for row in bounds]
    mut_b[0]["fail_closed"] = False
    fc["boundaries"] = mut_b
    with pytest.raises(ValueError, match="fail_closed"):
        assert_pgbo_qgt_product_integrity(fc)


def test_integrity_more_edges() -> None:
    registry = build_pgbo_qgt_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    not_map = dict(registry)
    not_map["capabilities"] = cast(list[dict[str, object]], ["x"])
    with pytest.raises(ValueError, match="mapping"):
        assert_pgbo_qgt_product_integrity(not_map)

    blank_id = dict(registry)
    bc = [dict(row) for row in caps]
    bc[0]["capability_id"] = "  "
    blank_id["capabilities"] = bc
    with pytest.raises(ValueError, match="blank"):
        assert_pgbo_qgt_product_integrity(blank_id)

    dup = dict(registry)
    dc = [dict(row) for row in caps]
    dc[1] = dict(dc[0])
    dup["capabilities"] = dc
    with pytest.raises(ValueError, match="duplicate capability_id"):
        assert_pgbo_qgt_product_integrity(dup)

    no_sym = dict(registry)
    ns = [dict(row) for row in caps]
    ns[0]["ambient_symbol"] = ""
    no_sym["capabilities"] = ns
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_pgbo_qgt_product_integrity(no_sym)

    no_tensor = dict(registry)
    filtered = [dict(row) for row in caps if row["capability_id"] != "pgbo_tensor"]
    no_tensor["capabilities"] = filtered
    no_tensor["capability_count"] = len(filtered)
    with pytest.raises(ValueError, match="pgbo_tensor|drift"):
        assert_pgbo_qgt_product_integrity(no_tensor)

    bounds = cast(list[dict[str, object]], registry["boundaries"])
    b_not = dict(registry)
    b_not["boundaries"] = cast(list[dict[str, object]], [1])
    with pytest.raises(ValueError, match="mapping"):
        assert_pgbo_qgt_product_integrity(b_not)

    blank_b = dict(registry)
    bb = [dict(row) for row in bounds]
    bb[0]["boundary_id"] = ""
    blank_b["boundaries"] = bb
    with pytest.raises(ValueError, match="boundary_id"):
        assert_pgbo_qgt_product_integrity(blank_b)

    dup_b = dict(registry)
    db = [dict(row) for row in bounds]
    db[1] = dict(db[0])
    dup_b["boundaries"] = db
    with pytest.raises(ValueError, match="duplicate boundary_id"):
        assert_pgbo_qgt_product_integrity(dup_b)

    count_m = dict(registry)
    count_m["capability_count"] = 99
    with pytest.raises(ValueError, match="capability_count"):
        assert_pgbo_qgt_product_integrity(count_m)

    count_b = dict(registry)
    count_b["boundary_count"] = 99
    with pytest.raises(ValueError, match="boundary_count"):
        assert_pgbo_qgt_product_integrity(count_b)


def test_dataclass_invariants() -> None:
    with pytest.raises(ValueError, match="capability_id"):
        QgtCapabilityRow(
            capability_id="",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="unknown capability kind"):
        QgtCapabilityRow(
            capability_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="title"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="summary"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="ambient_module"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="ambient_symbol"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="",
        )
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
            hardware_submit_allowed=True,
        )
    with pytest.raises(ValueError, match="support_posture"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
            support_posture=cast(Any, "bogus"),
        )
    with pytest.raises(ValueError, match="as_of"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
            as_of="",
        )
    ok = QgtCapabilityRow(
        capability_id="x",
        kind="pgbo_tensor",
        title="t",
        summary="s",
        ambient_module="m",
        ambient_symbol="x",
    )
    assert ok.to_dict()["capability_id"] == "x"

    with pytest.raises(ValueError, match="boundary_id"):
        QgtBoundaryRow(
            boundary_id="",
            kind="experimental_geometry_claim",
            title="t",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="unknown boundary kind"):
        QgtBoundaryRow(
            boundary_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="fail_closed"):
        QgtBoundaryRow(
            boundary_id="x",
            kind="experimental_geometry_claim",
            title="t",
            failure_class="f",
            summary="s",
            fail_closed=False,
        )
    with pytest.raises(ValueError, match="title"):
        QgtBoundaryRow(
            boundary_id="x",
            kind="experimental_geometry_claim",
            title="",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="failure_class"):
        QgtBoundaryRow(
            boundary_id="x",
            kind="experimental_geometry_claim",
            title="t",
            failure_class="",
            summary="s",
        )
    with pytest.raises(ValueError, match="summary"):
        QgtBoundaryRow(
            boundary_id="x",
            kind="experimental_geometry_claim",
            title="t",
            failure_class="f",
            summary="",
        )
    ok_b = QgtBoundaryRow(
        boundary_id="x",
        kind="experimental_geometry_claim",
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

    with pytest.raises(ValueError, match="capability_id"):
        MaterialisedPgboTensorProbe(
            capability_id="",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("K_01",),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="n_oscillators"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=1,
            n_parameters=0,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=(),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="upper-triangle"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=2,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("a", "b"),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("K_01",),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="x",
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_experimental_geometry"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("K_01",),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=True,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_live_qpu"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("K_01",),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("K_01",),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="",
        )
    with pytest.raises(ValueError, match="total_curvature"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=-1.0,
            parameter_labels=("K_01",),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="epsilon"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("K_01",),
            metric_frobenius=0.1,
            epsilon=0.0,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    ok_p = MaterialisedPgboTensorProbe(
        capability_id="pgbo_tensor",
        n_oscillators=2,
        n_parameters=1,
        metric_determinant=0.1,
        total_curvature=0.0,
        parameter_labels=("K_01",),
        metric_frobenius=0.1,
        epsilon=0.005,
        probe_digest="a" * 64,
        invent_green_experimental_geometry=False,
        invent_green_live_qpu=False,
        demo_label="d",
    )
    assert ok_p.to_dict()["n_parameters"] == 1


def test_ambient_payload_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    def _zero_params(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_parameters": 0,
            "metric_determinant": 0.0,
            "total_curvature": 0.0,
            "parameter_labels": [],
            "metric_frobenius": 0.0,
            "metric_shape": [0, 0],
            "metric_finite": True,
        }

    monkeypatch.setattr(qgt_product, "_run_ambient_pgbo_json", _zero_params)
    with pytest.raises(ValueError, match="positive n_parameters"):
        materialise_pgbo_tensor_probe()

    def _rect(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_parameters": 1,
            "metric_determinant": 0.0,
            "total_curvature": 0.0,
            "parameter_labels": ["K_01"],
            "metric_frobenius": 1.0,
            "metric_shape": [1, 2],
            "metric_finite": True,
        }

    monkeypatch.setattr(qgt_product, "_run_ambient_pgbo_json", _rect)
    with pytest.raises(ValueError, match="square"):
        materialise_pgbo_tensor_probe()

    def _nan(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_parameters": 1,
            "metric_determinant": 0.0,
            "total_curvature": 0.0,
            "parameter_labels": ["K_01"],
            "metric_frobenius": 1.0,
            "metric_shape": [1, 1],
            "metric_finite": False,
        }

    monkeypatch.setattr(qgt_product, "_run_ambient_pgbo_json", _nan)
    with pytest.raises(ValueError, match="finite"):
        materialise_pgbo_tensor_probe()

    def _missing(*_a: object, **_k: object) -> dict[str, object]:
        return {"n_parameters": 1}

    monkeypatch.setattr(qgt_product, "_run_ambient_pgbo_json", _missing)
    with pytest.raises(ValueError, match="missing fields"):
        materialise_pgbo_tensor_probe()


def test_demo_system_and_subprocess_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        qgt_product._demo_coupling_system(1)
    with pytest.raises(ValueError, match="MAX_OSCILLATORS|<="):
        qgt_product._demo_coupling_system(MAX_OSCILLATORS + 1)
    # n=2 and n=3 demo systems construct without error
    k2, o2 = qgt_product._demo_coupling_system(2)
    assert k2.shape == (2, 2)
    assert o2.shape == (2,)
    k3, o3 = qgt_product._demo_coupling_system(3)
    assert k3.shape == (3, 3)

    def _bad_labels(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_parameters": 1,
            "metric_determinant": 0.1,
            "total_curvature": 0.0,
            "parameter_labels": ["a", "b"],
            "metric_frobenius": 0.1,
            "metric_shape": [1, 1],
            "metric_finite": True,
        }

    monkeypatch.setattr(qgt_product, "_run_ambient_pgbo_json", _bad_labels)
    with pytest.raises(ValueError, match="parameter_labels"):
        materialise_pgbo_tensor_probe()

    def _bad_det(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_parameters": 1,
            "metric_determinant": float("nan"),
            "total_curvature": 0.0,
            "parameter_labels": ["K_01"],
            "metric_frobenius": 0.1,
            "metric_shape": [1, 1],
            "metric_finite": True,
        }

    monkeypatch.setattr(qgt_product, "_run_ambient_pgbo_json", _bad_det)
    with pytest.raises(ValueError, match="metric_determinant"):
        materialise_pgbo_tensor_probe()

    def _bad_curv(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_parameters": 1,
            "metric_determinant": 0.1,
            "total_curvature": -1.0,
            "parameter_labels": ["K_01"],
            "metric_frobenius": 0.1,
            "metric_shape": [1, 1],
            "metric_finite": True,
        }

    monkeypatch.setattr(qgt_product, "_run_ambient_pgbo_json", _bad_curv)
    with pytest.raises(ValueError, match="total_curvature"):
        materialise_pgbo_tensor_probe()

    def _bad_frob(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_parameters": 1,
            "metric_determinant": 0.1,
            "total_curvature": 0.0,
            "parameter_labels": ["K_01"],
            "metric_frobenius": -0.1,
            "metric_shape": [1, 1],
            "metric_finite": True,
        }

    monkeypatch.setattr(qgt_product, "_run_ambient_pgbo_json", _bad_frob)
    with pytest.raises(ValueError, match="metric_frobenius"):
        materialise_pgbo_tensor_probe()


def test_probe_dataclass_more() -> None:
    with pytest.raises(ValueError, match="product probes"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=MAX_OSCILLATORS + 1,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("K_01",),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="n_parameters"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=0,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=(),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="parameter_labels"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("",),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="metric_frobenius"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=0.1,
            total_curvature=0.0,
            parameter_labels=("K_01",),
            metric_frobenius=-1.0,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="metric_determinant"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
            metric_determinant=float("nan"),
            total_curvature=0.0,
            parameter_labels=("K_01",),
            metric_frobenius=0.1,
            epsilon=0.005,
            probe_digest="a" * 64,
            invent_green_experimental_geometry=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )


def test_module_exports_stable() -> None:
    assert "assert_pgbo_qgt_product_integrity" in qgt_product.__all__
    assert "materialise_demo_pgbo_tensor_probe" in qgt_product.__all__
    assert PGBO_QGT_PRODUCT_SCHEMA == "pgbo_qgt_product.v1"
    assert MAX_OSCILLATORS == 6
