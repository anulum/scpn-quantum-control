# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for PGBO QGT product
"""Real-surface tests for ``pgbo_qgt_product`` (one concern per test)."""

from __future__ import annotations

import subprocess
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


def test_list_capability_ids_covers_tensor_family_and_policy() -> None:
    ids = list_qgt_capability_ids()
    assert ids == (
        "pgbo_tensor",
        "fubini_study_metric",
        "berry_curvature",
        "size_cap_policy",
        "geometric_control_compose",
    )


def test_list_boundary_ids_covers_honesty_gaps() -> None:
    ids = list_qgt_boundary_ids()
    assert "experimental_geometry_claim" in ids
    assert "live_qpu_qgt" in ids
    assert "unbounded_system_size" in ids
    assert "fd_derivative_as_exact" in ids
    assert len(ids) == 4


def test_iter_capabilities_without_kind_returns_full_catalogue() -> None:
    rows = iter_qgt_capabilities()
    assert len(rows) == 5
    assert {row.capability_id for row in rows} == set(list_qgt_capability_ids())


def test_iter_capabilities_filters_by_kind() -> None:
    rows = iter_qgt_capabilities(kind="pgbo_tensor")
    assert len(rows) == 1
    assert rows[0].capability_id == "pgbo_tensor"


def test_iter_boundaries_without_kind_returns_full_catalogue() -> None:
    rows = iter_qgt_boundaries()
    assert len(rows) == 4
    assert {row.boundary_id for row in rows} == set(list_qgt_boundary_ids())


def test_iter_boundaries_filters_by_kind() -> None:
    rows = iter_qgt_boundaries(kind="live_qpu_qgt")
    assert len(rows) == 1
    assert rows[0].boundary_id == "live_qpu_qgt"


def test_get_qgt_capability_known_row() -> None:
    row = get_qgt_capability("pgbo_tensor")
    assert row.claim_boundary == PGBO_QGT_CLAIM_BOUNDARY
    assert row.hardware_submit_allowed is False
    assert row.ambient_symbol == "compute_pgbo_tensor"


def test_get_qgt_capability_rejects_blank() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        get_qgt_capability("  ")


def test_get_qgt_capability_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown capability_id"):
        get_qgt_capability("ghost")


def test_get_qgt_boundary_known_row() -> None:
    b = get_qgt_boundary("experimental_geometry_claim")
    assert b.fail_closed is True
    assert b.kind == "experimental_geometry_claim"


def test_get_qgt_boundary_rejects_blank() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        get_qgt_boundary("  ")


def test_get_qgt_boundary_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown boundary_id"):
        get_qgt_boundary("ghost")


def test_decide_qgt_path_allows_clean_local_probe() -> None:
    ok = decide_qgt_path("pgbo_tensor")
    assert ok.allowed is True
    assert ok.outcome == "allowed"
    assert ok.blockers == ()


def test_decide_qgt_path_refuses_experimental_geometry() -> None:
    decision = decide_qgt_path("pgbo_tensor", invent_green_experimental_geometry=True)
    assert decision.allowed is False
    assert any("experimental" in x.lower() for x in decision.blockers)


def test_decide_qgt_path_refuses_live_qpu() -> None:
    decision = decide_qgt_path("pgbo_tensor", invent_green_live_qpu=True)
    assert decision.allowed is False
    assert any("qpu" in x.lower() for x in decision.blockers)


def test_decide_qgt_path_refuses_unbounded_n() -> None:
    decision = decide_qgt_path("size_cap_policy", invent_green_unbounded_n=True)
    assert decision.allowed is False
    assert any("unbounded" in x.lower() for x in decision.blockers)


def test_decide_qgt_path_refuses_fd_as_exact() -> None:
    decision = decide_qgt_path("pgbo_tensor", invent_green_fd_as_exact=True)
    assert decision.allowed is False
    assert any("exact" in x.lower() or "fd" in x.lower() for x in decision.blockers)


def test_demo_pgbo_tensor_probe_real_ambient() -> None:
    probe = materialise_demo_pgbo_tensor_probe()
    assert probe.capability_id == "pgbo_tensor"
    assert probe.n_oscillators == 2
    assert probe.n_parameters == 1
    assert len(probe.probe_digest) == 64
    assert probe.invent_green_experimental_geometry is False
    assert probe.invent_green_live_qpu is False


def test_pgbo_tensor_probe_digest_deterministic() -> None:
    first = materialise_pgbo_tensor_probe("pgbo_tensor")
    again = materialise_pgbo_tensor_probe("pgbo_tensor")
    assert again.probe_digest == first.probe_digest
    assert again.metric_determinant == pytest.approx(first.metric_determinant)
    assert again.total_curvature == pytest.approx(first.total_curvature)


def test_fubini_study_metric_capability_probe() -> None:
    probe = materialise_pgbo_tensor_probe("fubini_study_metric")
    assert probe.capability_id == "fubini_study_metric"
    assert probe.metric_frobenius >= 0.0


def test_berry_curvature_capability_probe() -> None:
    probe = materialise_pgbo_tensor_probe("berry_curvature")
    assert probe.capability_id == "berry_curvature"
    assert probe.total_curvature >= 0.0


def test_n3_probe_parameter_count() -> None:
    probe = materialise_pgbo_tensor_probe(n_oscillators=3)
    assert probe.n_parameters == 3
    assert len(probe.parameter_labels) == 3


def test_probe_refuses_size_cap_exceeded() -> None:
    with pytest.raises(ValueError, match="exceeds product cap|max"):
        materialise_pgbo_tensor_probe(n_oscillators=MAX_OSCILLATORS + 1)


def test_probe_refuses_experimental_geometry() -> None:
    with pytest.raises(ValueError, match="experimental"):
        materialise_pgbo_tensor_probe(invent_green_experimental_geometry=True)


def test_probe_refuses_live_qpu() -> None:
    with pytest.raises(ValueError, match="QPU|qpu|live"):
        materialise_pgbo_tensor_probe(invent_green_live_qpu=True)


def test_probe_refuses_unbounded_n_flag() -> None:
    with pytest.raises(ValueError, match="unbounded|max_oscillators"):
        materialise_pgbo_tensor_probe(invent_green_unbounded_n=True)


def test_probe_refuses_fd_as_exact_flag() -> None:
    with pytest.raises(ValueError, match="exact|finite difference|FD|fd"):
        materialise_pgbo_tensor_probe(invent_green_fd_as_exact=True)


def test_probe_refuses_non_tensor_family_capability() -> None:
    with pytest.raises(ValueError, match="tensor-family"):
        materialise_pgbo_tensor_probe("size_cap_policy")


def test_probe_refuses_non_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        materialise_pgbo_tensor_probe(epsilon=-0.1)


def test_map_public_surfaces_includes_product_and_ambient() -> None:
    surfaces = map_pgbo_qgt_public_surfaces()
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.pgbo_qgt_product" in paths
    assert "scpn_quantum_control.pgbo.quantum_bridge" in paths
    assert "scpn_quantum_control.geometric_control_product" in paths


def test_build_registry_schema_and_policy_flags() -> None:
    registry = build_pgbo_qgt_product_registry()
    assert registry["schema"] == PGBO_QGT_PRODUCT_SCHEMA
    assert registry["max_oscillators"] == MAX_OSCILLATORS
    assert registry["experimental_geometry_claim_policy"] is False
    assert registry["unbounded_system_size_policy"] is False
    assert registry["hardware_submit_allowed_policy"] is False
    assert registry["fd_derivative_as_exact_policy"] is False
    assert registry["blank_entry_count"] == 0


def test_assert_integrity_on_live_registry() -> None:
    validated = assert_pgbo_qgt_product_integrity()
    assert validated["capability_count"] == 5
    assert validated["boundary_count"] == 4
    assert validated["blank_entry_count"] == 0


def test_assert_integrity_accepts_explicit_payload() -> None:
    registry = build_pgbo_qgt_product_registry()
    validated = assert_pgbo_qgt_product_integrity(registry)
    assert validated["schema"] == PGBO_QGT_PRODUCT_SCHEMA


def test_integrity_rejects_stale_schema() -> None:
    """Reject the exact superseded serialized contract."""
    registry = build_pgbo_qgt_product_registry()
    registry["schema"] = "pgbo_qgt_product.v1"
    with pytest.raises(ValueError, match="unexpected PGBO QGT product schema"):
        assert_pgbo_qgt_product_integrity(registry)


def test_integrity_rejects_unexpected_registry_key() -> None:
    """Reject compatibility aliases outside the canonical registry shape."""
    registry = build_pgbo_qgt_product_registry()
    registry["legacy_alias"] = "deprecated"
    with pytest.raises(ValueError, match="registry keys drift"):
        assert_pgbo_qgt_product_integrity(registry)


def test_integrity_rejects_claim_boundary_drift() -> None:
    """Reject a top-level claim boundary that diverges from the product."""
    registry = build_pgbo_qgt_product_registry()
    registry["claim_boundary"] = "legacy planning label"
    with pytest.raises(ValueError, match="claim boundary drift"):
        assert_pgbo_qgt_product_integrity(registry)


def test_integrity_rejects_public_surface_drift() -> None:
    """Reject public-surface metadata that diverges from the live mapper."""
    registry = build_pgbo_qgt_product_registry()
    registry["public_surfaces"] = []
    with pytest.raises(ValueError, match="public surface map drift"):
        assert_pgbo_qgt_product_integrity(registry)


def test_integrity_rejects_policy_note_drift() -> None:
    """Reject stale serialized policy language."""
    registry = build_pgbo_qgt_product_registry()
    registry["policy_note"] = "legacy planning label"
    with pytest.raises(ValueError, match="policy note drift"):
        assert_pgbo_qgt_product_integrity(registry)


def test_integrity_rejects_empty_capabilities() -> None:
    registry = build_pgbo_qgt_product_registry()
    empty = dict(registry)
    empty["capabilities"] = []
    empty["capability_count"] = 0
    with pytest.raises(ValueError, match="non-empty capabilities"):
        assert_pgbo_qgt_product_integrity(empty)


def test_integrity_rejects_empty_boundaries() -> None:
    registry = build_pgbo_qgt_product_registry()
    no_b = dict(registry)
    no_b["boundaries"] = []
    no_b["boundary_count"] = 0
    with pytest.raises(ValueError, match="non-empty boundaries"):
        assert_pgbo_qgt_product_integrity(no_b)


def test_integrity_rejects_capability_set_drift() -> None:
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


def test_integrity_rejects_boundary_set_drift() -> None:
    registry = build_pgbo_qgt_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    broken = dict(registry)
    broken["boundaries"] = bounds + [
        {
            "boundary_id": "ghost_boundary",
            "kind": "experimental_geometry_claim",
            "title": "t",
            "failure_class": "f",
            "summary": "s",
            "fail_closed": True,
            "claim_boundary": PGBO_QGT_CLAIM_BOUNDARY,
        }
    ]
    broken["boundary_count"] = len(cast(list[object], broken["boundaries"]))
    with pytest.raises(ValueError, match="boundary set drift|drift"):
        assert_pgbo_qgt_product_integrity(broken)


def test_integrity_rejects_hardware_submit_policy_true() -> None:
    registry = build_pgbo_qgt_product_registry()
    bad = dict(registry)
    bad["hardware_submit_allowed_policy"] = True
    with pytest.raises(ValueError, match="hardware_submit_allowed_policy"):
        assert_pgbo_qgt_product_integrity(bad)


def test_integrity_rejects_experimental_geometry_policy_true() -> None:
    registry = build_pgbo_qgt_product_registry()
    bad = dict(registry)
    bad["experimental_geometry_claim_policy"] = True
    with pytest.raises(ValueError, match="experimental_geometry_claim_policy"):
        assert_pgbo_qgt_product_integrity(bad)


def test_integrity_rejects_unbounded_size_policy_true() -> None:
    registry = build_pgbo_qgt_product_registry()
    bad = dict(registry)
    bad["unbounded_system_size_policy"] = True
    with pytest.raises(ValueError, match="unbounded_system_size_policy"):
        assert_pgbo_qgt_product_integrity(bad)


def test_integrity_rejects_fd_as_exact_policy_true() -> None:
    registry = build_pgbo_qgt_product_registry()
    bad = dict(registry)
    bad["fd_derivative_as_exact_policy"] = True
    with pytest.raises(ValueError, match="fd_derivative_as_exact_policy"):
        assert_pgbo_qgt_product_integrity(bad)


def test_integrity_rejects_row_hardware_submit_true() -> None:
    registry = build_pgbo_qgt_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    hw = dict(registry)
    mut = [dict(row) for row in caps]
    mut[0]["hardware_submit_allowed"] = True
    hw["capabilities"] = mut
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        assert_pgbo_qgt_product_integrity(hw)


def test_integrity_rejects_nonzero_blank_entry_count() -> None:
    registry = build_pgbo_qgt_product_registry()
    blank = dict(registry)
    blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_pgbo_qgt_product_integrity(blank)


def test_integrity_rejects_wrong_max_oscillators() -> None:
    registry = build_pgbo_qgt_product_registry()
    bad_max = dict(registry)
    bad_max["max_oscillators"] = 99
    with pytest.raises(ValueError, match="max_oscillators"):
        assert_pgbo_qgt_product_integrity(bad_max)


def test_integrity_rejects_default_epsilon_drift() -> None:
    """Reject a registry epsilon that differs from the runtime default."""
    registry = build_pgbo_qgt_product_registry()
    registry["default_epsilon"] = 0.1
    with pytest.raises(ValueError, match="default_epsilon"):
        assert_pgbo_qgt_product_integrity(registry)


def test_integrity_rejects_boundary_fail_closed_false() -> None:
    registry = build_pgbo_qgt_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    fc = dict(registry)
    mut_b = [dict(row) for row in bounds]
    mut_b[0]["fail_closed"] = False
    fc["boundaries"] = mut_b
    with pytest.raises(ValueError, match="fail_closed"):
        assert_pgbo_qgt_product_integrity(fc)


def test_integrity_rejects_boundary_row_drift() -> None:
    """Reject non-canonical serialized boundary content."""
    registry = build_pgbo_qgt_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    drift = dict(registry)
    rows = [dict(row) for row in bounds]
    rows[0]["summary"] = "legacy planning label"
    drift["boundaries"] = rows
    with pytest.raises(ValueError, match="catalogue row drift"):
        assert_pgbo_qgt_product_integrity(drift)


def test_integrity_rejects_capability_row_not_mapping() -> None:
    registry = build_pgbo_qgt_product_registry()
    not_map = dict(registry)
    not_map["capabilities"] = cast(list[dict[str, object]], ["x"])
    with pytest.raises(ValueError, match="mapping"):
        assert_pgbo_qgt_product_integrity(not_map)


def test_integrity_rejects_blank_capability_id_in_registry() -> None:
    registry = build_pgbo_qgt_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    blank_id = dict(registry)
    bc = [dict(row) for row in caps]
    bc[0]["capability_id"] = "  "
    blank_id["capabilities"] = bc
    with pytest.raises(ValueError, match="blank"):
        assert_pgbo_qgt_product_integrity(blank_id)


def test_integrity_rejects_duplicate_capability_id() -> None:
    registry = build_pgbo_qgt_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    dup = dict(registry)
    dc = [dict(row) for row in caps]
    dc[1] = dict(dc[0])
    dup["capabilities"] = dc
    with pytest.raises(ValueError, match="duplicate capability_id"):
        assert_pgbo_qgt_product_integrity(dup)


def test_integrity_rejects_empty_ambient_symbol() -> None:
    registry = build_pgbo_qgt_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    no_sym = dict(registry)
    ns = [dict(row) for row in caps]
    ns[0]["ambient_symbol"] = ""
    no_sym["capabilities"] = ns
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_pgbo_qgt_product_integrity(no_sym)


def test_integrity_rejects_capability_row_drift() -> None:
    """Reject non-canonical serialized capability content."""
    registry = build_pgbo_qgt_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    drift = dict(registry)
    rows = [dict(row) for row in caps]
    rows[0]["summary"] = "legacy planning label"
    drift["capabilities"] = rows
    with pytest.raises(ValueError, match="catalogue row drift"):
        assert_pgbo_qgt_product_integrity(drift)


def test_integrity_rejects_missing_pgbo_tensor() -> None:
    registry = build_pgbo_qgt_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    no_tensor = dict(registry)
    filtered = [dict(row) for row in caps if row["capability_id"] != "pgbo_tensor"]
    no_tensor["capabilities"] = filtered
    no_tensor["capability_count"] = len(filtered)
    with pytest.raises(ValueError, match="pgbo_tensor|drift"):
        assert_pgbo_qgt_product_integrity(no_tensor)


def test_integrity_rejects_boundary_row_not_mapping() -> None:
    registry = build_pgbo_qgt_product_registry()
    b_not = dict(registry)
    b_not["boundaries"] = cast(list[dict[str, object]], [1])
    with pytest.raises(ValueError, match="mapping"):
        assert_pgbo_qgt_product_integrity(b_not)


def test_integrity_rejects_blank_boundary_id() -> None:
    registry = build_pgbo_qgt_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    blank_b = dict(registry)
    bb = [dict(row) for row in bounds]
    bb[0]["boundary_id"] = ""
    blank_b["boundaries"] = bb
    with pytest.raises(ValueError, match="boundary_id"):
        assert_pgbo_qgt_product_integrity(blank_b)


def test_integrity_rejects_duplicate_boundary_id() -> None:
    registry = build_pgbo_qgt_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    dup_b = dict(registry)
    db = [dict(row) for row in bounds]
    db[1] = dict(db[0])
    dup_b["boundaries"] = db
    with pytest.raises(ValueError, match="duplicate boundary_id"):
        assert_pgbo_qgt_product_integrity(dup_b)


def test_integrity_rejects_capability_count_mismatch() -> None:
    registry = build_pgbo_qgt_product_registry()
    count_m = dict(registry)
    count_m["capability_count"] = 99
    with pytest.raises(ValueError, match="capability_count"):
        assert_pgbo_qgt_product_integrity(count_m)


def test_integrity_rejects_boundary_count_mismatch() -> None:
    registry = build_pgbo_qgt_product_registry()
    count_b = dict(registry)
    count_b["boundary_count"] = 99
    with pytest.raises(ValueError, match="boundary_count"):
        assert_pgbo_qgt_product_integrity(count_b)


def test_capability_row_rejects_blank_id() -> None:
    with pytest.raises(ValueError, match="capability_id"):
        QgtCapabilityRow(
            capability_id="",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="sym",
        )


def test_capability_row_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown capability kind"):
        QgtCapabilityRow(
            capability_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="sym",
        )


def test_capability_row_rejects_blank_title() -> None:
    with pytest.raises(ValueError, match="title"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="",
            summary="s",
            ambient_module="m",
            ambient_symbol="sym",
        )


def test_capability_row_rejects_blank_summary() -> None:
    with pytest.raises(ValueError, match="summary"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="",
            ambient_module="m",
            ambient_symbol="sym",
        )


def test_capability_row_rejects_blank_ambient_module() -> None:
    with pytest.raises(ValueError, match="ambient_module"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="",
            ambient_symbol="sym",
        )


def test_capability_row_rejects_blank_ambient_symbol() -> None:
    with pytest.raises(ValueError, match="ambient_symbol"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="",
        )


def test_capability_row_rejects_hardware_submit_true() -> None:
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="sym",
            hardware_submit_allowed=True,
        )


def test_capability_row_rejects_unknown_support_posture() -> None:
    with pytest.raises(ValueError, match="support_posture"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="sym",
            support_posture=cast(Any, "bogus"),
        )


def test_capability_row_rejects_blank_as_of() -> None:
    with pytest.raises(ValueError, match="as_of"):
        QgtCapabilityRow(
            capability_id="x",
            kind="pgbo_tensor",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="sym",
            as_of="",
        )


def test_capability_row_to_dict_round_trip_fields() -> None:
    row = QgtCapabilityRow(
        capability_id="x",
        kind="pgbo_tensor",
        title="t",
        summary="s",
        ambient_module="m",
        ambient_symbol="sym",
    )
    payload = row.to_dict()
    assert payload["capability_id"] == "x"
    assert payload["hardware_submit_allowed"] is False


def test_boundary_row_rejects_blank_id() -> None:
    with pytest.raises(ValueError, match="boundary_id"):
        QgtBoundaryRow(
            boundary_id="",
            kind="experimental_geometry_claim",
            title="t",
            failure_class="f",
            summary="s",
        )


def test_boundary_row_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown boundary kind"):
        QgtBoundaryRow(
            boundary_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            failure_class="f",
            summary="s",
        )


def test_boundary_row_rejects_fail_closed_false() -> None:
    with pytest.raises(ValueError, match="fail_closed"):
        QgtBoundaryRow(
            boundary_id="x",
            kind="experimental_geometry_claim",
            title="t",
            failure_class="f",
            summary="s",
            fail_closed=False,
        )


def test_boundary_row_rejects_blank_title() -> None:
    with pytest.raises(ValueError, match="title"):
        QgtBoundaryRow(
            boundary_id="x",
            kind="experimental_geometry_claim",
            title="",
            failure_class="f",
            summary="s",
        )


def test_boundary_row_rejects_blank_failure_class() -> None:
    with pytest.raises(ValueError, match="failure_class"):
        QgtBoundaryRow(
            boundary_id="x",
            kind="experimental_geometry_claim",
            title="t",
            failure_class="",
            summary="s",
        )


def test_boundary_row_rejects_blank_summary() -> None:
    with pytest.raises(ValueError, match="summary"):
        QgtBoundaryRow(
            boundary_id="x",
            kind="experimental_geometry_claim",
            title="t",
            failure_class="f",
            summary="",
        )


def test_boundary_row_to_dict() -> None:
    row = QgtBoundaryRow(
        boundary_id="x",
        kind="experimental_geometry_claim",
        title="t",
        failure_class="f",
        summary="s",
    )
    assert row.to_dict()["fail_closed"] is True


def test_path_decision_refuses_empty_blockers_on_refused() -> None:
    with pytest.raises(ValueError, match="blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )


def test_path_decision_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "maybe"),
            allowed=True,
            reason="r",
            blockers=(),
        )


def test_path_decision_rejects_blank_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="",
            blockers=(),
        )


def test_path_decision_rejects_allowed_with_refused_outcome() -> None:
    with pytest.raises(ValueError, match="outcome=allowed"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
        )


def test_path_decision_rejects_refused_with_allowed_outcome() -> None:
    with pytest.raises(ValueError, match="outcome=refused"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=False,
            reason="r",
            blockers=("b",),
        )


def test_path_decision_rejects_allowed_with_blockers() -> None:
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("b",),
        )


def test_path_decision_rejects_blank_blocker_entry() -> None:
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("",),
        )


def test_path_decision_to_dict() -> None:
    decision = PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason="ok",
        blockers=(),
    )
    assert decision.to_dict()["allowed"] is True


def test_probe_rejects_blank_capability_id() -> None:
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


def test_probe_rejects_n_oscillators_below_two() -> None:
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


def test_probe_rejects_n_oscillators_above_cap() -> None:
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


def test_probe_rejects_non_positive_n_parameters() -> None:
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


def test_probe_rejects_n_parameters_not_upper_triangle() -> None:
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


def test_probe_rejects_non_finite_metric_determinant() -> None:
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


def test_probe_rejects_negative_total_curvature() -> None:
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


def test_probe_rejects_parameter_labels_length_mismatch() -> None:
    with pytest.raises(ValueError, match="parameter_labels length"):
        MaterialisedPgboTensorProbe(
            capability_id="pgbo_tensor",
            n_oscillators=2,
            n_parameters=1,
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


def test_probe_rejects_blank_parameter_label_entry() -> None:
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


def test_probe_rejects_negative_metric_frobenius() -> None:
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


def test_probe_rejects_non_positive_epsilon() -> None:
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


def test_probe_rejects_short_digest() -> None:
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


def test_probe_rejects_invent_green_experimental_geometry() -> None:
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


def test_probe_rejects_invent_green_live_qpu() -> None:
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


def test_probe_rejects_blank_demo_label() -> None:
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


def test_probe_to_dict() -> None:
    probe = MaterialisedPgboTensorProbe(
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
    assert probe.to_dict()["n_parameters"] == 1


def test_ambient_payload_rejects_zero_n_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ambient_payload_rejects_non_square_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ambient_payload_rejects_non_finite_metric_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ambient_payload_rejects_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(*_a: object, **_k: object) -> dict[str, object]:
        return {"n_parameters": 1}

    monkeypatch.setattr(qgt_product, "_run_ambient_pgbo_json", _missing)
    with pytest.raises(ValueError, match="missing fields"):
        materialise_pgbo_tensor_probe()


def test_ambient_payload_rejects_labels_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ambient_payload_rejects_nan_metric_determinant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ambient_payload_rejects_negative_total_curvature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ambient_payload_rejects_negative_frobenius(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_demo_coupling_system_rejects_below_two() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        qgt_product._demo_coupling_system(1)


def test_demo_coupling_system_rejects_above_cap() -> None:
    with pytest.raises(ValueError, match="MAX_OSCILLATORS|<="):
        qgt_product._demo_coupling_system(MAX_OSCILLATORS + 1)


def test_demo_coupling_system_n2_shape() -> None:
    coupling, omega = qgt_product._demo_coupling_system(2)
    assert coupling.shape == (2, 2)
    assert omega.shape == (2,)


def test_demo_coupling_system_n3_shape() -> None:
    coupling, omega = qgt_product._demo_coupling_system(3)
    assert coupling.shape == (3, 3)
    assert omega.shape == (3,)


def test_ambient_subprocess_called_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: object, **_k: object) -> object:
        raise subprocess.CalledProcessError(1, "x", stderr="pgbo boom")

    monkeypatch.setattr(
        "scpn_quantum_control.pgbo_qgt_product.subprocess.run",
        boom,
    )
    with pytest.raises(ValueError, match="ambient PGBO subprocess failed"):
        qgt_product._run_ambient_pgbo_json(n_oscillators=2, epsilon=0.005)


def test_ambient_subprocess_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_a: object, **_k: object) -> object:
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(
        "scpn_quantum_control.pgbo_qgt_product.subprocess.run",
        timeout,
    )
    with pytest.raises(ValueError, match="timed out"):
        qgt_product._run_ambient_pgbo_json(n_oscillators=2, epsilon=0.005)


def test_ambient_subprocess_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Out:
        stdout = "not-json\n"

    monkeypatch.setattr(
        "scpn_quantum_control.pgbo_qgt_product.subprocess.run",
        lambda *_a, **_k: _Out(),
    )
    with pytest.raises(ValueError, match="non-JSON"):
        qgt_product._run_ambient_pgbo_json(n_oscillators=2, epsilon=0.005)


def test_ambient_subprocess_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Out:
        stdout = "[1, 2, 3]\n"

    monkeypatch.setattr(
        "scpn_quantum_control.pgbo_qgt_product.subprocess.run",
        lambda *_a, **_k: _Out(),
    )
    with pytest.raises(ValueError, match="must be an object"):
        qgt_product._run_ambient_pgbo_json(n_oscillators=2, epsilon=0.005)


def test_capability_map_rejects_empty_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qgt_product, "_CAPABILITIES", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        qgt_product._capability_map()


def test_capability_map_rejects_blank_id(monkeypatch: pytest.MonkeyPatch) -> None:
    blank = QgtCapabilityRow(
        capability_id="tmp",
        kind="pgbo_tensor",
        title="t",
        summary="s",
        ambient_module="m",
        ambient_symbol="x",
    )
    object.__setattr__(blank, "capability_id", "  ")
    monkeypatch.setattr(qgt_product, "_CAPABILITIES", (blank,))
    with pytest.raises(RuntimeError, match="blank capability_id"):
        qgt_product._capability_map()


def test_capability_map_rejects_duplicate_id(monkeypatch: pytest.MonkeyPatch) -> None:
    good = QgtCapabilityRow(
        capability_id="dup",
        kind="pgbo_tensor",
        title="t",
        summary="s",
        ambient_module="m",
        ambient_symbol="x",
    )
    monkeypatch.setattr(qgt_product, "_CAPABILITIES", (good, good))
    with pytest.raises(RuntimeError, match="duplicate capability_id"):
        qgt_product._capability_map()


def test_module_exports_stable() -> None:
    assert "assert_pgbo_qgt_product_integrity" in qgt_product.__all__
    assert "materialise_demo_pgbo_tensor_probe" in qgt_product.__all__
    assert PGBO_QGT_PRODUCT_SCHEMA == "pgbo_qgt_product.v2"
    assert MAX_OSCILLATORS == 6
