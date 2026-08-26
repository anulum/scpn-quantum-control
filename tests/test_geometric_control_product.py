# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for geometric control product
"""Real-surface tests for ``geometric_control_product`` (one concern per test)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.geometric_control_product as geo_product
from scpn_quantum_control.geometric_control_product import (
    GEOMETRIC_CONTROL_CLAIM_BOUNDARY,
    GEOMETRIC_CONTROL_PRODUCT_SCHEMA,
    GeometryBoundaryRow,
    GeometryCapabilityRow,
    MaterialisedMetricDiagnosticsProbe,
    MaterialisedQngDirectionProbe,
    PathEligibilityDecision,
    assert_geometric_control_product_integrity,
    build_geometric_control_product_registry,
    decide_geometry_path,
    get_geometry_boundary,
    get_geometry_capability,
    get_geometry_glossary_entry,
    iter_geometry_boundaries,
    iter_geometry_capabilities,
    list_geometry_ambient_inventory,
    list_geometry_boundary_ids,
    list_geometry_capability_ids,
    list_geometry_glossary_keys,
    map_geometric_control_public_surfaces,
    materialise_demo_metric_diagnostics_probe,
    materialise_metric_diagnostics_probe,
    materialise_qng_direction_probe,
)
from scpn_quantum_control.phase.variational_metric import mclachlan_metric


def _capability_kwargs(**overrides: Any) -> dict[str, Any]:
    """Valid GeometryCapabilityRow constructor kwargs."""
    base: dict[str, Any] = {
        "capability_id": "x",
        "kind": "mclachlan_metric",
        "title": "t",
        "summary": "s",
        "ambient_module": "m",
        "ambient_symbol": "x",
    }
    base.update(overrides)
    return base


def _boundary_kwargs(**overrides: Any) -> dict[str, Any]:
    """Valid GeometryBoundaryRow constructor kwargs."""
    base: dict[str, Any] = {
        "boundary_id": "x",
        "kind": "experimental_advantage_criticality",
        "title": "t",
        "failure_class": "f",
        "summary": "s",
    }
    base.update(overrides)
    return base


def _metric_probe_kwargs(**overrides: Any) -> dict[str, Any]:
    """Valid MaterialisedMetricDiagnosticsProbe constructor kwargs."""
    base: dict[str, Any] = {
        "capability_id": "c",
        "n_parameters": 3,
        "metric_rank": 3,
        "metric_nullity": 0,
        "condition_number": 2.0,
        "minimum_eigenvalue": 0.1,
        "maximum_eigenvalue": 1.0,
        "eigenvalues": (0.1, 0.5, 1.0),
        "probe_digest": "a" * 64,
        "invent_green_advantage": False,
        "invent_green_live_qpu": False,
        "demo_label": "d",
    }
    base.update(overrides)
    return base


def _qng_probe_kwargs(**overrides: Any) -> dict[str, Any]:
    """Valid MaterialisedQngDirectionProbe constructor kwargs."""
    base: dict[str, Any] = {
        "capability_id": "qng_regularised",
        "metric_rank": 3,
        "metric_nullity": 0,
        "condition_number": 2.0,
        "natural_gradient_norm": 0.1,
        "euclidean_gradient_norm": 0.2,
        "regularization_reason": "damped",
        "direction": (0.1, 0.2, 0.3),
        "probe_digest": "a" * 64,
        "invent_green_advantage": False,
        "invent_green_live_qpu": False,
        "demo_label": "d",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Catalogue / public surface
# ---------------------------------------------------------------------------


def test_list_and_filters() -> None:
    """Capability, boundary, glossary, and ambient inventory lists are non-empty."""
    ids = list_geometry_capability_ids()
    assert "mclachlan_metric" in ids
    assert "qng_regularised" in ids
    assert "qfi_spectral" in ids
    assert len(ids) == 5
    bounds = list_geometry_boundary_ids()
    assert "experimental_advantage_criticality" in bounds
    assert len(bounds) == 4
    assert list_geometry_glossary_keys()
    assert "QFI" in list_geometry_glossary_keys()
    inv = list_geometry_ambient_inventory()
    assert any("variational_metric" in str(row["module_path"]) for row in inv)


def test_iter_capabilities_without_kind_filter() -> None:
    """Unfiltered capability iteration returns the full catalogue."""
    rows = iter_geometry_capabilities()
    assert len(rows) == len(list_geometry_capability_ids())


def test_iter_capabilities_with_kind_filter() -> None:
    """Kind filter returns only matching capability rows."""
    qng = iter_geometry_capabilities(kind="qng_regularised")
    assert len(qng) == 1
    assert qng[0].capability_id == "qng_regularised"


def test_iter_boundaries_without_kind_filter() -> None:
    """Unfiltered boundary iteration returns the full catalogue."""
    rows = iter_geometry_boundaries()
    assert len(rows) == len(list_geometry_boundary_ids())


def test_iter_boundaries_with_kind_filter() -> None:
    """Kind filter returns only matching boundary rows."""
    rows = iter_geometry_boundaries(kind="live_qpu_geometry")
    assert len(rows) == 1
    assert rows[0].boundary_id == "live_qpu_geometry"


def test_get_known_capability() -> None:
    """Known capability resolves with product claim boundary and no HW submit."""
    row = get_geometry_capability("mclachlan_metric")
    assert row.claim_boundary == GEOMETRIC_CONTROL_CLAIM_BOUNDARY
    assert row.hardware_submit_allowed is False


def test_get_known_boundary() -> None:
    """Known boundary resolves with fail-closed True."""
    boundary = get_geometry_boundary("experimental_advantage_criticality")
    assert boundary.fail_closed is True


def test_get_glossary_entry_known() -> None:
    """Known glossary keys return non-empty definitions."""
    assert "Fisher" in get_geometry_glossary_entry("QFI")


def test_get_capability_rejects_blank_id() -> None:
    """Blank capability_id is refused."""
    with pytest.raises(ValueError, match="non-empty"):
        get_geometry_capability("  ")


def test_get_capability_rejects_unknown_id() -> None:
    """Unknown capability_id is refused."""
    with pytest.raises(ValueError, match="unknown capability_id"):
        get_geometry_capability("ghost")


def test_get_boundary_rejects_blank_id() -> None:
    """Blank boundary_id is refused."""
    with pytest.raises(ValueError, match="non-empty"):
        get_geometry_boundary("  ")


def test_get_boundary_rejects_unknown_id() -> None:
    """Unknown boundary_id is refused."""
    with pytest.raises(ValueError, match="unknown boundary_id"):
        get_geometry_boundary("ghost")


def test_get_glossary_rejects_blank_key() -> None:
    """Blank glossary key is refused."""
    with pytest.raises(ValueError, match="non-empty"):
        get_geometry_glossary_entry("")


def test_get_glossary_rejects_unknown_key() -> None:
    """Unknown glossary key is refused."""
    with pytest.raises(ValueError, match="unknown glossary"):
        get_geometry_glossary_entry("not_a_key")


# ---------------------------------------------------------------------------
# Path decisions / invent-green refuse
# ---------------------------------------------------------------------------


def test_decide_geometry_path_allows_honest_capability() -> None:
    """Honest local path is allowed."""
    ok = decide_geometry_path("mclachlan_metric")
    assert ok.allowed is True


def test_decide_geometry_path_refuses_invent_green_advantage() -> None:
    """Invent-green experimental advantage is refused."""
    adv = decide_geometry_path("criticality_diagnostics", invent_green_advantage=True)
    assert adv.allowed is False
    assert any("advantage" in item.lower() for item in adv.blockers)


def test_decide_geometry_path_refuses_invent_green_live_qpu() -> None:
    """Invent-green live QPU geometry is refused."""
    qpu = decide_geometry_path("qng_regularised", invent_green_live_qpu=True)
    assert qpu.allowed is False


def test_decide_geometry_path_refuses_indefinite_silent_repair() -> None:
    """Invent-green silent repair of indefinite metrics is refused."""
    silent = decide_geometry_path(
        "qng_regularised",
        invent_green_indefinite_silent_repair=True,
    )
    assert silent.allowed is False


# ---------------------------------------------------------------------------
# Ambient probes
# ---------------------------------------------------------------------------


def test_metric_diagnostics_probe_real_ambient() -> None:
    """Demo metric diagnostics probe uses ambient McLachlan metric honestly."""
    probe = materialise_demo_metric_diagnostics_probe()
    assert probe.capability_id == "criticality_diagnostics"
    assert probe.n_parameters == 3
    assert probe.metric_rank + probe.metric_nullity == probe.n_parameters
    assert len(probe.probe_digest) == 64
    assert probe.invent_green_advantage is False
    assert probe.invent_green_live_qpu is False
    # Smoke ambient import wiring used by product
    assert callable(mclachlan_metric)


def test_metric_probe_refuses_invent_green_advantage() -> None:
    """Metric probe refuses invent-green advantage."""
    with pytest.raises(ValueError, match="refused"):
        materialise_metric_diagnostics_probe(invent_green_advantage=True)


def test_metric_probe_refuses_invent_green_live_qpu() -> None:
    """Metric probe refuses invent-green live QPU."""
    with pytest.raises(ValueError, match="refused"):
        materialise_metric_diagnostics_probe(invent_green_live_qpu=True)


def test_metric_probe_rejects_non_metric_capability_kind() -> None:
    """Metric diagnostics require a metric-family capability kind."""
    with pytest.raises(ValueError, match="metric-family"):
        materialise_metric_diagnostics_probe("qng_regularised")


def test_qng_direction_probe() -> None:
    """QNG direction probe returns a finite direction vector."""
    probe = materialise_qng_direction_probe()
    assert probe.capability_id == "qng_regularised"
    assert len(probe.direction) >= 1
    assert all(np.isfinite(item) for item in probe.direction)
    assert probe.invent_green_advantage is False


def test_qng_probe_refuses_invent_green_advantage() -> None:
    """QNG probe refuses invent-green advantage."""
    with pytest.raises(ValueError, match="refused"):
        materialise_qng_direction_probe(invent_green_advantage=True)


def test_qng_probe_refuses_invent_green_live_qpu() -> None:
    """QNG probe refuses invent-green live QPU."""
    with pytest.raises(ValueError, match="refused"):
        materialise_qng_direction_probe(invent_green_live_qpu=True)


def test_qng_probe_rejects_non_qng_capability_kind() -> None:
    """QNG direction probe requires qng_regularised capability."""
    with pytest.raises(ValueError, match="qng_regularised"):
        materialise_qng_direction_probe("mclachlan_metric")


# ---------------------------------------------------------------------------
# Registry / integrity
# ---------------------------------------------------------------------------


def test_registry_and_integrity() -> None:
    """Built registry passes integrity and exposes expected counts."""
    surfaces = map_geometric_control_public_surfaces()
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.geometric_control_product" in paths
    assert "scpn_quantum_control.phase.variational_metric" in paths

    registry = build_geometric_control_product_registry()
    assert registry["schema"] == GEOMETRIC_CONTROL_PRODUCT_SCHEMA
    assert registry["experimental_advantage_criticality_policy"] is False
    assert registry["indefinite_metric_silent_repair_policy"] is False
    validated = assert_geometric_control_product_integrity(registry)
    assert validated["capability_count"] == 5
    assert validated["boundary_count"] == 4
    assert "QFI" in cast(dict[str, object], validated["glossary"])
    assert assert_geometric_control_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_capability_set_drift() -> None:
    """Extra capability rows produce set-drift refuse."""
    registry = build_geometric_control_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    broken = dict(registry)
    broken["capabilities"] = caps + [
        {
            "capability_id": "ghost",
            "kind": "mclachlan_metric",
            "title": "t",
            "summary": "s",
            "ambient_module": "m",
            "ambient_symbol": "x",
            "hardware_submit_allowed": False,
            "support_posture": "local_research",
            "as_of": "2026-07-24",
            "claim_boundary": GEOMETRIC_CONTROL_CLAIM_BOUNDARY,
        }
    ]
    broken["capability_count"] = len(cast(list[object], broken["capabilities"]))
    with pytest.raises(ValueError, match="drift"):
        assert_geometric_control_product_integrity(broken)


def test_integrity_rejects_boundary_set_drift() -> None:
    """Extra boundary rows produce boundary set-drift refuse."""
    registry = build_geometric_control_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    broken = dict(registry)
    ghost = dict(bounds[0])
    ghost["boundary_id"] = "ghost_boundary"
    broken["boundaries"] = bounds + [ghost]
    broken["boundary_count"] = len(cast(list[object], broken["boundaries"]))
    with pytest.raises(ValueError, match="boundary set drift"):
        assert_geometric_control_product_integrity(broken)


def test_integrity_rejects_empty_capabilities() -> None:
    """Empty capabilities list is refused."""
    registry = build_geometric_control_product_registry()
    empty: dict[str, object] = {
        "capabilities": [],
        "boundaries": registry["boundaries"],
        "blank_entry_count": 0,
        "capability_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty capabilities"):
        assert_geometric_control_product_integrity(empty)


def test_integrity_rejects_empty_boundaries() -> None:
    """Empty boundaries list is refused."""
    registry = build_geometric_control_product_registry()
    no_b = dict(registry)
    no_b["boundaries"] = []
    no_b["boundary_count"] = 0
    with pytest.raises(ValueError, match="non-empty boundaries"):
        assert_geometric_control_product_integrity(no_b)


def test_integrity_rejects_policy_flags_true() -> None:
    """Honesty policy flags must remain False."""
    registry = build_geometric_control_product_registry()
    for policy in (
        "hardware_submit_allowed_policy",
        "experimental_advantage_criticality_policy",
        "indefinite_metric_silent_repair_policy",
    ):
        bad = dict(registry)
        bad[policy] = True
        with pytest.raises(ValueError, match=policy):
            assert_geometric_control_product_integrity(bad)


def test_integrity_rejects_hardware_submit_on_capability() -> None:
    """Per-row hardware_submit_allowed True is refused."""
    registry = build_geometric_control_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    hw = dict(registry)
    mut = [dict(row) for row in caps]
    mut[0]["hardware_submit_allowed"] = True
    hw["capabilities"] = mut
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        assert_geometric_control_product_integrity(hw)


def test_integrity_rejects_nonzero_blank_entry_count() -> None:
    """blank_entry_count must be zero."""
    registry = build_geometric_control_product_registry()
    blank = dict(registry)
    blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_geometric_control_product_integrity(blank)


def test_integrity_rejects_empty_glossary() -> None:
    """Empty glossary mapping is refused."""
    registry = build_geometric_control_product_registry()
    no_gloss = dict(registry)
    no_gloss["glossary"] = {}
    with pytest.raises(ValueError, match="glossary"):
        assert_geometric_control_product_integrity(no_gloss)


def test_integrity_rejects_missing_glossary_key() -> None:
    """Missing required glossary keys are refused."""
    registry = build_geometric_control_product_registry()
    miss_key = dict(registry)
    miss_key["glossary"] = {"only": "one"}
    with pytest.raises(ValueError, match="glossary missing"):
        assert_geometric_control_product_integrity(miss_key)


def test_integrity_rejects_empty_ambient_inventory() -> None:
    """Empty ambient inventory is refused."""
    registry = build_geometric_control_product_registry()
    no_inv = dict(registry)
    no_inv["ambient_inventory"] = []
    with pytest.raises(ValueError, match="ambient_inventory"):
        assert_geometric_control_product_integrity(no_inv)


def test_integrity_rejects_fail_closed_false_boundary() -> None:
    """Boundary fail_closed must stay True."""
    registry = build_geometric_control_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    fc = dict(registry)
    mut_b = [dict(row) for row in bounds]
    mut_b[0]["fail_closed"] = False
    fc["boundaries"] = mut_b
    with pytest.raises(ValueError, match="fail_closed"):
        assert_geometric_control_product_integrity(fc)


def test_integrity_rejects_non_mapping_capability_row() -> None:
    """Capability rows must be mappings."""
    registry = build_geometric_control_product_registry()
    not_map = dict(registry)
    not_map["capabilities"] = cast(list[dict[str, object]], ["x"])
    with pytest.raises(ValueError, match="mapping"):
        assert_geometric_control_product_integrity(not_map)


def test_integrity_rejects_blank_capability_id_row() -> None:
    """Blank capability_id rows are refused."""
    registry = build_geometric_control_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    blank_id = dict(registry)
    bc = [dict(row) for row in caps]
    bc[0]["capability_id"] = "  "
    blank_id["capabilities"] = bc
    with pytest.raises(ValueError, match="blank"):
        assert_geometric_control_product_integrity(blank_id)


def test_integrity_rejects_duplicate_capability_id() -> None:
    """Duplicate capability_id values are refused."""
    registry = build_geometric_control_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    dup = dict(registry)
    dc = [dict(row) for row in caps]
    dc[1] = dict(dc[0])
    dup["capabilities"] = dc
    with pytest.raises(ValueError, match="duplicate capability_id"):
        assert_geometric_control_product_integrity(dup)


def test_integrity_rejects_blank_ambient_symbol() -> None:
    """Blank ambient_symbol is refused."""
    registry = build_geometric_control_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    no_sym = dict(registry)
    ns = [dict(row) for row in caps]
    ns[0]["ambient_symbol"] = ""
    no_sym["capabilities"] = ns
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_geometric_control_product_integrity(no_sym)


def test_integrity_rejects_missing_mclachlan_metric() -> None:
    """Catalogue without mclachlan_metric is refused (or drifts)."""
    registry = build_geometric_control_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    no_mcl = dict(registry)
    filtered = [dict(row) for row in caps if row["capability_id"] != "mclachlan_metric"]
    no_mcl["capabilities"] = filtered
    no_mcl["capability_count"] = len(filtered)
    with pytest.raises(ValueError, match="mclachlan_metric|drift"):
        assert_geometric_control_product_integrity(no_mcl)


def test_integrity_rejects_non_mapping_boundary_row() -> None:
    """Boundary rows must be mappings."""
    registry = build_geometric_control_product_registry()
    b_not = dict(registry)
    b_not["boundaries"] = cast(list[dict[str, object]], [1])
    with pytest.raises(ValueError, match="mapping"):
        assert_geometric_control_product_integrity(b_not)


def test_integrity_rejects_blank_boundary_id_row() -> None:
    """Blank boundary_id rows are refused."""
    registry = build_geometric_control_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    blank_b = dict(registry)
    bb = [dict(row) for row in bounds]
    bb[0]["boundary_id"] = ""
    blank_b["boundaries"] = bb
    with pytest.raises(ValueError, match="boundary_id"):
        assert_geometric_control_product_integrity(blank_b)


def test_integrity_rejects_duplicate_boundary_id() -> None:
    """Duplicate boundary_id values are refused."""
    registry = build_geometric_control_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    dup_b = dict(registry)
    db = [dict(row) for row in bounds]
    db[1] = dict(db[0])
    dup_b["boundaries"] = db
    with pytest.raises(ValueError, match="duplicate boundary_id"):
        assert_geometric_control_product_integrity(dup_b)


def test_integrity_rejects_capability_count_mismatch() -> None:
    """capability_count must match list length."""
    registry = build_geometric_control_product_registry()
    count_m = dict(registry)
    count_m["capability_count"] = 99
    with pytest.raises(ValueError, match="capability_count"):
        assert_geometric_control_product_integrity(count_m)


def test_integrity_rejects_boundary_count_mismatch() -> None:
    """boundary_count must match list length."""
    registry = build_geometric_control_product_registry()
    count_b = dict(registry)
    count_b["boundary_count"] = 99
    with pytest.raises(ValueError, match="boundary_count"):
        assert_geometric_control_product_integrity(count_b)


# ---------------------------------------------------------------------------
# GeometryCapabilityRow invariants (one concern each)
# ---------------------------------------------------------------------------


def test_capability_row_rejects_blank_id() -> None:
    """Capability rows require non-empty capability_id."""
    with pytest.raises(ValueError, match="capability_id"):
        GeometryCapabilityRow(**_capability_kwargs(capability_id=""))


def test_capability_row_rejects_unknown_kind() -> None:
    """Unknown capability kind is refused."""
    with pytest.raises(ValueError, match="unknown capability kind"):
        GeometryCapabilityRow(**_capability_kwargs(kind=cast(Any, "bogus")))


def test_capability_row_rejects_blank_title() -> None:
    """Title must be non-empty."""
    with pytest.raises(ValueError, match="title"):
        GeometryCapabilityRow(**_capability_kwargs(title=""))


def test_capability_row_rejects_blank_summary() -> None:
    """Summary must be non-empty."""
    with pytest.raises(ValueError, match="summary"):
        GeometryCapabilityRow(**_capability_kwargs(summary=""))


def test_capability_row_rejects_blank_ambient_module() -> None:
    """ambient_module must be non-empty."""
    with pytest.raises(ValueError, match="ambient_module"):
        GeometryCapabilityRow(**_capability_kwargs(ambient_module=""))


def test_capability_row_rejects_blank_ambient_symbol() -> None:
    """ambient_symbol must be non-empty."""
    with pytest.raises(ValueError, match="ambient_symbol"):
        GeometryCapabilityRow(**_capability_kwargs(ambient_symbol=""))


def test_capability_row_rejects_hardware_submit_allowed() -> None:
    """hardware_submit_allowed must remain False."""
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        GeometryCapabilityRow(**_capability_kwargs(hardware_submit_allowed=True))


def test_capability_row_rejects_unknown_support_posture() -> None:
    """Unknown support_posture is refused."""
    with pytest.raises(ValueError, match="support_posture"):
        GeometryCapabilityRow(**_capability_kwargs(support_posture=cast(Any, "bogus")))


def test_capability_row_rejects_blank_as_of() -> None:
    """as_of must be non-empty."""
    with pytest.raises(ValueError, match="as_of"):
        GeometryCapabilityRow(**_capability_kwargs(as_of=""))


def test_capability_row_to_dict() -> None:
    """Valid capability rows serialise capability_id."""
    ok = GeometryCapabilityRow(**_capability_kwargs())
    assert ok.to_dict()["capability_id"] == "x"


# ---------------------------------------------------------------------------
# GeometryBoundaryRow invariants
# ---------------------------------------------------------------------------


def test_boundary_row_rejects_blank_id() -> None:
    """Boundary rows require non-empty boundary_id."""
    with pytest.raises(ValueError, match="boundary_id"):
        GeometryBoundaryRow(**_boundary_kwargs(boundary_id=""))


def test_boundary_row_rejects_unknown_kind() -> None:
    """Unknown boundary kind is refused."""
    with pytest.raises(ValueError, match="unknown boundary kind"):
        GeometryBoundaryRow(**_boundary_kwargs(kind=cast(Any, "bogus")))


def test_boundary_row_rejects_fail_closed_false() -> None:
    """fail_closed must be True."""
    with pytest.raises(ValueError, match="fail_closed"):
        GeometryBoundaryRow(**_boundary_kwargs(fail_closed=False))


def test_boundary_row_rejects_blank_title() -> None:
    """Boundary title must be non-empty."""
    with pytest.raises(ValueError, match="title"):
        GeometryBoundaryRow(**_boundary_kwargs(title=""))


def test_boundary_row_rejects_blank_failure_class() -> None:
    """failure_class must be non-empty."""
    with pytest.raises(ValueError, match="failure_class"):
        GeometryBoundaryRow(**_boundary_kwargs(failure_class=""))


def test_boundary_row_rejects_blank_summary() -> None:
    """Boundary summary must be non-empty."""
    with pytest.raises(ValueError, match="summary"):
        GeometryBoundaryRow(**_boundary_kwargs(summary=""))


def test_boundary_row_to_dict() -> None:
    """Valid boundary rows serialise fail_closed True."""
    ok_b = GeometryBoundaryRow(**_boundary_kwargs())
    assert ok_b.to_dict()["fail_closed"] is True


# ---------------------------------------------------------------------------
# PathEligibilityDecision invariants
# ---------------------------------------------------------------------------


def test_path_decision_refused_requires_blockers() -> None:
    """Refused decisions require non-empty blockers."""
    with pytest.raises(ValueError, match="blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )


def test_path_decision_rejects_unknown_outcome() -> None:
    """Unknown path outcome is refused."""
    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "maybe"),
            allowed=True,
            reason="r",
            blockers=(),
        )


def test_path_decision_rejects_blank_reason() -> None:
    """Reason must be non-empty."""
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="",
            blockers=(),
        )


def test_path_decision_allowed_flag_must_match_allowed_outcome() -> None:
    """allowed=True requires outcome=allowed."""
    with pytest.raises(ValueError, match="outcome=allowed"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
        )


def test_path_decision_allowed_flag_must_match_refused_outcome() -> None:
    """allowed=False requires outcome=refused."""
    with pytest.raises(ValueError, match="outcome=refused"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=False,
            reason="r",
            blockers=("b",),
        )


def test_path_decision_allowed_cannot_list_blockers() -> None:
    """Allowed decisions cannot list blockers."""
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("b",),
        )


def test_path_decision_rejects_blank_blocker_entries() -> None:
    """Blocker strings must be non-empty."""
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("",),
        )


def test_path_decision_to_dict() -> None:
    """Valid allowed path decisions serialise allowed True."""
    ok_d = PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason="ok",
        blockers=(),
    )
    assert ok_d.to_dict()["allowed"] is True


# ---------------------------------------------------------------------------
# MaterialisedMetricDiagnosticsProbe invariants
# ---------------------------------------------------------------------------


def test_metric_probe_rejects_blank_capability_id() -> None:
    """Metric probe requires non-empty capability_id."""
    with pytest.raises(ValueError, match="capability_id"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(capability_id=""))


def test_metric_probe_rejects_non_positive_n_parameters() -> None:
    """n_parameters must be positive."""
    with pytest.raises(ValueError, match="n_parameters"):
        MaterialisedMetricDiagnosticsProbe(
            **_metric_probe_kwargs(
                n_parameters=0,
                metric_rank=0,
                metric_nullity=0,
                eigenvalues=(),
            )
        )


def test_metric_probe_rejects_metric_rank_out_of_range() -> None:
    """metric_rank must lie in [0, n_parameters]."""
    with pytest.raises(ValueError, match="metric_rank"):
        MaterialisedMetricDiagnosticsProbe(
            **_metric_probe_kwargs(metric_rank=4, metric_nullity=-1)
        )


def test_metric_probe_rejects_metric_nullity_mismatch() -> None:
    """metric_nullity must equal n_parameters - metric_rank."""
    with pytest.raises(ValueError, match="metric_nullity"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(metric_rank=2, metric_nullity=0))


def test_metric_probe_rejects_non_positive_condition_number() -> None:
    """condition_number must be finite and positive."""
    with pytest.raises(ValueError, match="condition_number"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(condition_number=-1.0))


def test_metric_probe_rejects_nonfinite_minimum_eigenvalue() -> None:
    """minimum_eigenvalue must be finite."""
    with pytest.raises(ValueError, match="minimum_eigenvalue"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(minimum_eigenvalue=float("nan")))


def test_metric_probe_rejects_nonfinite_maximum_eigenvalue() -> None:
    """maximum_eigenvalue must be finite."""
    with pytest.raises(ValueError, match="maximum_eigenvalue"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(maximum_eigenvalue=float("inf")))


def test_metric_probe_rejects_eigenvalues_length_mismatch() -> None:
    """eigenvalues length must equal n_parameters."""
    with pytest.raises(ValueError, match="eigenvalues"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(eigenvalues=(0.1, 0.5)))


def test_metric_probe_rejects_short_probe_digest() -> None:
    """probe_digest must be 64-char hex."""
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(probe_digest="x"))


def test_metric_probe_rejects_invent_green_advantage() -> None:
    """invent_green_advantage must be False on materialised probes."""
    with pytest.raises(ValueError, match="invent_green_advantage"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(invent_green_advantage=True))


def test_metric_probe_rejects_invent_green_live_qpu() -> None:
    """invent_green_live_qpu must be False on materialised probes."""
    with pytest.raises(ValueError, match="invent_green_live_qpu"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(invent_green_live_qpu=True))


def test_metric_probe_rejects_blank_demo_label() -> None:
    """demo_label must be non-empty."""
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs(demo_label=""))


def test_metric_probe_to_dict() -> None:
    """Valid metric probes serialise metric_rank."""
    ok_m = MaterialisedMetricDiagnosticsProbe(**_metric_probe_kwargs())
    assert ok_m.to_dict()["metric_rank"] == 3


# ---------------------------------------------------------------------------
# MaterialisedQngDirectionProbe invariants
# ---------------------------------------------------------------------------


def test_qng_probe_rejects_blank_capability_id() -> None:
    """QNG probe requires non-empty capability_id."""
    with pytest.raises(ValueError, match="capability_id"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(capability_id=""))


def test_qng_probe_rejects_negative_metric_rank() -> None:
    """metric_rank must be non-negative."""
    with pytest.raises(ValueError, match="metric_rank"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(metric_rank=-1))


def test_qng_probe_rejects_negative_metric_nullity() -> None:
    """metric_nullity must be non-negative."""
    with pytest.raises(ValueError, match="metric_nullity"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(metric_nullity=-1))


def test_qng_probe_rejects_non_positive_condition_number() -> None:
    """condition_number must be finite and positive."""
    with pytest.raises(ValueError, match="condition_number"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(condition_number=-1.0))


def test_qng_probe_rejects_negative_natural_gradient_norm() -> None:
    """natural_gradient_norm must be finite and non-negative."""
    with pytest.raises(ValueError, match="natural_gradient_norm"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(natural_gradient_norm=-0.1))


def test_qng_probe_rejects_negative_euclidean_gradient_norm() -> None:
    """euclidean_gradient_norm must be finite and non-negative."""
    with pytest.raises(ValueError, match="euclidean_gradient_norm"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(euclidean_gradient_norm=-0.2))


def test_qng_probe_rejects_blank_regularization_reason() -> None:
    """regularization_reason must be non-empty."""
    with pytest.raises(ValueError, match="regularization_reason"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(regularization_reason=""))


def test_qng_probe_rejects_empty_direction() -> None:
    """direction must be non-empty."""
    with pytest.raises(ValueError, match="direction"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(direction=()))


def test_qng_probe_rejects_nonfinite_direction_entries() -> None:
    """direction entries must be finite."""
    with pytest.raises(ValueError, match="direction entries"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(direction=(0.1, float("nan"), 0.3)))


def test_qng_probe_rejects_short_probe_digest() -> None:
    """probe_digest must be 64-char hex."""
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(probe_digest="x"))


def test_qng_probe_rejects_invent_green_advantage() -> None:
    """invent_green_advantage must be False."""
    with pytest.raises(ValueError, match="invent_green_advantage"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(invent_green_advantage=True))


def test_qng_probe_rejects_invent_green_live_qpu() -> None:
    """invent_green_live_qpu must be False."""
    with pytest.raises(ValueError, match="invent_green_live_qpu"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(invent_green_live_qpu=True))


def test_qng_probe_rejects_blank_demo_label() -> None:
    """demo_label must be non-empty."""
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedQngDirectionProbe(**_qng_probe_kwargs(demo_label=""))


def test_qng_probe_to_dict() -> None:
    """Valid QNG probes serialise metric_rank."""
    ok_q = MaterialisedQngDirectionProbe(**_qng_probe_kwargs())
    assert ok_q.to_dict()["metric_rank"] == 3


# ---------------------------------------------------------------------------
# Probe input validation / catalogue map guards
# ---------------------------------------------------------------------------


def test_demo_state_derivatives_rejects_non_positive_n_parameters() -> None:
    """Demo state derivatives require positive n_parameters."""
    with pytest.raises(ValueError, match="n_parameters"):
        geo_product._demo_state_derivatives(n_parameters=0)


def test_demo_state_derivatives_rejects_small_dim() -> None:
    """Demo state derivatives require dim >= 2."""
    with pytest.raises(ValueError, match="dim"):
        geo_product._demo_state_derivatives(dim=1)


def test_metric_probe_rejects_negative_eigenvalue_floor() -> None:
    """eigenvalue_floor must be finite and non-negative."""
    with pytest.raises(ValueError, match="eigenvalue_floor"):
        materialise_metric_diagnostics_probe(eigenvalue_floor=-1.0)


def test_metric_probe_rejects_non_square_ambient_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-square ambient metric is refused."""

    def _bad_metric(*_a: object, **_k: object) -> Any:
        return np.asarray([[1.0, 0.0]], dtype=np.float64)

    monkeypatch.setattr(geo_product, "mclachlan_metric", _bad_metric)
    with pytest.raises(ValueError, match="square"):
        materialise_metric_diagnostics_probe()


def test_metric_probe_rejects_nonfinite_ambient_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-finite ambient metric is refused."""

    def _nan_metric(*_a: object, **_k: object) -> Any:
        m = np.eye(3, dtype=np.float64)
        m[0, 0] = float("nan")
        return m

    monkeypatch.setattr(geo_product, "mclachlan_metric", _nan_metric)
    with pytest.raises(ValueError, match="finite"):
        materialise_metric_diagnostics_probe()


def test_metric_probe_rejects_size_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambient metric size must match n_parameters."""

    def _wrong_size(*_a: object, **_k: object) -> Any:
        return np.eye(2, dtype=np.float64)

    monkeypatch.setattr(geo_product, "mclachlan_metric", _wrong_size)
    with pytest.raises(ValueError, match="n_parameters"):
        materialise_metric_diagnostics_probe(n_parameters=3)


def test_metric_probe_rejects_zero_metric_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero metric yields invalid condition number path."""

    def _zero_metric(*_a: object, **_k: object) -> Any:
        return np.zeros((3, 3), dtype=np.float64)

    monkeypatch.setattr(geo_product, "mclachlan_metric", _zero_metric)
    with pytest.raises(ValueError, match="condition"):
        materialise_metric_diagnostics_probe()


def test_capability_map_rejects_blank_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalogue map refuses blank capability_id after strip."""
    good = get_geometry_capability("mclachlan_metric")
    blank = GeometryCapabilityRow(**_capability_kwargs(capability_id="tmp"))
    object.__setattr__(blank, "capability_id", "  ")
    monkeypatch.setattr(geo_product, "_CAPABILITIES", (blank,))
    with pytest.raises(RuntimeError, match="blank capability_id"):
        geo_product._capability_map()
    # restore is automatic; good kept for clarity in next tests if needed
    assert good.capability_id == "mclachlan_metric"


def test_capability_map_rejects_duplicate_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalogue map refuses duplicate capability_id."""
    row = get_geometry_capability("mclachlan_metric")
    monkeypatch.setattr(geo_product, "_CAPABILITIES", (row, row))
    with pytest.raises(RuntimeError, match="duplicate capability_id"):
        geo_product._capability_map()


def test_capability_map_rejects_empty_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalogue map refuses empty capability tuple."""
    monkeypatch.setattr(geo_product, "_CAPABILITIES", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        geo_product._capability_map()


def test_capability_map_accepts_valid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalogue map accepts a single valid row."""
    good = get_geometry_capability("mclachlan_metric")
    monkeypatch.setattr(geo_product, "_CAPABILITIES", (good,))
    mapping = geo_product._capability_map()
    assert mapping[good.capability_id].capability_id == good.capability_id


def test_module_exports_stable() -> None:
    """Public __all__ includes integrity and demo probe entry points."""
    assert "assert_geometric_control_product_integrity" in geo_product.__all__
    assert "materialise_demo_metric_diagnostics_probe" in geo_product.__all__
    assert GEOMETRIC_CONTROL_PRODUCT_SCHEMA == "geometric_control_product.v1"
