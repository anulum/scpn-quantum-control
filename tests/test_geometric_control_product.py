# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for geometric control product (BL-50)
"""Real-surface tests for ``geometric_control_product``."""

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


def test_list_and_filters() -> None:
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
    qng = iter_geometry_capabilities(kind="qng_regularised")
    assert len(qng) == 1
    assert iter_geometry_boundaries(kind="live_qpu_geometry")
    inv = list_geometry_ambient_inventory()
    assert any("variational_metric" in str(row["module_path"]) for row in inv)


def test_get_known_and_unknown() -> None:
    row = get_geometry_capability("mclachlan_metric")
    assert row.claim_boundary == GEOMETRIC_CONTROL_CLAIM_BOUNDARY
    assert row.hardware_submit_allowed is False
    assert "Fisher" in get_geometry_glossary_entry("QFI")
    b = get_geometry_boundary("experimental_advantage_criticality")
    assert b.fail_closed is True
    with pytest.raises(ValueError, match="non-empty"):
        get_geometry_capability("  ")
    with pytest.raises(ValueError, match="unknown capability_id"):
        get_geometry_capability("ghost")
    with pytest.raises(ValueError, match="unknown boundary_id"):
        get_geometry_boundary("ghost")
    with pytest.raises(ValueError, match="unknown glossary"):
        get_geometry_glossary_entry("not_a_key")
    with pytest.raises(ValueError, match="non-empty"):
        get_geometry_glossary_entry("")


def test_decide_geometry_path() -> None:
    ok = decide_geometry_path("mclachlan_metric")
    assert ok.allowed is True
    adv = decide_geometry_path("criticality_diagnostics", invent_green_advantage=True)
    assert adv.allowed is False
    assert any("advantage" in x.lower() for x in adv.blockers)
    qpu = decide_geometry_path("qng_regularised", invent_green_live_qpu=True)
    assert qpu.allowed is False
    silent = decide_geometry_path("qng_regularised", invent_green_indefinite_silent_repair=True)
    assert silent.allowed is False


def test_metric_diagnostics_probe_real_ambient() -> None:
    probe = materialise_demo_metric_diagnostics_probe()
    assert probe.capability_id == "criticality_diagnostics"
    assert probe.n_parameters == 3
    assert probe.metric_rank + probe.metric_nullity == probe.n_parameters
    assert len(probe.probe_digest) == 64
    assert probe.invent_green_advantage is False
    # Cross-check ambient McLachlan path (same synthetic derivatives construction).
    rng = np.random.default_rng(50)
    derivs = (rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))).astype(np.complex128)
    metric = mclachlan_metric(derivs)
    evals = np.linalg.eigvalsh(0.5 * (metric + metric.T))
    assert probe.minimum_eigenvalue == pytest.approx(float(evals[0]))
    assert probe.maximum_eigenvalue == pytest.approx(float(evals[-1]))

    again = materialise_metric_diagnostics_probe("mclachlan_metric")
    assert (
        again.probe_digest == materialise_metric_diagnostics_probe("mclachlan_metric").probe_digest
    )


def test_metric_probe_refuses() -> None:
    with pytest.raises(ValueError, match="advantage"):
        materialise_metric_diagnostics_probe(invent_green_advantage=True)
    with pytest.raises(ValueError, match="QPU|qpu|live"):
        materialise_metric_diagnostics_probe(invent_green_live_qpu=True)
    with pytest.raises(ValueError, match="metric-family"):
        materialise_metric_diagnostics_probe("qng_regularised")


def test_qng_direction_probe() -> None:
    probe = materialise_qng_direction_probe()
    assert probe.capability_id == "qng_regularised"
    assert len(probe.direction) == 3
    assert probe.natural_gradient_norm >= 0.0
    assert probe.regularization_reason
    assert len(probe.probe_digest) == 64
    again = materialise_qng_direction_probe()
    assert again.probe_digest == probe.probe_digest
    with pytest.raises(ValueError, match="qng_regularised"):
        materialise_qng_direction_probe("mclachlan_metric")
    with pytest.raises(ValueError, match="advantage"):
        materialise_qng_direction_probe(invent_green_advantage=True)


def test_registry_and_integrity() -> None:
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


def test_integrity_rejects_drift() -> None:
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

    empty: dict[str, object] = {
        "capabilities": [],
        "boundaries": registry["boundaries"],
        "blank_entry_count": 0,
        "capability_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty capabilities"):
        assert_geometric_control_product_integrity(empty)

    no_b = dict(registry)
    no_b["boundaries"] = []
    no_b["boundary_count"] = 0
    with pytest.raises(ValueError, match="non-empty boundaries"):
        assert_geometric_control_product_integrity(no_b)

    for policy in (
        "hardware_submit_allowed_policy",
        "experimental_advantage_criticality_policy",
        "indefinite_metric_silent_repair_policy",
    ):
        bad = dict(registry)
        bad[policy] = True
        with pytest.raises(ValueError, match=policy):
            assert_geometric_control_product_integrity(bad)

    hw = dict(registry)
    mut = [dict(row) for row in caps]
    mut[0]["hardware_submit_allowed"] = True
    hw["capabilities"] = mut
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        assert_geometric_control_product_integrity(hw)

    blank = dict(registry)
    blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_geometric_control_product_integrity(blank)

    no_gloss = dict(registry)
    no_gloss["glossary"] = {}
    with pytest.raises(ValueError, match="glossary"):
        assert_geometric_control_product_integrity(no_gloss)

    miss_key = dict(registry)
    miss_key["glossary"] = {"only": "one"}
    with pytest.raises(ValueError, match="glossary missing"):
        assert_geometric_control_product_integrity(miss_key)

    no_inv = dict(registry)
    no_inv["ambient_inventory"] = []
    with pytest.raises(ValueError, match="ambient_inventory"):
        assert_geometric_control_product_integrity(no_inv)

    bounds = cast(list[dict[str, object]], registry["boundaries"])
    fc = dict(registry)
    mut_b = [dict(row) for row in bounds]
    mut_b[0]["fail_closed"] = False
    fc["boundaries"] = mut_b
    with pytest.raises(ValueError, match="fail_closed"):
        assert_geometric_control_product_integrity(fc)


def test_integrity_more_edges() -> None:
    registry = build_geometric_control_product_registry()
    caps = cast(list[dict[str, object]], registry["capabilities"])
    not_map = dict(registry)
    not_map["capabilities"] = cast(list[dict[str, object]], ["x"])
    with pytest.raises(ValueError, match="mapping"):
        assert_geometric_control_product_integrity(not_map)

    blank_id = dict(registry)
    bc = [dict(row) for row in caps]
    bc[0]["capability_id"] = "  "
    blank_id["capabilities"] = bc
    with pytest.raises(ValueError, match="blank"):
        assert_geometric_control_product_integrity(blank_id)

    dup = dict(registry)
    dc = [dict(row) for row in caps]
    dc[1] = dict(dc[0])
    dup["capabilities"] = dc
    with pytest.raises(ValueError, match="duplicate capability_id"):
        assert_geometric_control_product_integrity(dup)

    no_sym = dict(registry)
    ns = [dict(row) for row in caps]
    ns[0]["ambient_symbol"] = ""
    no_sym["capabilities"] = ns
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_geometric_control_product_integrity(no_sym)

    no_mcl = dict(registry)
    filtered = [dict(row) for row in caps if row["capability_id"] != "mclachlan_metric"]
    no_mcl["capabilities"] = filtered
    no_mcl["capability_count"] = len(filtered)
    with pytest.raises(ValueError, match="mclachlan_metric|drift"):
        assert_geometric_control_product_integrity(no_mcl)

    bounds = cast(list[dict[str, object]], registry["boundaries"])
    b_not = dict(registry)
    b_not["boundaries"] = cast(list[dict[str, object]], [1])
    with pytest.raises(ValueError, match="mapping"):
        assert_geometric_control_product_integrity(b_not)

    blank_b = dict(registry)
    bb = [dict(row) for row in bounds]
    bb[0]["boundary_id"] = ""
    blank_b["boundaries"] = bb
    with pytest.raises(ValueError, match="boundary_id"):
        assert_geometric_control_product_integrity(blank_b)

    dup_b = dict(registry)
    db = [dict(row) for row in bounds]
    db[1] = dict(db[0])
    dup_b["boundaries"] = db
    with pytest.raises(ValueError, match="duplicate boundary_id"):
        assert_geometric_control_product_integrity(dup_b)

    count_m = dict(registry)
    count_m["capability_count"] = 99
    with pytest.raises(ValueError, match="capability_count"):
        assert_geometric_control_product_integrity(count_m)

    count_b = dict(registry)
    count_b["boundary_count"] = 99
    with pytest.raises(ValueError, match="boundary_count"):
        assert_geometric_control_product_integrity(count_b)


def test_dataclass_invariants() -> None:
    with pytest.raises(ValueError, match="capability_id"):
        GeometryCapabilityRow(
            capability_id="",
            kind="mclachlan_metric",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="unknown capability kind"):
        GeometryCapabilityRow(
            capability_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="title"):
        GeometryCapabilityRow(
            capability_id="x",
            kind="mclachlan_metric",
            title="",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="summary"):
        GeometryCapabilityRow(
            capability_id="x",
            kind="mclachlan_metric",
            title="t",
            summary="",
            ambient_module="m",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="ambient_module"):
        GeometryCapabilityRow(
            capability_id="x",
            kind="mclachlan_metric",
            title="t",
            summary="s",
            ambient_module="",
            ambient_symbol="x",
        )
    with pytest.raises(ValueError, match="ambient_symbol"):
        GeometryCapabilityRow(
            capability_id="x",
            kind="mclachlan_metric",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="",
        )
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        GeometryCapabilityRow(
            capability_id="x",
            kind="mclachlan_metric",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
            hardware_submit_allowed=True,
        )
    with pytest.raises(ValueError, match="support_posture"):
        GeometryCapabilityRow(
            capability_id="x",
            kind="mclachlan_metric",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
            support_posture=cast(Any, "bogus"),
        )
    with pytest.raises(ValueError, match="as_of"):
        GeometryCapabilityRow(
            capability_id="x",
            kind="mclachlan_metric",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="x",
            as_of="",
        )
    ok = GeometryCapabilityRow(
        capability_id="x",
        kind="mclachlan_metric",
        title="t",
        summary="s",
        ambient_module="m",
        ambient_symbol="x",
    )
    assert ok.to_dict()["capability_id"] == "x"

    with pytest.raises(ValueError, match="boundary_id"):
        GeometryBoundaryRow(
            boundary_id="",
            kind="experimental_advantage_criticality",
            title="t",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="unknown boundary kind"):
        GeometryBoundaryRow(
            boundary_id="x",
            kind=cast(Any, "bogus"),
            title="t",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="fail_closed"):
        GeometryBoundaryRow(
            boundary_id="x",
            kind="experimental_advantage_criticality",
            title="t",
            failure_class="f",
            summary="s",
            fail_closed=False,
        )
    with pytest.raises(ValueError, match="title"):
        GeometryBoundaryRow(
            boundary_id="x",
            kind="experimental_advantage_criticality",
            title="",
            failure_class="f",
            summary="s",
        )
    with pytest.raises(ValueError, match="failure_class"):
        GeometryBoundaryRow(
            boundary_id="x",
            kind="experimental_advantage_criticality",
            title="t",
            failure_class="",
            summary="s",
        )
    with pytest.raises(ValueError, match="summary"):
        GeometryBoundaryRow(
            boundary_id="x",
            kind="experimental_advantage_criticality",
            title="t",
            failure_class="f",
            summary="",
        )
    ok_b = GeometryBoundaryRow(
        boundary_id="x",
        kind="experimental_advantage_criticality",
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
        MaterialisedMetricDiagnosticsProbe(
            capability_id="",
            n_parameters=3,
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            minimum_eigenvalue=0.1,
            maximum_eigenvalue=1.0,
            eigenvalues=(0.1, 0.5, 1.0),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="n_parameters"):
        MaterialisedMetricDiagnosticsProbe(
            capability_id="c",
            n_parameters=0,
            metric_rank=0,
            metric_nullity=0,
            condition_number=2.0,
            minimum_eigenvalue=0.1,
            maximum_eigenvalue=1.0,
            eigenvalues=(),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="metric_nullity"):
        MaterialisedMetricDiagnosticsProbe(
            capability_id="c",
            n_parameters=3,
            metric_rank=2,
            metric_nullity=0,
            condition_number=2.0,
            minimum_eigenvalue=0.1,
            maximum_eigenvalue=1.0,
            eigenvalues=(0.1, 0.5, 1.0),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="condition_number"):
        MaterialisedMetricDiagnosticsProbe(
            capability_id="c",
            n_parameters=3,
            metric_rank=3,
            metric_nullity=0,
            condition_number=-1.0,
            minimum_eigenvalue=0.1,
            maximum_eigenvalue=1.0,
            eigenvalues=(0.1, 0.5, 1.0),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="eigenvalues"):
        MaterialisedMetricDiagnosticsProbe(
            capability_id="c",
            n_parameters=3,
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            minimum_eigenvalue=0.1,
            maximum_eigenvalue=1.0,
            eigenvalues=(0.1, 0.5),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedMetricDiagnosticsProbe(
            capability_id="c",
            n_parameters=3,
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            minimum_eigenvalue=0.1,
            maximum_eigenvalue=1.0,
            eigenvalues=(0.1, 0.5, 1.0),
            probe_digest="x",
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_advantage"):
        MaterialisedMetricDiagnosticsProbe(
            capability_id="c",
            n_parameters=3,
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            minimum_eigenvalue=0.1,
            maximum_eigenvalue=1.0,
            eigenvalues=(0.1, 0.5, 1.0),
            probe_digest="a" * 64,
            invent_green_advantage=True,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_live_qpu"):
        MaterialisedMetricDiagnosticsProbe(
            capability_id="c",
            n_parameters=3,
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            minimum_eigenvalue=0.1,
            maximum_eigenvalue=1.0,
            eigenvalues=(0.1, 0.5, 1.0),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedMetricDiagnosticsProbe(
            capability_id="c",
            n_parameters=3,
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            minimum_eigenvalue=0.1,
            maximum_eigenvalue=1.0,
            eigenvalues=(0.1, 0.5, 1.0),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="",
        )
    ok_m = MaterialisedMetricDiagnosticsProbe(
        capability_id="c",
        n_parameters=3,
        metric_rank=3,
        metric_nullity=0,
        condition_number=2.0,
        minimum_eigenvalue=0.1,
        maximum_eigenvalue=1.0,
        eigenvalues=(0.1, 0.5, 1.0),
        probe_digest="a" * 64,
        invent_green_advantage=False,
        invent_green_live_qpu=False,
        demo_label="d",
    )
    assert ok_m.to_dict()["metric_rank"] == 3

    with pytest.raises(ValueError, match="direction"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="damped",
            direction=(),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="regularization_reason"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="",
            direction=(0.1, 0.2, 0.3),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="capability_id"):
        MaterialisedQngDirectionProbe(
            capability_id="",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="damped",
            direction=(0.1, 0.2, 0.3),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="condition_number"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=-1.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="damped",
            direction=(0.1, 0.2, 0.3),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="natural_gradient_norm"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=-0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="damped",
            direction=(0.1, 0.2, 0.3),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="euclidean_gradient_norm"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=-0.2,
            regularization_reason="damped",
            direction=(0.1, 0.2, 0.3),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="direction entries"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="damped",
            direction=(0.1, float("nan"), 0.3),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="probe_digest"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="damped",
            direction=(0.1, 0.2, 0.3),
            probe_digest="x",
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_advantage"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="damped",
            direction=(0.1, 0.2, 0.3),
            probe_digest="a" * 64,
            invent_green_advantage=True,
            invent_green_live_qpu=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_live_qpu"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="damped",
            direction=(0.1, 0.2, 0.3),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedQngDirectionProbe(
            capability_id="qng_regularised",
            metric_rank=3,
            metric_nullity=0,
            condition_number=2.0,
            natural_gradient_norm=0.1,
            euclidean_gradient_norm=0.2,
            regularization_reason="damped",
            direction=(0.1, 0.2, 0.3),
            probe_digest="a" * 64,
            invent_green_advantage=False,
            invent_green_live_qpu=False,
            demo_label="",
        )
    ok_q = MaterialisedQngDirectionProbe(
        capability_id="qng_regularised",
        metric_rank=3,
        metric_nullity=0,
        condition_number=2.0,
        natural_gradient_norm=0.1,
        euclidean_gradient_norm=0.2,
        regularization_reason="damped",
        direction=(0.1, 0.2, 0.3),
        probe_digest="a" * 64,
        invent_green_advantage=False,
        invent_green_live_qpu=False,
        demo_label="d",
    )
    assert ok_q.to_dict()["metric_rank"] == 3


def test_probe_input_validation() -> None:
    with pytest.raises(ValueError, match="n_parameters"):
        geo_product._demo_state_derivatives(n_parameters=0)
    with pytest.raises(ValueError, match="dim"):
        geo_product._demo_state_derivatives(dim=1)
    with pytest.raises(ValueError, match="eigenvalue_floor"):
        materialise_metric_diagnostics_probe(eigenvalue_floor=-1.0)

    def _bad_metric(*_a: object, **_k: object) -> Any:
        return np.asarray([[1.0, 0.0]], dtype=np.float64)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(geo_product, "mclachlan_metric", _bad_metric)
    try:
        with pytest.raises(ValueError, match="square"):
            materialise_metric_diagnostics_probe()
    finally:
        monkey.undo()

    def _nan_metric(*_a: object, **_k: object) -> Any:
        m = np.eye(3, dtype=np.float64)
        m[0, 0] = float("nan")
        return m

    monkey = pytest.MonkeyPatch()
    monkey.setattr(geo_product, "mclachlan_metric", _nan_metric)
    try:
        with pytest.raises(ValueError, match="finite"):
            materialise_metric_diagnostics_probe()
    finally:
        monkey.undo()

    def _wrong_size(*_a: object, **_k: object) -> Any:
        return np.eye(2, dtype=np.float64)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(geo_product, "mclachlan_metric", _wrong_size)
    try:
        with pytest.raises(ValueError, match="n_parameters"):
            materialise_metric_diagnostics_probe(n_parameters=3)
    finally:
        monkey.undo()

    def _zero_metric(*_a: object, **_k: object) -> Any:
        return np.zeros((3, 3), dtype=np.float64)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(geo_product, "mclachlan_metric", _zero_metric)
    try:
        with pytest.raises(ValueError, match="condition"):
            materialise_metric_diagnostics_probe()
    finally:
        monkey.undo()


def test_module_exports_stable() -> None:
    assert "assert_geometric_control_product_integrity" in geo_product.__all__
    assert "materialise_demo_metric_diagnostics_probe" in geo_product.__all__
    assert GEOMETRIC_CONTROL_PRODUCT_SCHEMA == "geometric_control_product.v1"
