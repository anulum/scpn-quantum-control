# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for Wirtinger + implicit product (BL-64)
"""Real-surface tests for ``scpn_quantum_control.wirtinger_implicit_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.wirtinger_implicit_product as wirtinger_implicit_product
from scpn_quantum_control.wirtinger_implicit_product import (
    BL46_COMPLEX_WITHOUT_WIRTINGER_LAW,
    BL53_COMPLEX_WITHOUT_WIRTINGER,
    WIRTINGER_IMPLICIT_CLAIM_BOUNDARY,
    WIRTINGER_IMPLICIT_PRODUCT_SCHEMA,
    ComplexContractDecision,
    MaterialisedImplicitProbe,
    MaterialisedWirtingerProbe,
    WirtingerImplicitSurfaceRow,
    assert_wirtinger_implicit_product_integrity,
    build_wirtinger_implicit_product_registry,
    decide_complex_objective_contract,
    get_wirtinger_implicit_surface,
    iter_wirtinger_implicit_surfaces,
    list_wirtinger_implicit_surface_ids,
    map_wirtinger_implicit_public_surfaces,
    materialise_demo_implicit_stationary_probe,
    materialise_demo_wirtinger_probe,
)


def test_list_surfaces_and_filters() -> None:
    ids = list_wirtinger_implicit_surface_ids()
    assert "wirtinger_partials" in ids
    assert "holomorphic_gradient" in ids
    assert "real_objective_cr_gradient" in ids
    assert "implicit_stationary_sensitivity" in ids
    assert "implicit_fixed_point_sensitivity" in ids
    assert "complex_without_wirtinger_refuse" in ids
    assert ids == list_wirtinger_implicit_surface_ids()
    partials = iter_wirtinger_implicit_surfaces(kind="wirtinger_partials")
    assert len(partials) == 1
    refuse = iter_wirtinger_implicit_surfaces(support_posture="refuse_only")
    assert refuse
    assert all(row.support_posture == "refuse_only" for row in refuse)


def test_get_known_and_unknown_fail_closed() -> None:
    row = get_wirtinger_implicit_surface("wirtinger_partials")
    assert row.claim_boundary == WIRTINGER_IMPLICIT_CLAIM_BOUNDARY
    assert row.bl53_pointer == BL53_COMPLEX_WITHOUT_WIRTINGER
    assert row.bl46_pointer == BL46_COMPLEX_WITHOUT_WIRTINGER_LAW
    with pytest.raises(ValueError, match="non-empty"):
        get_wirtinger_implicit_surface("  ")
    with pytest.raises(ValueError, match="unknown surface_id"):
        get_wirtinger_implicit_surface("not_a_surface")


def test_complex_contract_refuse_and_allow() -> None:
    refused = decide_complex_objective_contract(has_wirtinger_contract=False)
    assert refused.allowed is False
    assert refused.blockers
    assert any("wirtinger" in item.lower() for item in refused.blockers)
    assert BL53_COMPLEX_WITHOUT_WIRTINGER in refused.bl53_id
    assert BL46_COMPLEX_WITHOUT_WIRTINGER_LAW in refused.bl46_id

    allowed = decide_complex_objective_contract(has_wirtinger_contract=True)
    assert allowed.allowed is True
    assert allowed.has_wirtinger_contract is True
    assert not allowed.blockers


def test_materialise_demo_wirtinger_holomorphic() -> None:
    probe = materialise_demo_wirtinger_probe(
        demo="holomorphic_square",
        z0=1.0 + 0.5j,
    )
    assert probe.demo_label == "holomorphic_square"
    assert probe.is_holomorphic is True
    assert probe.holomorphic_residual <= 1e-5
    # f(z)=z^2 => df/dz = 2z = 2+1j at z=1+0.5j
    re, im = probe.df_dz[0]
    assert abs(re - 2.0) < 1e-4
    assert abs(im - 1.0) < 1e-4
    payload = probe.to_dict()
    assert payload["is_holomorphic"] is True


def test_materialise_demo_wirtinger_nonholomorphic() -> None:
    probe = materialise_demo_wirtinger_probe(demo="modulus_squared", z0=0.8 + 0.3j)
    assert probe.demo_label == "modulus_squared"
    assert probe.holomorphic_residual > 1e-6
    assert probe.is_holomorphic is False
    with pytest.raises(ValueError, match="unknown wirtinger demo"):
        materialise_demo_wirtinger_probe(demo=cast(Any, "nope"))


def test_materialise_demo_implicit_stationary() -> None:
    probe = materialise_demo_implicit_stationary_probe(
        hessian_scale=2.0,
        cross_scale=1.0,
    )
    assert probe.method == "implicit_stationary_sensitivity"
    assert probe.shape == (1, 1)
    assert abs(probe.sensitivity[0] - (-0.5)) < 1e-9
    assert probe.condition_number >= 1.0
    payload = probe.to_dict()
    assert payload["demo_label"] == "stationary_1d_scale"
    with pytest.raises(ValueError, match="hessian_scale"):
        materialise_demo_implicit_stationary_probe(hessian_scale=0.0)


def test_public_surfaces_and_registry() -> None:
    surfaces = map_wirtinger_implicit_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.wirtinger_calculus" in paths
    assert "scpn_quantum_control.differentiable_implicit_sensitivity" in paths
    assert "scpn_quantum_control.unsuitable_scenario_registry" in paths

    registry = build_wirtinger_implicit_product_registry()
    assert registry["schema"] == WIRTINGER_IMPLICIT_PRODUCT_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_surface_id"] == "wirtinger_partials"
    assert registry["bl53_complex_without_wirtinger"] == BL53_COMPLEX_WITHOUT_WIRTINGER
    validated = assert_wirtinger_implicit_product_integrity(registry)
    assert validated["surface_count"] == len(list_wirtinger_implicit_surface_ids())
    assert assert_wirtinger_implicit_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift() -> None:
    registry = build_wirtinger_implicit_product_registry()
    surfaces = cast(list[dict[str, object]], list(registry["surfaces"]))

    broken = dict(registry)
    broken["surfaces"] = surfaces + [
        {
            "surface_id": "ghost",
            "kind": "wirtinger_partials",
            "title": "t",
            "summary": "s",
            "module_path": "m",
            "symbol_name": "x",
            "support_posture": "local_materialised",
            "bl53_pointer": BL53_COMPLEX_WITHOUT_WIRTINGER,
            "bl46_pointer": BL46_COMPLEX_WITHOUT_WIRTINGER_LAW,
            "as_of": "2026-07-24",
            "claim_boundary": WIRTINGER_IMPLICIT_CLAIM_BOUNDARY,
        }
    ]
    broken["surface_count"] = len(cast(list[object], broken["surfaces"]))
    with pytest.raises(ValueError, match="drift"):
        assert_wirtinger_implicit_product_integrity(broken)

    empty: dict[str, object] = {
        "surfaces": [],
        "blank_entry_count": 0,
        "surface_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty surfaces"):
        assert_wirtinger_implicit_product_integrity(empty)


def test_integrity_rejects_blank_invalid() -> None:
    registry = build_wirtinger_implicit_product_registry()
    surfaces = cast(list[dict[str, object]], list(registry["surfaces"]))

    non_map = dict(registry)
    non_map["surfaces"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_wirtinger_implicit_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in surfaces]
    rows[0]["surface_id"] = "  "
    blank_id["surfaces"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_wirtinger_implicit_product_integrity(blank_id)

    bad_kind = dict(registry)
    krows = [dict(row) for row in surfaces]
    krows[1]["kind"] = "nope"
    bad_kind["surfaces"] = krows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_wirtinger_implicit_product_integrity(bad_kind)

    no_symbol = dict(registry)
    srows = [dict(row) for row in surfaces]
    srows[0]["symbol_name"] = ""
    no_symbol["surfaces"] = srows
    with pytest.raises(ValueError, match="symbol_name"):
        assert_wirtinger_implicit_product_integrity(no_symbol)

    no_bl53 = dict(registry)
    brows = [dict(row) for row in surfaces]
    brows[0]["bl53_pointer"] = ""
    no_bl53["surfaces"] = brows
    with pytest.raises(ValueError, match="bl53_pointer"):
        assert_wirtinger_implicit_product_integrity(no_bl53)

    no_default = dict(registry)
    renamed = [dict(row) for row in surfaces]
    for row in renamed:
        if row.get("surface_id") == "wirtinger_partials":
            row["surface_id"] = "renamed"
    no_default["surfaces"] = renamed
    with pytest.raises(ValueError, match="missing wirtinger_partials|drift"):
        assert_wirtinger_implicit_product_integrity(no_default)

    no_refuse = dict(registry)
    without = [
        dict(row)
        for row in surfaces
        if row.get("surface_id") != "complex_without_wirtinger_refuse"
    ]
    no_refuse["surfaces"] = without
    no_refuse["surface_count"] = len(without)
    with pytest.raises(ValueError, match="missing complex_without_wirtinger|drift"):
        assert_wirtinger_implicit_product_integrity(no_refuse)

    dup = dict(registry)
    drows = [dict(row) for row in surfaces]
    drows.append(dict(drows[0]))
    dup["surfaces"] = drows
    dup["surface_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate surface_id"):
        assert_wirtinger_implicit_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_wirtinger_implicit_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["surface_count"] = 0
    with pytest.raises(ValueError, match="surface_count"):
        assert_wirtinger_implicit_product_integrity(count_mismatch)


def test_module_exports() -> None:
    assert "decide_complex_objective_contract" in wirtinger_implicit_product.__all__
    assert "materialise_demo_wirtinger_probe" in wirtinger_implicit_product.__all__
    assert "materialise_demo_implicit_stationary_probe" in wirtinger_implicit_product.__all__
    assert "list_wirtinger_implicit_surface_ids" in wirtinger_implicit_product.__all__


def test_surface_row_validation() -> None:
    base: dict[str, Any] = {
        "surface_id": "x",
        "kind": "wirtinger_partials",
        "title": "t",
        "summary": "s",
        "module_path": "m",
        "symbol_name": "fn",
        "support_posture": "local_materialised",
    }
    assert WirtingerImplicitSurfaceRow(**base).surface_id == "x"
    with pytest.raises(ValueError, match="surface_id"):
        WirtingerImplicitSurfaceRow(**{**base, "surface_id": ""})
    with pytest.raises(ValueError, match="kind"):
        WirtingerImplicitSurfaceRow(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        WirtingerImplicitSurfaceRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        WirtingerImplicitSurfaceRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="module_path"):
        WirtingerImplicitSurfaceRow(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="symbol_name"):
        WirtingerImplicitSurfaceRow(**{**base, "symbol_name": ""})
    with pytest.raises(ValueError, match="support_posture"):
        WirtingerImplicitSurfaceRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="as_of"):
        WirtingerImplicitSurfaceRow(**{**base, "as_of": ""})
    with pytest.raises(ValueError, match="bl53_pointer"):
        WirtingerImplicitSurfaceRow(**{**base, "bl53_pointer": ""})
    with pytest.raises(ValueError, match="bl46_pointer"):
        WirtingerImplicitSurfaceRow(**{**base, "bl46_pointer": ""})


def test_probe_and_decision_validation() -> None:
    with pytest.raises(ValueError, match="z must be non-empty"):
        MaterialisedWirtingerProbe(
            z=(),
            df_dz=(),
            df_dconj_z=(),
            holomorphic_residual=0.0,
            is_holomorphic=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="df_dz length"):
        MaterialisedWirtingerProbe(
            z=((1.0, 0.0),),
            df_dz=(),
            df_dconj_z=((0.0, 0.0),),
            holomorphic_residual=0.0,
            is_holomorphic=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="df_dconj_z length"):
        MaterialisedWirtingerProbe(
            z=((1.0, 0.0),),
            df_dz=((0.0, 0.0),),
            df_dconj_z=(),
            holomorphic_residual=0.0,
            is_holomorphic=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="holomorphic_residual"):
        MaterialisedWirtingerProbe(
            z=((1.0, 0.0),),
            df_dz=((0.0, 0.0),),
            df_dconj_z=((0.0, 0.0),),
            holomorphic_residual=-0.1,
            is_holomorphic=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedWirtingerProbe(
            z=((1.0, 0.0),),
            df_dz=((0.0, 0.0),),
            df_dconj_z=((0.0, 0.0),),
            holomorphic_residual=0.0,
            is_holomorphic=True,
            demo_label="",
        )
    with pytest.raises(ValueError, match="sensitivity must be non-empty"):
        MaterialisedImplicitProbe(
            method="m",
            sensitivity=(),
            shape=(1, 1),
            condition_number=1.0,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="method"):
        MaterialisedImplicitProbe(
            method="",
            sensitivity=(0.0,),
            shape=(1, 1),
            condition_number=1.0,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="shape dimensions"):
        MaterialisedImplicitProbe(
            method="m",
            sensitivity=(0.0,),
            shape=(0, 1),
            condition_number=1.0,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="sensitivity length"):
        MaterialisedImplicitProbe(
            method="m",
            sensitivity=(0.0,),
            shape=(2, 2),
            condition_number=1.0,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="condition_number"):
        MaterialisedImplicitProbe(
            method="m",
            sensitivity=(0.0,),
            shape=(1, 1),
            condition_number=-1.0,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedImplicitProbe(
            method="m",
            sensitivity=(0.0,),
            shape=(1, 1),
            condition_number=1.0,
            demo_label="",
        )
    with pytest.raises(ValueError, match="allowed requires"):
        ComplexContractDecision(
            allowed=True,
            has_wirtinger_contract=False,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="require blockers"):
        ComplexContractDecision(
            allowed=False,
            has_wirtinger_contract=False,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="reason"):
        ComplexContractDecision(
            allowed=False,
            has_wirtinger_contract=False,
            reason="",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        ComplexContractDecision(
            allowed=True,
            has_wirtinger_contract=True,
            reason="ok",
            blockers=("x",),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        ComplexContractDecision(
            allowed=False,
            has_wirtinger_contract=False,
            reason="r",
            blockers=("",),
        )
    with pytest.raises(ValueError, match="cross_scale"):
        materialise_demo_implicit_stationary_probe(cross_scale=float("nan"))
