# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for governed multi-ecosystem route matrix
"""Real-surface tests for ``scpn_quantum_control.governed_route_matrix``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest

import scpn_quantum_control.governed_route_matrix as governed_route_matrix
from scpn_quantum_control.governed_route_matrix import (
    GOVERNED_ROUTE_MATRIX_CLAIM_BOUNDARY,
    GOVERNED_ROUTE_MATRIX_SCHEMA,
    GovernedRouteRecord,
    RouteCapability,
    RouteExplanation,
    assert_no_blank_matrix_cells,
    build_governed_route_matrix,
    explain_route,
    get_governed_route,
    iter_governed_routes,
    list_governed_route_ids,
)


def test_list_ids_are_stable_nonempty_and_unique() -> None:
    """List stable, non-blank, unique canonical route identifiers."""
    ids = list_governed_route_ids()
    assert ids
    assert len(ids) == len(set(ids))
    assert ids == list_governed_route_ids()
    assert all(":" in route_id for route_id in ids)


def test_get_governed_route_supported_and_boundary() -> None:
    """Return both supported and boundary catalogue records."""
    supported = get_governed_route("transform:native.grad_vmap")
    assert supported.closure_status == "supported"
    assert supported.family == "transform"
    assert supported.claim_boundary == GOVERNED_ROUTE_MATRIX_CLAIM_BOUNDARY
    assert not supported.closure_reason

    boundary = get_governed_route("compiler:catalyst.qjit_vmap")
    assert boundary.closure_status == "permanent_boundary"
    assert boundary.closure_reason
    assert "vmap" in boundary.closure_reason.lower() or "batch" in boundary.closure_reason.lower()


def test_ssgf_latent_gradient_routes_are_explicit() -> None:
    """Keep supported and boundary SSGF gradient routes explicit."""
    supported = get_governed_route("transform:ssgf.latent_finite_difference")
    boundary = get_governed_route("transform:ssgf.latent_parameter_shift")

    assert supported.closure_status == "supported"
    assert boundary.closure_status == "permanent_boundary"
    assert supported.rejected_alternatives == (boundary.route_id,)
    assert "softplus" in boundary.closure_reason


def test_l16_indicator_and_hardware_routes_are_explicit() -> None:
    """Support bounded local indicators and permanently refuse autonomous actuation."""
    supported = get_governed_route("adapter:l16.local_indicator")
    boundary = get_governed_route("adapter:l16.autonomous_hardware_control")

    assert supported.closure_status == "supported"
    assert supported.rejected_alternatives == (boundary.route_id,)
    assert boundary.closure_status == "permanent_boundary"
    assert "not a Lyapunov" in boundary.closure_reason


def test_get_governed_route_rejects_blank_and_unknown() -> None:
    """Reject blank and unknown route identifiers."""
    with pytest.raises(ValueError, match="non-empty"):
        get_governed_route("  ")
    with pytest.raises(ValueError, match="unknown governed route_id"):
        get_governed_route("invented:green.cell")


def test_iter_governed_routes_filters() -> None:
    """Filter the catalogue by family and closure status."""
    adapters = iter_governed_routes(family="adapter")
    assert adapters
    assert all(row.family == "adapter" for row in adapters)

    permanent = iter_governed_routes(closure_status="permanent_boundary")
    assert permanent
    assert all(row.closure_status == "permanent_boundary" for row in permanent)

    both = iter_governed_routes(family="adapter", closure_status="implementation_path")
    assert both
    assert all(
        row.family == "adapter" and row.closure_status == "implementation_path" for row in both
    )


def test_build_matrix_has_zero_blanks_and_schema() -> None:
    """Build a schema-tagged matrix without blank cells."""
    matrix = build_governed_route_matrix()
    routes = cast(list[dict[str, object]], matrix["routes"])
    assert matrix["schema"] == GOVERNED_ROUTE_MATRIX_SCHEMA
    assert matrix["blank_cell_count"] == 0
    assert matrix["route_count"] == len(routes)
    assert (
        cast(int, matrix["supported_count"])
        + cast(int, matrix["permanent_boundary_count"])
        + cast(int, matrix["implementation_path_count"])
        == matrix["route_count"]
    )
    validated = assert_no_blank_matrix_cells(matrix)
    assert validated["blank_cell_count"] == 0
    for row in routes:
        assert row["closure_status"] in {
            "supported",
            "permanent_boundary",
            "implementation_path",
        }
        if row["closure_status"] != "supported":
            assert row["closure_reason"]


def test_assert_no_blank_matrix_cells_rejects_invalid_payload() -> None:
    """Reject malformed rows, blank cells, and count drift."""
    with pytest.raises(ValueError, match="non-empty routes"):
        assert_no_blank_matrix_cells({"routes": []})
    with pytest.raises(ValueError, match="blank"):
        assert_no_blank_matrix_cells(
            {
                "routes": [{"route_id": "", "closure_status": "supported"}],
                "blank_cell_count": 0,
                "route_count": 1,
            }
        )
    with pytest.raises(ValueError, match="closure_reason"):
        assert_no_blank_matrix_cells(
            {
                "routes": [
                    {
                        "route_id": "x",
                        "closure_status": "permanent_boundary",
                        "closure_reason": "",
                    }
                ],
                "blank_cell_count": 0,
                "route_count": 1,
            }
        )


def test_explain_route_supported_transform_with_rejected_alternatives() -> None:
    """Explain a supported transform and its rejected alternatives."""
    explanation = explain_route(
        "transform:native.grad_vmap",
        RouteCapability(ecosystem="native", method="grad"),
    )
    assert isinstance(explanation, RouteExplanation)
    assert explanation.selected.route_id == "transform:native.grad_vmap"
    assert explanation.selected.closure_status == "supported"
    assert explanation.rejected
    assert any(row.closure_status != "supported" for row in explanation.rejected)
    payload = explanation.to_dict()
    selected = cast(Mapping[str, object], payload["selected"])
    assert selected["route_id"] == "transform:native.grad_vmap"
    assert payload["claim_boundary"] == GOVERNED_ROUTE_MATRIX_CLAIM_BOUNDARY


def test_explain_route_adapter_and_implementation_path() -> None:
    """Explain supported adapters and implementation-path routes."""
    jax_supported = explain_route(
        "adapter:jax.value_and_grad_local",
        {"ecosystem": "JAX", "method": "value_and_grad"},
    )
    assert jax_supported.capability.ecosystem == "jax"
    assert jax_supported.selected.closure_status == "supported"

    impl = explain_route(
        "adapter:jax.provider_arbitrary_simulator",
        RouteCapability(ecosystem="jax"),
    )
    assert impl.selected.closure_status == "implementation_path"
    assert impl.selected.closure_reason
    assert "not implemented" in impl.selected.closure_reason.lower()


def test_explain_route_competitor_boundary_fixture() -> None:
    """Explain a competitor failure fixture as a permanent boundary."""
    explanation = explain_route(
        "competitor:differentiation_interface.silent_wrong_grads",
        RouteCapability(ecosystem="julia", method="reverse"),
    )
    assert explanation.selected.closure_status == "permanent_boundary"
    assert "silent" in explanation.selected.closure_reason.lower()
    assert explanation.selected.family == "competitor_boundary"


def test_explain_route_unknown_fail_closed_policies() -> None:
    """Apply raise and boundary policies to unknown routes."""
    with pytest.raises(ValueError, match="unknown governed route_id"):
        explain_route("blank.invent.green", RouteCapability(ecosystem="native"))

    boundary = explain_route(
        "blank.invent.green",
        RouteCapability(ecosystem="native"),
        unknown_policy="boundary",
    )
    assert boundary.selected.closure_status == "permanent_boundary"
    assert boundary.selected.route_id.startswith("unknown:")
    assert "invent-green" in boundary.selected.closure_reason or "not in the governed" in (
        boundary.selected.closure_reason
    )


def test_explain_route_rejects_blank_ids_and_bad_capability() -> None:
    """Reject blank IDs and unsupported capability input types."""
    with pytest.raises(ValueError, match="non-empty"):
        explain_route("", RouteCapability(ecosystem="native"))
    with pytest.raises(TypeError, match="capability"):
        explain_route("transform:native.grad_vmap", capability=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ecosystem"):
        RouteCapability(ecosystem="   ")
    with pytest.raises(ValueError, match="method"):
        RouteCapability(ecosystem="native", method="")


def test_explain_route_hardware_and_finite_shot_notes() -> None:
    """Report hardware and finite-shot policy notes without promotion."""
    closed = explain_route(
        "provider:hardware.gradient_live",
        RouteCapability(ecosystem="provider", allow_hardware=False),
    )
    assert closed.selected.closure_status == "permanent_boundary"
    assert any("allow_hardware=False" in note for note in closed.notes)

    ticketed = explain_route(
        "provider:hardware.gradient_live",
        RouteCapability(ecosystem="provider", allow_hardware=True),
    )
    assert ticketed.selected.closure_status == "permanent_boundary"
    assert any("owner-ticket" in note for note in ticketed.notes)

    finite = explain_route(
        "transform:native.vmap_grad",
        RouteCapability(ecosystem="native", finite_shot=True),
    )
    assert finite.selected.closure_status == "supported"
    assert any("finite_shot=True" in note for note in finite.notes)


def test_governed_route_record_invariants() -> None:
    """Enforce supported and non-supported record invariants."""
    with pytest.raises(ValueError, match="closure_reason"):
        GovernedRouteRecord(
            route_id="x:y",
            family="transform",
            closure_status="permanent_boundary",
            summary="boundary without reason",
            evidence=("e",),
            rejected_alternatives=(),
            closure_reason="",
        )
    with pytest.raises(ValueError, match="must not carry a closure_reason"):
        GovernedRouteRecord(
            route_id="x:y",
            family="transform",
            closure_status="supported",
            summary="supported with reason",
            evidence=("e",),
            rejected_alternatives=(),
            closure_reason="should not be here",
        )


def test_default_capability_when_none() -> None:
    """Use the native automatic capability when none is supplied."""
    explanation = explain_route("rust:program_ad.static_registry_replay")
    assert explanation.capability.ecosystem == "native"
    assert explanation.capability.method == "auto"
    assert explanation.selected.closure_status == "supported"


def test_record_validation_edge_paths() -> None:
    """Reject invalid record fields and blank evidence entries."""
    with pytest.raises(ValueError, match="route_id"):
        GovernedRouteRecord(
            route_id="",
            family="transform",
            closure_status="supported",
            summary="x",
            evidence=("e",),
            rejected_alternatives=(),
        )
    with pytest.raises(ValueError, match="unknown route family"):
        GovernedRouteRecord(
            route_id="x:y",
            family="not_a_family",  # type: ignore[arg-type]
            closure_status="supported",
            summary="x",
            evidence=("e",),
            rejected_alternatives=(),
        )
    with pytest.raises(ValueError, match="unknown closure_status"):
        GovernedRouteRecord(
            route_id="x:y",
            family="transform",
            closure_status="maybe",  # type: ignore[arg-type]
            summary="x",
            evidence=("e",),
            rejected_alternatives=(),
        )
    with pytest.raises(ValueError, match="summary"):
        GovernedRouteRecord(
            route_id="x:y",
            family="transform",
            closure_status="supported",
            summary="  ",
            evidence=("e",),
            rejected_alternatives=(),
        )
    with pytest.raises(ValueError, match="evidence"):
        GovernedRouteRecord(
            route_id="x:y",
            family="transform",
            closure_status="supported",
            summary="ok",
            evidence=("",),
            rejected_alternatives=(),
        )
    with pytest.raises(ValueError, match="rejected_alternatives"):
        GovernedRouteRecord(
            route_id="x:y",
            family="transform",
            closure_status="supported",
            summary="ok",
            evidence=("e",),
            rejected_alternatives=("  ",),
        )


def test_explanation_validation_and_matrix_count_drift() -> None:
    """Reject inconsistent explanations and matrix counts."""
    with pytest.raises(ValueError, match="route_id must be non-empty"):
        RouteExplanation(
            route_id="",
            capability=RouteCapability(ecosystem="native"),
            selected=get_governed_route("transform:native.grad_vmap"),
            rejected=(),
            notes=(),
        )
    good = get_governed_route("transform:native.grad_vmap")
    other = get_governed_route("transform:native.vmap_grad")
    with pytest.raises(ValueError, match="selected route_id"):
        RouteExplanation(
            route_id=good.route_id,
            capability=RouteCapability(ecosystem="native"),
            selected=other,
            rejected=(),
            notes=(),
        )
    with pytest.raises(ValueError, match="blank_cell_count"):
        assert_no_blank_matrix_cells(
            {
                "routes": [good.to_dict()],
                "blank_cell_count": 1,
                "route_count": 1,
            }
        )
    with pytest.raises(ValueError, match="route_count"):
        assert_no_blank_matrix_cells(
            {
                "routes": [good.to_dict()],
                "blank_cell_count": 0,
                "route_count": 99,
            }
        )
    with pytest.raises(ValueError, match="mapping"):
        assert_no_blank_matrix_cells(
            {
                "routes": ["not-a-mapping"],
                "blank_cell_count": 0,
                "route_count": 1,
            }
        )
    with pytest.raises(ValueError, match="blank"):
        assert_no_blank_matrix_cells(
            {
                "routes": [
                    {
                        "route_id": "x",
                        "closure_status": "not-a-status",
                    }
                ],
                "blank_cell_count": 0,
                "route_count": 1,
            }
        )


def test_explain_route_invalid_unknown_policy_rejected() -> None:
    """Reject unknown policies outside the closed vocabulary."""
    with pytest.raises(ValueError, match="unknown_policy"):
        explain_route(
            "not.in.catalogue",
            RouteCapability(ecosystem="native"),
            unknown_policy="invent",  # type: ignore[arg-type]
        )


def test_record_to_dict_round_trip_fields() -> None:
    """Serialise every immutable route-record field."""
    row = get_governed_route("adapter:torch.func_local")
    payload = row.to_dict()
    assert payload["route_id"] == row.route_id
    assert payload["family"] == "adapter"
    assert isinstance(payload["evidence"], list)
    assert payload["claim_boundary"] == GOVERNED_ROUTE_MATRIX_CLAIM_BOUNDARY


def test_catalogue_map_rejects_duplicate_route_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject duplicate identifiers in canonical catalogue construction."""
    row = get_governed_route("transform:native.grad_vmap")
    monkeypatch.setattr(governed_route_matrix, "_CANONICAL_ROUTES", (row, row))
    with pytest.raises(RuntimeError, match="duplicate route_id"):
        governed_route_matrix._catalogue_map()


def test_explain_route_skips_missing_rejected_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip missing alternative pointers while retaining valid rejections."""
    base = get_governed_route("transform:native.vmap_grad")
    patched = replace(base, rejected_alternatives=("missing:not.in.catalogue",))
    mapping = dict(governed_route_matrix._ROUTE_BY_ID)
    mapping[patched.route_id] = patched
    monkeypatch.setattr(governed_route_matrix, "_ROUTE_BY_ID", mapping)
    explanation = explain_route(
        patched.route_id,
        RouteCapability(ecosystem="native"),
    )
    assert explanation.selected.route_id == patched.route_id
    # Missing alt must not invent a green row.
    assert all(row.route_id != "missing:not.in.catalogue" for row in explanation.rejected)
