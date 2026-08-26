# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for error mitigation product
"""Real-surface tests for ``error_mitigation_product``."""

from __future__ import annotations

import builtins
import subprocess
from types import ModuleType
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


def _valid_taxonomy_row(**overrides: object) -> MitigatorTaxonomyRow:
    """Build a valid taxonomy row; overrides replace constructor kwargs."""
    payload: dict[str, object] = {
        "mitigator_id": "x",
        "kind": "zne",
        "title": "t",
        "summary": "s",
        "differentiability": "fd_only",
        "ambient_module": "m",
        "ambient_symbol": "x",
    }
    payload.update(overrides)
    return MitigatorTaxonomyRow(**cast(Any, payload))


def _valid_boundary_row(**overrides: object) -> MitigationBoundaryRow:
    """Build a valid boundary row; overrides replace constructor kwargs."""
    payload: dict[str, object] = {
        "boundary_id": "x",
        "kind": "ideal_gradient_restore",
        "title": "t",
        "failure_class": "f",
        "summary": "s",
    }
    payload.update(overrides)
    return MitigationBoundaryRow(**cast(Any, payload))


def _valid_zne_probe(**overrides: object) -> MaterialisedZneProbe:
    """Build a valid materialised ZNE probe; overrides replace constructor kwargs."""
    payload: dict[str, object] = {
        "mitigator_id": "zne_richardson",
        "zero_noise_estimate": 1.0,
        "fit_residual": 0.0,
        "order": 1,
        "n_points": 3,
        "probe_digest": "a" * 64,
        "invent_green_ideal_gradient_restore": False,
        "invent_green_live_qpu": False,
        "demo_label": "d",
    }
    payload.update(overrides)
    return MaterialisedZneProbe(**cast(Any, payload))


def _valid_readout_probe(**overrides: object) -> MaterialisedReadoutProbe:
    """Build a valid materialised readout probe; overrides replace constructor kwargs."""
    payload: dict[str, object] = {
        "mitigator_id": "readout_confusion",
        "n_qubits": 1,
        "n_basis": 2,
        "mitigated_probability_sum": 1.0,
        "probe_digest": "a" * 64,
        "invent_green_ideal_gradient_restore": False,
        "demo_label": "d",
    }
    payload.update(overrides)
    return MaterialisedReadoutProbe(**cast(Any, payload))


def test_list_and_filters() -> None:
    """Exercise catalogue listing and differentiability filters."""
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


def test_iter_mitigation_boundaries_without_kind_filter() -> None:
    """Unfiltered boundary iteration returns the full catalogue."""
    all_rows = iter_mitigation_boundaries()
    assert len(all_rows) == len(list_mitigation_boundary_ids())
    assert {row.boundary_id for row in all_rows} == set(list_mitigation_boundary_ids())


def test_get_known_and_unknown() -> None:
    """Resolve known rows and reject unknown identifiers."""
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


def test_get_mitigation_boundary_rejects_blank_id() -> None:
    """Blank boundary_id fails closed before catalogue lookup."""
    with pytest.raises(ValueError, match="non-empty"):
        get_mitigation_boundary("  ")
    with pytest.raises(ValueError, match="non-empty"):
        get_mitigation_boundary("")


def test_decide_mitigation_path() -> None:
    """Enforce mitigation-path eligibility boundaries."""
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
    """Materialise a ZNE probe through the ambient implementation."""
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
    """Refuse unsupported ZNE promotion requests."""
    with pytest.raises(ValueError, match="ideal"):
        materialise_zne_probe("zne_richardson", invent_green_ideal_gradient_restore=True)
    with pytest.raises(ValueError, match="QPU|qpu|live"):
        materialise_zne_probe("zne_richardson", invent_green_live_qpu=True)
    with pytest.raises(ValueError, match="ZNE"):
        materialise_zne_probe("readout_confusion")


def test_readout_probe() -> None:
    """Materialise a local readout-mitigation probe."""
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
    """Expose the Studio boundary in the complete registry."""
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


def test_integrity_rejects_extra_mitigator_drift() -> None:
    """Reject undeclared mitigator catalogue drift."""
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


def test_integrity_rejects_empty_mitigators() -> None:
    """Reject an empty mitigator catalogue."""
    registry = build_error_mitigation_product_registry()
    empty: dict[str, object] = {
        "mitigators": [],
        "boundaries": registry["boundaries"],
        "blank_entry_count": 0,
        "mitigator_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty mitigators"):
        assert_error_mitigation_product_integrity(empty)


def test_integrity_rejects_empty_boundaries() -> None:
    """Reject an empty mitigation-boundary catalogue."""
    registry = build_error_mitigation_product_registry()
    no_b = dict(registry)
    no_b["boundaries"] = []
    no_b["boundary_count"] = 0
    with pytest.raises(ValueError, match="non-empty boundaries"):
        assert_error_mitigation_product_integrity(no_b)


def test_integrity_rejects_invent_green_policies() -> None:
    """Reject enabled invent-green policies."""
    registry = build_error_mitigation_product_registry()
    for policy in (
        "hardware_submit_allowed_policy",
        "ideal_gradient_restore_policy",
        "mitiq_hard_dependency_policy",
    ):
        bad = dict(registry)
        bad[policy] = True
        with pytest.raises(ValueError, match=policy):
            assert_error_mitigation_product_integrity(bad)


def test_integrity_rejects_hardware_submit_on_mitigator_row() -> None:
    """Reject hardware submission on mitigator rows."""
    registry = build_error_mitigation_product_registry()
    mitigators = cast(list[dict[str, object]], registry["mitigators"])
    hw = dict(registry)
    mut = [dict(row) for row in mitigators]
    mut[0]["hardware_submit_allowed"] = True
    hw["mitigators"] = mut
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        assert_error_mitigation_product_integrity(hw)


def test_integrity_rejects_nonzero_blank_entry_count() -> None:
    """Reject a nonzero blank-entry count."""
    registry = build_error_mitigation_product_registry()
    blank = dict(registry)
    blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_error_mitigation_product_integrity(blank)


def test_integrity_rejects_blank_studio_claim_boundary() -> None:
    """Reject a blank Studio claim boundary."""
    registry = build_error_mitigation_product_registry()
    no_studio = dict(registry)
    no_studio["studio_mitigate_claim_boundary"] = ""
    with pytest.raises(ValueError, match="studio_mitigate_claim_boundary"):
        assert_error_mitigation_product_integrity(no_studio)


def test_integrity_rejects_boundary_not_fail_closed() -> None:
    """Require every mitigation boundary to fail closed."""
    registry = build_error_mitigation_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    fc = dict(registry)
    mut_b = [dict(row) for row in bounds]
    mut_b[0]["fail_closed"] = False
    fc["boundaries"] = mut_b
    with pytest.raises(ValueError, match="fail_closed"):
        assert_error_mitigation_product_integrity(fc)


def test_integrity_rejects_boundary_set_drift() -> None:
    """Registry boundary set must match catalogue identifiers exactly."""
    registry = build_error_mitigation_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    extra = dict(registry)
    mutated = [dict(row) for row in bounds]
    mutated.append(
        {
            "boundary_id": "extra_boundary",
            "kind": "ideal_gradient_restore",
            "title": "t",
            "failure_class": "f",
            "summary": "s",
            "fail_closed": True,
            "claim_boundary": ERROR_MITIGATION_CLAIM_BOUNDARY,
        }
    )
    extra["boundaries"] = mutated
    extra["boundary_count"] = len(mutated)
    with pytest.raises(ValueError, match="boundary set drift|extra"):
        assert_error_mitigation_product_integrity(extra)


def test_mitigator_taxonomy_row_rejects_blank_mitigator_id() -> None:
    """Reject blank mitigator identifiers."""
    with pytest.raises(ValueError, match="mitigator_id"):
        _valid_taxonomy_row(mitigator_id="")


def test_mitigator_taxonomy_row_rejects_unknown_kind() -> None:
    """Reject unknown mitigator kinds."""
    with pytest.raises(ValueError, match="unknown mitigator kind"):
        _valid_taxonomy_row(kind=cast(Any, "bogus"))


def test_mitigator_taxonomy_row_rejects_blank_title() -> None:
    """Reject blank mitigator titles."""
    with pytest.raises(ValueError, match="title"):
        _valid_taxonomy_row(title="")


def test_mitigator_taxonomy_row_rejects_blank_summary() -> None:
    """Reject blank mitigator summaries."""
    with pytest.raises(ValueError, match="summary"):
        _valid_taxonomy_row(summary="")


def test_mitigator_taxonomy_row_rejects_unknown_differentiability() -> None:
    """Reject unknown differentiability classes."""
    with pytest.raises(ValueError, match="differentiability"):
        _valid_taxonomy_row(differentiability=cast(Any, "bogus"))


def test_mitigator_taxonomy_row_rejects_blank_ambient_module() -> None:
    """Reject blank ambient-module paths."""
    with pytest.raises(ValueError, match="ambient_module"):
        _valid_taxonomy_row(ambient_module="")


def test_mitigator_taxonomy_row_rejects_blank_ambient_symbol() -> None:
    """Reject blank ambient symbols."""
    with pytest.raises(ValueError, match="ambient_symbol"):
        _valid_taxonomy_row(ambient_symbol="")


def test_mitigator_taxonomy_row_rejects_hardware_submit_allowed() -> None:
    """Reject hardware-enabled mitigator rows."""
    with pytest.raises(ValueError, match="hardware_submit_allowed"):
        _valid_taxonomy_row(hardware_submit_allowed=True)


def test_mitigator_taxonomy_row_rejects_unknown_support_posture() -> None:
    """Reject unknown support postures."""
    with pytest.raises(ValueError, match="support_posture"):
        _valid_taxonomy_row(support_posture=cast(Any, "bogus"))


def test_mitigator_taxonomy_row_rejects_blank_as_of() -> None:
    """Reject blank inventory dates."""
    with pytest.raises(ValueError, match="as_of"):
        _valid_taxonomy_row(as_of="")


def test_mitigator_taxonomy_row_to_dict() -> None:
    """Serialize mitigator taxonomy rows."""
    ok = _valid_taxonomy_row()
    assert ok.to_dict()["mitigator_id"] == "x"


def test_mitigation_boundary_row_rejects_blank_boundary_id() -> None:
    """Reject blank mitigation-boundary identifiers."""
    with pytest.raises(ValueError, match="boundary_id"):
        _valid_boundary_row(boundary_id="")


def test_mitigation_boundary_row_rejects_unknown_kind() -> None:
    """Reject unknown mitigation-boundary kinds."""
    with pytest.raises(ValueError, match="unknown boundary kind"):
        _valid_boundary_row(kind=cast(Any, "bogus"))


def test_mitigation_boundary_row_rejects_fail_closed_false() -> None:
    """Reject mitigation boundaries that do not fail closed."""
    with pytest.raises(ValueError, match="fail_closed"):
        _valid_boundary_row(fail_closed=False)


def test_mitigation_boundary_row_rejects_blank_title() -> None:
    """Reject blank mitigation-boundary titles."""
    with pytest.raises(ValueError, match="title"):
        _valid_boundary_row(title="")


def test_mitigation_boundary_row_rejects_blank_failure_class() -> None:
    """Reject blank mitigation failure classes."""
    with pytest.raises(ValueError, match="failure_class"):
        _valid_boundary_row(failure_class="")


def test_mitigation_boundary_row_rejects_blank_summary() -> None:
    """Reject blank mitigation-boundary summaries."""
    with pytest.raises(ValueError, match="summary"):
        _valid_boundary_row(summary="")


def test_mitigation_boundary_row_to_dict() -> None:
    """Serialize mitigation-boundary rows."""
    ok_b = _valid_boundary_row()
    assert ok_b.to_dict()["fail_closed"] is True


def test_path_eligibility_rejects_refused_without_blockers() -> None:
    """Require blockers on refused path decisions."""
    with pytest.raises(ValueError, match="blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )


def test_path_eligibility_rejects_unknown_outcome() -> None:
    """Reject unknown path-decision outcomes."""
    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "maybe"),
            allowed=True,
            reason="r",
            blockers=(),
        )


def test_path_eligibility_rejects_blank_reason() -> None:
    """Reject blank path-decision reasons."""
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="",
            blockers=(),
        )


def test_path_eligibility_rejects_allowed_flag_with_refused_outcome() -> None:
    """Reject allowed flags on refused outcomes."""
    with pytest.raises(ValueError, match="outcome=allowed"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
        )


def test_path_eligibility_rejects_refused_flag_with_allowed_outcome() -> None:
    """Reject refused flags on allowed outcomes."""
    with pytest.raises(ValueError, match="outcome=refused"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=False,
            reason="r",
            blockers=("b",),
        )


def test_path_eligibility_rejects_allowed_with_blockers() -> None:
    """Reject blockers on allowed path decisions."""
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("b",),
        )


def test_path_eligibility_rejects_blank_blocker_entries() -> None:
    """Reject blank blocker entries."""
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("",),
        )


def test_path_eligibility_to_dict() -> None:
    """Serialize mitigation path decisions."""
    ok_d = PathEligibilityDecision(
        outcome="allowed",
        allowed=True,
        reason="ok",
        blockers=(),
    )
    assert ok_d.to_dict()["allowed"] is True


def test_materialised_zne_probe_rejects_blank_mitigator_id() -> None:
    """Reject blank mitigator IDs on ZNE probes."""
    with pytest.raises(ValueError, match="mitigator_id"):
        _valid_zne_probe(mitigator_id="")


def test_materialised_zne_probe_rejects_non_finite_estimate() -> None:
    """Reject non-finite zero-noise estimates."""
    with pytest.raises(ValueError, match="zero_noise_estimate"):
        _valid_zne_probe(zero_noise_estimate=float("nan"))


def test_materialised_zne_probe_rejects_negative_residual() -> None:
    """Reject negative ZNE fit residuals."""
    with pytest.raises(ValueError, match="fit_residual"):
        _valid_zne_probe(fit_residual=-1.0)


def test_materialised_zne_probe_rejects_non_positive_order() -> None:
    """Reject non-positive extrapolation orders."""
    with pytest.raises(ValueError, match="order"):
        _valid_zne_probe(order=0)


def test_materialised_zne_probe_rejects_insufficient_points() -> None:
    """Require enough ZNE sample points."""
    with pytest.raises(ValueError, match="n_points"):
        _valid_zne_probe(n_points=1)


def test_materialised_zne_probe_rejects_bad_digest() -> None:
    """Reject malformed ZNE probe digests."""
    with pytest.raises(ValueError, match="probe_digest"):
        _valid_zne_probe(probe_digest="x")


def test_materialised_zne_probe_rejects_ideal_gradient_invent_green() -> None:
    """Reject invented ideal-gradient restoration."""
    with pytest.raises(ValueError, match="invent_green_ideal_gradient_restore"):
        _valid_zne_probe(invent_green_ideal_gradient_restore=True)


def test_materialised_zne_probe_rejects_live_qpu_invent_green() -> None:
    """Reject invented live-QPU ZNE claims."""
    with pytest.raises(ValueError, match="invent_green_live_qpu"):
        _valid_zne_probe(invent_green_live_qpu=True)


def test_materialised_zne_probe_rejects_blank_demo_label() -> None:
    """Reject blank ZNE demo labels."""
    with pytest.raises(ValueError, match="demo_label"):
        _valid_zne_probe(demo_label="")


def test_materialised_zne_probe_to_dict() -> None:
    """Serialize materialised ZNE probes."""
    ok_z = _valid_zne_probe()
    assert ok_z.to_dict()["n_points"] == 3


def test_materialised_readout_probe_rejects_non_positive_n_qubits() -> None:
    """Reject non-positive readout qubit counts."""
    with pytest.raises(ValueError, match="n_qubits"):
        _valid_readout_probe(n_qubits=0, n_basis=1)


def test_materialised_readout_probe_rejects_n_basis_mismatch() -> None:
    """Reject readout basis-size mismatches."""
    with pytest.raises(ValueError, match="n_basis"):
        _valid_readout_probe(n_qubits=1, n_basis=3)


def test_materialised_readout_probe_rejects_blank_mitigator_id() -> None:
    """Reject blank mitigator IDs on readout probes."""
    with pytest.raises(ValueError, match="mitigator_id"):
        _valid_readout_probe(mitigator_id="")


def test_materialised_readout_probe_rejects_non_finite_sum() -> None:
    """Reject non-finite mitigated probability sums."""
    with pytest.raises(ValueError, match="mitigated_probability_sum"):
        _valid_readout_probe(mitigated_probability_sum=float("nan"))


def test_materialised_readout_probe_rejects_bad_digest() -> None:
    """Reject malformed readout probe digests."""
    with pytest.raises(ValueError, match="probe_digest"):
        _valid_readout_probe(probe_digest="x")


def test_materialised_readout_probe_rejects_ideal_gradient_invent_green() -> None:
    """Reject invented readout-gradient restoration."""
    with pytest.raises(ValueError, match="invent_green_ideal_gradient_restore"):
        _valid_readout_probe(invent_green_ideal_gradient_restore=True)


def test_materialised_readout_probe_rejects_blank_demo_label() -> None:
    """Reject blank readout demo labels."""
    with pytest.raises(ValueError, match="demo_label"):
        _valid_readout_probe(demo_label="")


def test_materialised_readout_probe_to_dict() -> None:
    """Serialize materialised readout probes."""
    ok_r = _valid_readout_probe()
    assert ok_r.to_dict()["n_basis"] == 2


def test_integrity_rejects_non_mapping_mitigator_row() -> None:
    """Reject non-mapping mitigator payload rows."""
    registry = build_error_mitigation_product_registry()
    not_map = dict(registry)
    not_map["mitigators"] = cast(list[dict[str, object]], ["x"])
    with pytest.raises(ValueError, match="mapping"):
        assert_error_mitigation_product_integrity(not_map)


def test_integrity_rejects_blank_mitigator_id_in_payload() -> None:
    """Reject blank mitigator IDs in registry payloads."""
    registry = build_error_mitigation_product_registry()
    mitigators = cast(list[dict[str, object]], registry["mitigators"])
    blank_id = dict(registry)
    bc = [dict(row) for row in mitigators]
    bc[0]["mitigator_id"] = "  "
    blank_id["mitigators"] = bc
    with pytest.raises(ValueError, match="blank"):
        assert_error_mitigation_product_integrity(blank_id)


def test_integrity_rejects_duplicate_mitigator_id() -> None:
    """Reject duplicate mitigator IDs."""
    registry = build_error_mitigation_product_registry()
    mitigators = cast(list[dict[str, object]], registry["mitigators"])
    dup = dict(registry)
    dc = [dict(row) for row in mitigators]
    dc[1] = dict(dc[0])
    dup["mitigators"] = dc
    with pytest.raises(ValueError, match="duplicate mitigator_id"):
        assert_error_mitigation_product_integrity(dup)


def test_integrity_rejects_invalid_differentiability() -> None:
    """Reject invalid registry differentiability classes."""
    registry = build_error_mitigation_product_registry()
    mitigators = cast(list[dict[str, object]], registry["mitigators"])
    bad_diff = dict(registry)
    bd = [dict(row) for row in mitigators]
    bd[0]["differentiability"] = "magic"
    bad_diff["mitigators"] = bd
    with pytest.raises(ValueError, match="differentiability"):
        assert_error_mitigation_product_integrity(bad_diff)


def test_integrity_rejects_blank_ambient_symbol() -> None:
    """Reject blank registry ambient symbols."""
    registry = build_error_mitigation_product_registry()
    mitigators = cast(list[dict[str, object]], registry["mitigators"])
    no_sym = dict(registry)
    ns = [dict(row) for row in mitigators]
    ns[0]["ambient_symbol"] = ""
    no_sym["mitigators"] = ns
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_error_mitigation_product_integrity(no_sym)


def test_integrity_rejects_missing_zne_richardson() -> None:
    """Require the Richardson ZNE catalogue row."""
    registry = build_error_mitigation_product_registry()
    mitigators = cast(list[dict[str, object]], registry["mitigators"])
    no_zne = dict(registry)
    filtered = [dict(row) for row in mitigators if row["mitigator_id"] != "zne_richardson"]
    no_zne["mitigators"] = filtered
    no_zne["mitigator_count"] = len(filtered)
    with pytest.raises(ValueError, match="zne_richardson|drift"):
        assert_error_mitigation_product_integrity(no_zne)


def test_integrity_rejects_non_mapping_boundary_row() -> None:
    """Reject non-mapping boundary payload rows."""
    registry = build_error_mitigation_product_registry()
    b_not = dict(registry)
    b_not["boundaries"] = cast(list[dict[str, object]], [1])
    with pytest.raises(ValueError, match="mapping"):
        assert_error_mitigation_product_integrity(b_not)


def test_integrity_rejects_blank_boundary_id_in_payload() -> None:
    """Reject blank boundary IDs in registry payloads."""
    registry = build_error_mitigation_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    blank_b = dict(registry)
    bb = [dict(row) for row in bounds]
    bb[0]["boundary_id"] = ""
    blank_b["boundaries"] = bb
    with pytest.raises(ValueError, match="boundary_id"):
        assert_error_mitigation_product_integrity(blank_b)


def test_integrity_rejects_duplicate_boundary_id() -> None:
    """Reject duplicate mitigation-boundary IDs."""
    registry = build_error_mitigation_product_registry()
    bounds = cast(list[dict[str, object]], registry["boundaries"])
    dup_b = dict(registry)
    db = [dict(row) for row in bounds]
    db[1] = dict(db[0])
    dup_b["boundaries"] = db
    with pytest.raises(ValueError, match="duplicate boundary_id"):
        assert_error_mitigation_product_integrity(dup_b)


def test_integrity_rejects_mitigator_count_mismatch() -> None:
    """Reject mitigator count mismatches."""
    registry = build_error_mitigation_product_registry()
    count_m = dict(registry)
    count_m["mitigator_count"] = 99
    with pytest.raises(ValueError, match="mitigator_count"):
        assert_error_mitigation_product_integrity(count_m)


def test_integrity_rejects_boundary_count_mismatch() -> None:
    """Reject mitigation-boundary count mismatches."""
    registry = build_error_mitigation_product_registry()
    count_b = dict(registry)
    count_b["boundary_count"] = 99
    with pytest.raises(ValueError, match="boundary_count"):
        assert_error_mitigation_product_integrity(count_b)


def test_zne_probe_rejects_missing_ambient_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject incomplete ambient ZNE results."""

    def _bad_zne(*_a: object, **_k: object) -> dict[str, object]:
        return {"not": "enough"}

    monkeypatch.setattr(mit_product, "_run_ambient_mitigation_json", _bad_zne)
    with pytest.raises(ValueError, match="missing fields"):
        materialise_zne_probe("zne_richardson")


def test_zne_probe_rejects_non_finite_ambient_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject non-finite ambient ZNE estimates."""

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


def test_zne_probe_rejects_negative_ambient_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject negative ambient ZNE residuals."""

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


def test_zne_probe_rejects_insufficient_ambient_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject insufficient ambient ZNE points."""

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


def test_readout_probe_rejects_missing_ambient_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject incomplete ambient readout results."""

    def _bad_ro(*_a: object, **_k: object) -> dict[str, object]:
        return {"n_qubits": 1}

    monkeypatch.setattr(mit_product, "_run_ambient_mitigation_json", _bad_ro)
    with pytest.raises(ValueError, match="missing fields"):
        materialise_readout_probe()


def test_readout_probe_rejects_non_finite_ambient_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject non-finite ambient readout sums."""

    def _nan_ro(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "n_qubits": 1,
            "n_basis": 2,
            "mitigated_probability_sum": float("nan"),
        }

    monkeypatch.setattr(mit_product, "_run_ambient_mitigation_json", _nan_ro)
    with pytest.raises(ValueError, match="finite"):
        materialise_readout_probe()


def test_studio_mitigate_claim_boundary_rejects_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a blank ambient Studio mitigation boundary."""
    monkeypatch.setattr(mit_product, "_studio_mitigate_claim_boundary_text", lambda: "")
    with pytest.raises(ValueError, match="non-empty"):
        studio_mitigate_claim_boundary()


def test_studio_mitigate_claim_boundary_rejects_missing_honesty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require honesty language in the Studio boundary."""
    monkeypatch.setattr(
        mit_product,
        "_studio_mitigate_claim_boundary_text",
        lambda: "promotional claim without honesty",
    )
    with pytest.raises(ValueError, match="honesty"):
        studio_mitigate_claim_boundary()


def test_studio_boundary_loader_uses_importable_ambient_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the ambient Studio constant when its module is importable."""
    ambient = ModuleType("scpn_quantum_control.studio.executive_mitigate")
    boundary = "measured expectation extrapolation; does not run circuits"
    ambient.__dict__["MITIGATE_CLAIM_BOUNDARY"] = boundary

    def import_ambient(name: str, *_args: object, **_kwargs: object) -> ModuleType:
        assert name.endswith("studio.executive_mitigate")
        return ambient

    monkeypatch.setattr(builtins, "__import__", import_ambient)
    assert mit_product._studio_mitigate_claim_boundary_text() == boundary


def test_studio_boundary_loader_uses_mirror_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the local honesty mirror when Studio cannot import."""

    def reject_import(name: str, *_args: object, **_kwargs: object) -> ModuleType:
        assert name.endswith("studio.executive_mitigate")
        raise ImportError("Studio unavailable")

    monkeypatch.setattr(builtins, "__import__", reject_import)
    assert (
        mit_product._studio_mitigate_claim_boundary_text()
        == mit_product._STUDIO_MITIGATE_CLAIM_BOUNDARY_MIRROR
    )


def test_studio_boundary_loader_rejects_blank_ambient_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a blank constant from an importable Studio module."""
    ambient = ModuleType("scpn_quantum_control.studio.executive_mitigate")
    ambient.__dict__["MITIGATE_CLAIM_BOUNDARY"] = "  "

    def import_ambient(name: str, *_args: object, **_kwargs: object) -> ModuleType:
        assert name.endswith("studio.executive_mitigate")
        return ambient

    monkeypatch.setattr(builtins, "__import__", import_ambient)
    with pytest.raises(ValueError, match="must be non-empty"):
        mit_product._studio_mitigate_claim_boundary_text()


def test_studio_mitigate_claim_boundary_mirror_matches_ambient_when_importable() -> None:
    """When Studio package loads, ambient constant and product mirror stay lockstep."""
    try:
        from scpn_quantum_control.studio.executive_mitigate import (
            MITIGATE_CLAIM_BOUNDARY as ambient,
        )
    except ImportError:
        pytest.skip("scpn_studio_platform / Studio package not installed on this matrix cell")
    assert ambient == mit_product._STUDIO_MITIGATE_CLAIM_BOUNDARY_MIRROR
    assert studio_mitigate_claim_boundary() == ambient


def test_taxonomy_map_rejects_blank_mitigator_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalogue map fails closed when a row carries a blank id after construction."""
    blank = _valid_taxonomy_row()
    object.__setattr__(blank, "mitigator_id", "  ")
    monkeypatch.setattr(mit_product, "_TAXONOMY", (blank,))
    with pytest.raises(RuntimeError, match="blank mitigator_id"):
        mit_product._taxonomy_map()


def test_taxonomy_map_rejects_duplicate_mitigator_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject duplicate IDs while building the taxonomy map."""
    row = get_mitigator("zne_richardson")
    monkeypatch.setattr(mit_product, "_TAXONOMY", (row, row))
    with pytest.raises(RuntimeError, match="duplicate mitigator_id"):
        mit_product._taxonomy_map()


def test_taxonomy_map_rejects_empty_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an empty taxonomy map source."""
    monkeypatch.setattr(mit_product, "_TAXONOMY", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        mit_product._taxonomy_map()


def test_ambient_subprocess_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """CalledProcessError from ambient mitigation subprocess is fail-closed."""

    def boom(*_a: object, **_k: object) -> object:
        raise subprocess.CalledProcessError(1, "x", stderr="mitigation boom")

    monkeypatch.setattr(
        "scpn_quantum_control.error_mitigation_product.subprocess.run",
        boom,
    )
    with pytest.raises(ValueError, match="ambient mitigation subprocess failed"):
        materialise_zne_probe("zne_richardson")


def test_ambient_subprocess_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """TimeoutExpired from ambient mitigation subprocess is fail-closed."""

    def timeout(*_a: object, **_k: object) -> object:
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(
        "scpn_quantum_control.error_mitigation_product.subprocess.run",
        timeout,
    )
    with pytest.raises(ValueError, match="timed out"):
        materialise_zne_probe("zne_richardson")


def test_ambient_subprocess_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-JSON stdout from ambient mitigation subprocess is fail-closed."""

    class _Out:
        stdout = "not-json\n"

    monkeypatch.setattr(
        "scpn_quantum_control.error_mitigation_product.subprocess.run",
        lambda *_a, **_k: _Out(),
    )
    with pytest.raises(ValueError, match="non-JSON"):
        materialise_zne_probe("zne_richardson")


def test_ambient_subprocess_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON array (not object) payload from ambient mitigation is fail-closed."""

    class _Out:
        stdout = "[1, 2, 3]\n"

    monkeypatch.setattr(
        "scpn_quantum_control.error_mitigation_product.subprocess.run",
        lambda *_a, **_k: _Out(),
    )
    with pytest.raises(ValueError, match="must be an object"):
        materialise_zne_probe("zne_richardson")


def test_module_exports_stable() -> None:
    """Keep the public mitigation exports stable."""
    assert "assert_error_mitigation_product_integrity" in mit_product.__all__
    assert "materialise_demo_zne_probe" in mit_product.__all__
    assert ERROR_MITIGATION_PRODUCT_SCHEMA == "error_mitigation_product.v1"
