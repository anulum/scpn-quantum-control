# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the circuit-cutting product
"""Real-surface tests for ``circuit_cutting_product``."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.circuit_cutting_product as cutting
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.circuit_cutting_product import (
    CIRCUIT_CUTTING_PRODUCT_SCHEMA,
    CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA,
    CIRCUIT_CUTTING_RESOURCE_SCHEMA,
    CuttingPathDecision,
    CuttingResourceCertificate,
    CuttingSurfaceRow,
    SyntheticReconstructionCertificate,
    assert_circuit_cutting_product_integrity,
    build_circuit_cutting_product_registry,
    build_cutting_resource_certificate,
    certify_synthetic_reconstruction,
    decide_cutting_path,
    get_cutting_surface,
    iter_cutting_surfaces,
    list_cutting_surface_ids,
)
from scpn_quantum_control.hardware_safe_execution import get_execution_policy


def test_surface_catalogue_and_registry() -> None:
    """The frozen inventory exposes only real, no-submit ambient surfaces."""
    ids = list_cutting_surface_ids()
    assert ids == ("resource_planner", "partition_local_simulator")
    assert list_cutting_surface_ids() == ids
    planner = get_cutting_surface("resource_planner")
    assert planner.support_posture == "bounded_planner"
    assert planner.to_dict()["live_submit"] is False
    simulated = iter_cutting_surfaces(support_posture="partition_local_simulator")
    assert simulated[0].surface_id == "partition_local_simulator"
    assert iter_cutting_surfaces(support_posture="missing") == ()
    assert tuple(row.surface_id for row in iter_cutting_surfaces()) == ids
    registry = build_circuit_cutting_product_registry()
    assert registry["schema"] == CIRCUIT_CUTTING_PRODUCT_SCHEMA
    assert registry["live_submit"] is False
    assert registry["general_reconstruction"] is False
    assert_circuit_cutting_product_integrity()


def test_surface_lookup_fails_closed() -> None:
    """Blank and unknown inventory identifiers are refused."""
    with pytest.raises(ValueError, match="non-empty"):
        get_cutting_surface(" ")
    with pytest.raises(ValueError, match="unknown"):
        get_cutting_surface("ghost")


def test_bounded_single_partition_resource_certificate() -> None:
    """A no-cut local plan has one fragment and exact execution-policy accounting."""
    certificate = build_cutting_resource_certificate(
        build_knm_paper27(L=4),
        max_partition_size=4,
        shots_per_fragment=128,
    )
    assert certificate.schema == CIRCUIT_CUTTING_RESOURCE_SCHEMA
    assert certificate.n_partitions == 1
    assert certificate.n_cuts == 0
    assert certificate.fragment_evaluations == 1
    assert certificate.estimated_total_shots == 128
    assert certificate.feasible is True
    assert certificate.outcome == "allowed_plan"
    payload = certificate.to_dict()
    assert payload["partition_sizes"] == [4]
    assert payload["blockers"] == []
    assert payload["no_submit"] is True


def test_small_partitioned_resource_certificate_and_budget_refusal() -> None:
    """Finite cuts are costed exactly and refused when the execution budget is exceeded."""
    coupling = np.zeros((4, 4), dtype=np.float64)
    coupling[0, 2] = coupling[2, 0] = 0.5
    allowed = build_cutting_resource_certificate(
        coupling,
        max_partition_size=2,
        shots_per_fragment=256,
    )
    assert allowed.n_cuts == 1
    assert allowed.fragment_evaluations == 4
    assert allowed.estimated_total_shots == 1024
    assert allowed.feasible is True

    over_budget = build_cutting_resource_certificate(
        coupling,
        max_partition_size=2,
        shots_per_fragment=2048,
        would_submit=True,
    )
    assert over_budget.feasible is False
    assert any("live submit" in item for item in over_budget.blockers)
    assert any("shots_per_fragment" in item for item in over_budget.blockers)

    five_cuts = np.zeros((6, 6), dtype=np.float64)
    for left, right in ((0, 3), (0, 4), (1, 3), (1, 4), (2, 3)):
        five_cuts[left, right] = five_cuts[right, left] = 0.5
    total_over_budget = build_cutting_resource_certificate(
        five_cuts,
        max_partition_size=3,
        shots_per_fragment=256,
    )
    assert total_over_budget.fragment_evaluations == 1024
    assert any("estimated_total_shots" in item for item in total_over_budget.blockers)


def test_resource_certificate_requires_no_submit_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drifted hardware-safe policy cannot open a submission-capable cutting plan."""
    policy = get_execution_policy("default_no_submit")
    monkeypatch.setattr(
        cutting,
        "get_execution_policy",
        lambda policy_id: replace(policy, no_submit=False, owner_allow_submit=True),
    )
    certificate = build_cutting_resource_certificate(build_knm_paper27(L=4), max_partition_size=4)
    assert certificate.feasible is False
    assert any("no-submit policy" in item for item in certificate.blockers)


def test_dense_large_n_fails_closed_as_unbounded() -> None:
    """Dense N=32 cutting is not promoted when 4^cuts exceeds the finite model."""
    certificate = build_cutting_resource_certificate(
        build_knm_paper27(L=32),
        max_partition_size=16,
    )
    assert certificate.fragment_evaluations is None
    assert certificate.estimated_total_shots is None
    assert certificate.feasible is False
    assert any("finite planner bound" in item for item in certificate.blockers)


def test_resource_certificate_validates_shots_and_target() -> None:
    """Shot types and target capacity are explicit contracts."""
    coupling = build_knm_paper27(L=4)
    with pytest.raises(TypeError, match="integer"):
        build_cutting_resource_certificate(coupling, shots_per_fragment=cast(int, 1.5))
    with pytest.raises(ValueError, match="positive"):
        build_cutting_resource_certificate(coupling, shots_per_fragment=0)
    refused = build_cutting_resource_certificate(
        coupling,
        max_partition_size=4,
        target_qubits=2,
    )
    assert refused.fits_target is False
    assert any("target-qubit" in item for item in refused.blockers)


def test_synthetic_reconstruction_certificate_pass_and_fail() -> None:
    """Synthetic evidence reports observed error without becoming hardware evidence."""
    passed = certify_synthetic_reconstruction(
        observable_id="R_global",
        exact_value=0.8,
        reconstructed_value=0.79,
        declared_error_bound=0.02,
    )
    assert passed.schema == CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA
    assert passed.absolute_error == pytest.approx(0.01)
    assert passed.within_bound is True
    assert passed.synthetic_only is True
    assert passed.hardware_result is False
    assert passed.to_dict()["hardware_result"] is False

    failed = certify_synthetic_reconstruction(
        observable_id="energy",
        exact_value=1.0,
        reconstructed_value=1.5,
        declared_error_bound=0.1,
    )
    assert failed.within_bound is False


def test_synthetic_reconstruction_rejects_bad_inputs() -> None:
    """Non-finite values and negative declared bounds fail closed."""
    with pytest.raises(ValueError, match="finite"):
        certify_synthetic_reconstruction(
            observable_id="R",
            exact_value=np.nan,
            reconstructed_value=0.0,
            declared_error_bound=0.1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        certify_synthetic_reconstruction(
            observable_id="R",
            exact_value=0.0,
            reconstructed_value=0.0,
            declared_error_bound=-0.1,
        )
    with pytest.raises(ValueError, match="observable_id"):
        certify_synthetic_reconstruction(
            observable_id=" ",
            exact_value=0.0,
            reconstructed_value=0.0,
            declared_error_bound=0.0,
        )


def test_path_decisions_cover_allowed_and_refused_boundaries() -> None:
    """Planning/synthetic paths are bounded; energy and live paths refuse."""
    resource = build_cutting_resource_certificate(
        np.zeros((4, 4), dtype=np.float64), max_partition_size=2
    )
    reconstruction = certify_synthetic_reconstruction(
        observable_id="R_global",
        exact_value=0.5,
        reconstructed_value=0.49,
        declared_error_bound=0.02,
    )
    dry = decide_cutting_path("dry_run_plan", resource=resource)
    assert dry.allowed is True
    assert dry.to_dict()["blockers"] == []
    synthetic = decide_cutting_path(
        "synthetic_reconstruction",
        resource=resource,
        reconstruction=reconstruction,
    )
    assert synthetic.allowed is True
    local = decide_cutting_path(
        "partition_local_diagnostic",
        resource=resource,
        accept_partition_local_energy=True,
    )
    assert local.allowed is True

    missing = decide_cutting_path("synthetic_reconstruction", resource=resource)
    assert missing.allowed is False
    rejected_local = decide_cutting_path("partition_local_diagnostic", resource=resource)
    assert rejected_local.allowed is False
    full = decide_cutting_path("full_system_energy", resource=resource)
    assert full.allowed is False
    live = decide_cutting_path("live_submit", resource=resource)
    assert live.allowed is False


def test_failed_reconstruction_and_resource_propagate_blockers() -> None:
    """Failed synthetic evidence and resource plans cannot become green decisions."""
    resource = build_cutting_resource_certificate(build_knm_paper27(L=32), max_partition_size=16)
    reconstruction = certify_synthetic_reconstruction(
        observable_id="R",
        exact_value=0.0,
        reconstructed_value=1.0,
        declared_error_bound=0.1,
    )
    decision = decide_cutting_path(
        "synthetic_reconstruction",
        resource=resource,
        reconstruction=reconstruction,
    )
    assert decision.allowed is False
    assert any("finite planner" in item for item in decision.blockers)
    assert any("exceeds" in item for item in decision.blockers)
    with pytest.raises(ValueError, match="unknown circuit-cutting path"):
        decide_cutting_path(cast(cutting.CuttingPath, "ghost"), resource=resource)


def test_dataclass_invariants_and_integrity_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed records and drifted registries fail closed."""
    with pytest.raises(ValueError, match="surface_id"):
        CuttingSurfaceRow("", "pointer", "bounded_planner", "summary", False)
    with pytest.raises(ValueError, match="authority_pointer"):
        CuttingSurfaceRow("id", "", "bounded_planner", "summary", False)
    with pytest.raises(ValueError, match="support_posture"):
        CuttingSurfaceRow("id", "pointer", "bad", "summary", False)
    with pytest.raises(ValueError, match="summary"):
        CuttingSurfaceRow("id", "pointer", "bounded_planner", "", False)
    with pytest.raises(ValueError, match="cannot submit"):
        CuttingSurfaceRow("id", "pointer", "bounded_planner", "summary", False, True)

    base = CuttingResourceCertificate(
        schema=CIRCUIT_CUTTING_RESOURCE_SCHEMA,
        policy_id="policy",
        n_oscillators=4,
        n_partitions=2,
        partition_sizes=(2, 2),
        n_cuts=1,
        fragment_evaluations=4,
        shots_per_fragment=10,
        estimated_total_shots=40,
        cost_model_status="unavailable",
        fits_target=True,
        feasible=True,
        outcome="allowed_plan",
        reason="ok",
        blockers=(),
    )
    with pytest.raises(ValueError, match="schema"):
        replace(base, schema="circuit_cutting_resource.v1")
    with pytest.raises(ValueError, match="policy_id"):
        replace(base, policy_id="")
    with pytest.raises(ValueError, match="dimensions"):
        replace(base, n_cuts=-1)
    with pytest.raises(ValueError, match="cover"):
        replace(base, partition_sizes=(2,))
    with pytest.raises(ValueError, match="positive"):
        replace(base, shots_per_fragment=0)
    with pytest.raises(ValueError, match="unbounded"):
        replace(base, fragment_evaluations=None)
    with pytest.raises(ValueError, match="fragments times shots"):
        replace(base, estimated_total_shots=41)
    with pytest.raises(ValueError, match="feasible must agree"):
        replace(base, outcome="refused")
    with pytest.raises(ValueError, match="cannot list"):
        replace(base, blockers=("bad",))
    with pytest.raises(ValueError, match="require blockers"):
        replace(base, feasible=False, outcome="refused")
    with pytest.raises(ValueError, match="reason"):
        replace(base, reason="")
    with pytest.raises(ValueError, match="no-submit"):
        replace(base, no_submit=False)

    with pytest.raises(ValueError, match="schema"):
        SyntheticReconstructionCertificate(
            "circuit_cutting_reconstruction.v1", "R", 0, 0, 0, 0, True
        )
    with pytest.raises(ValueError, match="finite"):
        SyntheticReconstructionCertificate(
            CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA, "R", 0, 0, np.inf, 0, True
        )
    with pytest.raises(ValueError, match="non-negative"):
        SyntheticReconstructionCertificate(
            CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA, "R", 0, 0, -1, 0, True
        )
    with pytest.raises(ValueError, match="must match"):
        SyntheticReconstructionCertificate(
            CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA, "R", 0, 1, 0, 1, True
        )
    with pytest.raises(ValueError, match="within_bound"):
        SyntheticReconstructionCertificate(
            CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA, "R", 0, 1, 1, 1, False
        )
    with pytest.raises(ValueError, match="synthetic-only"):
        SyntheticReconstructionCertificate(
            CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA,
            "R",
            0,
            0,
            0,
            0,
            True,
            synthetic_only=False,
        )

    decision = CuttingPathDecision(
        path="dry_run_plan",
        outcome="allowed",
        allowed=True,
        reason="ok",
        blockers=(),
    )
    with pytest.raises(ValueError, match="unknown path"):
        replace(decision, path=cast(cutting.CuttingPath, "bad"))
    with pytest.raises(ValueError, match="unknown outcome"):
        replace(decision, outcome=cast(cutting.DecisionOutcome, "bad"))
    with pytest.raises(ValueError, match="agree"):
        replace(decision, allowed=False)
    with pytest.raises(ValueError, match="cannot list"):
        replace(decision, blockers=("bad",))
    with pytest.raises(ValueError, match="require blockers"):
        CuttingPathDecision(
            path="dry_run_plan",
            outcome="refused",
            allowed=False,
            reason="no",
            blockers=(),
        )
    with pytest.raises(ValueError, match="reason"):
        replace(decision, reason="")

    monkeypatch.setattr(cutting, "_SURFACES", (cutting._SURFACES[0],) * 2)
    with pytest.raises(RuntimeError, match="unique"):
        assert_circuit_cutting_product_integrity()


def test_integrity_rejects_submit_and_registry_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integrity checks reject submit and registry claim drift."""
    original_surfaces = cutting._SURFACES
    unsafe = object.__new__(CuttingSurfaceRow)
    object.__setattr__(unsafe, "surface_id", "unsafe")
    object.__setattr__(unsafe, "authority_pointer", "pointer")
    object.__setattr__(unsafe, "support_posture", "bounded_planner")
    object.__setattr__(unsafe, "summary", "unsafe")
    object.__setattr__(unsafe, "full_system_energy", False)
    object.__setattr__(unsafe, "live_submit", True)
    object.__setattr__(unsafe, "claim_boundary", cutting.CIRCUIT_CUTTING_CLAIM_BOUNDARY)
    monkeypatch.setattr(cutting, "_SURFACES", (unsafe,))
    with pytest.raises(RuntimeError, match="live submission"):
        assert_circuit_cutting_product_integrity()

    monkeypatch.setattr(cutting, "_SURFACES", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        assert_circuit_cutting_product_integrity()

    monkeypatch.setattr(cutting, "_SURFACES", original_surfaces)
    original_registry = cutting.build_circuit_cutting_product_registry
    monkeypatch.setattr(
        cutting,
        "build_circuit_cutting_product_registry",
        lambda: {**original_registry(), "schema": "circuit_cutting_product.v1"},
    )
    with pytest.raises(RuntimeError, match="schema drift"):
        assert_circuit_cutting_product_integrity()

    monkeypatch.setattr(
        cutting,
        "build_circuit_cutting_product_registry",
        lambda: {
            **original_registry(),
            "live_submit": True,
            "general_reconstruction": True,
        },
    )
    with pytest.raises(RuntimeError, match="claim boundary drift"):
        assert_circuit_cutting_product_integrity()


def test_full_system_energy_single_partition_is_allowed() -> None:
    """An uncut single partition may retain its full-system energy label."""
    resource = build_cutting_resource_certificate(build_knm_paper27(L=4), max_partition_size=4)
    decision = decide_cutting_path("full_system_energy", resource=resource)
    assert decision.allowed is True
