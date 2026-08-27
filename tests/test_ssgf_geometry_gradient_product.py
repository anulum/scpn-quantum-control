# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — geometry-gradient SSGF geometry-gradient product tests
"""Real-surface and fail-closed tests for the geometry-gradient product."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.ssgf_geometry_gradient_product as product
from scpn_quantum_control.ssgf.quantum_gradient import QuantumGradientResult
from scpn_quantum_control.ssgf_geometry_gradient_product import (
    MAX_OSCILLATORS,
    SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY,
    SSGF_GEOMETRY_GRADIENT_SCHEMA,
    GeometryGradientCertificate,
    SsgfGeometryObserverRecord,
    SsgfPublicSurfaceRow,
    assert_ssgf_geometry_gradient_integrity,
    build_ssgf_geometry_gradient_registry,
    certify_geometry_gradient,
    certify_quantum_cost,
    decide_ssgf_gradient_route,
    geometry_observer_from_certificate,
    list_ssgf_public_surfaces,
    materialise_outer_cycle_evidence,
    ssgf_gradient_unsuitable_scenarios,
)


def _problem() -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    return np.array([0.2, -0.4, 0.7]), np.array([0.1, 0.6, 1.4])


def test_inventory_freezes_real_ssgf_surfaces() -> None:
    """Keep the public simulator and bridge inventory complete."""
    rows = list_ssgf_public_surfaces()
    ids = {row.surface_id for row in rows}

    assert len(rows) == 8
    assert {"quantum_cost", "quantum_gradient", "quantum_outer_cycle"} <= ids
    assert {"hamiltonian_bridge", "state_bridge", "quantum_spectral"} <= ids
    assert all(row.hardware_submit_allowed is False for row in rows)
    assert all(row.claim_boundary == SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY for row in rows)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"surface_id": ""}, "surface_id"),
        ({"module_path": ""}, "module_path"),
        ({"symbol": ""}, "module_path"),
        ({"hardware_submit_allowed": True}, "hardware"),
    ],
)
def test_surface_row_validation(kwargs: dict[str, object], message: str) -> None:
    """Reject incomplete or hardware-enabled public surface rows."""
    values: dict[str, object] = {
        "surface_id": "x",
        "module_path": "m",
        "symbol": "s",
        "role": "quantum_cost",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SsgfPublicSurfaceRow(**values)  # type: ignore[arg-type]


def test_unsuitable_scenarios_cover_negative_space() -> None:
    """Expose the documented unsupported-use boundaries."""
    scenarios = ssgf_gradient_unsuitable_scenarios()

    assert len(scenarios) == 6
    assert any("parameter-shift" in item for item in scenarios)
    assert any("analytic" in item for item in scenarios)
    assert any("operational controller" in item for item in scenarios)


def test_route_policy_supports_fd_and_refuses_latent_parameter_shift() -> None:
    """Allow finite differences and refuse the invalid latent shift route."""
    finite_difference = decide_ssgf_gradient_route(" Finite_Difference ")
    parameter_shift = decide_ssgf_gradient_route("parameter_shift")

    assert finite_difference.allowed is True
    assert finite_difference.route_id == "transform:ssgf.latent_finite_difference"
    assert finite_difference.blockers == ()
    assert parameter_shift.allowed is False
    assert parameter_shift.route_id == "transform:ssgf.latent_parameter_shift"
    assert "softplus" in parameter_shift.blockers[0]


def test_route_policy_rejects_unknown_and_detects_matrix_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unknown methods and drift from the governed route matrix."""
    with pytest.raises(ValueError, match="method"):
        decide_ssgf_gradient_route("adjoint")
    with pytest.raises(ValueError, match="method"):
        decide_ssgf_gradient_route(cast(str, None))

    monkeypatch.setattr(
        product,
        "get_governed_route",
        lambda _route_id: SimpleNamespace(
            closure_status="implementation_path", closure_reason="x"
        ),
    )
    with pytest.raises(RuntimeError, match="not supported"):
        decide_ssgf_gradient_route("finite_difference")
    with pytest.raises(RuntimeError, match="not a permanent boundary"):
        decide_ssgf_gradient_route("parameter_shift")


def test_real_quantum_cost_certificate_cross_checks_c_equals_one_minus_r() -> None:
    """Cross-check the real cost bundle against its complement law."""
    z, theta = _problem()
    certificate = certify_quantum_cost(z, 3, theta, trotter_reps=1)

    assert certificate.schema == SSGF_GEOMETRY_GRADIENT_SCHEMA
    assert certificate.n_parameters == 3
    assert certificate.cost == pytest.approx(1.0 - certificate.r_global, abs=1e-12)
    assert certificate.cost == pytest.approx(certificate.c_micro, abs=1e-12)
    assert certificate.complement_residual <= 1e-12
    assert certificate.cross_surface_residual <= 1e-12
    assert certificate.geometry_symmetry_residual == 0.0
    assert len(certificate.certificate_digest) == 64
    assert certificate.to_dict()["schema"] == SSGF_GEOMETRY_GRADIENT_SCHEMA


def test_cost_certificate_digest_is_deterministic_and_omega_aware() -> None:
    """Keep the cost digest deterministic for explicit frequencies."""
    z, theta = _problem()
    omega = np.array([-0.1, 0.0, 0.1])
    first = certify_quantum_cost(z, 3, theta, omega=omega, trotter_reps=1)
    again = certify_quantum_cost(z, 3, theta, omega=omega, trotter_reps=1)

    assert again.certificate_digest == first.certificate_digest
    assert again.cost == pytest.approx(first.cost)


@pytest.mark.parametrize(
    ("z", "n", "theta", "omega", "kwargs", "message"),
    [
        (np.zeros(1), cast(int, 2.0), np.zeros(2), None, {}, "integer"),
        (np.zeros(0), 1, np.zeros(1), None, {}, r"\[2"),
        (np.zeros(1), MAX_OSCILLATORS + 1, np.zeros(MAX_OSCILLATORS + 1), None, {}, r"\[2"),
        (np.zeros(2), 3, np.zeros(3), None, {}, "z shape"),
        (np.zeros(3), 3, np.zeros(2), None, {}, "theta_init shape"),
        (np.zeros(3), 3, np.zeros(3), np.zeros(2), {}, "omega shape"),
        (np.array([np.nan, 0.0, 0.0]), 3, np.zeros(3), None, {}, "z must"),
        (np.zeros(3), 3, np.array([0.0, np.inf, 0.0]), None, {}, "theta_init must"),
        (np.zeros(3), 3, np.zeros(3), np.array([0.0, np.nan, 0.0]), {}, "omega must"),
        (np.zeros(3), 3, np.zeros(3), None, {"dt": 0.0}, "dt"),
        (np.zeros(3), 3, np.zeros(3), None, {"trotter_reps": 0}, "trotter_reps"),
        (np.zeros(3), 3, np.zeros(3), None, {"trotter_reps": cast(int, True)}, "trotter_reps"),
    ],
)
def test_problem_contract_fails_before_ambient_evaluation(
    z: np.ndarray[Any, np.dtype[np.float64]],
    n: int,
    theta: np.ndarray[Any, np.dtype[np.float64]],
    omega: np.ndarray[Any, np.dtype[np.float64]] | None,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Reject invalid bounded-problem inputs before simulator evaluation."""
    with pytest.raises((TypeError, ValueError), match=message):
        certify_quantum_cost(z, n, theta, omega=omega, **kwargs)  # type: ignore[arg-type]


def test_cost_certificate_rejects_bad_atol_and_ambient_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid tolerance and malformed ambient cost results."""
    z = np.array([0.0])
    theta = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="atol"):
        certify_quantum_cost(z, 2, theta, atol=-1.0)

    monkeypatch.setattr(product, "quantum_cost", lambda *_args: float("nan"))
    monkeypatch.setattr(
        product,
        "compute_quantum_costs",
        lambda *_args: SimpleNamespace(r_global=0.5, c_micro=0.5),
    )
    with pytest.raises(ValueError, match="non-finite"):
        certify_quantum_cost(z, 2, theta)

    monkeypatch.setattr(product, "quantum_cost", lambda *_args: 1.2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        certify_quantum_cost(z, 2, theta)

    monkeypatch.setattr(product, "quantum_cost", lambda *_args: 0.4)
    monkeypatch.setattr(
        product,
        "compute_quantum_costs",
        lambda *_args: SimpleNamespace(r_global=0.5, c_micro=0.5),
    )
    with pytest.raises(ValueError, match="cross-surface"):
        certify_quantum_cost(z, 2, theta, atol=1e-12)


def test_cost_certificate_rejects_invalid_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject asymmetric or negative latent geometry."""
    monkeypatch.setattr(product, "_w_from_z", lambda *_args: np.array([[0.0, -1.0], [0.0, 0.0]]))
    monkeypatch.setattr(product, "quantum_cost", lambda *_args: 0.5)
    monkeypatch.setattr(
        product,
        "compute_quantum_costs",
        lambda *_args: SimpleNamespace(r_global=0.5, c_micro=0.5),
    )
    with pytest.raises(ValueError, match="symmetric"):
        certify_quantum_cost(np.zeros(1), 2, np.zeros(2))


def test_real_gradient_certificate_enforces_metamorphic_laws() -> None:
    """Certify refinement and phase-periodicity on the real gradient path."""
    z, theta = _problem()
    certificate = certify_geometry_gradient(z, 3, theta, trotter_reps=1)

    assert isinstance(certificate, GeometryGradientCertificate)
    assert certificate.method == "finite_difference"
    assert certificate.n_parameters == 3
    assert certificate.expected_evaluations_per_gradient == 7
    assert certificate.n_evaluations == 21
    assert certificate.gradient_norm > 0.0
    assert certificate.refinement_max_abs_delta < 1e-6
    assert certificate.periodic_cost_residual < 1e-12
    assert certificate.periodic_gradient_max_abs_delta < 1e-12
    assert certificate.geometry_symmetry_residual == 0.0
    assert certificate.to_dict()["route_id"] == certificate.route_id


def test_gradient_certificate_rejects_parameter_shift_and_bad_tolerances() -> None:
    """Reject the unsupported derivative route and invalid tolerances."""
    z, theta = _problem()
    with pytest.raises(ValueError, match="refused"):
        certify_geometry_gradient(z, 3, theta, method="parameter_shift")
    with pytest.raises(ValueError, match="refinement_atol"):
        certify_geometry_gradient(z, 3, theta, refinement_atol=-1.0)
    with pytest.raises(ValueError, match="periodicity_atol"):
        certify_geometry_gradient(z, 3, theta, periodicity_atol=float("inf"))
    with pytest.raises(ValueError, match="epsilon"):
        certify_geometry_gradient(z, 3, theta, epsilon=0.0)


def _gradient_result(
    gradient: np.ndarray[Any, np.dtype[np.float64]],
    *,
    cost: float = 0.4,
    r_global: float = 0.6,
    n_evaluations: int = 3,
) -> QuantumGradientResult:
    return QuantumGradientResult(
        cost=cost,
        gradient=gradient,
        r_global=r_global,
        n_evaluations=n_evaluations,
    )


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([_gradient_result(np.zeros(2))] * 3, "shape drift"),
        ([_gradient_result(np.zeros(1), n_evaluations=2)] * 3, "evaluation-count"),
        ([_gradient_result(np.array([np.nan]))] * 3, "non-finite"),
        ([_gradient_result(np.zeros(1), cost=float("inf"))] * 3, "non-finite"),
    ],
)
def test_gradient_certificate_rejects_ambient_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
    results: list[QuantumGradientResult],
    message: str,
) -> None:
    """Reject malformed gradient results from the ambient simulator."""
    iterator = iter(results)
    monkeypatch.setattr(
        product, "compute_quantum_gradient", lambda *_args, **_kwargs: next(iterator)
    )
    with pytest.raises(ValueError, match=message):
        certify_geometry_gradient(np.zeros(1), 2, np.array([0.1, 0.2]))


def test_gradient_certificate_rejects_refinement_and_periodicity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject failed refinement and periodicity metamorphic laws."""
    primary = _gradient_result(np.array([0.0]))
    refined = _gradient_result(np.array([1.0]))
    periodic = _gradient_result(np.array([0.0]))
    iterator = iter((primary, refined, periodic))
    monkeypatch.setattr(
        product, "compute_quantum_gradient", lambda *_args, **_kwargs: next(iterator)
    )
    with pytest.raises(ValueError, match="step-refinement"):
        certify_geometry_gradient(np.zeros(1), 2, np.array([0.1, 0.2]))

    periodic_bad = _gradient_result(np.array([0.1]), cost=0.5)
    iterator = iter((primary, primary, periodic_bad))
    monkeypatch.setattr(
        product, "compute_quantum_gradient", lambda *_args, **_kwargs: next(iterator)
    )
    with pytest.raises(ValueError, match="phase-periodicity"):
        certify_geometry_gradient(np.zeros(1), 2, np.array([0.1, 0.2]))


def test_geometry_observer_composes_control_and_codesign_telemetry() -> None:
    """Map a certificate into non-operational observer telemetry."""
    z, theta = _problem()
    certificate = certify_geometry_gradient(z, 3, theta, trotter_reps=1)
    observer = geometry_observer_from_certificate(certificate)

    assert observer.cost == certificate.cost
    assert observer.gradient_norm == certificate.gradient_norm
    assert observer.geometry_symmetry_residual == certificate.geometry_symmetry_residual
    assert observer.operational_control_claim is False
    assert observer.to_dict()["schema"] == SSGF_GEOMETRY_GRADIENT_SCHEMA
    with pytest.raises(ValueError, match="operational"):
        replace(observer, operational_control_claim=True)


def test_real_outer_cycle_evidence_is_functional_and_deterministic() -> None:
    """Keep the bounded outer-cycle evidence deterministic and functional."""
    z, theta = _problem()
    first = materialise_outer_cycle_evidence(
        n_oscillators=3,
        z_init=z,
        theta_init=theta,
        max_iterations=3,
    )
    again = materialise_outer_cycle_evidence(
        n_oscillators=3,
        z_init=z,
        theta_init=theta,
        max_iterations=3,
    )

    assert first.n_parameters == 3
    assert first.n_iterations >= 1
    assert first.evidence_label == "functional_non_isolated_local_simulation"
    assert first.geometry_symmetry_residual == 0.0
    assert first.minimum_coupling >= 0.0
    assert first.cost_delta <= 0.0
    assert first.evidence_digest == again.evidence_digest
    assert first.to_dict()["schema"] == SSGF_GEOMETRY_GRADIENT_SCHEMA


def test_outer_cycle_defaults_are_bounded() -> None:
    """Keep the default outer-cycle probe small and bounded."""
    evidence = materialise_outer_cycle_evidence(max_iterations=1)

    assert evidence.n_oscillators == 2
    assert evidence.n_parameters == 1
    assert evidence.n_iterations == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_oscillators": 1}, r"\[2"),
        ({"n_oscillators": MAX_OSCILLATORS + 1}, r"\[2"),
        ({"z_init": np.zeros(2)}, "z shape"),
        ({"theta_init": np.zeros(3)}, "theta_init shape"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": cast(int, True)}, "max_iterations"),
        ({"convergence_threshold": -1.0}, "convergence_threshold"),
    ],
)
def test_outer_cycle_input_guards(kwargs: dict[str, object], message: str) -> None:
    """Reject invalid outer-cycle inputs before execution."""
    with pytest.raises(ValueError, match=message):
        materialise_outer_cycle_evidence(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            SimpleNamespace(
                cost_history=[],
                r_global_history=[],
                n_iterations=0,
                W_optimised=np.zeros((2, 2)),
            ),
            "history contract",
        ),
        (
            SimpleNamespace(
                cost_history=[float("nan")],
                r_global_history=[0.5],
                n_iterations=1,
                W_optimised=np.zeros((2, 2)),
            ),
            "non-finite",
        ),
        (
            SimpleNamespace(
                cost_history=[0.5],
                r_global_history=[1.2],
                n_iterations=1,
                W_optimised=np.zeros((2, 2)),
            ),
            r"\[0, 1\]",
        ),
        (
            SimpleNamespace(
                cost_history=[0.5],
                r_global_history=[0.5],
                n_iterations=1,
                W_optimised=np.array([[0.0, -1.0], [0.0, 0.0]]),
            ),
            "geometry contract",
        ),
    ],
)
def test_outer_cycle_rejects_ambient_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
    result: SimpleNamespace,
    message: str,
) -> None:
    """Reject malformed histories and geometry from the ambient cycle."""
    monkeypatch.setattr(product, "quantum_outer_cycle", lambda **_kwargs: result)
    with pytest.raises(ValueError, match=message):
        materialise_outer_cycle_evidence()


def test_registry_is_complete_and_fail_closed() -> None:
    """Keep the public registry complete and fail closed."""
    registry = assert_ssgf_geometry_gradient_integrity()

    assert registry["schema"] == SSGF_GEOMETRY_GRADIENT_SCHEMA
    assert registry["surface_count"] == 8
    assert registry["hardware_submit_allowed"] is False
    assert registry["analytic_ad_claim_allowed"] is False
    assert registry["parameter_shift_on_latent_z_allowed"] is False
    assert registry["composition"]["geometric_control"]  # type: ignore[index]
    explicit = assert_ssgf_geometry_gradient_integrity(build_ssgf_geometry_gradient_registry())
    assert explicit == registry


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda r: r.update(surfaces=[]), "non-empty surfaces"),
        (lambda r: r.update(gradient_routes=[]), "exactly two"),
        (lambda r: r.update(unsuitable_scenarios=[]), "unsuitable"),
        (lambda r: cast(list[object], r["surfaces"]).__setitem__(0, "bad"), "mappings"),
        (
            lambda r: cast(list[dict[str, object]], r["surfaces"])[0].update(surface_id=""),
            "blank surface_id",
        ),
        (
            lambda r: cast(list[dict[str, object]], r["surfaces"])[1].update(
                surface_id=cast(list[dict[str, object]], r["surfaces"])[0]["surface_id"]
            ),
            "duplicate",
        ),
        (
            lambda r: cast(list[dict[str, object]], r["surfaces"])[0].update(
                hardware_submit_allowed=True
            ),
            "hardware_submit_allowed",
        ),
        (
            lambda r: cast(list[dict[str, object]], r["surfaces"]).pop(),
            "inventory drift",
        ),
        (
            lambda r: cast(list[dict[str, object]], r["gradient_routes"])[0].update(
                method="ghost"
            ),
            "method set drift",
        ),
        (
            lambda r: cast(list[dict[str, object]], r["gradient_routes"])[0].update(allowed=False),
            "finite-difference",
        ),
        (
            lambda r: cast(list[dict[str, object]], r["gradient_routes"])[1].update(allowed=True),
            "parameter-shift",
        ),
        (lambda r: r.update(hardware_submit_allowed=True), "policy"),
        (lambda r: r.update(analytic_ad_claim_allowed=True), "policy"),
        (lambda r: r.update(parameter_shift_on_latent_z_allowed=True), "policy"),
        (lambda r: r.update(surface_count=99), "surface_count"),
        (lambda r: r.update(blank_entry_count=1), "blank_entry_count"),
    ],
)
def test_registry_integrity_rejects_drift(
    mutator: Any,
    message: str,
) -> None:
    """Reject registry inventory and policy drift."""
    registry = build_ssgf_geometry_gradient_registry()
    mutator(registry)
    with pytest.raises(ValueError, match=message):
        assert_ssgf_geometry_gradient_integrity(registry)


def test_public_exports_are_explicit() -> None:
    """Keep the public exports, schema, and observer boundary explicit."""
    assert "certify_geometry_gradient" in product.__all__
    assert "materialise_outer_cycle_evidence" in product.__all__
    assert SSGF_GEOMETRY_GRADIENT_SCHEMA == "ssgf_geometry_gradient_product.v1"
    observer = SsgfGeometryObserverRecord(
        cost=0.5,
        r_global=0.5,
        gradient_norm=0.0,
        geometry_symmetry_residual=0.0,
        method="finite_difference",
        route_id="transform:ssgf.latent_finite_difference",
    )
    assert observer.claim_boundary == SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY
