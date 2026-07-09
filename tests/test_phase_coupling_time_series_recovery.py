# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coupling-Recovery Tests
"""Tests for bounded Kuramoto/XY coupling time-series recovery."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import (
    COUPLING_RECOVERY_CLAIM_BOUNDARY,
    COUPLING_RECOVERY_EVIDENCE_CLASS,
    CouplingRecoveryBoundaryRow,
    CouplingRecoveryCase,
    CouplingRecoveryRecord,
    CouplingRecoverySuiteResult,
    coupling_recovery_boundary_rows,
    default_coupling_recovery_cases,
    inject_time_series_noise_and_missing,
    recover_kuramoto_couplings_from_time_series,
    recover_xy_couplings_from_pair_energy_series,
    run_coupling_recovery_suite,
    simulate_kuramoto_phase_time_series,
    simulate_xy_pair_energy_time_series,
)
from scpn_quantum_control.phase.coupling_time_series_recovery import (
    BoundaryStatus,
    CouplingRecoveryFamily,
    _normalise_edges,
    _solve_ridge,
)


@pytest.fixture(scope="module")
def ground_truth() -> NDArray[np.float64]:
    """Return the default BL-17 ground-truth coupling matrix."""
    return default_coupling_recovery_cases()[0].true_couplings


def test_default_coupling_recovery_suite_passes() -> None:
    suite = run_coupling_recovery_suite()

    assert suite.passed
    assert suite.evidence_class == COUPLING_RECOVERY_EVIDENCE_CLASS
    assert suite.claim_boundary == COUPLING_RECOVERY_CLAIM_BOUNDARY
    assert {record.family for record in suite.records} == {"kuramoto_phase", "xy_pair_energy"}
    assert len(suite.records_for_family("kuramoto_phase")) == 2
    assert len(suite.records_for_family("xy_pair_energy")) == 1
    assert suite.records_for_family("kuramoto_phase")[0].max_abs_error < 0.01
    assert suite.boundary_rows == coupling_recovery_boundary_rows()


def test_coupling_recovery_suite_serializes_records_and_boundaries() -> None:
    payload = run_coupling_recovery_suite().to_dict()

    assert payload["passed"] is True
    assert payload["evidence_class"] == "functional_non_isolated"
    rows = cast(list[dict[str, object]], payload["records"])
    boundaries = cast(list[dict[str, object]], payload["boundary_rows"])
    assert rows[0]["case_id"] == "kuramoto_clean_three_node"
    assert rows[1]["missing_fraction"] == pytest.approx(0.03)
    assert boundaries[0]["boundary_id"] == "partial_observation_inference_boundary"
    assert boundaries[1]["status"] == "hard_gap"


def test_kuramoto_clean_time_series_recovers_ground_truth(
    ground_truth: NDArray[np.float64],
) -> None:
    case = default_coupling_recovery_cases()[0]
    phases = simulate_kuramoto_phase_time_series(
        ground_truth,
        case.omega,
        case.theta0,
        dt=case.dt,
        n_steps=case.n_steps,
    )

    record = recover_kuramoto_couplings_from_time_series(
        phases,
        case.omega,
        ground_truth,
        dt=case.dt,
        tolerance=case.tolerance,
    )

    assert record.passed
    assert record.valid_fraction == 1.0
    assert record.design_rank == 3
    assert record.max_abs_error <= case.tolerance
    assert np.max(np.abs(record.learned_couplings - ground_truth)) < 0.002


def test_noisy_missing_kuramoto_and_xy_cases_pass() -> None:
    records = run_coupling_recovery_suite().records

    noisy_kuramoto = next(
        record for record in records if record.case_id.startswith("kuramoto_noisy")
    )
    xy_record = next(record for record in records if record.family == "xy_pair_energy")

    assert noisy_kuramoto.passed
    assert noisy_kuramoto.valid_fraction < 1.0
    assert xy_record.passed
    assert xy_record.valid_fraction < 1.0
    assert xy_record.rmse < xy_record.tolerance


def test_xy_pair_energy_recovery_supports_edge_subsets(
    ground_truth: NDArray[np.float64],
) -> None:
    case = default_coupling_recovery_cases()[2]
    phases = simulate_kuramoto_phase_time_series(
        ground_truth,
        case.omega,
        case.theta0,
        dt=case.dt,
        n_steps=case.n_steps,
    )
    pair_energy = simulate_xy_pair_energy_time_series(ground_truth, phases)

    record = recover_xy_couplings_from_pair_energy_series(
        pair_energy,
        phases,
        ground_truth,
        edges=((0, 1), (2, 1)),
        tolerance=case.tolerance,
    )

    assert record.passed
    assert record.learned_couplings[0, 1] == pytest.approx(ground_truth[0, 1], abs=1.0e-8)
    assert record.learned_couplings[1, 2] == pytest.approx(ground_truth[1, 2], abs=1.0e-8)
    assert record.learned_couplings[0, 2] == 0.0

    phases_with_missing_row = phases.copy()
    phases_with_missing_row[0, 0] = np.nan
    missing_row_record = recover_xy_couplings_from_pair_energy_series(
        pair_energy,
        phases_with_missing_row,
        ground_truth,
        tolerance=case.tolerance,
    )
    assert missing_row_record.passed
    assert missing_row_record.valid_fraction < 1.0


def test_noise_and_missing_injection_is_deterministic() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)

    first = inject_time_series_noise_and_missing(
        values,
        noise_std=0.1,
        missing_fraction=0.25,
        seed=123,
    )
    second = inject_time_series_noise_and_missing(
        values,
        noise_std=0.1,
        missing_fraction=0.25,
        seed=123,
    )

    assert np.array_equal(first, second, equal_nan=True)
    assert np.isnan(first).any()
    assert np.nanmax(np.abs(first - values)) > 0.0


def test_public_case_and_record_to_dict_contract() -> None:
    case = default_coupling_recovery_cases()[0]
    record = run_coupling_recovery_suite(cases=(case,)).records[0]

    assert case.to_dict()["case_id"] == "kuramoto_clean_three_node"
    assert record.to_dict()["family"] == "kuramoto_phase"
    assert record.to_dict()["passed"] is True
    assert record.claim_boundary == COUPLING_RECOVERY_CLAIM_BOUNDARY


def test_boundary_row_contracts() -> None:
    row = CouplingRecoveryBoundaryRow(
        boundary_id="manual-boundary",
        status="hard_gap",
        reason="requires a separate identifiability proof",
    )

    assert row.to_dict()["status"] == "hard_gap"
    with pytest.raises(ValueError, match="boundary_id"):
        CouplingRecoveryBoundaryRow(boundary_id="", status="hard_gap", reason="x")
    with pytest.raises(ValueError, match="status"):
        CouplingRecoveryBoundaryRow(
            boundary_id="bad",
            status=cast(BoundaryStatus, "open"),
            reason="x",
        )
    with pytest.raises(ValueError, match="reason"):
        CouplingRecoveryBoundaryRow(boundary_id="bad", status="hard_gap", reason="")
    with pytest.raises(ValueError, match="claim_boundary"):
        CouplingRecoveryBoundaryRow(
            boundary_id="bad",
            status="hard_gap",
            reason="x",
            claim_boundary="",
        )


def _case(
    *,
    case_id: str | None = None,
    family: object | None = None,
    true_couplings: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
    theta0: NDArray[np.float64] | None = None,
    dt: float | None = None,
    n_steps: int | None = None,
    noise_std: float | None = None,
    missing_fraction: float | None = None,
    seed: object | None = None,
    tolerance: float | None = None,
) -> CouplingRecoveryCase:
    base = default_coupling_recovery_cases()[0]
    return CouplingRecoveryCase(
        case_id=base.case_id if case_id is None else case_id,
        family=base.family if family is None else cast(CouplingRecoveryFamily, family),
        true_couplings=base.true_couplings if true_couplings is None else true_couplings,
        omega=base.omega if omega is None else omega,
        theta0=base.theta0 if theta0 is None else theta0,
        dt=base.dt if dt is None else dt,
        n_steps=base.n_steps if n_steps is None else n_steps,
        noise_std=base.noise_std if noise_std is None else noise_std,
        missing_fraction=base.missing_fraction if missing_fraction is None else missing_fraction,
        seed=base.seed if seed is None else cast(int, seed),
        tolerance=base.tolerance if tolerance is None else tolerance,
    )


def _record(
    base: CouplingRecoveryRecord,
    *,
    case_id: str | None = None,
    family: object | None = None,
    learned_couplings: NDArray[np.float64] | None = None,
    true_couplings: NDArray[np.float64] | None = None,
    abs_error: NDArray[np.float64] | None = None,
    max_abs_error: float | None = None,
    rmse: float | None = None,
    valid_fraction: float | None = None,
    design_rank: int | None = None,
    condition_number: float | None = None,
    noise_std: float | None = None,
    missing_fraction: float | None = None,
    tolerance: float | None = None,
    passed: object | None = None,
    claim_boundary: str | None = None,
) -> CouplingRecoveryRecord:
    return CouplingRecoveryRecord(
        case_id=base.case_id if case_id is None else case_id,
        family=base.family if family is None else cast(CouplingRecoveryFamily, family),
        learned_couplings=base.learned_couplings
        if learned_couplings is None
        else learned_couplings,
        true_couplings=base.true_couplings if true_couplings is None else true_couplings,
        abs_error=base.abs_error if abs_error is None else abs_error,
        max_abs_error=base.max_abs_error if max_abs_error is None else max_abs_error,
        rmse=base.rmse if rmse is None else rmse,
        valid_fraction=base.valid_fraction if valid_fraction is None else valid_fraction,
        design_rank=base.design_rank if design_rank is None else design_rank,
        condition_number=(base.condition_number if condition_number is None else condition_number),
        noise_std=base.noise_std if noise_std is None else noise_std,
        missing_fraction=base.missing_fraction if missing_fraction is None else missing_fraction,
        tolerance=base.tolerance if tolerance is None else tolerance,
        passed=base.passed if passed is None else cast(bool, passed),
        claim_boundary=base.claim_boundary if claim_boundary is None else claim_boundary,
    )


def test_case_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="case_id"):
        _case(case_id="")
    with pytest.raises(ValueError, match="family"):
        _case(family="bad")
    with pytest.raises(ValueError, match="dimensions"):
        _case(omega=np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="omega must contain"):
        _case(omega=np.array([1.0, float("nan"), 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="theta0 must be a non-empty"):
        _case(theta0=np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="true_couplings must be a square"):
        _case(true_couplings=np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="at least two nodes"):
        _case(true_couplings=np.array([[0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="only finite"):
        _case(
            true_couplings=np.array(
                [[0.0, float("nan")], [float("nan"), 0.0]],
                dtype=np.float64,
            ),
            omega=np.array([0.1, 0.2], dtype=np.float64),
            theta0=np.array([0.1, 0.2], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="symmetric"):
        _case(
            true_couplings=np.array([[0.0, 0.2], [0.3, 0.0]], dtype=np.float64),
            omega=np.array([0.1, 0.2], dtype=np.float64),
            theta0=np.array([0.1, 0.2], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="diagonal"):
        _case(
            true_couplings=np.array([[0.1, 0.2], [0.2, 0.0]], dtype=np.float64),
            omega=np.array([0.1, 0.2], dtype=np.float64),
            theta0=np.array([0.1, 0.2], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="dt"):
        _case(dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        _case(dt=cast(float, True))
    with pytest.raises(ValueError, match="noise_std"):
        _case(noise_std=float("nan"))
    with pytest.raises(ValueError, match="n_steps"):
        _case(n_steps=1)
    with pytest.raises(ValueError, match="noise_std"):
        _case(noise_std=-1.0)
    with pytest.raises(ValueError, match="missing_fraction"):
        _case(missing_fraction=1.0)
    with pytest.raises(ValueError, match="seed"):
        _case(seed=cast(int, 1.2))
    with pytest.raises(ValueError, match="tolerance"):
        _case(tolerance=0.0)


def test_recovery_record_validation_rejects_bad_inputs(
    ground_truth: NDArray[np.float64],
) -> None:
    valid = run_coupling_recovery_suite().records[0]
    with pytest.raises(ValueError, match="case_id"):
        _record(valid, case_id="")
    with pytest.raises(ValueError, match="family"):
        _record(valid, family="bad")
    with pytest.raises(ValueError, match="share one shape"):
        _record(valid, abs_error=ground_truth[:2, :2])
    with pytest.raises(ValueError, match="finite non-negative"):
        _record(valid, abs_error=np.full_like(valid.abs_error, -1.0))
    with pytest.raises(ValueError, match="non-negative"):
        _record(valid, max_abs_error=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        _record(valid, rmse=-1.0)
    with pytest.raises(ValueError, match="valid_fraction"):
        _record(valid, valid_fraction=1.5)
    with pytest.raises(ValueError, match="design_rank"):
        _record(valid, design_rank=0)
    with pytest.raises(ValueError, match="condition_number"):
        _record(valid, condition_number=0.5)
    with pytest.raises(ValueError, match="noise and missing"):
        _record(valid, noise_std=-1.0)
    with pytest.raises(ValueError, match="noise and missing"):
        _record(valid, missing_fraction=1.0)
    with pytest.raises(ValueError, match="tolerance"):
        _record(valid, tolerance=0.0)
    with pytest.raises(ValueError, match="passed"):
        _record(valid, passed="yes")
    with pytest.raises(ValueError, match="claim_boundary"):
        _record(valid, claim_boundary="")


def test_suite_validation_rejects_empty_records_and_boundaries() -> None:
    record = run_coupling_recovery_suite().records[0]
    boundary = coupling_recovery_boundary_rows()[0]

    with pytest.raises(ValueError, match="records"):
        CouplingRecoverySuiteResult(records=(), boundary_rows=(boundary,))
    with pytest.raises(ValueError, match="boundary_rows"):
        CouplingRecoverySuiteResult(records=(record,), boundary_rows=())
    with pytest.raises(ValueError, match="evidence_class"):
        CouplingRecoverySuiteResult(
            records=(record,), boundary_rows=(boundary,), evidence_class=""
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        CouplingRecoverySuiteResult(
            records=(record,), boundary_rows=(boundary,), claim_boundary=""
        )


def test_simulators_and_recovery_reject_invalid_shapes(
    ground_truth: NDArray[np.float64],
) -> None:
    case = default_coupling_recovery_cases()[0]
    phases = simulate_kuramoto_phase_time_series(
        ground_truth,
        case.omega,
        case.theta0,
        dt=case.dt,
        n_steps=case.n_steps,
    )

    with pytest.raises(ValueError, match="couplings, omega, and theta0"):
        simulate_kuramoto_phase_time_series(
            ground_truth, case.omega[:2], case.theta0, dt=0.1, n_steps=2
        )
    with pytest.raises(ValueError, match="dt"):
        simulate_kuramoto_phase_time_series(
            ground_truth, case.omega, case.theta0, dt=0.0, n_steps=2
        )
    with pytest.raises(ValueError, match="n_steps"):
        simulate_kuramoto_phase_time_series(
            ground_truth, case.omega, case.theta0, dt=0.1, n_steps=1
        )
    with pytest.raises(ValueError, match="phases must have shape"):
        simulate_xy_pair_energy_time_series(ground_truth, phases[:, :2])
    with pytest.raises(ValueError, match="phases must contain"):
        simulate_xy_pair_energy_time_series(ground_truth, np.full_like(phases, np.nan))
    with pytest.raises(ValueError, match="values must be finite"):
        inject_time_series_noise_and_missing(
            np.array([np.nan]), noise_std=0.0, missing_fraction=0.0, seed=1
        )
    with pytest.raises(ValueError, match="noise_std"):
        inject_time_series_noise_and_missing(
            np.array([1.0]), noise_std=-0.1, missing_fraction=0.0, seed=1
        )
    with pytest.raises(ValueError, match="missing_fraction"):
        inject_time_series_noise_and_missing(
            np.array([1.0]), noise_std=0.0, missing_fraction=1.0, seed=1
        )
    with pytest.raises(ValueError, match="seed"):
        inject_time_series_noise_and_missing(
            np.array([1.0]), noise_std=0.0, missing_fraction=0.0, seed=cast(int, 1.2)
        )


def test_recovery_rejects_invalid_metadata_and_missing_rows(
    ground_truth: NDArray[np.float64],
) -> None:
    case = default_coupling_recovery_cases()[0]
    phases = simulate_kuramoto_phase_time_series(
        ground_truth,
        case.omega,
        case.theta0,
        dt=case.dt,
        n_steps=case.n_steps,
    )
    pair_energy = simulate_xy_pair_energy_time_series(ground_truth, phases)

    with pytest.raises(ValueError, match="phases must have shape"):
        recover_kuramoto_couplings_from_time_series(
            phases[:, :2], case.omega, ground_truth, dt=case.dt
        )
    with pytest.raises(ValueError, match="omega dimension"):
        recover_kuramoto_couplings_from_time_series(
            phases, case.omega[:2], ground_truth, dt=case.dt
        )
    with pytest.raises(ValueError, match="dt"):
        recover_kuramoto_couplings_from_time_series(phases, case.omega, ground_truth, dt=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        recover_kuramoto_couplings_from_time_series(
            phases, case.omega, ground_truth, dt=case.dt, tolerance=0.0
        )
    with pytest.raises(ValueError, match="edges"):
        recover_kuramoto_couplings_from_time_series(
            phases,
            case.omega,
            ground_truth,
            dt=case.dt,
            edges=((0, 0),),
        )
    with pytest.raises(ValueError, match="exactly two"):
        recover_kuramoto_couplings_from_time_series(
            phases,
            case.omega,
            ground_truth,
            dt=case.dt,
            edges=((0, 1, 2),),
        )
    with pytest.raises(ValueError, match="out of bounds"):
        recover_kuramoto_couplings_from_time_series(
            phases,
            case.omega,
            ground_truth,
            dt=case.dt,
            edges=((0, 3),),
        )
    with pytest.raises(ValueError, match="unique"):
        recover_kuramoto_couplings_from_time_series(
            phases,
            case.omega,
            ground_truth,
            dt=case.dt,
            edges=((0, 1), (1, 0)),
        )
    with pytest.raises(ValueError, match="at least one"):
        recover_kuramoto_couplings_from_time_series(
            phases,
            case.omega,
            ground_truth,
            dt=case.dt,
            edges=(),
        )
    with pytest.raises(ValueError, match="no valid Kuramoto"):
        recover_kuramoto_couplings_from_time_series(
            np.full_like(phases, np.nan),
            case.omega,
            ground_truth,
            dt=case.dt,
        )
    with pytest.raises(ValueError, match="phases must have shape"):
        recover_xy_couplings_from_pair_energy_series(pair_energy, phases[:, :2], ground_truth)
    with pytest.raises(ValueError, match="pair_energy"):
        recover_xy_couplings_from_pair_energy_series(pair_energy[:, :2, :2], phases, ground_truth)
    with pytest.raises(ValueError, match="tolerance"):
        recover_xy_couplings_from_pair_energy_series(
            pair_energy, phases, ground_truth, tolerance=0.0
        )
    with pytest.raises(ValueError, match="no valid XY"):
        recover_xy_couplings_from_pair_energy_series(
            np.full_like(pair_energy, np.nan), phases, ground_truth
        )
    with pytest.raises(ValueError, match="cases"):
        run_coupling_recovery_suite(cases=())


def test_internal_fail_closed_helpers_cover_unreachable_branches() -> None:
    with pytest.raises(ValueError, match="n_nodes"):
        _normalise_edges(None, 1)
    with pytest.raises(ValueError, match="dimensions"):
        _solve_ridge(
            np.ones((1, 1), dtype=np.float64),
            np.ones((2,), dtype=np.float64),
            0.0,
        )
    with pytest.raises(ValueError, match="non-empty"):
        _solve_ridge(
            np.ones((0, 1), dtype=np.float64),
            np.ones((0,), dtype=np.float64),
            0.0,
        )
    with pytest.raises(ValueError, match="ridge"):
        _solve_ridge(
            np.ones((1, 1), dtype=np.float64),
            np.ones((1,), dtype=np.float64),
            -1.0,
        )

    solution, rank, condition_number = _solve_ridge(
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((2,), dtype=np.float64),
        0.0,
    )

    assert solution.tolist() == [0.0, 0.0]
    assert rank == 0
    assert condition_number > 1.0e300
