# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Open-System Objective Tests
"""Tests for bounded Lindblad and MCWF objective evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import (
    OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY,
    OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS,
    BoundedOpenSystemObjectiveCase,
    DensityMatrixInvariantCertificate,
    MCWFReproducibilityCertificate,
    OpenSystemObjectiveBoundaryRow,
    OpenSystemObjectiveRecord,
    OpenSystemObjectiveSuiteResult,
    certify_density_matrix_invariants,
    certify_mcwf_reproducibility,
    default_open_system_objective_cases,
    evaluate_lindblad_objective,
    evaluate_mcwf_objective,
    open_system_objective_boundary_rows,
    run_open_system_objective_suite,
)
from scpn_quantum_control.phase.open_system_objectives import BoundaryStatus, OpenSystemBackend


@pytest.fixture(scope="module")
def suite() -> OpenSystemObjectiveSuiteResult:
    """Run the default BL-16 open-system suite once."""
    return run_open_system_objective_suite()


def test_default_open_system_objective_suite_passes(
    suite: OpenSystemObjectiveSuiteResult,
) -> None:
    assert suite.passed
    assert suite.case_count == 2
    assert suite.record_count == 4
    assert suite.backend_names == ("lindblad_density", "mcwf_ensemble")
    assert suite.evidence_class == OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS
    assert suite.claim_boundary == OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY
    assert len(suite.boundary_rows) == 2

    for record in suite.records:
        assert record.passed
        assert record.evidence_class == OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS
        assert np.all(np.isfinite(record.gradient))
        if record.backend == "lindblad_density":
            assert record.invariant_certificate is not None
            assert record.invariant_certificate.trace_error <= 5.0e-8
            assert record.invariant_certificate.min_eigenvalue >= -5.0e-8
            assert record.reproducibility_certificate is None
        else:
            assert record.invariant_certificate is None
            assert record.reproducibility_certificate is not None
            assert record.reproducibility_certificate.same_seed_max_abs_diff == 0.0
            assert record.reproducibility_certificate.trajectory_shape[0] == 12


def test_open_system_suite_serializes_cases_records_and_boundaries(
    suite: OpenSystemObjectiveSuiteResult,
) -> None:
    payload = suite.to_dict()

    assert payload["passed"] is True
    assert payload["case_count"] == 2
    assert payload["record_count"] == 4
    assert payload["backend_names"] == ["lindblad_density", "mcwf_ensemble"]
    assert len(cast(list[dict[str, object]], payload["cases"])) == 2
    assert len(cast(list[dict[str, object]], payload["records"])) == 4
    boundaries = cast(list[dict[str, object]], payload["boundary_rows"])
    assert boundaries[0]["failure_class"] == "unsupported_adjoint_lindblad_gradient"
    assert boundaries[1]["failure_class"] == "no_live_provider_attestation"


def test_open_system_records_for_case_contract(
    suite: OpenSystemObjectiveSuiteResult,
) -> None:
    case_id = suite.cases[0].case_id

    records = suite.records_for_case(case_id)

    assert {record.backend for record in records} == {"lindblad_density", "mcwf_ensemble"}
    with pytest.raises(KeyError, match="unknown open-system objective case"):
        suite.records_for_case("missing-case")


def test_custom_open_system_suite_subset_omits_boundary_rows() -> None:
    suite = run_open_system_objective_suite(
        cases=default_open_system_objective_cases()[:1],
        backends=("mcwf_ensemble", "mcwf_ensemble"),
        include_boundary_rows=False,
    )

    assert suite.passed
    assert suite.case_count == 1
    assert suite.record_count == 1
    assert suite.backend_names == ("mcwf_ensemble",)
    assert suite.boundary_rows == ()


def test_open_system_objective_evaluators_return_certificates() -> None:
    case = default_open_system_objective_cases()[0]

    lindblad_value, lindblad_r, invariant = evaluate_lindblad_objective(case, case.initial_params)
    mcwf_value, mcwf_r, reproducibility = evaluate_mcwf_objective(case, case.initial_params)

    assert lindblad_value >= 0.0
    assert 0.0 <= lindblad_r <= 1.0
    assert invariant.passed
    assert mcwf_value >= 0.0
    assert 0.0 <= mcwf_r <= 1.0
    assert reproducibility.passed


def test_open_system_case_validates_and_scales_inputs() -> None:
    case = default_open_system_objective_cases()[0]
    scaled_k, gamma_amp, gamma_deph = case.scaled_inputs(np.array([2.0, 3.0]))

    assert scaled_k[0, 1] == pytest.approx(case.coupling_matrix[0, 1] * 2.0)
    assert gamma_amp == pytest.approx(case.gamma_amp * 3.0)
    assert gamma_deph == pytest.approx(case.gamma_deph * 3.0)
    assert case.to_dict()["case_id"] == "two_qubit_relaxing_sync"

    with pytest.raises(ValueError, match="params must have shape"):
        case.scaled_inputs(np.array([1.0]))
    with pytest.raises(ValueError, match="positive coupling"):
        case.scaled_inputs(np.array([1.0, 0.0]))


def _case(
    *,
    case_id: str = "valid",
    n_oscillators: int = 2,
    coupling_matrix: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
    gamma_amp: float = 0.1,
    gamma_deph: float = 0.01,
    initial_params: NDArray[np.float64] | None = None,
    target_order_parameter: float = 0.5,
    target_purity: float = 0.8,
    t_max: float = 0.2,
    dt: float = 0.1,
    n_trajectories: int = 4,
    seed: int = 7,
    finite_difference_step: float = 1.0e-3,
) -> BoundedOpenSystemObjectiveCase:
    """Build a typed objective case fixture."""
    return BoundedOpenSystemObjectiveCase(
        case_id=case_id,
        n_oscillators=n_oscillators,
        coupling_matrix=(
            np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64)
            if coupling_matrix is None
            else coupling_matrix
        ),
        omega=np.array([0.5, 0.9], dtype=np.float64) if omega is None else omega,
        gamma_amp=gamma_amp,
        gamma_deph=gamma_deph,
        initial_params=(
            np.array([1.0, 1.0], dtype=np.float64) if initial_params is None else initial_params
        ),
        target_order_parameter=target_order_parameter,
        target_purity=target_purity,
        t_max=t_max,
        dt=dt,
        n_trajectories=n_trajectories,
        seed=seed,
        finite_difference_step=finite_difference_step,
    )


def test_open_system_case_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="case_id must be non-empty"):
        _case(case_id="")
    with pytest.raises(ValueError, match="n_oscillators"):
        _case(n_oscillators=0)
    with pytest.raises(ValueError, match="coupling_matrix must have shape"):
        _case(coupling_matrix=np.eye(3, dtype=np.float64))
    with pytest.raises(ValueError, match="omega must have shape"):
        _case(omega=np.array([0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="omega must be 1-dimensional"):
        _case(omega=np.array([[0.5, 0.9]], dtype=np.float64))
    with pytest.raises(ValueError, match="params must be non-empty"):
        _case(initial_params=np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="omega must contain only finite values"):
        _case(omega=np.array([0.5, float("nan")], dtype=np.float64))
    with pytest.raises(ValueError, match="symmetric"):
        _case(coupling_matrix=np.array([[0.0, 0.2], [0.3, 0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="gamma_amp"):
        _case(gamma_amp=-0.1)
    with pytest.raises(ValueError, match="gamma_amp"):
        _case(gamma_amp=cast(float, True))
    with pytest.raises(ValueError, match="gamma_deph"):
        _case(gamma_deph=-0.1)
    with pytest.raises(ValueError, match="gamma_deph"):
        _case(gamma_deph=float("nan"))
    with pytest.raises(ValueError, match="gamma_deph"):
        _case(gamma_deph=cast(float, 0.0 + 1.0j))
    with pytest.raises(ValueError, match="target_order_parameter"):
        _case(target_order_parameter=1.1)
    with pytest.raises(ValueError, match="target_order_parameter"):
        _case(target_order_parameter=cast(float, True))
    with pytest.raises(ValueError, match="target_purity"):
        _case(target_purity=1.1)
    with pytest.raises(ValueError, match="target_purity"):
        _case(target_purity=cast(float, True))
    with pytest.raises(ValueError, match="target_purity"):
        _case(target_purity=cast(float, "bad"))
    with pytest.raises(ValueError, match="target_purity"):
        _case(target_purity=0.0)
    with pytest.raises(ValueError, match="dt"):
        _case(dt=0.0)
    with pytest.raises(ValueError, match="n_trajectories"):
        _case(n_trajectories=0)
    with pytest.raises(ValueError, match="seed"):
        _case(seed=-1)
    with pytest.raises(ValueError, match="finite_difference_step"):
        _case(initial_params=np.array([0.001, 1.0]), finite_difference_step=0.002)


def test_density_matrix_invariant_certificate_rejects_wrong_shape() -> None:
    case = default_open_system_objective_cases()[0]

    with pytest.raises(ValueError, match="rho shape"):
        certify_density_matrix_invariants(case, np.eye(2, dtype=np.complex128))


def test_density_matrix_invariant_certificate_detects_invalid_matrix() -> None:
    case = default_open_system_objective_cases()[0]
    rho = np.diag(np.array([1.1, -0.1, 0.0, 0.0], dtype=np.complex128))

    certificate = certify_density_matrix_invariants(case, rho)

    assert certificate.finite
    assert certificate.min_eigenvalue == pytest.approx(-0.1)
    assert not certificate.passed
    assert certificate.to_dict()["passed"] is False


def test_mcwf_reproducibility_certificate_rejects_bad_batches() -> None:
    case = _case()
    valid = {
        "R_trajectories": np.zeros((4, 3), dtype=np.float64),
        "R_mean": np.zeros(3, dtype=np.float64),
        "R_std": np.zeros(3, dtype=np.float64),
        "total_jumps": 0,
    }

    with pytest.raises(ValueError, match="share shape"):
        certify_mcwf_reproducibility(
            case,
            valid,
            {**valid, "R_trajectories": np.zeros((4, 2), dtype=np.float64)},
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        certify_mcwf_reproducibility(
            case,
            {**valid, "R_trajectories": np.zeros(3, dtype=np.float64)},
            {**valid, "R_trajectories": np.zeros(3, dtype=np.float64)},
        )
    with pytest.raises(ValueError, match="row count"):
        certify_mcwf_reproducibility(
            case,
            {**valid, "R_trajectories": np.zeros((3, 3), dtype=np.float64)},
            {**valid, "R_trajectories": np.zeros((3, 3), dtype=np.float64)},
        )
    with pytest.raises(ValueError, match="mean and standard-deviation"):
        certify_mcwf_reproducibility(
            case,
            {**valid, "R_std": np.zeros(2, dtype=np.float64)},
            valid,
        )
    with pytest.raises(ValueError, match="total_jumps"):
        certify_mcwf_reproducibility(
            case,
            {**valid, "total_jumps": cast(int, True)},
            valid,
        )


def test_boundary_rows_validate_and_serialize() -> None:
    rows = open_system_objective_boundary_rows()

    assert rows[0].to_dict()["status"] == "hard_gap"
    with pytest.raises(ValueError, match="case_id"):
        OpenSystemObjectiveBoundaryRow("", "backend", "hard_gap", "failure", "setup")
    with pytest.raises(ValueError, match="backend"):
        OpenSystemObjectiveBoundaryRow("case", "", "hard_gap", "failure", "setup")
    with pytest.raises(ValueError, match="status"):
        OpenSystemObjectiveBoundaryRow(
            "case",
            "backend",
            cast(BoundaryStatus, "open"),
            "failure",
            "setup",
        )
    with pytest.raises(ValueError, match="failure_class"):
        OpenSystemObjectiveBoundaryRow("case", "backend", "hard_gap", "", "setup")
    with pytest.raises(ValueError, match="setup_instructions"):
        OpenSystemObjectiveBoundaryRow("case", "backend", "hard_gap", "failure", "")


def test_record_passed_property_requires_backend_certificate() -> None:
    bad_invariant = DensityMatrixInvariantCertificate(
        trace_error=1.0,
        hermiticity_error=0.0,
        min_eigenvalue=0.0,
        purity=1.0,
        finite=True,
        passed=False,
    )
    bad_record = OpenSystemObjectiveRecord(
        case_id="case",
        backend="lindblad_density",
        params=(1.0, 1.0),
        value=0.0,
        gradient=(0.0, 0.0),
        evaluations=1,
        final_order_parameter=0.5,
        invariant_certificate=bad_invariant,
        reproducibility_certificate=None,
    )
    good_replay = MCWFReproducibilityCertificate(
        n_trajectories=4,
        seed=1,
        trajectory_shape=(4, 3),
        same_seed_max_abs_diff=0.0,
        final_mean_order_parameter=0.5,
        final_std_order_parameter=0.1,
        total_jumps=0,
        finite=True,
        passed=True,
    )
    good_record = OpenSystemObjectiveRecord(
        case_id="case",
        backend="mcwf_ensemble",
        params=(1.0, 1.0),
        value=0.0,
        gradient=(0.0, 0.0),
        evaluations=1,
        final_order_parameter=0.5,
        invariant_certificate=None,
        reproducibility_certificate=good_replay,
    )

    assert not bad_record.passed
    assert bad_record.to_dict()["passed"] is False
    assert good_record.passed
    assert good_record.to_dict()["reproducibility_certificate"] == good_replay.to_dict()


def test_open_system_suite_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="at least one open-system objective case"):
        run_open_system_objective_suite(cases=())
    with pytest.raises(ValueError, match="at least one backend"):
        run_open_system_objective_suite(backends=())
    with pytest.raises(ValueError, match="unknown open-system objective backend"):
        run_open_system_objective_suite(backends=cast(Sequence[OpenSystemBackend], ("unknown",)))
