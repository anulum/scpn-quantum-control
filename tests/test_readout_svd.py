# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Readout SVD workspace tests
"""Condition-number correctness and admission through calibration construction."""

import math
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.dense_budget import GIB, DenseAllocationError
from scpn_quantum_control.mitigation import _readout_svd as svd_module
from scpn_quantum_control.mitigation import readout_matrix as readout


def test_svd_workspace_refusal_precedes_basis_enumeration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two matrices fitting does not imply that the queried SVD workspace fits."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("basis enumeration reached before SVD admission")

    monkeypatch.setattr(readout, "computational_basis_labels", forbidden)
    with pytest.raises(DenseAllocationError, match="workspace"):
        readout.build_readout_confusion_matrix({}, 2, max_dense_gib=320 / GIB)


@pytest.mark.parametrize("error_count", [0, 1, 20, 49, 50])
def test_condition_number_preserves_calibrated_matrix(error_count: int) -> None:
    """Symmetric binary readout has condition 1/abs(1-2p), infinite at p=1/2."""
    counts = {
        "0": {"0": 100 - error_count, "1": error_count},
        "1": {"0": error_count, "1": 100 - error_count},
    }
    result = readout.build_readout_confusion_matrix(counts, 1)
    expected = np.array([[100 - error_count, error_count], [error_count, 100 - error_count]]) / 100
    np.testing.assert_array_equal(result.matrix, expected)
    if error_count < 50:
        assert result.condition_number == pytest.approx(1 / (1 - 2 * error_count / 100))
    else:
        assert result.condition_number >= 1e15


def test_exact_rank_deficiency_reports_infinite_condition() -> None:
    """Identical deterministic readouts carry no information about preparation."""
    result = readout.build_readout_confusion_matrix({"0": {"0": 1}, "1": {"0": 1}}, 1)
    assert np.isinf(result.condition_number)
    np.testing.assert_array_equal(result.matrix, [[1, 1], [0, 0]])


def test_exact_declared_buffer_budget_and_one_byte_shortfall() -> None:
    """Query-sized numeric buffers fit exactly; one byte less must refuse."""
    work, info = svd_module.dgesdd_lwork(2, 2, compute_uv=0)
    assert info == 0
    required = 2 * 4 * 8 + (math.ceil(float(work)) + 2 + 2) * 8
    required += 8 * 2 * np.dtype(svd_module.dgesdd.int_dtype).itemsize
    counts = {"0": {"0": 1}, "1": {"1": 1}}
    result = readout.build_readout_confusion_matrix(counts, 1, max_dense_gib=required / GIB)
    assert result.condition_number == pytest.approx(1.0)
    with pytest.raises(DenseAllocationError, match="workspace"):
        readout.build_readout_confusion_matrix(counts, 1, max_dense_gib=(required - 1) / GIB)


def test_asymmetric_calibration_condition_matches_gram_eigenvalues() -> None:
    """General stochastic columns use singular values, not eigenvalues of the calibration."""
    result = readout.build_readout_confusion_matrix(
        {"0": {"0": 90, "1": 10}, "1": {"0": 20, "1": 80}}, 1
    )
    expected_matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
    eigenvalues = np.linalg.eigvalsh(expected_matrix.T @ expected_matrix)
    assert result.condition_number == pytest.approx(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    np.testing.assert_array_equal(result.matrix, expected_matrix)


@pytest.mark.parametrize(("work", "info"), [(float("nan"), 0), (0.0, 0), (10.0, -1)])
def test_invalid_workspace_query_fails_closed(
    work: float, info: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed or malformed LAPACK queries cannot authorise calibration construction."""
    monkeypatch.setattr(svd_module, "dgesdd_lwork", lambda *args, **kwargs: (work, info))
    with pytest.raises(np.linalg.LinAlgError, match="workspace query"):
        readout.build_readout_confusion_matrix({}, 1)


def test_lapack_integer_workspace_limit_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrepresentable LWORK cannot reach the solver's integer conversion."""
    excessive = float(np.iinfo(svd_module.dgesdd.int_dtype).max) + 1.0
    monkeypatch.setattr(svd_module, "dgesdd_lwork", lambda *args, **kwargs: (excessive, 0))
    with pytest.raises(DenseAllocationError, match="integer range"):
        readout.build_readout_confusion_matrix({}, 1)


@pytest.mark.parametrize("info", [-1, 1])
def test_solver_failure_does_not_publish_a_condition(
    info: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both argument failure and nonconvergence stop the public calibration build."""
    solver = svd_module.dgesdd

    class FailedSolver:
        """Preserve actual solver outputs and integer ABI while injecting its status."""

        int_dtype = solver.int_dtype

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Run the actual solver, changing only its completion status."""
            u, s, vt, _ = solver(*args, **kwargs)
            return u, s, vt, info

    monkeypatch.setattr(svd_module, "dgesdd", FailedSolver())
    with pytest.raises(np.linalg.LinAlgError, match="LAPACK info"):
        readout.build_readout_confusion_matrix({"0": {"0": 1}, "1": {"1": 1}}, 1)
