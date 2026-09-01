# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — exact Program AD linalg contracts
"""Exact fail-closed and runtime coverage for Program AD linalg contracts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import program_ad_linalg_primitives as linalg
from scpn_quantum_control.program_ad_registry import PrimitiveContract


class TraceADArray:
    """Minimal structural trace value for shape and dtype dispatch."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Store one static trace shape."""
        self.shape = shape


def _contract_with(**overrides: object) -> PrimitiveContract:
    """Return the registered determinant contract with selected fields replaced."""
    contract = linalg._require_program_ad_linalg_contract("det")
    return replace(contract, **overrides)  # type: ignore[arg-type]


def _call_rule(rule: object, name: str, *args: object) -> NDArray[np.float64]:
    """Invoke a dynamically stored derivative-rule callable."""
    function = getattr(rule, name)
    assert function is not None
    return cast(NDArray[np.float64], cast(Any, function)(*args))


def test_dispatch_and_diagnostic_guards_fail_closed() -> None:
    """Reject incomplete dispatch contracts and malformed diagnostics."""
    matrix = np.eye(2)
    bad_contracts = (
        (_contract_with(static_argument_rule=None), "static argument rule"),
        (_contract_with(static_argument_rule=lambda _args: []), "must return a tuple"),
        (_contract_with(shape_rule=None), "shape rule"),
        (_contract_with(shape_rule=lambda _args: (-1,)), "non-negative integer"),
        (_contract_with(dtype_rule=None), "dtype rule"),
        (_contract_with(dtype_rule=lambda _args: ""), "dtype name"),
    )
    for contract, message in bad_contracts:
        with pytest.raises(ValueError, match=message):
            linalg._validate_program_ad_linalg_contract_dispatch(contract, (matrix,))

    with pytest.raises(ValueError, match="operator-intercepted"):
        linalg._program_ad_linalg_direct_jvp(matrix, matrix)

    valid = dict(
        primitive="det",
        shape=(2, 2),
        status="well_conditioned",
        differentiability_ready=True,
        condition_number=1.0,
        rank=2,
        smallest_scale=1.0,
        largest_scale=1.0,
        minimum_gap=None,
        threshold=2.0,
        required_boundary="boundary",
        message="ready",
    )
    invalid = (
        ({"primitive": ""}, "primitive"),
        ({"shape": (-1, 2)}, "shape"),
        ({"status": "unknown"}, "status"),
        ({"condition_number": np.inf}, "condition_number"),
        ({"smallest_scale": -1.0}, "smallest_scale"),
        ({"minimum_gap": np.nan}, "minimum_gap"),
        ({"rank": -1}, "rank"),
        ({"required_boundary": ""}, "required_boundary"),
        ({"message": ""}, "message"),
    )
    for changes, message in invalid:
        with pytest.raises(ValueError, match=message):
            linalg.ProgramADLinalgConditioningDiagnostic(**(valid | changes))  # type: ignore[arg-type]

    diagnostic = linalg.ProgramADLinalgConditioningDiagnostic(**valid)  # type: ignore[arg-type]
    assert diagnostic.as_dict()["shape"] == [2, 2]


def test_conditioning_diagnostics_cover_every_boundary() -> None:
    """Exercise matrix, spectrum, rank, norm, and threshold diagnostics."""
    with pytest.raises(ValueError, match="rank-2"):
        linalg._program_ad_linalg_conditioning_matrix("det", [1.0, 2.0])
    with pytest.raises(ValueError, match="non-empty"):
        linalg._program_ad_linalg_conditioning_matrix("det", np.empty((0, 2)))
    assert linalg._program_ad_linalg_condition_number(np.array([])) == (0.0, 0.0, 0.0)
    assert np.isinf(linalg._program_ad_linalg_condition_number(np.array([0.0, 1.0]))[0])
    assert linalg._program_ad_linalg_minimum_gap(np.array([1.0])) is None

    with pytest.raises(ValueError, match="unsupported"):
        linalg.diagnose_program_ad_linalg_conditioning("unknown", np.eye(2))
    with pytest.raises(ValueError, match="threshold"):
        linalg.diagnose_program_ad_linalg_conditioning("det", np.eye(2), condition_threshold=0)
    with pytest.raises(ValueError, match="rank tolerance"):
        linalg.diagnose_program_ad_linalg_conditioning("det", np.eye(2), rank_tolerance=-1)

    zero = linalg.diagnose_program_ad_linalg_conditioning("norm", np.zeros(2))
    ready = linalg.diagnose_program_ad_linalg_conditioning("norm", np.ones(2))
    assert zero.status == "zero_norm_boundary"
    assert ready.status == "well_conditioned"
    assert linalg.diagnose_program_ad_linalg_conditioning("det", np.diag([1.0, 0.0])).status == (
        "rank_deficient"
    )
    assert (
        linalg.diagnose_program_ad_linalg_conditioning(
            "det", np.diag([1.0, 1.0e-6]), condition_threshold=10.0
        ).status
        == "ill_conditioned"
    )
    assert (
        linalg.diagnose_program_ad_linalg_conditioning(
            "eig", np.array([[0.0, -1.0], [1.0, 0.0]])
        ).status
        == "rank_deficient"
    )
    assert (
        linalg.diagnose_program_ad_linalg_conditioning("eigvals", np.diag([1.0, 1.0])).status
        == "rank_deficient"
    )
    assert linalg.diagnose_program_ad_linalg_conditioning(
        "eigh", np.diag([1.0, 2.0])
    ).minimum_gap == pytest.approx(1.0)
    assert (
        linalg.diagnose_program_ad_linalg_conditioning("eigvalsh", np.diag([1.0, 1.0])).status
        == "rank_deficient"
    )
    assert linalg.diagnose_program_ad_linalg_conditioning(
        "svd", np.diag([1.0, 2.0])
    ).minimum_gap == pytest.approx(1.0)


def test_direct_kernel_shape_guards_and_small_matrices() -> None:
    """Exercise direct kernel special cases and mismatched tangent/cotangent guards."""
    with pytest.raises(ValueError, match="flattened square"):
        linalg._program_ad_linalg_square_matrix("det", np.ones(2))
    with pytest.raises(ValueError, match="square matrix"):
        linalg._program_ad_linalg_det_cofactor_matrix(np.ones((2, 3)))
    assert linalg._program_ad_linalg_det_cofactor_matrix(np.empty((0, 0))).shape == (0, 0)
    np.testing.assert_array_equal(
        linalg._program_ad_linalg_det_cofactor_matrix(np.ones((1, 1))), np.ones((1, 1))
    )
    with pytest.raises(ValueError, match="tangent shape"):
        linalg._program_ad_linalg_det_jvp(np.eye(2).reshape(-1), np.ones(9))
    with pytest.raises(ValueError, match="scalar cotangent"):
        linalg._program_ad_linalg_scalar_cotangent("det", np.ones(2))
    with pytest.raises(ValueError, match="tangent shape"):
        linalg._program_ad_linalg_inv_jvp(np.eye(2).reshape(-1), np.ones(9))
    with pytest.raises(ValueError, match="cotangent shape"):
        linalg._program_ad_linalg_inv_vjp(np.eye(2).reshape(-1), np.ones(9))
    with pytest.raises(ValueError, match="flattened square matrix"):
        linalg._program_ad_linalg_solve_split("solve", np.ones(4))
    values = np.concatenate((np.eye(2).reshape(-1), np.ones(2)))
    with pytest.raises(ValueError, match="tangent shape"):
        linalg._program_ad_linalg_solve_jvp(
            values, np.concatenate((np.eye(3).reshape(-1), np.ones(3)))
        )
    with pytest.raises(ValueError, match="cotangent shape"):
        linalg._program_ad_linalg_solve_vjp(values, np.ones(3))
    with pytest.raises(ValueError, match="tangent shape"):
        linalg._program_ad_linalg_trace_jvp(np.eye(2).reshape(-1), np.ones(9))
    with pytest.raises(ValueError, match="scalar"):
        linalg._program_ad_linalg_trace_vjp(np.eye(2).reshape(-1), np.ones(2))


def test_static_solve_power_and_multi_dot_guards() -> None:
    """Cover fixed-signature factories, matrix RHS adjoints, and invalid signatures."""
    invalid_solve_shapes = (
        ((2,), (2,), "square matrix"),
        ((-2, -2), (-2,), "non-negative"),
        ((2, 2), (2, 1, 1), "rank-1 or rank-2"),
        ((2, 2), (3,), "rows must match"),
    )
    for matrix_shape, rhs_shape, message in invalid_solve_shapes:
        with pytest.raises(ValueError, match=message):
            linalg.program_ad_linalg_solve_derivative_rule(matrix_shape, rhs_shape)
    solve = linalg.program_ad_linalg_solve_derivative_rule((2, 2), (2, 2))
    values = np.concatenate((np.eye(2).reshape(-1), np.eye(2).reshape(-1)))
    assert _call_rule(solve, "vjp_rule", values, np.eye(2).reshape(-1)).shape == (8,)
    with pytest.raises(ValueError, match="flattened matrix"):
        _call_rule(solve, "value_fn", np.ones(7))
    with pytest.raises(ValueError, match="cotangent shape"):
        _call_rule(solve, "vjp_rule", values, np.ones(3))

    with pytest.raises(ValueError, match="integer power"):
        linalg.program_ad_linalg_matrix_power_derivative_rule(True)
    power = linalg.program_ad_linalg_matrix_power_derivative_rule(2)
    with pytest.raises(ValueError, match="tangent shape"):
        _call_rule(power, "jvp_rule", np.eye(2).reshape(-1), np.ones(9))
    with pytest.raises(ValueError, match="cotangent shape"):
        _call_rule(power, "vjp_rule", np.eye(2).reshape(-1), np.ones(9))

    invalid_shapes = (
        ([(2,)], "at least two"),
        ([(2, 2, 1), (2,)], "rank-1/rank-2"),
        ([(2, 0), (0, 2)], "positive"),
        ([(2,), (2,), (2,)], "middle operands"),
    )
    for shapes, message in invalid_shapes:
        with pytest.raises(ValueError, match=message):
            linalg.program_ad_linalg_multi_dot_derivative_rule(shapes)
    multi = linalg.program_ad_linalg_multi_dot_derivative_rule(((2,), (2, 2)))
    flat = np.arange(6.0)
    with pytest.raises(ValueError, match="size must match"):
        _call_rule(multi, "value_fn", np.ones(5))
    with pytest.raises(ValueError, match="cotangent shape"):
        _call_rule(multi, "vjp_rule", flat, np.ones(3))


def test_trace_diag_and_diagflat_factory_guards() -> None:
    """Cover invalid offsets, axis pairs, shape sizes, and output cotangents."""
    with pytest.raises(ValueError, match="integer offset"):
        linalg._program_ad_linalg_offset("trace", True)
    with pytest.raises(ValueError, match="rank-2"):
        linalg._program_ad_linalg_rank2_shape("trace", (2,))
    with pytest.raises(ValueError, match="positive"):
        linalg._program_ad_linalg_rank2_shape("trace", (2, 0))
    with pytest.raises(ValueError, match="empty diagonal"):
        linalg._program_ad_linalg_trace_positions((2, 2), 4)
    with pytest.raises(ValueError, match="axis1=0"):
        linalg.program_ad_linalg_trace_derivative_rule((2, 2), axis1=1)
    trace = linalg.program_ad_linalg_trace_derivative_rule((2, 3), offset=1)
    with pytest.raises(ValueError, match="size must match"):
        _call_rule(trace, "value_fn", np.ones(5))
    with pytest.raises(ValueError, match="scalar"):
        _call_rule(trace, "vjp_rule", np.ones(6), np.ones(2))

    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        linalg._program_ad_linalg_diag_positions((1, 1, 1), 0)
    with pytest.raises(ValueError, match="empty diagonal"):
        linalg._program_ad_linalg_diag_positions((2, 2), 3)
    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        linalg.program_ad_linalg_diag_derivative_rule((2, 2, 2))
    with pytest.raises(ValueError, match="positive"):
        linalg.program_ad_linalg_diag_derivative_rule((0,))
    diag = linalg.program_ad_linalg_diag_derivative_rule((2,), k=1)
    with pytest.raises(ValueError, match="values size"):
        _call_rule(diag, "value_fn", np.ones(3))
    with pytest.raises(ValueError, match="cotangent size"):
        _call_rule(diag, "vjp_rule", np.ones(2), np.ones(8))

    with pytest.raises(ValueError, match="positive"):
        linalg.program_ad_linalg_diagflat_derivative_rule((0,))
    diagflat = linalg.program_ad_linalg_diagflat_derivative_rule((2,), k=-1)
    with pytest.raises(ValueError, match="values size"):
        _call_rule(diagflat, "value_fn", np.ones(3))
    with pytest.raises(ValueError, match="cotangent size"):
        _call_rule(diagflat, "vjp_rule", np.ones(2), np.ones(8))


def test_spectral_and_pseudoinverse_helper_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise real-simple spectrum, SVD, UPLO, and full-rank pinv boundaries."""
    with pytest.raises(ValueError, match="symmetric"):
        linalg._program_ad_linalg_require_symmetric("eigh", np.array([[1.0, 1.0], [0.0, 2.0]]))
    linalg._program_ad_linalg_require_distinct_eigenvalues(np.array([1.0]), "eig")
    with pytest.raises(ValueError, match="distinct"):
        linalg._program_ad_linalg_require_distinct_eigenvalues(np.array([1.0, 1.0]), "eig")
    with pytest.raises(ValueError, match="rank-2"):
        linalg._program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", np.ones(2))
    with pytest.raises(ValueError, match="square"):
        linalg._program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", np.ones((2, 3)))
    with pytest.raises(ValueError, match="real eigenvalues"):
        linalg._program_ad_linalg_real_simple_eig_decomposition_from_matrix(
            "eig", np.array([[0.0, -1.0], [1.0, 0.0]])
        )
    with pytest.raises(ValueError, match="distinct positive"):
        linalg._program_ad_linalg_require_distinct_positive_singular_values(np.array([]), "svd")
    with pytest.raises(ValueError, match="distinct positive"):
        linalg._program_ad_linalg_require_distinct_positive_singular_values(np.array([0.0]), "svd")
    linalg._program_ad_linalg_require_distinct_positive_singular_values(np.array([1.0]), "svd")
    with pytest.raises(ValueError, match="distinct positive"):
        linalg._program_ad_linalg_require_distinct_positive_singular_values(
            np.array([1.0, 1.0]), "svd"
        )

    assert linalg._program_ad_linalg_normalise_rcond(None) == pytest.approx(1.0e-15)
    with pytest.raises(ValueError, match="static real scalar"):
        linalg._program_ad_linalg_normalise_rcond(True)
    with pytest.raises(ValueError, match="finite and non-negative"):
        linalg._program_ad_linalg_normalise_rcond(-1.0)
    with pytest.raises(ValueError, match="rank-2 singular"):
        linalg._program_ad_linalg_require_constant_full_rank(np.eye(2), np.ones(1), rcond=0)
    with pytest.raises(ValueError, match="constant full-rank"):
        linalg._program_ad_linalg_require_constant_full_rank(
            np.eye(2), np.array([1.0, 0.0]), rcond=0
        )
    with pytest.raises(ValueError, match="rank-2"):
        linalg._program_ad_linalg_pinv_value_matrix(np.ones(2))
    with pytest.raises(ValueError, match="non-empty"):
        linalg._program_ad_linalg_pinv_value_matrix(np.empty((0, 2)))
    matrix = np.diag([1.0, 2.0])
    pinv = np.linalg.pinv(matrix)
    with pytest.raises(ValueError, match="tangent shape"):
        linalg._program_ad_linalg_pinv_jvp_matrix(matrix, pinv, np.ones((3, 3)))
    with pytest.raises(ValueError, match="cotangent shape"):
        linalg._program_ad_linalg_pinv_vjp_matrix(matrix, pinv, np.ones((3, 3)))
    with pytest.raises(ValueError, match="cotangent size"):
        linalg._program_ad_linalg_pinv_square_vjp(matrix.reshape(-1), np.ones(3))
    with pytest.raises(ValueError, match="UPLO"):
        linalg._program_ad_linalg_uplo("x", "eigh")

    original_eig = np.linalg.eig
    monkeypatch.setattr(
        np.linalg,
        "eig",
        lambda _matrix: (np.array([1.0, 2.0]), np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0]])),
    )
    with pytest.raises(ValueError, match="real eigenvectors"):
        linalg._program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", matrix)
    monkeypatch.setattr(np.linalg, "eig", original_eig)


def test_spectral_factories_cover_values_jvps_vjps_and_guards() -> None:
    """Run direct eig, eigh, eigvals, eigvalsh, SVD, and pinv contracts."""
    matrix = np.array([[2.0, 0.0], [0.0, 1.0]])
    tangent = np.array([[0.2, 0.1], [0.1, -0.1]])
    for name in ("eig", "eigh"):
        rule = getattr(linalg, f"program_ad_linalg_{name}_derivative_rule")((2, 2))
        output = _call_rule(rule, "value_fn", matrix.reshape(-1))
        assert (
            _call_rule(rule, "jvp_rule", matrix.reshape(-1), tangent.reshape(-1)).shape
            == output.shape
        )
        assert _call_rule(rule, "vjp_rule", matrix.reshape(-1), np.ones_like(output)).shape == (4,)
        with pytest.raises(ValueError, match="values size"):
            _call_rule(rule, "value_fn", np.ones(3))
        with pytest.raises(ValueError, match="cotangent size"):
            _call_rule(rule, "vjp_rule", matrix.reshape(-1), np.ones(2))

    for name in ("eigvals", "eigvalsh"):
        rule = getattr(linalg, f"program_ad_linalg_{name}_derivative_rule")((2, 2))
        output = _call_rule(rule, "value_fn", matrix.reshape(-1))
        assert (
            _call_rule(rule, "jvp_rule", matrix.reshape(-1), tangent.reshape(-1)).shape
            == output.shape
        )
        assert _call_rule(rule, "vjp_rule", matrix.reshape(-1), np.ones_like(output)).shape == (4,)
        with pytest.raises(ValueError, match="cotangent size"):
            _call_rule(rule, "vjp_rule", matrix.reshape(-1), np.ones(3))
        with pytest.raises(ValueError, match="values size"):
            _call_rule(rule, "value_fn", np.ones(3))

    svd = linalg.program_ad_linalg_svdvals_derivative_rule((2, 2))
    singular_values = _call_rule(svd, "value_fn", matrix.reshape(-1))
    assert _call_rule(svd, "jvp_rule", matrix.reshape(-1), tangent.reshape(-1)).shape == (2,)
    assert _call_rule(
        svd, "vjp_rule", matrix.reshape(-1), np.ones_like(singular_values)
    ).shape == (4,)
    with pytest.raises(ValueError, match="values size"):
        _call_rule(svd, "value_fn", np.ones(3))
    with pytest.raises(ValueError, match="cotangent size"):
        _call_rule(svd, "vjp_rule", matrix.reshape(-1), np.ones(3))

    pinv = linalg.program_ad_linalg_pinv_derivative_rule((2, 3), rcond=0)
    rectangular = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    output = _call_rule(pinv, "value_fn", rectangular.reshape(-1))
    assert _call_rule(pinv, "jvp_rule", rectangular.reshape(-1), np.ones(6)).shape == output.shape
    assert _call_rule(pinv, "vjp_rule", rectangular.reshape(-1), np.ones_like(output)).shape == (
        6,
    )
    with pytest.raises(ValueError, match="values size"):
        _call_rule(pinv, "value_fn", np.ones(5))
    with pytest.raises(ValueError, match="cotangent size"):
        _call_rule(pinv, "vjp_rule", rectangular.reshape(-1), np.ones(5))


def test_registry_spectral_kernels_and_factory_boundaries() -> None:
    """Exercise registry-owned spectral kernels and non-square factory guards."""
    matrix = np.diag([2.0, 1.0])
    values = matrix.reshape(-1)
    tangent = np.array([[0.2, 0.1], [0.1, -0.1]]).reshape(-1)
    direct_sizes = {
        "eig": 6,
        "eigh": 6,
        "eigvals": 2,
        "eigvalsh": 2,
        "pinv": 4,
    }
    for name, output_size in direct_sizes.items():
        rule = linalg._program_ad_linalg_derivative_rule(name)
        output = _call_rule(rule, "value_fn", values)
        assert output.size == output_size
        assert _call_rule(rule, "jvp_rule", values, tangent).size == output_size
        assert _call_rule(rule, "vjp_rule", values, np.ones(output_size)).size == 4

    with pytest.raises(ValueError, match="tangent shape"):
        linalg._program_ad_linalg_eigvals_jvp(values, np.ones(9))
    with pytest.raises(ValueError, match="cotangent size"):
        linalg._program_ad_linalg_eigvals_vjp(values, np.ones(3))
    with pytest.raises(ValueError, match="tangent shape"):
        linalg._program_ad_linalg_eig_jvp(values, np.ones(9))
    with pytest.raises(ValueError, match="cotangent size"):
        linalg._program_ad_linalg_eig_vjp(values, np.ones(3))
    with pytest.raises(ValueError, match="tangent shape"):
        linalg._program_ad_linalg_eigvalsh_jvp(values, np.ones(9))
    with pytest.raises(ValueError, match="symmetric"):
        linalg._program_ad_linalg_eigvalsh_jvp(values, np.array([1.0, 1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="cotangent size"):
        linalg._program_ad_linalg_eigvalsh_vjp(values, np.ones(3))
    with pytest.raises(ValueError, match="tangent shape"):
        linalg._program_ad_linalg_eigh_jvp(values, np.ones(9))
    with pytest.raises(ValueError, match="symmetric"):
        linalg._program_ad_linalg_eigh_jvp(values, np.array([1.0, 1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="cotangent size"):
        linalg._program_ad_linalg_eigh_vjp(values, np.ones(3))

    for factory in (
        linalg.program_ad_linalg_eig_derivative_rule,
        linalg.program_ad_linalg_eigvals_derivative_rule,
        linalg.program_ad_linalg_eigh_derivative_rule,
        linalg.program_ad_linalg_eigvalsh_derivative_rule,
    ):
        with pytest.raises(ValueError, match="square matrix"):
            factory((2, 3))
    eigvals = linalg.program_ad_linalg_eigvals_derivative_rule((2, 2))
    with pytest.raises(ValueError, match="values size"):
        _call_rule(eigvals, "value_fn", np.ones(3))


def test_eigendecomposition_exception_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate singular and ill-conditioned eigenbases into fail-closed errors."""
    matrix = np.diag([1.0, 2.0])
    original_inv = np.linalg.inv
    monkeypatch.setattr(
        np.linalg,
        "inv",
        lambda _matrix: (_ for _ in ()).throw(np.linalg.LinAlgError("singular")),
    )
    with pytest.raises(ValueError, match="diagonalizable"):
        linalg._program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", matrix)
    monkeypatch.setattr(np.linalg, "inv", original_inv)

    original_cond = np.linalg.cond
    monkeypatch.setattr(np.linalg, "cond", lambda _matrix: np.inf)
    with pytest.raises(ValueError, match="well-conditioned"):
        linalg._program_ad_linalg_real_simple_eig_decomposition_from_matrix("eig", matrix)
    monkeypatch.setattr(np.linalg, "cond", original_cond)

    assert linalg._program_ad_linalg_real_simple_eig_decomposition("eig", matrix.reshape(-1))[
        0
    ].shape == (2, 2)
    with pytest.raises(ValueError, match="must be static"):
        linalg._program_ad_linalg_normalise_rcond(TraceADArray(()))


def test_shape_static_dtype_and_batching_edges() -> None:
    """Exercise every static shape family and batching boundary."""
    matrix = np.eye(2)
    vector = np.ones(2)
    assert linalg._program_ad_linalg_shape_of(TraceADArray((2, 3))) == (2, 3)
    with pytest.raises(ValueError, match="rank-2"):
        linalg._program_ad_linalg_require_matrix_shape("det", vector)
    with pytest.raises(ValueError, match="square"):
        linalg._program_ad_linalg_require_matrix_shape("det", np.ones((2, 3)))

    one_arg_rules = (
        (linalg._program_ad_linalg_det_shape, "det"),
        (linalg._program_ad_linalg_inv_shape, "inv"),
        (linalg._program_ad_linalg_eigh_shape, "eigh"),
        (linalg._program_ad_linalg_eig_shape, "eig"),
        (linalg._program_ad_linalg_eigvalsh_shape, "eigvalsh"),
        (linalg._program_ad_linalg_eigvals_shape, "eigvals"),
        (linalg._program_ad_linalg_svd_shape, "svd"),
        (linalg._program_ad_linalg_pinv_shape, "pinv"),
    )
    for rule, name in one_arg_rules:
        with pytest.raises(ValueError, match=name):
            rule(())
    with pytest.raises(ValueError, match="vector length"):
        linalg._program_ad_linalg_solve_shape((matrix, np.ones(3)))
    with pytest.raises(ValueError, match="rhs rows"):
        linalg._program_ad_linalg_solve_shape((matrix, np.ones((3, 1))))
    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        linalg._program_ad_linalg_solve_shape((matrix, np.ones((2, 1, 1))))
    with pytest.raises(ValueError, match="matrix and right-hand side"):
        linalg._program_ad_linalg_solve_shape((matrix,))

    for rule in (
        linalg._program_ad_linalg_trace_shape,
        linalg._program_ad_linalg_diag_shape,
        linalg._program_ad_linalg_diagflat_shape,
    ):
        with pytest.raises(ValueError, match="requires"):
            rule(())
    with pytest.raises(ValueError, match="rank-2"):
        linalg._program_ad_linalg_trace_shape((vector,))
    with pytest.raises(ValueError, match="axis1=0"):
        linalg._program_ad_linalg_trace_shape((matrix, 0, 1, 0))
    assert linalg._program_ad_linalg_trace_shape((matrix,)) == ()
    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        linalg._program_ad_linalg_diag_shape((np.ones((1, 1, 1)),))
    with pytest.raises(ValueError, match="positive"):
        linalg._program_ad_linalg_diag_shape((np.empty((0,)),))
    with pytest.raises(ValueError, match="non-empty"):
        linalg._program_ad_linalg_diagflat_shape((np.empty((0,)),))
    with pytest.raises(ValueError, match="integer power"):
        linalg._program_ad_linalg_matrix_power_shape((matrix, True))
    with pytest.raises(ValueError, match="matrix and power"):
        linalg._program_ad_linalg_matrix_power_shape((matrix,))
    assert linalg._program_ad_linalg_trace_shape((matrix, 0, 0, 1)) == ()

    invalid_multi_dot = (
        ((matrix,), "operand sequence"),
        (([matrix],), "at least two"),
        (([np.ones((1, 1, 1)), vector],), "rank-1 and rank-2"),
        (([vector, vector, vector],), "middle operands"),
        (([np.ones(2), np.ones(3)],), "align"),
        (([np.ones(2), np.ones((3, 2))],), "align"),
        (([np.ones((2, 3)), np.ones(2)],), "align"),
        (([np.ones((2, 3)), np.ones((2, 2))],), "align"),
    )
    for args, message in invalid_multi_dot:
        with pytest.raises(ValueError, match=message):
            linalg._program_ad_linalg_multi_dot_shape(args)
    with pytest.raises(ValueError, match="one operand sequence"):
        linalg._program_ad_linalg_multi_dot_shape(())
    assert linalg._program_ad_linalg_multi_dot_shape(([np.ones((2, 3)), np.ones(3)],)) == (2,)
    assert linalg._program_ad_linalg_multi_dot_shape(([np.ones((2, 3)), np.ones((3, 4))],)) == (
        2,
        4,
    )
    with pytest.raises(ValueError, match="non-empty dimensions"):
        linalg._program_ad_linalg_svd_shape((np.empty((0, 2)),))
    with pytest.raises(ValueError, match="non-empty dimensions"):
        linalg._program_ad_linalg_pinv_shape((np.empty((2, 0)),))

    assert linalg._program_ad_linalg_dtype_rule((TraceADArray((2,)), 2)) == "float64"
    assert (
        linalg._program_ad_linalg_dtype_rule(([TraceADArray((2,)), np.ones(2, dtype=np.float32)],))
        == "float64"
    )
    assert linalg._program_ad_linalg_no_static_arguments((matrix,)) == ()
    with pytest.raises(ValueError, match="matrix and power"):
        linalg._program_ad_linalg_matrix_power_static_arguments((matrix,))
    with pytest.raises(ValueError, match="integer power"):
        linalg._program_ad_linalg_matrix_power_static_arguments((matrix, True))
    with pytest.raises(ValueError, match="static rule"):
        linalg._program_ad_linalg_trace_static_arguments(())
    with pytest.raises(ValueError, match="rank-2"):
        linalg._program_ad_linalg_trace_static_arguments((vector,))
    with pytest.raises(ValueError, match="axis1=0"):
        linalg._program_ad_linalg_trace_static_arguments((matrix, 0, 1, 0))
    assert linalg._program_ad_linalg_trace_static_arguments((matrix,)) == ((2, 2), 0, 0, 1)
    assert linalg._program_ad_linalg_trace_static_arguments((matrix, 0, 0, 1)) == (
        (2, 2),
        0,
        0,
        1,
    )
    with pytest.raises(ValueError, match="diag static"):
        linalg._program_ad_linalg_diag_static_arguments(())
    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        linalg._program_ad_linalg_diag_static_arguments((np.ones((1, 1, 1)),))
    with pytest.raises(ValueError, match="diagflat static"):
        linalg._program_ad_linalg_diagflat_static_arguments(())
    with pytest.raises(ValueError, match="non-empty"):
        linalg._program_ad_linalg_diagflat_static_arguments((np.empty((0,)),))
    with pytest.raises(ValueError, match="multi_dot static"):
        linalg._program_ad_linalg_multi_dot_static_arguments(())
    with pytest.raises(ValueError, match="static operand"):
        linalg._program_ad_linalg_multi_dot_static_arguments((matrix,))
    with pytest.raises(ValueError, match="at least two"):
        linalg._program_ad_linalg_multi_dot_static_arguments(([matrix],))

    with pytest.raises(ValueError, match="argument count"):
        linalg._program_ad_linalg_batching_rule(np.add, (matrix,), (0, None), 0)
    with pytest.raises(ValueError, match="non-empty"):
        linalg._program_ad_linalg_batching_rule(lambda value: value, (np.empty((0, 2)),), (0,), 0)
    with pytest.raises(ValueError, match="one batch size"):
        linalg._program_ad_linalg_batching_rule(
            np.add, (np.ones((2, 1)), np.ones((3, 1))), (0, 0), 0
        )
    with pytest.raises(ValueError, match="at least one mapped"):
        linalg._program_ad_linalg_batching_rule(lambda value: value, (matrix,), (None,), 0)
    batched = linalg._program_ad_linalg_batching_rule(
        lambda left, right: left + right,
        (np.arange(6.0).reshape(2, 3), np.ones(3)),
        (0, None),
        1,
    )
    assert np.asarray(batched).shape == (3, 2)


def test_lowering_metadata_and_registry_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover every lowering family and incomplete registry contract guard."""
    names = tuple(linalg._PROGRAM_AD_LINALG_SHAPE_RULES)
    metadata = {name: linalg._program_ad_linalg_lowering_metadata(name) for name in names}
    assert all(row["mlir_op"].endswith(name) for name, row in metadata.items())

    with pytest.raises(ValueError, match="unsupported"):
        linalg._require_program_ad_linalg_contract("unknown")
    base = linalg._require_program_ad_linalg_contract("det")
    invalid: tuple[tuple[dict[str, object], str], ...] = (
        ({"nondifferentiable_policy": "wrong"}, "declare policy"),
        ({"effect": "io"}, "must be pure"),
        ({"batching_rule": None}, "batching_rule"),
        ({"lowering_metadata": {}}, "lowering_metadata"),
        (
            {
                "lowering_metadata": {
                    "nondifferentiable_boundary": "x",
                    "nondifferentiable_boundary_policy": "fail_closed",
                }
            },
            "mlir_op",
        ),
        (
            {
                "lowering_metadata": {
                    "mlir_op": "x",
                    "nondifferentiable_boundary_policy": "fail_closed",
                }
            },
            "nondifferentiable_boundary",
        ),
        (
            {
                "lowering_metadata": {
                    "mlir_op": "x",
                    "nondifferentiable_boundary": "x",
                    "nondifferentiable_boundary_policy": "wrong",
                }
            },
            "nondifferentiable_boundary_policy",
        ),
        ({"shape_rule": None}, "shape_rule"),
        ({"dtype_rule": None}, "dtype_rule"),
        ({"static_argument_rule": None}, "static_argument_rule"),
    )
    for changes, message in invalid:
        contract = replace(base, **changes)  # type: ignore[arg-type]
        registry = SimpleNamespace(require_contract=lambda _identity, value=contract: value)
        monkeypatch.setattr(linalg, "DEFAULT_CUSTOM_DERIVATIVE_REGISTRY", registry)
        with pytest.raises(ValueError, match=message):
            linalg._require_program_ad_linalg_contract("det")


def test_registration_populates_an_empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register every linalg primitive when no contract is already present."""
    registered: list[object] = []
    registry = SimpleNamespace(
        contract_for=lambda _identity: None,
        register_transform=registered.append,
    )
    monkeypatch.setattr(linalg, "DEFAULT_CUSTOM_DERIVATIVE_REGISTRY", registry)
    linalg._register_program_ad_linalg_primitive_contracts()
    assert len(registered) == len(linalg._PROGRAM_AD_LINALG_SHAPE_RULES)

    registered.clear()
    registry = SimpleNamespace(
        contract_for=lambda _identity: object(),
        register_transform=registered.append,
    )
    monkeypatch.setattr(linalg, "DEFAULT_CUSTOM_DERIVATIVE_REGISTRY", registry)
    linalg._register_program_ad_linalg_primitive_contracts()
    assert registered == []
