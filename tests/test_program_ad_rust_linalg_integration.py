# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD Rust Linalg Integration Tests
"""Real-engine alias and linear-algebra replay integration tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)


def _objective_reshape_sumsq(values: Any) -> Any:
    return np.sum(np.reshape(values, (2, 2)) ** 2)


def _objective_mutation(values: Any) -> Any:
    work = values * 1.0
    work[0] = work[0] * work[1]
    return np.sum(work)


def test_bridge_replays_inert_view_alias_program_with_real_engine() -> None:
    """With the real engine, a reshape view-alias program replays bit-exact."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_reshape_sumsq, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine does not yet replay view aliases: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_reshape_sumsq, sample)
    np.testing.assert_array_equal(np.asarray(rust.gradient), reference)
    assert rust.claim_boundary.endswith(
        "view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_bridge_fails_closed_on_mutation_alias_with_real_engine() -> None:
    """A mutation-aliasing program stays outside the bounded Rust replay."""
    pytest.importorskip("scpn_quantum_engine")

    sample = np.array([2.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_mutation, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is False
    assert any("non-view alias" in reason for reason in rust.blocked_reasons)


def _objective_trace_2x2(values: Any) -> Any:
    return np.trace(np.reshape(values, (2, 2)))


def _objective_det_2x2(values: Any) -> Any:
    return np.linalg.det(np.reshape(values, (2, 2)))


def test_bridge_replays_linalg_trace_with_real_engine() -> None:
    """With the real engine, a 2x2 trace program replays bit-exact."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_trace_2x2, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:trace replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_trace_2x2, sample)
    np.testing.assert_array_equal(np.asarray(rust.gradient), reference)


def test_bridge_replays_linalg_det_2x2_with_real_engine() -> None:
    """With the real engine, a 2x2 determinant program replays within float64 tolerance."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 1.0, 1.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_det_2x2, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:det replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_det_2x2, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)
    assert "dynamic_boundary_fail_closed_audit" in rust.claim_boundary
    assert rust.claim_boundary.endswith(
        "executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def _objective_inv_2x2_sum(values: Any) -> Any:
    return np.sum(np.linalg.inv(np.reshape(values, (2, 2))))


def _objective_solve_2x2_sum(values: Any) -> Any:
    return np.sum(np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:]))


def _objective_solve_2x2_indexed(values: Any) -> Any:
    return np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:])[0]


def test_bridge_replays_linalg_inverse_with_real_engine() -> None:
    """With the real engine, a reduced 2x2 inverse program replays within tolerance."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 1.0, 1.0, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_inv_2x2_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:inv replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_inv_2x2_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def test_bridge_replays_linalg_solve_with_real_engine() -> None:
    """With the real engine, a reduced 2x2 linear solve replays within tolerance."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 1.0, 1.0, 3.0, 1.0, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_solve_2x2_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:solve replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_solve_2x2_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def test_bridge_fails_closed_on_indexed_multi_output_linalg_with_real_engine() -> None:
    """A bare indexed solve component stays outside the bounded Rust replay."""
    pytest.importorskip("scpn_quantum_engine")

    sample = np.array([3.0, 1.0, 2.0, 4.0, 5.0, 6.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_solve_2x2_indexed, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    assert rust.supported is False
    assert any("indexed multi-output linalg" in reason for reason in rust.blocked_reasons)


def _objective_det_3x3(values: Any) -> Any:
    return np.linalg.det(np.reshape(values, (3, 3)))


def test_bridge_replays_linalg_det_3x3_with_real_engine() -> None:
    """With the real engine, a 3x3 determinant program replays within float64 tolerance."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_det_3x3, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:det:3x3 replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_det_3x3, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def _objective_inv_3x3_sum(values: Any) -> Any:
    return np.sum(np.linalg.inv(np.reshape(values, (3, 3))))


def _objective_solve_3x3_sum(values: Any) -> Any:
    return np.sum(np.linalg.solve(np.reshape(values[:9], (3, 3)), values[9:]))


def test_bridge_replays_linalg_inverse_3x3_with_real_engine() -> None:
    """With the real engine, a reduced 3x3 inverse program replays bit-exact."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array([2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0], dtype=np.float64)
    result = whole_program_value_and_grad(_objective_inv_3x3_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:inv:3x3 replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_inv_3x3_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def test_bridge_replays_linalg_solve_3x3_with_real_engine() -> None:
    """With the real engine, a reduced 3x3 linear solve replays within tolerance."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = np.array(
        [2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0, 1.0, 2.0, 3.0], dtype=np.float64
    )
    result = whole_program_value_and_grad(_objective_solve_3x3_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks linalg:solve:3x3 replay: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_solve_3x3_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, atol=1.0e-12)


def _diagonally_dominant(n: int, seed: int) -> Any:
    rng = np.random.default_rng(seed)
    return (np.eye(n) * (n + 3.0) + rng.random((n, n))).ravel()


def _objective_det_nxn(values: Any) -> Any:
    n = int(round(float(np.sqrt(values.size))))
    return np.linalg.det(np.reshape(values, (n, n)))


def _objective_inv_nxn_sum(values: Any) -> Any:
    n = int(round(float(np.sqrt(values.size))))
    return np.sum(np.linalg.inv(np.reshape(values, (n, n))))


def test_bridge_replays_general_linalg_det_4x4_with_real_engine() -> None:
    """With the real engine, a 4x4 determinant replays via the general LU path."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = _diagonally_dominant(4, seed=11)
    result = whole_program_value_and_grad(_objective_det_nxn, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks general linalg:det: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_det_nxn, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-9, atol=1.0e-9)


def test_bridge_replays_general_linalg_inverse_5x5_with_real_engine() -> None:
    """With the real engine, a reduced 5x5 inverse replays via the general LU path."""
    pytest.importorskip("scpn_quantum_engine")
    from scpn_quantum_control import program_adjoint_value_and_grad

    sample = _diagonally_dominant(5, seed=23)
    result = whole_program_value_and_grad(_objective_inv_nxn_sum, sample)
    assert result.program_ir is not None
    rust = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, sample)
    if not rust.supported:
        pytest.skip(f"installed engine lacks general linalg:inv: {rust.blocked_reasons}")
    _, reference = program_adjoint_value_and_grad(_objective_inv_nxn_sum, sample)
    np.testing.assert_allclose(np.asarray(rust.gradient), reference, rtol=1.0e-9, atol=1.0e-9)
