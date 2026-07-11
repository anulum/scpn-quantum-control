# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Quantum Benchmark Tests
"""Behavioral and structural tests for differentiable quantum benchmark cases."""

from __future__ import annotations

import ast
import inspect
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.benchmarks.differentiable_programming as dp_benchmarks
import scpn_quantum_control.benchmarks.differentiable_programming_quantum as quantum_benchmarks
from scpn_quantum_control.benchmarks import (
    QuantumGradientBenchmarkResult,
    run_quantum_gradient_benchmark_suite,
)


def _require_torch_backend() -> None:
    pytest.importorskip("torch", reason="native Torch quantum-gradient rows require PyTorch")


def test_quantum_gradient_benchmark_suite_matches_analytic_references() -> None:
    """Quantum-gradient rows should expose parameter-shift verification evidence."""

    results = run_quantum_gradient_benchmark_suite()
    dp_any = cast(Any, dp_benchmarks)

    expected_case_ids = [
        "single_rotation_parameter_shift",
        "two_parameter_phase_expectation",
        "sparse_ising_chain_six_qubit_expectation",
    ]
    if dp_any.is_phase_torch_available():
        expected_case_ids.append("torch_registered_phase_qnode_statevector_lowering")
        expected_case_ids.append("torch_registered_phase_qnode_func_transform_lowering")
        expected_case_ids.append("torch_registered_phase_qnode_compile_lowering")
        expected_case_ids.append("torch_registered_phase_qnode_compile_boundary_diagnostic")
    if dp_any.is_phase_jax_available():
        expected_case_ids.append("jax_registered_phase_qnode_native_transform_lowering")
        expected_case_ids.append("jax_registered_phase_qnode_pytree_transform_lowering")
        expected_case_ids.append("jax_registered_phase_qnode_pmap_sharding_lowering")
        expected_case_ids.append("jax_registered_phase_qnode_aot_export_lowering")
    assert [row.case_id for row in results] == expected_case_ids
    for row in results:
        assert isinstance(row, QuantumGradientBenchmarkResult)
        assert row.category == "quantum-gradient"
        assert row.passed
        assert row.verification_passed
        assert row.evaluations > 0
        assert row.max_abs_reference_error <= 1.0e-12
        assert row.max_abs_finite_difference_error <= 1.0e-5
        assert "no wall-clock performance" in row.claim_boundary
        assert "hardware" in row.claim_boundary
        np.testing.assert_allclose(
            row.parameter_shift_gradient,
            row.analytic_gradient,
            atol=1.0e-12,
        )
    sparse_row = next(row for row in results if row.case_id.startswith("sparse_ising"))
    assert sparse_row.parameter_shift_gradient.shape == (6,)
    assert sparse_row.evaluations >= 24
    assert "sparse Hamiltonian" in sparse_row.claim_boundary
    if dp_any.is_phase_torch_available():
        torch_row = next(row for row in results if row.case_id.startswith("torch_registered"))
        assert torch_row.parameter_shift_gradient.shape == (2,)
        assert torch_row.max_abs_reference_error <= 1.0e-8
        assert "native PyTorch autograd statevector lowering" in torch_row.claim_boundary
        assert "no provider, hardware, isolated benchmark, or performance promotion" in (
            torch_row.claim_boundary
        )
    if dp_any.is_phase_jax_available():
        jax_row = next(row for row in results if row.case_id.startswith("jax_registered"))
        assert jax_row.parameter_shift_gradient.shape == (2,)
        assert jax_row.max_abs_reference_error <= 1.0e-8
        assert "native JAX grad" in jax_row.claim_boundary
        assert "no provider, hardware, isolated benchmark, or performance promotion" in (
            jax_row.claim_boundary
        )
        jax_aot_row = next(
            row
            for row in results
            if row.case_id == "jax_registered_phase_qnode_aot_export_lowering"
        )
        assert "jax.export serialization/deserialization diagnostic" in (
            jax_aot_row.claim_boundary
        )
        assert "persistent cross-platform execution" in jax_aot_row.claim_boundary
        assert "no exported VJP" in jax_aot_row.claim_boundary


def test_quantum_benchmark_leaf_has_no_facade_backedge() -> None:
    """Keep quantum cases independent of suite orchestration and other benchmark families."""
    tree = ast.parse(inspect.getsource(quantum_benchmarks))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.endswith("differentiable_programming") for module in imported_modules)


def test_quantum_benchmark_cases_are_exact_facade_aliases() -> None:
    """Preserve every private case-builder identity used by diagnostics and tests."""
    names = (
        "_single_rotation_quantum_gradient_case",
        "_two_parameter_quantum_gradient_case",
        "_sparse_ising_chain_quantum_gradient_case",
        "_torch_registered_phase_qnode_statevector_case",
        "_torch_registered_phase_qnode_func_transform_case",
        "_torch_registered_phase_qnode_compile_case",
        "_torch_registered_phase_qnode_compile_boundary_case",
        "_jax_registered_phase_qnode_native_transform_case",
        "_jax_registered_phase_qnode_pytree_transform_case",
        "_jax_registered_phase_qnode_sharding_transform_case",
        "_jax_registered_phase_qnode_aot_export_case",
        "_quantum_gradient_case",
    )

    for name in names:
        assert getattr(dp_benchmarks, name) is getattr(quantum_benchmarks, name)


def test_benchmark_facade_does_not_redefine_quantum_cases() -> None:
    """Keep moved quantum builders single-owned by the quantum leaf."""
    tree = ast.parse(inspect.getsource(dp_benchmarks))
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    moved = {
        node.name
        for node in ast.parse(inspect.getsource(quantum_benchmarks)).body
        if isinstance(node, ast.FunctionDef)
    }

    assert len(moved) == 12
    assert definitions.isdisjoint(moved)


def test_quantum_benchmark_leaf_excludes_suite_orchestration() -> None:
    """Keep optional-backend ordering and monkeypatch seams in the facade."""
    assert not hasattr(quantum_benchmarks, "run_quantum_gradient_benchmark_suite")
