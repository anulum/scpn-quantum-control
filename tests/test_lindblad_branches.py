# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Lindblad Branch Hardening Tests
"""Branch-level hardening tests for the Lindblad open-system solver."""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase.lindblad import (
    LindbladKuramotoSolver,
    _as_nonnegative_rate,
    _as_real_numeric_array,
)


def _two_node_coupling() -> NDArray[np.float64]:
    """Return a symmetric two-oscillator coupling matrix."""
    return np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)


def _two_node_omega() -> NDArray[np.float64]:
    """Return a two-oscillator natural-frequency vector."""
    return np.array([1.0, 1.2], dtype=np.float64)


def test_rhs_requires_built_hamiltonian_fails_closed() -> None:
    """RHS evaluation should fail explicitly when build() has not populated H."""
    solver = LindbladKuramotoSolver(2, _two_node_coupling(), _two_node_omega())
    rho = np.eye(solver.dim, dtype=np.complex128) / solver.dim

    with pytest.raises(RuntimeError, match="build"):
        solver._rhs(0.0, rho.ravel())


def test_rhs_contains_no_runtime_asserts() -> None:
    """Optimized bytecode must not remove the RHS Hamiltonian guard."""
    source = textwrap.dedent(inspect.getsource(LindbladKuramotoSolver._rhs))
    tree = ast.parse(source)

    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_real_numeric_array_rejects_ragged_input() -> None:
    """Ragged numeric payloads should fail before object coercion."""
    with pytest.raises(ValueError, match="rectangular numeric array"):
        _as_real_numeric_array("K_coupling", [[1.0], [1.0, 2.0]])


def test_real_numeric_array_rejects_complex_values() -> None:
    """Complex inputs should not be silently truncated to real numbers."""
    with pytest.raises(ValueError, match="real numeric scalars"):
        _as_real_numeric_array("K_coupling", np.array([1.0 + 1.0j], dtype=np.complex128))


def test_real_numeric_array_rejects_structured_dtype() -> None:
    """Structured arrays should fail the final numeric conversion guard."""
    structured = np.array([(1, 2)], dtype=[("left", "i4"), ("right", "i4")])

    with pytest.raises(ValueError, match="real numeric scalars"):
        _as_real_numeric_array("K_coupling", structured)


def test_nonnegative_rate_rejects_vector_payload() -> None:
    """Damping rates must be scalar, not one-element vectors."""
    with pytest.raises(ValueError, match="finite non-negative real scalar"):
        _as_nonnegative_rate("gamma_amp", np.array([0.1], dtype=np.float64))


def test_constructor_rejects_boolean_oscillator_count() -> None:
    """Boolean oscillator counts must not pass through int semantics."""
    with pytest.raises(ValueError, match="positive integer"):
        LindbladKuramotoSolver(True, _two_node_coupling(), _two_node_omega())


def test_run_reuses_existing_build_state() -> None:
    """A pre-built solver should not rebuild its Hamiltonian during run()."""
    solver = LindbladKuramotoSolver(2, _two_node_coupling(), _two_node_omega())
    solver.build()
    hamiltonian = solver._H

    result = solver.run(t_max=0.1, dt=0.1)

    assert solver._H is hamiltonian
    assert result["rho_final"].shape == (solver.dim, solver.dim)
