# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Execution Tests
"""Identity and dependency tests for the Phase-QNode execution leaf."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.qnode_circuit as qnode_circuit
import scpn_quantum_control.phase.qnode_circuit_execution as execution

EXECUTION_FUNCTIONS = (
    "execute_phase_qnode_circuit",
    "execute_phase_qnode_density_matrix",
)
PRIVATE_EXECUTION_HELPERS = (
    "_as_density_circuit",
    "_apply_operation",
    "_operation_matrix",
    "_gate_matrix",
    "_controlled",
    "_ccnot_matrix",
    "_cswap_matrix",
    "_apply_gate_matrix",
    "_expanded_operator",
    "_apply_unitary_density_matrix",
    "_apply_noise_channel_density_matrix",
    "_noise_channel_kraus",
    "_expectation_value",
    "_density_expectation_value",
    "_density_term_expectation",
    "_term_expectation",
    "_execute_state",
    "_covariance_expectation",
    "_symmetrized_product_expectation",
    "_term_product_expectation",
    "_density_covariance_expectation",
    "_term_operator",
    "_apply_term_operator",
)


def test_qnode_execution_leaf_has_no_executable_facade_back_edge() -> None:
    """Keep numerical execution independent from orchestration and construction."""
    tree = ast.parse(inspect.getsource(execution))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert relative_imports.isdisjoint({"__init__", "qnode_circuit", "qnode_circuit_builders"})


def test_qnode_execution_keeps_leaf_facade_and_phase_identity() -> None:
    """Re-export both public execution routes as the same function objects."""
    for name in EXECUTION_FUNCTIONS:
        leaf_function = getattr(execution, name)
        assert getattr(qnode_circuit, name) is leaf_function
        assert getattr(phase, name) is leaf_function


def test_qnode_private_execution_helpers_remain_exact_facade_aliases() -> None:
    """Keep state, density, gate, and observable kernels stable."""
    for name in PRIVATE_EXECUTION_HELPERS:
        assert getattr(qnode_circuit, name) is getattr(execution, name)


def test_qnode_executable_facade_defines_no_duplicate_execution_functions() -> None:
    """Prevent numerical execution functions from drifting back into the facade."""
    tree = ast.parse(inspect.getsource(qnode_circuit))
    facade_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(EXECUTION_FUNCTIONS)
    assert facade_functions.isdisjoint(PRIVATE_EXECUTION_HELPERS)
