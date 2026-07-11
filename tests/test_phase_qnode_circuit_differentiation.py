# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Differentiation Tests
"""Identity and dependency tests for the Phase-QNode differentiation leaf."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.qnode_circuit as qnode_circuit
import scpn_quantum_control.phase.qnode_circuit_differentiation as differentiation

DIFFERENTIATION_FUNCTIONS = (
    "phase_qnode_computational_basis_fisher_support_report",
    "parameter_shift_phase_qnode_gradient",
    "phase_qnode_quantum_fisher_information",
    "phase_qnode_computational_basis_fisher_information",
    "phase_qnode_natural_gradient_metric",
)
PRIVATE_DIFFERENTIATION_HELPERS = (
    "_parametric_operations_by_parameter",
    "_as_min_probability",
    "_as_shot_count",
    "_as_confidence_level",
    "_as_confidence_z",
    "_as_observed_count_record",
    "_classical_fisher_from_probabilities",
    "_classical_fisher_delta_method_standard_error",
    "_finite_shot_fisher_evidence",
    "_execute_state_and_parameter_derivatives",
    "_operation_derivative_matrix",
    "_gate_derivative_matrix",
    "_covariance_product_rule_gradient",
)


def test_qnode_differentiation_leaf_has_no_facade_back_edge() -> None:
    """Keep differentiation independent from the compatibility facade."""
    tree = ast.parse(inspect.getsource(differentiation))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert relative_imports.isdisjoint({"__init__", "qnode_circuit", "qnode_circuit_builders"})


def test_qnode_differentiation_keeps_leaf_facade_and_phase_identity() -> None:
    """Re-export every public differentiation route as the same function object."""
    for name in DIFFERENTIATION_FUNCTIONS:
        leaf_function = getattr(differentiation, name)
        assert getattr(qnode_circuit, name) is leaf_function
        assert getattr(phase, name) is leaf_function


def test_qnode_private_differentiation_helpers_remain_exact_facade_aliases() -> None:
    """Keep derivative, Fisher, validation, and statistics helpers stable."""
    for name in PRIVATE_DIFFERENTIATION_HELPERS:
        assert getattr(qnode_circuit, name) is getattr(differentiation, name)


def test_qnode_compatibility_facade_defines_no_functions() -> None:
    """Keep the completed compatibility facade free of executable definitions."""
    tree = ast.parse(inspect.getsource(qnode_circuit))
    assert not [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
