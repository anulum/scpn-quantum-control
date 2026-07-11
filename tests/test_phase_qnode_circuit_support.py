# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Support Tests
"""Identity and dependency tests for the Phase-QNode support leaf."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.qnode_circuit as qnode_circuit
import scpn_quantum_control.phase.qnode_circuit_support as support

SUPPORT_FUNCTIONS = (
    "build_registered_phase_qnode_circuit",
    "phase_qnode_depth_profile",
    "phase_qnode_support_report",
    "phase_qnode_density_support_report",
    "phase_qnode_gradient_support_report",
    "plan_phase_qnode_parameter_shift_evaluations",
    "phase_qnode_metric_support_report",
)
PRIVATE_SUPPORT_HELPERS = (
    "_parsed_operations",
    "_parsed_density_operations",
    "_blocked_density_route_support_report",
    "_observable_kind",
    "_as_parameter_vector",
    "_as_optional_positive_int",
    "_parameter_generator_key",
    "_operations_commute_for_shared_parameter",
    "_base_generator_frequencies",
    "_group_generator_frequencies",
    "_parameter_shift_terms_for_group",
    "_collapsible_shared_parameter_group",
    "_parameter_generators_commute",
    "_is_controlled_parametric_gate",
    "_generator_pauli_map",
)


def test_qnode_support_leaf_has_no_executable_facade_back_edge() -> None:
    """Keep support analysis independent from execution and construction."""
    tree = ast.parse(inspect.getsource(support))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert relative_imports.isdisjoint({"__init__", "qnode_circuit", "qnode_circuit_builders"})


def test_qnode_support_keeps_leaf_facade_and_phase_identity() -> None:
    """Re-export every public support route as the same function object."""
    for name in SUPPORT_FUNCTIONS:
        leaf_function = getattr(support, name)
        assert getattr(qnode_circuit, name) is leaf_function
        assert getattr(phase, name) is leaf_function


def test_qnode_private_support_helpers_remain_exact_facade_aliases() -> None:
    """Keep parsing, validation, and shift-planning helpers stable."""
    for name in PRIVATE_SUPPORT_HELPERS:
        assert getattr(qnode_circuit, name) is getattr(support, name)


def test_qnode_executable_facade_defines_no_duplicate_support_functions() -> None:
    """Prevent declarative support functions from drifting back into the facade."""
    tree = ast.parse(inspect.getsource(qnode_circuit))
    facade_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(SUPPORT_FUNCTIONS)
    assert facade_functions.isdisjoint(PRIVATE_SUPPORT_HELPERS)
