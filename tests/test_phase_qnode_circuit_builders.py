# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Builder Tests
"""Identity and dependency tests for the Phase-QNode construction leaf."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.qnode_circuit as qnode_circuit
import scpn_quantum_control.phase.qnode_circuit_builders as builders

BUILDER_FUNCTIONS = (
    "registered_phase_qnode_gates",
    "registered_phase_qnode_observables",
    "registered_phase_qnode_templates",
    "registered_phase_qnode_decompositions",
    "registered_phase_qnode_noise_channels",
    "build_sparse_ising_chain_hamiltonian",
    "decompose_phase_qnode_controlled_gate",
    "build_phase_qnode_template",
)
PRIVATE_BUILDER_HELPERS = (
    "_as_template_width",
    "_as_ising_chain_width",
    "_as_template_layers",
    "_as_template_entangler",
    "_as_broadcast_coefficients",
    "_normalise_template_observable",
    "_z_magnetization_observable",
    "_ghz_chain_operations",
    "_hardware_efficient_operations",
    "_entangler_operations",
    "_require_qubit_width",
)


def test_qnode_builder_leaf_has_no_executable_facade_back_edge() -> None:
    """Keep circuit construction independent from the executable facade."""
    tree = ast.parse(inspect.getsource(builders))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "qnode_circuit" not in relative_imports
    assert "__init__" not in relative_imports


def test_qnode_builders_keep_leaf_facade_and_phase_identity() -> None:
    """Re-export every public builder as the same function object."""
    for name in BUILDER_FUNCTIONS:
        leaf_function = getattr(builders, name)
        assert getattr(qnode_circuit, name) is leaf_function
        assert getattr(phase, name) is leaf_function


def test_qnode_private_builder_helpers_remain_exact_facade_aliases() -> None:
    """Keep construction validation and operation helpers stable."""
    for name in PRIVATE_BUILDER_HELPERS:
        assert getattr(qnode_circuit, name) is getattr(builders, name)


def test_qnode_executable_facade_defines_no_duplicate_builders() -> None:
    """Prevent construction functions from drifting back into the facade."""
    tree = ast.parse(inspect.getsource(qnode_circuit))
    facade_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(BUILDER_FUNCTIONS)
    assert facade_functions.isdisjoint(PRIVATE_BUILDER_HELPERS)
