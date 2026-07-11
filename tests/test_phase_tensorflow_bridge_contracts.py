# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TensorFlow Bridge Contract Tests
"""Identity and dependency tests for TensorFlow bridge contracts."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.tensorflow_bridge as tensorflow_bridge
import scpn_quantum_control.phase.tensorflow_bridge_contracts as contracts

CONTRACT_CLASSES = (
    "PhaseTensorFlowParameterShiftResult",
    "PhaseTensorFlowQNNGradientResult",
    "PhaseTensorFlowGradientTapeCompatibilityResult",
    "PhaseTensorFlowFunctionCompatibilityResult",
    "PhaseTensorFlowXLACompatibilityResult",
    "PhaseTensorFlowKerasLayerWrapperAuditResult",
    "PhaseTensorFlowMaturityAuditResult",
    "PhaseTensorFlowPhaseQNodeLoweringRoute",
    "PhaseTensorFlowPhaseQNodeLoweringMatrixResult",
)
PRIVATE_CONTRACT_SYMBOLS = ("FloatArray", "_result_to_dict")


def test_tensorflow_contract_leaf_has_no_bridge_back_edge() -> None:
    """Keep declarations independent from executable TensorFlow routes."""
    tree = ast.parse(inspect.getsource(contracts))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert not relative_imports


def test_tensorflow_contract_classes_keep_leaf_bridge_and_phase_identity() -> None:
    """Re-export every public TensorFlow contract as the same class object."""
    for name in CONTRACT_CLASSES:
        leaf_class = getattr(contracts, name)
        assert getattr(tensorflow_bridge, name) is leaf_class
        assert getattr(phase, name) is leaf_class


def test_tensorflow_private_contract_symbols_remain_exact_bridge_aliases() -> None:
    """Keep the array alias and serializer stable through the bridge."""
    for name in PRIVATE_CONTRACT_SYMBOLS:
        assert getattr(tensorflow_bridge, name) is getattr(contracts, name)


def test_tensorflow_bridge_defines_no_duplicate_contract_symbols() -> None:
    """Prevent contract classes, aliases, or helpers from drifting back."""
    tree = ast.parse(inspect.getsource(tensorflow_bridge))
    bridge_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    bridge_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    bridge_definitions = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }
    assert bridge_classes.isdisjoint(CONTRACT_CLASSES)
    assert bridge_functions.isdisjoint(PRIVATE_CONTRACT_SYMBOLS)
    assert "FloatArray" not in bridge_definitions
