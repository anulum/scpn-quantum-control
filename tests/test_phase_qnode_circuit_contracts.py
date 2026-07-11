# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Contract Tests
"""Identity and dependency tests for the Phase-QNode contract leaf."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.qnode_circuit as qnode_circuit
import scpn_quantum_control.phase.qnode_circuit_contracts as contracts

CONTRACT_CLASSES = (
    "PhaseQNodeOperation",
    "PhaseQNodeNoiseChannel",
    "PauliTerm",
    "SparsePauliHamiltonian",
    "DenseHermitianObservable",
    "PauliCovarianceObservable",
    "PhaseQNodeSupportReport",
    "PhaseQNodeSupportError",
    "PhaseQNodeCircuit",
    "PhaseQNodeDensityCircuit",
    "PhaseQNodeTemplateSpec",
    "PhaseQNodeDepthProfile",
    "PhaseQNodeRegisteredCircuitSpec",
    "PhaseQNodeExecutionResult",
    "PhaseQNodeDensityExecutionResult",
    "PhaseQNodeGradientResult",
    "PhaseQNodeGradientEvaluationGroup",
    "PhaseQNodeGradientEvaluationPlan",
    "PhaseQNodeMetricTensorResult",
    "PhaseQNodeClassicalFisherResult",
)
PRIVATE_CONTRACT_SYMBOLS = (
    "FloatArray",
    "ComplexArray",
    "OperationSpec",
    "DensityOperationSpec",
    "DensityOperation",
    "_NON_PARAMETRIC_GATES",
    "_PARAMETRIC_GATES",
    "_REGISTERED_GATES",
    "_GATE_ARITY",
    "_REGISTERED_OBSERVABLES",
    "_REGISTERED_TEMPLATES",
    "_REGISTERED_DECOMPOSITIONS",
    "_REGISTERED_NOISE_CHANNELS",
    "_I",
    "_X",
    "_Y",
    "_Z",
    "_H",
    "_S",
    "_T",
    "_SX",
    "_PAULI",
    "_PAULI_MATRICES",
    "_FiniteShotFisherEvidence",
    "_parse_operation",
    "_parse_density_operation",
    "_normalise_observable",
    "_as_finite_scalar",
    "_as_probability",
    "_optional_float_array_to_list",
)


def test_qnode_contract_leaf_has_no_executable_facade_back_edge() -> None:
    """Keep shared declarations independent from the executable facade."""
    tree = ast.parse(inspect.getsource(contracts))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "qnode_circuit" not in relative_imports
    assert "__init__" not in relative_imports


def test_qnode_contract_classes_keep_leaf_facade_and_phase_identity() -> None:
    """Re-export every public contract as the same class object."""
    for name in CONTRACT_CLASSES:
        leaf_class = getattr(contracts, name)
        assert getattr(qnode_circuit, name) is leaf_class
        assert getattr(phase, name) is leaf_class


def test_qnode_private_contract_symbols_remain_exact_facade_aliases() -> None:
    """Keep registries, matrices, aliases, and constructor helpers stable."""
    for name in PRIVATE_CONTRACT_SYMBOLS:
        assert getattr(qnode_circuit, name) is getattr(contracts, name)


def test_qnode_executable_facade_defines_no_duplicate_contract_classes() -> None:
    """Prevent declaration records from drifting back into the executable facade."""
    tree = ast.parse(inspect.getsource(qnode_circuit))
    facade_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert facade_classes.isdisjoint(CONTRACT_CLASSES)
    assert "_FiniteShotFisherEvidence" not in facade_classes
