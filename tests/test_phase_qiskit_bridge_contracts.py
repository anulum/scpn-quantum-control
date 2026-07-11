# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qiskit Bridge Contract Tests
"""Identity and dependency tests for the Qiskit bridge contract leaf."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.qiskit_bridge as qiskit_bridge
import scpn_quantum_control.phase.qiskit_bridge_contracts as contracts

CONTRACT_CLASSES = (
    "QiskitParameterShiftRecord",
    "QiskitParameterShiftGradientResult",
    "QiskitRuntimePrimitiveExecutionArtifact",
    "QiskitRuntimeQPUExecutionArtifact",
    "QiskitRawCountReplayArtifact",
    "QiskitCalibrationStatevectorComparisonArtifact",
    "QiskitProviderGradientWorkflowArtifact",
    "QiskitRuntimeQPUProviderEvidenceBundle",
    "QiskitMaturityAuditResult",
)
PRIVATE_CONTRACT_SYMBOLS = (
    "FloatArray",
    "QISKIT_PROVIDER_GRADIENT_METHODS",
    "QISKIT_PROVIDER_EVIDENCE_REVIEW_AS_OF_UTC",
    "_QISKIT_PROVIDER_GRADIENT_METHOD_COMMON_METADATA_KEYS",
    "_QISKIT_PROVIDER_GRADIENT_METHOD_SCHEMAS",
    "_QISKIT_PROVIDER_GRADIENT_METHOD_SPECIFIC_METADATA_KEYS",
    "_result_to_dict",
    "_as_finite_vector",
    "_as_finite_scalar",
    "_as_positive_scalar",
    "_validate_sha256_digest",
    "_normalise_metadata_text",
    "_normalise_utc_timestamp",
    "_utc_timestamp",
    "_normalise_qiskit_runtime_primitive",
    "_normalise_qiskit_provider_gradient_method",
    "_normalise_provider_gradient_method_metadata",
    "_require_provider_gradient_method_metadata_value",
    "_normalise_sha256_metadata_digest",
    "_validate_provider_gradient_method_metadata_consistency",
    "_validate_runtime_qpu_mode",
    "_validate_runtime_qpu_evidence_chain",
    "_require_matching_evidence_field",
    "_normalise_positive_int",
    "_normalise_non_negative_int",
    "_normalise_shots",
)


def test_qiskit_contract_leaf_has_no_bridge_back_edge() -> None:
    """Keep record validation independent from executable orchestration."""
    tree = ast.parse(inspect.getsource(contracts))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert not relative_imports


def test_qiskit_contract_classes_keep_leaf_bridge_and_phase_identity() -> None:
    """Re-export every public contract as the same class object."""
    for name in CONTRACT_CLASSES:
        leaf_class = getattr(contracts, name)
        assert getattr(qiskit_bridge, name) is leaf_class
        assert getattr(phase, name) is leaf_class


def test_qiskit_private_contract_symbols_remain_exact_bridge_aliases() -> None:
    """Keep registries, aliases, serialization, and validators stable."""
    for name in PRIVATE_CONTRACT_SYMBOLS:
        assert getattr(qiskit_bridge, name) is getattr(contracts, name)


def test_qiskit_bridge_defines_no_duplicate_contract_symbols() -> None:
    """Prevent contract classes, definitions, or helpers from drifting back."""
    tree = ast.parse(inspect.getsource(qiskit_bridge))
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
    assert bridge_definitions.isdisjoint(PRIVATE_CONTRACT_SYMBOLS)
