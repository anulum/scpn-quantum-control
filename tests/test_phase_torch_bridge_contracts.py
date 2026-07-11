# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Bridge Contract Tests
"""Identity and dependency tests for the Torch result-contract leaf."""

from __future__ import annotations

import ast
import inspect
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.torch_bridge as torch_bridge
import scpn_quantum_control.phase.torch_bridge_contracts as torch_contracts
import scpn_quantum_control.phase.torch_training_loop_matrix as training_loop_matrix

CONTRACT_NAMES = (
    "PhaseTorchAutogradQNNGradientResult",
    "PhaseTorchCloudValidationRunSpec",
    "PhaseTorchCompileBoundaryAuditResult",
    "PhaseTorchCompileBoundaryRoute",
    "PhaseTorchCompileCompatibilityResult",
    "PhaseTorchEcosystemMaturityAuditResult",
    "PhaseTorchEcosystemMaturityRoute",
    "PhaseTorchFuncCompatibilityResult",
    "PhaseTorchLiveOverlayEvidence",
    "PhaseTorchMaturityAuditResult",
    "PhaseTorchModuleWrapperAuditResult",
    "PhaseTorchParameterShiftResult",
    "PhaseTorchPhaseQNodeCompileResult",
    "PhaseTorchPhaseQNodeLoweringMatrixResult",
    "PhaseTorchPhaseQNodeLoweringRoute",
    "PhaseTorchPhaseQNodeStatevectorResult",
    "PhaseTorchPhaseQNodeTransformResult",
    "PhaseTorchQNNGradientResult",
    "PhaseTorchTrainingLoopAuditResult",
)


class _NestedEvidence:
    def to_dict(self) -> dict[str, object]:
        """Return nested values that exercise contract JSON conversion."""
        return {
            "vector": np.array([1.0, 2.0], dtype=np.float64),
            "artifact": Path("evidence.json"),
        }


def test_torch_contract_leaf_has_no_execution_back_edge() -> None:
    """Keep immutable Torch contracts independent from execution modules."""
    tree = ast.parse(inspect.getsource(torch_contracts))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert not relative_imports


def test_torch_contract_identity_is_stable_across_public_facades() -> None:
    """Re-export every Torch contract as the same class object."""
    assert set(torch_contracts.__all__) == set(CONTRACT_NAMES)
    for name in CONTRACT_NAMES:
        contract = getattr(torch_contracts, name)
        assert is_dataclass(contract)
        assert getattr(torch_bridge, name) is contract
        assert getattr(phase, name) is contract


def test_training_loop_matrix_imports_its_contract_directly() -> None:
    """Route the matrix annotation to the leaf while retaining the bridge runner."""
    tree = ast.parse(inspect.getsource(training_loop_matrix))
    relative_imports = {
        node.module: {alias.name for alias in node.names}
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "PhaseTorchTrainingLoopAuditResult" in relative_imports["torch_bridge_contracts"]
    assert "PhaseTorchTrainingLoopAuditResult" not in relative_imports["torch_bridge"]
    assert "run_torch_training_loop_audit" in relative_imports["torch_bridge"]


def test_torch_maturity_contract_serializes_nested_evidence() -> None:
    """Preserve recursive NumPy and path conversion in the aggregate contract."""
    result = torch_contracts.PhaseTorchMaturityAuditResult(
        bounded_model_ready=True,
        ready_for_provider_exceedance=False,
        evidence={"nested": _NestedEvidence()},
        required_capabilities={"provider": "blocked"},
        open_gaps=("provider",),
    )

    payload = result.to_dict()

    assert payload["evidence"] == {"nested": {"vector": [1.0, 2.0], "artifact": "evidence.json"}}
