# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TensorFlow Bridge Contract Tests
"""Identity, dependency and import-probe tests for the TensorFlow bridge."""

from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import inspect
import sys
import types
from collections.abc import Sequence

import pytest

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


class _BrokenTensorFlowFinder(importlib.abc.MetaPathFinder):
    """Fail the TensorFlow import the way a broken binary install does."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Raise a plain ImportError for TensorFlow and defer every other module."""
        if fullname == "tensorflow":
            raise ImportError("TensorFlow binary failed to load")
        return None


def _block_tensorflow(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Make ``import tensorflow`` fail as an absent or a broken install."""
    if mode == "absent":
        monkeypatch.setitem(sys.modules, "tensorflow", None)
        return
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenTensorFlowFinder(), *sys.meta_path])


@pytest.mark.parametrize("mode", ["absent", "broken"])
def test_tensorflow_bridge_reports_and_refuses_an_unimportable_tensorflow(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """An absent or broken TensorFlow makes the probe false and routes name the extra."""
    _block_tensorflow(monkeypatch, mode)

    assert tensorflow_bridge.is_phase_tensorflow_available() is False
    assert phase.is_phase_tensorflow_available() is False
    message = r"^TensorFlow is unavailable; install scpn-quantum-control\[tensorflow\]$"
    with pytest.raises(ImportError, match=message) as refusal:
        tensorflow_bridge.tensorflow_bounded_qnn_value_and_grad([[0.0], [3.0]], [0.0, 1.0], [0.8])
    assert isinstance(refusal.value.__cause__, ImportError)


def test_tensorflow_bridge_probe_accepts_an_importable_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any importable ``tensorflow`` module makes the availability probe true."""
    monkeypatch.setitem(sys.modules, "tensorflow", types.ModuleType("tensorflow"))

    assert tensorflow_bridge.is_phase_tensorflow_available() is True
    assert phase.is_phase_tensorflow_available() is True
