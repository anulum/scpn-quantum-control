# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — reviewed QPY artefact loader tests
"""Tests for the path-restricted reviewed QPY loader (campaign scripts)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from qiskit import QuantumCircuit, qpy

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "qpy_artifact_io.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qpy_artifact_io_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_loads_circuits_from_data_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_ARTIFACT_ROOT", tmp_path)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    artefact = tmp_path / "matrix.qpy"
    with artefact.open("wb") as stream:
        qpy.dump([circuit, circuit], stream)
    loaded = module.reviewed_qpy_load_circuits(artefact)
    assert len(loaded) == 2
    assert loaded[0] == circuit


def test_rejects_non_qpy_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_ARTIFACT_ROOT", tmp_path)
    payload = tmp_path / "matrix.json"
    payload.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing non-QPY artefact"):
        module.reviewed_qpy_load_circuits(payload)


def test_rejects_path_outside_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_ARTIFACT_ROOT", tmp_path / "data")
    outside = tmp_path / "matrix.qpy"
    with outside.open("wb") as stream:
        qpy.dump([QuantumCircuit(1)], stream)
    with pytest.raises(ValueError, match="outside the repository data tree"):
        module.reviewed_qpy_load_circuits(outside)


def test_default_artifact_root_is_repo_data() -> None:
    module = _load_module()
    assert _MODULE_PATH.parent.parent / "data" == module._ARTIFACT_ROOT
