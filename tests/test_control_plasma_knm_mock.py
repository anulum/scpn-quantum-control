# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Control Plasma Knm Mock
"""Mock-based tests for bridge/control_plasma_knm.py optional-dependency paths."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scpn_quantum_control.bridge.control_plasma_knm import (
    build_knm_plasma,
    build_knm_plasma_from_config,
    build_knm_plasma_spec,
    plasma_omega,
)


def _make_mock_module(L: int = 8) -> ModuleType:
    """Build a fake scpn_control.phase.plasma_knm module."""
    mod = ModuleType("scpn_control.phase.plasma_knm")

    def build_knm_plasma(**kwargs):
        n = kwargs.get("L", L)
        return SimpleNamespace(
            K=np.eye(n) * 0.45,
            zeta=np.ones(n) * 0.1,
            layer_names=[f"L{i}" for i in range(n)],
        )

    def build_knm_plasma_from_config(**kwargs):
        n = kwargs.get("L", L)
        return SimpleNamespace(K=np.eye(n) * 0.45)

    def _plasma_omega(L=8):
        return np.arange(L, dtype=np.float64)

    mod.build_knm_plasma = build_knm_plasma
    mod.build_knm_plasma_from_config = build_knm_plasma_from_config
    mod.plasma_omega = _plasma_omega
    return mod


@pytest.fixture
def mock_plasma_module(monkeypatch):
    """Monkeypatch importlib.import_module to return mock plasma module."""
    mock_mod = _make_mock_module()

    def _mock_import(name):
        if name == "scpn_control.phase.plasma_knm":
            return mock_mod
        raise ImportError(name)

    monkeypatch.setattr(
        "scpn_quantum_control.bridge.control_plasma_knm.import_module", _mock_import
    )
    return mock_mod


def test_build_knm_plasma_returns_matrix(mock_plasma_module):
    K = build_knm_plasma(L=4)
    assert K.shape == (4, 4)
    assert K.dtype == np.float64


def test_build_knm_plasma_spec_returns_dict(mock_plasma_module):
    spec = build_knm_plasma_spec(L=4)
    assert "K" in spec
    assert "zeta" in spec
    assert "layer_names" in spec
    assert spec["K"].shape == (4, 4)
    assert len(spec["layer_names"]) == 4


def test_build_knm_plasma_from_config_returns_matrix(mock_plasma_module):
    K = build_knm_plasma_from_config(R0=1.65, a=0.5, B0=5.3, Ip=15e6, n_e=1e20, L=4)
    assert K.shape == (4, 4)
    assert K.dtype == np.float64


def test_plasma_omega_returns_vector(mock_plasma_module):
    omega = plasma_omega(L=4)
    assert omega.shape == (4,)
    assert omega.dtype == np.float64


def test_sys_path_insertion_and_cleanup(monkeypatch):
    """Passing repo_src inserts into sys.path and cleans up."""
    mock_mod = _make_mock_module()
    calls = []

    def _mock_import(name):
        calls.append(name)
        if name == "scpn_control.phase.plasma_knm":
            return mock_mod
        raise ImportError(name)

    monkeypatch.setattr(
        "scpn_quantum_control.bridge.control_plasma_knm.import_module", _mock_import
    )

    original_path = sys.path.copy()
    K = build_knm_plasma(L=4, repo_src="/tmp/fake_scpn_control/src")
    assert K.shape == (4, 4)
    # sys.path should be restored (fake path removed)
    assert "/tmp/fake_scpn_control/src" not in sys.path or sys.path == original_path
