# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Control Plasma Knm Mock
"""Mock-based tests for bridge/control_plasma_knm.py — multi-angle coverage."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.bridge.control_plasma_knm import (
    build_knm_plasma,
    build_knm_plasma_from_config,
    build_knm_plasma_spec,
    plasma_omega,
)


def _make_mock_module(L: int = 8) -> ModuleType:
    """Build a fake scpn_control.phase.plasma_knm module."""
    mod = ModuleType("scpn_control.phase.plasma_knm")

    def _build_knm_plasma(**kwargs: object) -> SimpleNamespace:
        n = int(cast(int, kwargs.get("L", L)))
        return SimpleNamespace(
            K=np.eye(n) * 0.45,
            zeta=np.ones(n) * 0.1,
            layer_names=[f"L{i}" for i in range(n)],
        )

    def _build_knm_plasma_from_config(**kwargs: object) -> SimpleNamespace:
        n = int(cast(int, kwargs.get("L", L)))
        return SimpleNamespace(K=np.eye(n) * 0.45)

    def _plasma_omega(L: int = 8) -> NDArray[np.float64]:
        return np.arange(L, dtype=np.float64)

    # A synthesised stand-in module: `ModuleType` declares no such attributes,
    # so populate its namespace directly rather than assert them into existence.
    vars(mod).update(
        build_knm_plasma=_build_knm_plasma,
        build_knm_plasma_from_config=_build_knm_plasma_from_config,
        plasma_omega=_plasma_omega,
    )
    return mod


@pytest.fixture
def mock_plasma_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    mock_mod = _make_mock_module()

    def _mock_import(name: str) -> ModuleType:
        if name == "scpn_control.phase.plasma_knm":
            return mock_mod
        raise ImportError(name)

    monkeypatch.setattr(
        "scpn_quantum_control.bridge.control_plasma_knm.import_module", _mock_import
    )
    return mock_mod


# ---------------------------------------------------------------------------
# build_knm_plasma
# ---------------------------------------------------------------------------


class TestBuildKnmPlasma:
    def test_returns_matrix(self, mock_plasma_module: ModuleType) -> None:
        K = build_knm_plasma(L=4)
        assert K.shape == (4, 4)
        assert K.dtype == np.float64

    def test_default_L(self, mock_plasma_module: ModuleType) -> None:
        K = build_knm_plasma()
        assert K.shape == (8, 8)

    def test_values_from_mock(self, mock_plasma_module: ModuleType) -> None:
        K = build_knm_plasma(L=4)
        np.testing.assert_allclose(np.diag(K), 0.45)


# ---------------------------------------------------------------------------
# build_knm_plasma_spec
# ---------------------------------------------------------------------------


class TestBuildKnmPlasmaSpec:
    def test_returns_dict(self, mock_plasma_module: ModuleType) -> None:
        spec = build_knm_plasma_spec(L=4)
        assert "K" in spec
        assert "zeta" in spec
        assert "layer_names" in spec

    def test_shapes(self, mock_plasma_module: ModuleType) -> None:
        spec = build_knm_plasma_spec(L=4)
        assert spec["K"].shape == (4, 4)
        assert len(spec["layer_names"]) == 4

    def test_zeta_dtype(self, mock_plasma_module: ModuleType) -> None:
        spec = build_knm_plasma_spec(L=4)
        assert spec["zeta"].dtype == np.float64


# ---------------------------------------------------------------------------
# build_knm_plasma_from_config
# ---------------------------------------------------------------------------


class TestBuildKnmPlasmaFromConfig:
    def test_returns_matrix(self, mock_plasma_module: ModuleType) -> None:
        K = build_knm_plasma_from_config(R0=1.65, a=0.5, B0=5.3, Ip=15e6, n_e=1e20, L=4)
        assert K.shape == (4, 4)
        assert K.dtype == np.float64


# ---------------------------------------------------------------------------
# plasma_omega
# ---------------------------------------------------------------------------


class TestPlasmaOmega:
    def test_returns_vector(self, mock_plasma_module: ModuleType) -> None:
        omega = plasma_omega(L=4)
        assert omega.shape == (4,)
        assert omega.dtype == np.float64

    def test_values_from_mock(self, mock_plasma_module: ModuleType) -> None:
        omega = plasma_omega(L=4)
        np.testing.assert_array_equal(omega, np.arange(4, dtype=np.float64))


# ---------------------------------------------------------------------------
# sys.path management
# ---------------------------------------------------------------------------


class TestSysPathManagement:
    def test_sys_path_insertion_and_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_mod = _make_mock_module()

        def _mock_import(name: str) -> ModuleType:
            if name == "scpn_control.phase.plasma_knm":
                return mock_mod
            raise ImportError(name)

        monkeypatch.setattr(
            "scpn_quantum_control.bridge.control_plasma_knm.import_module", _mock_import
        )

        original_path = sys.path.copy()
        K = build_knm_plasma(L=4, repo_src="/tmp/fake_scpn_control/src")
        assert K.shape == (4, 4)
        assert "/tmp/fake_scpn_control/src" not in sys.path or sys.path == original_path

    def test_without_repo_src(self, mock_plasma_module: ModuleType) -> None:
        """Default call (no repo_src) should not modify sys.path."""
        path_before = sys.path.copy()
        build_knm_plasma(L=4)
        assert sys.path == path_before
