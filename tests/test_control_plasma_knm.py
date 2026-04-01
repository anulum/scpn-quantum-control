# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Control Plasma Knm
"""Tests for bridge/control_plasma_knm.py — elite multi-angle coverage."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scpn_quantum_control.bridge.control_plasma_knm import (
    build_knm_plasma,
    build_knm_plasma_from_config,
    build_knm_plasma_spec,
    plasma_omega,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_import_error(*args, **kwargs):
    raise ImportError("Unable to import scpn_control.phase.plasma_knm")


def _make_mock_module(L: int = 8):
    """Return a mock scpn_control.phase.plasma_knm module."""
    K = np.eye(L) * 0.3
    zeta = np.ones(L) * 0.1
    names = [f"layer_{i}" for i in range(L)]

    spec = SimpleNamespace(K=K, zeta=zeta, layer_names=names)
    mod = MagicMock()
    mod.build_knm_plasma.return_value = spec
    mod.build_knm_plasma_from_config.return_value = spec
    mod.plasma_omega.return_value = np.linspace(0.1, 1.0, L)
    return mod


_PATCH_TARGET = "scpn_quantum_control.bridge.control_plasma_knm._import_plasma_knm_module"


# ---------------------------------------------------------------------------
# ImportError paths — all 4 public functions
# ---------------------------------------------------------------------------


class TestImportErrorPaths:
    """All public functions must raise ImportError when scpn_control is absent."""

    @patch(_PATCH_TARGET, side_effect=_make_import_error)
    def test_build_knm_plasma_raises(self, _mock):
        with pytest.raises(ImportError, match="scpn_control"):
            build_knm_plasma()

    @patch(_PATCH_TARGET, side_effect=_make_import_error)
    def test_plasma_omega_raises(self, _mock):
        with pytest.raises(ImportError, match="scpn_control"):
            plasma_omega()

    @patch(_PATCH_TARGET, side_effect=_make_import_error)
    def test_build_knm_plasma_from_config_raises(self, _mock):
        with pytest.raises(ImportError, match="scpn_control"):
            build_knm_plasma_from_config(R0=6.2, a=2.0, B0=5.3, Ip=15.0, n_e=10.1)

    @patch(_PATCH_TARGET, side_effect=_make_import_error)
    def test_build_knm_plasma_spec_raises(self, _mock):
        with pytest.raises(ImportError, match="scpn_control"):
            build_knm_plasma_spec()


# ---------------------------------------------------------------------------
# Happy path with mocked module
# ---------------------------------------------------------------------------


class TestBuildKnmPlasma:
    """build_knm_plasma returns correct ndarray."""

    @patch(_PATCH_TARGET, return_value=_make_mock_module(8))
    def test_returns_ndarray(self, _mock):
        result = build_knm_plasma()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    @patch(_PATCH_TARGET, return_value=_make_mock_module(4))
    def test_shape_matches_L(self, _mock):
        result = build_knm_plasma(L=4)
        assert result.shape == (4, 4)

    @patch(_PATCH_TARGET, return_value=_make_mock_module(8))
    def test_default_params_forwarded(self, mock_import):
        build_knm_plasma()
        mod = mock_import.return_value
        mod.build_knm_plasma.assert_called_once_with(
            mode="baseline",
            L=8,
            K_base=0.30,
            zeta_uniform=0.0,
            custom_overrides=None,
            layer_names=None,
        )


class TestBuildKnmPlasmaSpec:
    """build_knm_plasma_spec returns portable dict."""

    @patch(_PATCH_TARGET, return_value=_make_mock_module(8))
    def test_returns_dict_with_expected_keys(self, _mock):
        result = build_knm_plasma_spec()
        assert set(result.keys()) == {"K", "zeta", "layer_names"}
        assert isinstance(result["K"], np.ndarray)
        assert result["K"].dtype == np.float64

    @patch(_PATCH_TARGET, return_value=_make_mock_module(8))
    def test_zeta_is_ndarray(self, _mock):
        result = build_knm_plasma_spec()
        assert isinstance(result["zeta"], np.ndarray)
        assert result["zeta"].dtype == np.float64

    @patch(_PATCH_TARGET, return_value=_make_mock_module(8))
    def test_layer_names_is_list(self, _mock):
        result = build_knm_plasma_spec()
        assert isinstance(result["layer_names"], list)
        assert len(result["layer_names"]) == 8

    @patch(_PATCH_TARGET)
    def test_none_zeta_passthrough(self, mock_import):
        mod = _make_mock_module(4)
        mod.build_knm_plasma.return_value = SimpleNamespace(
            K=np.eye(4), zeta=None, layer_names=None
        )
        mock_import.return_value = mod
        result = build_knm_plasma_spec(L=4)
        assert result["zeta"] is None
        assert result["layer_names"] is None


class TestBuildKnmPlasmaFromConfig:
    """build_knm_plasma_from_config with tokamak parameters."""

    @patch(_PATCH_TARGET, return_value=_make_mock_module(8))
    def test_returns_ndarray(self, _mock):
        result = build_knm_plasma_from_config(R0=6.2, a=2.0, B0=5.3, Ip=15.0, n_e=10.1)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    @patch(_PATCH_TARGET, return_value=_make_mock_module(8))
    def test_forwards_all_params(self, mock_import):
        build_knm_plasma_from_config(
            R0=6.2, a=2.0, B0=5.3, Ip=15.0, n_e=10.1, mode="enhanced", L=4
        )
        mod = mock_import.return_value
        mod.build_knm_plasma_from_config.assert_called_once_with(
            R0=6.2,
            a=2.0,
            B0=5.3,
            Ip=15.0,
            n_e=10.1,
            mode="enhanced",
            L=4,
            zeta_uniform=0.0,
        )


class TestPlasmaOmega:
    """plasma_omega returns omega vector."""

    @patch(_PATCH_TARGET, return_value=_make_mock_module(8))
    def test_returns_ndarray(self, _mock):
        result = plasma_omega()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    @patch(_PATCH_TARGET, return_value=_make_mock_module(8))
    def test_shape_default_L(self, _mock):
        result = plasma_omega()
        assert result.shape == (8,)

    @patch(_PATCH_TARGET, return_value=_make_mock_module(4))
    def test_forwards_L(self, mock_import):
        plasma_omega(L=4)
        mod = mock_import.return_value
        mod.plasma_omega.assert_called_once_with(L=4)


# ---------------------------------------------------------------------------
# _import_plasma_knm_module internals
# ---------------------------------------------------------------------------


class TestImportMechanism:
    """Verify sys.path manipulation in _import_plasma_knm_module."""

    def test_module_not_found_raises_import_error(self):
        """Without scpn_control installed, direct call raises."""
        from scpn_quantum_control.bridge.control_plasma_knm import (
            _import_plasma_knm_module,
        )

        with pytest.raises(ImportError, match="scpn_control"):
            _import_plasma_knm_module()

    def test_syspath_restored_after_failure(self):
        """sys.path must not leak after a failed import."""
        from scpn_quantum_control.bridge.control_plasma_knm import (
            _import_plasma_knm_module,
        )

        path_before = sys.path.copy()
        with pytest.raises(ImportError):
            _import_plasma_knm_module(repo_src="/nonexistent/path")
        assert sys.path == path_before


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestExports:
    """__all__ must list exactly the public API."""

    def test_all_exports(self):
        from scpn_quantum_control.bridge import control_plasma_knm

        expected = {
            "build_knm_plasma",
            "build_knm_plasma_spec",
            "build_knm_plasma_from_config",
            "plasma_omega",
        }
        assert set(control_plasma_knm.__all__) == expected
