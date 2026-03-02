"""Tests for bridge/control_plasma_knm.py ImportError paths."""

from unittest.mock import patch

import pytest

from scpn_quantum_control.bridge.control_plasma_knm import (
    build_knm_plasma,
    build_knm_plasma_from_config,
    plasma_omega,
)


def _import_raises(*args, **kwargs):
    raise ModuleNotFoundError("No module named 'scpn_control'")


@patch(
    "scpn_quantum_control.bridge.control_plasma_knm._import_plasma_knm_module",
    side_effect=ImportError("Unable to import scpn_control.phase.plasma_knm"),
)
def test_build_knm_plasma_raises_without_scpn_control(_mock):
    with pytest.raises(ImportError, match="scpn_control"):
        build_knm_plasma()


@patch(
    "scpn_quantum_control.bridge.control_plasma_knm._import_plasma_knm_module",
    side_effect=ImportError("Unable to import scpn_control.phase.plasma_knm"),
)
def test_plasma_omega_raises_without_scpn_control(_mock):
    with pytest.raises(ImportError, match="scpn_control"):
        plasma_omega()


@patch(
    "scpn_quantum_control.bridge.control_plasma_knm._import_plasma_knm_module",
    side_effect=ImportError("Unable to import scpn_control.phase.plasma_knm"),
)
def test_build_knm_plasma_from_config_raises(_mock):
    with pytest.raises(ImportError, match="scpn_control"):
        build_knm_plasma_from_config(R0=6.2, a=2.0, B0=5.3, Ip=15.0, n_e=10.1)
