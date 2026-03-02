"""Verify public API exports are stable and importable."""

from __future__ import annotations

import numpy as np


def test_top_level_version():
    """Package exposes __version__."""
    import scpn_quantum_control

    assert hasattr(scpn_quantum_control, "__version__")
    assert scpn_quantum_control.__version__ == "0.8.0"


def _check_exports(submod: str) -> None:
    mod = __import__(f"scpn_quantum_control.{submod}", fromlist=["__all__"])
    for name in mod.__all__:
        obj = getattr(mod, name)
        assert callable(obj) or isinstance(obj, (type, np.ndarray, dict, list, str)), (
            f"{submod}.{name} has unexpected type {type(obj)}"
        )


def test_bridge_exports():
    """bridge.__all__ exports are importable and typed."""
    _check_exports("bridge")


def test_phase_exports():
    """phase.__all__ exports are importable and typed."""
    _check_exports("phase")


def test_control_exports():
    """control.__all__ exports are importable and typed."""
    _check_exports("control")


def test_qsnn_exports():
    """qsnn.__all__ exports are importable and typed."""
    _check_exports("qsnn")


def test_mitigation_exports():
    """mitigation.__all__ exports are importable and typed."""
    _check_exports("mitigation")


def test_hardware_exports():
    """hardware.__all__ exports are importable and typed."""
    _check_exports("hardware")


def test_no_private_in_all():
    """No __all__ contains underscore-prefixed names."""

    for submod_name in ["bridge", "phase", "control", "qsnn", "mitigation", "hardware", "qec"]:
        mod = __import__(f"scpn_quantum_control.{submod_name}", fromlist=["__all__"])
        if hasattr(mod, "__all__"):
            private = [n for n in mod.__all__ if n.startswith("_")]
            assert not private, f"{submod_name}.__all__ has private names: {private}"
