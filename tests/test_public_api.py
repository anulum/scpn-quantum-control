# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Public Api
"""Verify public API exports are stable and importable."""

from __future__ import annotations

import numpy as np


def test_top_level_version():
    """Package exposes __version__ matching pyproject.toml."""
    import importlib.metadata

    import scpn_quantum_control

    assert hasattr(scpn_quantum_control, "__version__")
    try:
        expected = importlib.metadata.version("scpn-quantum-control")
        assert scpn_quantum_control.__version__ == expected
    except importlib.metadata.PackageNotFoundError:
        # Fallback for development environments where package is not installed
        assert isinstance(scpn_quantum_control.__version__, str)
        assert len(scpn_quantum_control.__version__.split(".")) >= 3


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


# ---------------------------------------------------------------------------
# Top-level __all__ completeness
# ---------------------------------------------------------------------------


def test_top_level_all_nonempty():
    import scpn_quantum_control

    assert len(scpn_quantum_control.__all__) > 50


def test_top_level_all_no_duplicates():
    import scpn_quantum_control

    names = scpn_quantum_control.__all__
    assert len(names) == len(set(names))


def test_qec_exports():
    """qec.__all__ exports are importable and typed."""
    _check_exports("qec")


def test_all_submodules_importable():
    """Every known subpackage imports without error."""
    for submod in [
        "bridge",
        "phase",
        "control",
        "qsnn",
        "mitigation",
        "hardware",
        "qec",
        "analysis",
        "identity",
        "crypto",
    ]:
        mod = __import__(f"scpn_quantum_control.{submod}")
        assert mod is not None
