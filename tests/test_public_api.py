"""Verify public API exports are stable and importable."""

from __future__ import annotations


def test_top_level_version():
    """Package exposes __version__."""
    import scpn_quantum_control

    assert hasattr(scpn_quantum_control, "__version__")
    assert scpn_quantum_control.__version__ == "0.7.0"


def test_bridge_exports():
    """bridge.__all__ exports are importable."""
    from scpn_quantum_control.bridge import __all__ as bridge_all

    for name in bridge_all:
        obj = getattr(__import__("scpn_quantum_control.bridge", fromlist=[name]), name)
        assert obj is not None, f"bridge.{name} is None"


def test_phase_exports():
    """phase.__all__ exports are importable."""
    from scpn_quantum_control.phase import __all__ as phase_all

    for name in phase_all:
        obj = getattr(__import__("scpn_quantum_control.phase", fromlist=[name]), name)
        assert obj is not None


def test_control_exports():
    """control.__all__ exports are importable."""
    from scpn_quantum_control.control import __all__ as control_all

    for name in control_all:
        obj = getattr(__import__("scpn_quantum_control.control", fromlist=[name]), name)
        assert obj is not None


def test_qsnn_exports():
    """qsnn.__all__ exports are importable."""
    from scpn_quantum_control.qsnn import __all__ as qsnn_all

    for name in qsnn_all:
        obj = getattr(__import__("scpn_quantum_control.qsnn", fromlist=[name]), name)
        assert obj is not None


def test_mitigation_exports():
    """mitigation.__all__ exports are importable."""
    from scpn_quantum_control.mitigation import __all__ as mit_all

    for name in mit_all:
        obj = getattr(__import__("scpn_quantum_control.mitigation", fromlist=[name]), name)
        assert obj is not None


def test_hardware_exports():
    """hardware.__all__ exports are importable."""
    from scpn_quantum_control.hardware import __all__ as hw_all

    for name in hw_all:
        obj = getattr(__import__("scpn_quantum_control.hardware", fromlist=[name]), name)
        assert obj is not None


def test_no_private_in_all():
    """No __all__ contains underscore-prefixed names."""

    for submod_name in ["bridge", "phase", "control", "qsnn", "mitigation", "hardware", "qec"]:
        mod = __import__(f"scpn_quantum_control.{submod_name}", fromlist=["__all__"])
        if hasattr(mod, "__all__"):
            private = [n for n in mod.__all__ if n.startswith("_")]
            assert not private, f"{submod_name}.__all__ has private names: {private}"
