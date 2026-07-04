# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto relocation shim tests
"""The relocated Kuramoto surface keeps its old import paths working, deprecated.

The Kuramoto toolkit moved to the standalone :mod:`oscillatools` distribution; the old
``scpn_quantum_control.kuramoto`` / ``scpn_quantum_control.accel`` /
``scpn_quantum_control.forecasting.kuramoto_neural_operator`` import paths remain as
re-export shims for the deprecation window. These tests assert that each shim (a) emits a
``DeprecationWarning`` naming the new path, (b) re-exports the same objects as the new home,
and (c) that importing ``scpn_quantum_control`` itself stays warning-free (its own code was
repointed to :mod:`oscillatools`).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import pytest

_SHIMS = [
    ("scpn_quantum_control.kuramoto", "oscillatools", "order_parameter"),
    ("scpn_quantum_control.accel", "oscillatools.accel", "order_parameter"),
    (
        "scpn_quantum_control.forecasting.kuramoto_neural_operator",
        "oscillatools.neural_operator",
        "simulate_operator_dataset",
    ),
]


def _reimport_recording(name: str) -> tuple[object, list[warnings.WarningMessage]]:
    """Drop ``name`` (and cached submodules) from ``sys.modules`` and re-import, recording warnings."""
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(name)
    return module, [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize("old_path,new_path,symbol", _SHIMS)
def test_shim_warns_and_reexports(old_path: str, new_path: str, symbol: str) -> None:
    """Each shim emits a DeprecationWarning naming the new home and re-exports its objects."""
    module, deprecations = _reimport_recording(old_path)
    messages = [str(w.message) for w in deprecations]
    assert any(new_path in m for m in messages), (
        f"{old_path} should warn naming {new_path}; got {messages}"
    )
    new_module = importlib.import_module(new_path)
    assert getattr(module, symbol) is getattr(new_module, symbol)


def test_accel_submodule_identity_preserved() -> None:
    """The accel shim resolves submodules to the *same* object as the new package."""
    from oscillatools.accel.networked_kuramoto import networked_kuramoto_force as canonical
    from scpn_quantum_control.accel.networked_kuramoto import networked_kuramoto_force as shimmed

    assert shimmed is canonical


def test_importing_parent_package_is_shim_warning_free() -> None:
    """``import scpn_quantum_control`` must not emit a relocation DeprecationWarning.

    Its internal consumers were repointed to :mod:`oscillatools`, so only *external* callers
    of the old paths should ever see the shim warning.
    """
    probe = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    import scpn_quantum_control\n"
        "moved = [str(x.message) for x in w if 'has moved to oscillatools' in str(x.message)]\n"
        "assert not moved, moved\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
