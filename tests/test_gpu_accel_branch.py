# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the GPU accelerator detection
"""No-device branch test for the CuPy accelerator probe."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scpn_quantum_control.hardware.gpu_accel import _detect_cupy_accelerator


def test_detect_cupy_reports_no_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CuPy install reporting zero devices yields no accelerator."""
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    available, module = _detect_cupy_accelerator()
    assert available is False
    assert module is None
