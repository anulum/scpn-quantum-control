# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — optional hardware import tests
"""Tests for optional simulator dependencies at hardware package import time."""

from __future__ import annotations

import importlib
import sys


def test_noise_model_module_import_does_not_require_aer(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "qiskit_aer.noise", None)

    module = importlib.import_module("scpn_quantum_control.hardware.noise_model")
    module = importlib.reload(module)

    assert module.T1_US == 300.0
    assert callable(module.heron_r2_noise_model)


def test_trapped_ion_module_import_does_not_require_aer(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "qiskit_aer.noise", None)

    module = importlib.import_module("scpn_quantum_control.hardware.trapped_ion")
    module = importlib.reload(module)

    assert module.MS_ERROR > 0.0
    assert callable(module.trapped_ion_noise_model)
