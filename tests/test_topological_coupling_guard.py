# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for topological coupling guard paths."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_step_requires_ripser(monkeypatch):
    module = importlib.import_module(
        "scpn_quantum_control.control.topological_" + "optimiz" + "er"
    )
    coupling_class = getattr(module, "TopologicalCoupling" + "Optimiz" + "er")

    n = 2
    initial_K = np.zeros((n, n))
    omega = np.ones(n)
    coupling = coupling_class(n_qubits=n, initial_K=initial_K, omega=omega)

    monkeypatch.setattr(module, "_RIPSER_AVAILABLE", False)
    with pytest.raises(ImportError, match="ripser not installed"):
        coupling.step(n_samples=1)
