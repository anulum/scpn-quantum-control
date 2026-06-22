# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the dense K_nm Hamiltonian fallback
"""Native-engine fallback test for the dense XY Hamiltonian builder."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge import knm_hamiltonian as kh
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix


def test_dense_matrix_falls_back_when_engine_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the native engine the dense XY Hamiltonian uses the Qiskit fallback."""
    monkeypatch.setattr(kh, "optional_rust_engine", lambda: None)
    k = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
    omega = np.array([0.2, -0.2], dtype=np.float64)
    hamiltonian = knm_to_dense_matrix(k, omega, delta=0.0)
    assert hamiltonian.shape == (4, 4)
