# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Unsupported-Pauli guard test for the classical backend
"""Guard test for the classical single-qubit Pauli expectation helper."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.hardware import classical


def test_expectation_pauli_rejects_unsupported_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the native engine absent, an unsupported Pauli label is rejected."""
    monkeypatch.setattr(classical, "optional_rust_engine", lambda: None)
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError, match="unsupported Pauli label"):
        classical._expectation_pauli(psi, 1, 0, "W")
