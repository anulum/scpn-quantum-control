# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the JAX accelerator guard
"""Unavailable-backend guard test for the JAX XY Hamiltonian builder."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.hardware import jax_accel


def test_jax_hamiltonian_requires_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the JAX NumPy backend the builder raises."""
    monkeypatch.setattr(jax_accel, "_jnp", None)
    with pytest.raises(RuntimeError, match="JAX NumPy backend is unavailable"):
        jax_accel._build_xy_hamiltonian_jax(
            np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64), 2
        )
