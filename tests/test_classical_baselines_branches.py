# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the classical baselines
"""Branch and guard tests for the documented classical Kuramoto baselines.

Covers the SciPy ODE theta0 shape/finiteness guards and integration-failure
path, the QuTiP Hamiltonian zero-coupling skip and the trace/norm state branch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.benchmarks import classical_baselines as cb
from scpn_quantum_control.benchmarks.classical_baselines import (
    _qutip_trace_or_norm,
    _qutip_xy_hamiltonian,
    scipy_ode_baseline,
)

_K = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_OMEGA = np.array([0.1, 0.2], dtype=np.float64)


def test_scipy_ode_rejects_theta0_shape_mismatch() -> None:
    """A theta0 of the wrong shape is rejected."""
    with pytest.raises(ValueError, match="theta0 must have shape"):
        scipy_ode_baseline(_K, _OMEGA, theta0=np.zeros(3, dtype=np.float64))


def test_scipy_ode_rejects_non_finite_theta0() -> None:
    """A non-finite theta0 is rejected."""
    theta0 = np.array([0.0, np.inf], dtype=np.float64)
    with pytest.raises(ValueError, match="theta0 must contain only finite values"):
        scipy_ode_baseline(_K, _OMEGA, theta0=theta0)


def test_scipy_ode_raises_on_integration_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed SciPy integration surfaces as a runtime error."""

    def _failed(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(success=False, message="forced failure")

    monkeypatch.setattr(cb, "solve_ivp", _failed)
    with pytest.raises(RuntimeError, match="SciPy Kuramoto ODE integration failed"):
        scipy_ode_baseline(_K, _OMEGA)


def test_qutip_hamiltonian_skips_zero_couplings() -> None:
    """Near-zero couplings are skipped while building the QuTiP Hamiltonian."""
    qutip = pytest.importorskip("qutip")
    k = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    omega = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    hamiltonian = _qutip_xy_hamiltonian(qutip, k, omega)
    assert hamiltonian.dims == [[2, 2, 2], [2, 2, 2]]


def test_qutip_trace_or_norm_uses_state_norm_for_kets() -> None:
    """A pure-state (non-operator) input is scored by its squared norm."""
    fake_ket = cast(Any, SimpleNamespace(isoper=False, norm=lambda: 1.0))
    assert _qutip_trace_or_norm(fake_ket) == 1.0
