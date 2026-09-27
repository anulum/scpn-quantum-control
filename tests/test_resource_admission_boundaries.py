# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — public resource-admission boundaries
"""Exercise product admission through its public resource-budget facade."""

from __future__ import annotations

import pytest

from scpn_quantum_control._rust_accel import optional_rust_engine
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.resource_budget_gate import (
    ExecutionBuffer,
    ExecutionMemoryPlan,
    MemoryCapacity,
    check_execution_memory,
    estimate_resource_budget,
    require_execution_memory,
)


@pytest.mark.parametrize("delta, allowed", [(-1, True), (0, True), (1, False)])
def test_resource_admission_boundaries_01(delta: int, allowed: bool) -> None:
    """Use independent byte-sized buffers at B-1/B/B+1 through the facade."""
    plan = ExecutionMemoryPlan(
        (ExecutionBuffer("output", "dense_output", (256 + delta,), "uint8"),)
    )
    decision = check_execution_memory(plan, MemoryCapacity(4096), max_bytes=256)
    assert decision.allowed is allowed
    assert decision.bytes_required == 256 + delta


def test_resource_admission_boundaries_02() -> None:
    """Product estimates refuse impossible exponents before building a dense shape."""
    with pytest.raises(DenseAllocationError, match="addressable"):
        estimate_resource_budget("dense_hilbert_default", n_qubits=10**100)


def test_resource_admission_missing_native_symbol_never_runs_python() -> None:
    """A native-only unavailable symbol refuses even with a valid tiny memory plan."""
    plan = ExecutionMemoryPlan((ExecutionBuffer("state", "forward", (2,), "float64"),))
    with pytest.raises(DenseAllocationError, match="native"):
        require_execution_memory(plan, native_symbol="unavailable_resource_admission_kernel")


def test_resource_admission_boundaries_03() -> None:
    """Exercise a real native-only export in the actual runner's backend configuration."""
    import numpy as np

    from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

    coupling = np.zeros((2, 2))
    frequencies = np.array([1.0, 2.0])
    engine = optional_rust_engine()
    if engine is None or not callable(getattr(engine, "build_xy_hamiltonian_dense", None)):
        with pytest.raises(DenseAllocationError, match="native"):
            knm_to_dense_matrix(coupling, frequencies, backend="rust")
    else:
        result = knm_to_dense_matrix(coupling, frequencies, backend="rust")
        np.testing.assert_allclose(result, np.diag([-3, -1, 1, 3]), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(coupling, np.zeros((2, 2)))
    np.testing.assert_array_equal(frequencies, [1.0, 2.0])
