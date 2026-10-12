# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MPS allocation and discarded-weight tests
"""Physical tensor component counts and local-loss reporting contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_quantum_control.phase.mps_diagnostics import (
    DiscardedWeightLog,
    MPSAllocationError,
    MPSObservation,
    MPSRunDiagnostics,
    require_mps_allocation,
    validate_mps_model,
)


@pytest.mark.parametrize("bond,expected_bytes", [(1, 128), (2, 384), (4, 640), (64, 640)])
def test_physical_schmidt_ranks_bound_declared_storage(bond: int, expected_bytes: int) -> None:
    """Bound four-site complex tensors by actual physical cut dimensions.

    Parameters
    ----------
    bond
        Requested maximum bond dimension.
    expected_bytes
        Independently counted open-boundary complex128 tensor components.

    """
    estimate = require_mps_allocation(4, bond, max_memory_bytes=1024 * 1024)
    assert estimate.state_bytes == expected_bytes
    assert estimate.bond_limit == bond
    assert estimate.n_sites == 4
    assert estimate.state_bytes < estimate.workspace_bytes <= estimate.budget_bytes
    if bond == 1:
        assert estimate.state_bytes < 16 * 2**4
    else:
        assert estimate.state_bytes > 16 * 2**4


def test_component_budget_boundary_and_recovery() -> None:
    """Admit the exact component boundary, refuse one byte less, then recover."""
    estimate = require_mps_allocation(4, 4, observations=3, max_memory_bytes=1024 * 1024)
    boundary = estimate.workspace_bytes
    assert (
        require_mps_allocation(4, 4, observations=3, max_memory_bytes=boundary).workspace_bytes
        == boundary
    )
    with pytest.raises(MPSAllocationError, match="MPS tensor"):
        require_mps_allocation(4, 4, observations=3, max_memory_bytes=boundary - 1)
    assert (
        require_mps_allocation(4, 4, observations=3, max_memory_bytes=boundary).budget_bytes
        == boundary
    )
    with pytest.raises(MPSAllocationError, match="input and observation"):
        require_mps_allocation(4, 4, observations=10**30, max_memory_bytes=boundary)


def test_caller_ceiling_cannot_raise_active_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the process policy ceiling even when a caller requests more bytes.

    Parameters
    ----------
    monkeypatch
        Scoped environment policy override, restored after the test.

    """
    monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str(8192 / 1024**3))
    assert require_mps_allocation(2, 2, max_memory_bytes=16384).budget_bytes == 8192
    assert require_mps_allocation(2, 2).budget_bytes == 8192
    monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str(1024 / 1024**3))
    with pytest.raises(MPSAllocationError, match="MPS tensor"):
        require_mps_allocation(2, 2, max_memory_bytes=16384)


@pytest.mark.parametrize(
    "sites,bond,observations",
    [
        (1, 2, 1),
        (True, 2, 1),
        (2, 0, 1),
        (2, False, 1),
        (2, 1.5, 1),
        (2, 2, 0),
        (2, 2, True),
    ],
)
def test_invalid_component_counts_refuse(sites: int, bond: int, observations: int) -> None:
    """Reject malformed counts without silently coercing them.

    Parameters
    ----------
    sites
        Deliberately malformed site metadata or valid companion count.
    bond
        Deliberately malformed bond metadata or valid companion count.
    observations
        Deliberately malformed record count or valid companion count.

    """
    with pytest.raises(ValueError, match="must be an integer"):
        require_mps_allocation(sites, bond, observations=observations)


@pytest.mark.parametrize("ceiling", [0, -1, True, 1.5])
def test_invalid_byte_ceiling_refuses(ceiling: int) -> None:
    """Reject invalid byte ceilings before any component admission.

    Parameters
    ----------
    ceiling
        Deliberately invalid caller byte metadata.

    """
    with pytest.raises(ValueError, match="positive integer"):
        require_mps_allocation(2, 2, max_memory_bytes=ceiling)


def test_local_discarded_norms_are_squared_without_global_promotion() -> None:
    """Retain actual local reports and their scope as a separate companion."""
    reports = DiscardedWeightLog()
    assert reports.events == 0
    reports["error"] = 0.0
    reports["error"] = 0.25
    reports["error"] = 0.5
    assert reports.events == 3
    assert reports.weight == 5 / 16
    assert reports["error"] == 0.5
    allocation = require_mps_allocation(2, 2)
    observation = MPSObservation(0.0, (2,), 128, 1.0, (0.0, 0.0))
    diagnostics = MPSRunDiagnostics(
        allocation,
        "canonical sweeps",
        (observation,),
        reports.weight,
        "local discarded singular-value norms; not a global bound",
        reports.events,
        0.01,
        cutoff=0.0,
        trotter_order=2,
        time_step=0.05,
        backend_version="fixture; not runtime evidence",
    )
    payload = json.loads(json.dumps(diagnostics.to_dict()))
    assert payload["observations"][0]["local_z"] == [0.0, 0.0]
    assert payload["discarded_weight"] == 5 / 16
    assert payload["trotter_error_estimate"] == 0.01
    assert payload["dense_result_exported"] is False
    assert diagnostics.memory_budget_bytes == allocation.budget_bytes
    assert "mps" not in payload and "amplitudes" not in payload


@pytest.mark.parametrize(
    "key,norm",
    [
        ("unexpected", 0.5),
        ("error", -0.1),
        ("error", np.inf),
        ("error", np.nan),
        ("error", 1e308),
    ],
)
def test_invalid_local_report_preserves_previous_reports(key: str, norm: float) -> None:
    """Reject malformed SVD reports without altering prior accepted metadata.

    Parameters
    ----------
    key
        Diagnostic key sent by the external splitter.
    norm
        Deliberately invalid local norm, or valid norm with an invalid key.

    """
    reports = DiscardedWeightLog()
    reports["error"] = 0.5
    with pytest.raises(ValueError, match="discarded singular-value norm"):
        reports[key] = norm
    assert reports.weight == 0.25 and reports.events == 1
    assert reports["error"] == 0.5
    reports["error"] = 0.25
    assert reports.weight == 5 / 16 and reports.events == 2


def test_finite_symmetric_model_validation_preserves_input() -> None:
    """Admit a strided real model without modifying its caller-owned arrays."""
    base = np.array(
        [[1.0, 2.0, 0.0, 0.0], [2.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    coupling = base[:2, :2]
    frequencies = np.array([0.0, 1.0])
    before = (base.tobytes(), frequencies.tobytes())
    assert validate_mps_model(coupling, frequencies) == 2
    assert (base.tobytes(), frequencies.tobytes()) == before


@pytest.mark.parametrize(
    "coupling,frequencies,reason",
    [
        (np.zeros(2), np.zeros(2), "square"),
        (np.zeros((2, 3)), np.zeros(2), "square"),
        (np.zeros((1, 1)), np.zeros(1), "square"),
        (np.zeros((2, 2)), np.zeros(3), "frequency"),
        (np.zeros((2, 2), dtype=np.complex128), np.zeros(2), "real"),
        (np.zeros((2, 2)), np.zeros(2, dtype=np.complex128), "real"),
        (np.full((2, 2), np.nan), np.zeros(2), "finite"),
        (np.zeros((2, 2)), np.full(2, np.inf), "finite"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros(2), "symmetric"),
    ],
)
def test_invalid_model_refuses(coupling: np.ndarray, frequencies: np.ndarray, reason: str) -> None:
    """Refuse invalid model data and leave its original bytes unchanged.

    Parameters
    ----------
    coupling
        Deliberately malformed square-model input.
    frequencies
        Deliberately malformed or valid companion frequencies.
    reason
        Expected located validation reason.

    """
    before = (coupling.tobytes(), frequencies.tobytes())
    with pytest.raises(ValueError, match=reason):
        validate_mps_model(coupling, frequencies)
    assert (coupling.tobytes(), frequencies.tobytes()) == before
