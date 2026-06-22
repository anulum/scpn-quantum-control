# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for DLA-protected scar memory
"""Branch and fail-closed tests for the DLA-protected scar-memory helpers.

Covers the scar-word resolution and default sync-pair guards, the native-engine
trajectory-metric fallback, the failure-reason ladder, the non-cat preparation
circuit, the count-to-probability validators and the short-word agreement edge.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.qec import DLAProtectedScarSpec, DLAProtectedSubspaceSpec
from scpn_quantum_control.qec.dla_protected_scar import (
    _default_sync_pair,
    _logical_agreement,
    _probabilities_from_counts,
    _resolve_scar_words,
    _scar_failure_reasons,
    _scar_preparation_circuit,
    simulate_dla_protected_scar_memory,
)
from scpn_quantum_control.qec.dla_protected_subspace import DLAProtectionCertificate

_Word = tuple[int, ...]


def _cert(
    *,
    protected: tuple[_Word, ...],
    sync: tuple[_Word, ...],
    threshold: float = 1.0,
) -> DLAProtectionCertificate:
    spec = DLAProtectedSubspaceSpec(
        n_logical=2, code_distance=3, target_parity=0, sync_agreement_threshold=threshold
    )
    return DLAProtectionCertificate(
        spec=spec,
        physical_dla_dimension=4,
        even_sector_dim=2,
        odd_sector_dim=2,
        protected_logical_dim=2,
        protected_basis_indices=(0,),
        sync_basis_indices=(0,),
        protected_logical_words=protected,
        sync_logical_words=sync,
        proof_obligations={"ok": True},
    )


def test_resolve_scar_words_requires_two_words() -> None:
    """A single requested scar word cannot drive a revival."""
    cert = _cert(protected=((0, 0), (1, 1)), sync=((0, 0), (1, 1)))
    with pytest.raises(ValueError, match="at least two scar logical words"):
        _resolve_scar_words(cert, [(0, 0)])


def test_resolve_scar_words_enforces_agreement_threshold() -> None:
    """A protected but low-agreement word fails the synchronisation threshold."""
    cert = _cert(protected=((0, 0), (0, 1)), sync=((0, 0),), threshold=0.5)
    with pytest.raises(ValueError, match="synchronisation threshold"):
        _resolve_scar_words(cert, [(0, 0), (0, 1)])


def test_default_sync_pair_uses_first_two_sync_words() -> None:
    """Without the cat pair, the first two sync words seed the revival."""
    cert = _cert(protected=((0, 1), (1, 0)), sync=((0, 1), (1, 0)))
    assert _default_sync_pair(cert) == ((0, 1), (1, 0))


def test_default_sync_pair_requires_two_sync_words() -> None:
    """A sync sector with a single word cannot seed a revival."""
    cert = _cert(protected=((0, 1),), sync=((0, 1),))
    with pytest.raises(ValueError, match="at least two scar words"):
        _default_sync_pair(cert)


def test_trajectory_metrics_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising native trajectory kernel falls back to the Python evaluator."""

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("engine refused the trajectory metrics")

    stub = types.ModuleType("scpn_quantum_engine")
    stub.dla_protected_trajectory_metrics = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    spec = DLAProtectedScarSpec(
        memory_spec=DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0),
        revival_period=1.0,
        n_time_steps=4,
    )
    result = simulate_dla_protected_scar_memory(spec=spec)
    assert result.backend == "python:evaluate_dla_protected_memory"


def test_scar_failure_reasons_collects_all_threshold_breaches() -> None:
    """Every breached criterion is reported as a distinct failure reason."""
    prototype = types.SimpleNamespace(
        certificate=types.SimpleNamespace(is_provable=False),
        spec=types.SimpleNamespace(
            min_revival_fidelity=1.0,
            min_protected_weight=1.0,
            max_parity_leakage=0.0,
            min_scar_support=1.0,
        ),
    )
    reasons = _scar_failure_reasons(
        prototype,  # type: ignore[arg-type]
        final_revival_fidelity=0.0,
        min_protected_weight=0.0,
        max_parity_leakage=1.0,
        min_scar_support=0.0,
    )
    assert "protection_certificate_failed" in reasons
    assert "revival_fidelity_below_threshold" in reasons


def test_preparation_circuit_initialises_non_cat_words() -> None:
    """A word set that is not the {zeros, ones} cat pair uses explicit initialisation."""
    memory_spec = DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0)
    state = np.zeros(memory_spec.hilbert_dim, dtype=np.complex128)
    state[0] = 1.0
    circuit = _scar_preparation_circuit(memory_spec, ((0, 0), (1, 1), (0, 1)), state)
    assert circuit.num_qubits == memory_spec.n_physical


def test_probabilities_from_counts_rejects_empty_total() -> None:
    """A zero shot total is rejected."""
    with pytest.raises(ValueError, match="counts must contain positive shot total"):
        _probabilities_from_counts({}, 6, 64)


def test_probabilities_from_counts_rejects_bad_bitstring() -> None:
    """A bitstring of the wrong length is rejected."""
    with pytest.raises(ValueError, match="bitstrings must have length 6"):
        _probabilities_from_counts({"01": 3}, 6, 64)


def test_probabilities_from_counts_rejects_negative_shots() -> None:
    """A negative shot count for a valid bitstring is rejected."""
    with pytest.raises(ValueError, match="counts must be non-negative"):
        _probabilities_from_counts({"000000": 5, "111111": -1}, 6, 64)


def test_logical_agreement_unit_for_short_word() -> None:
    """A word shorter than two bits is perfectly self-consistent."""
    assert _logical_agreement((0,)) == 1.0
