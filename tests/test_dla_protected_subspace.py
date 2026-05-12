# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for DLA-Protected Logical Subspaces
"""Tests for DLA-protected logical synchronisation memory."""

import numpy as np
import pytest

from scpn_quantum_control.analysis.logical_sync_witness import LogicalSyncWitness
from scpn_quantum_control.qec import (
    DLAProtectedLogicalSyncWitness,
    DLAProtectedSubspaceSpec,
    build_dla_protected_memory_prototype,
    certify_dla_protected_subspace,
    evaluate_dla_protected_memory,
    protected_memory_mask,
    sync_memory_mask,
)
from scpn_quantum_control.qec.dla_protected_subspace import (
    _memory_metrics_numpy,
    _protected_memory_mask_numpy,
)


def test_certificate_matches_dla_parity_sector_dimensions():
    spec = DLAProtectedSubspaceSpec(n_logical=3, code_distance=3, target_parity=1)
    certificate = certify_dla_protected_subspace(spec)

    assert certificate.is_provable is True
    assert certificate.physical_dla_dimension == 2 ** (2 * spec.n_physical - 1) - 2
    assert certificate.even_sector_dim == 2 ** (spec.n_physical - 1)
    assert certificate.odd_sector_dim == 2 ** (spec.n_physical - 1)
    assert certificate.protected_logical_dim == 4
    assert len(certificate.protected_basis_indices) == 4
    assert set(certificate.sync_basis_indices).issubset(certificate.protected_basis_indices)


def test_protected_memory_mask_selects_fixed_parity_repetition_words():
    spec = DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0)
    mask = protected_memory_mask(spec)
    fallback = _protected_memory_mask_numpy(spec)

    np.testing.assert_array_equal(mask, fallback)
    assert mask.sum() == 2
    assert mask[0b000_000]
    assert mask[0b111_111]
    assert not mask[0b000_111]
    assert not mask[0b010_111]


def test_sync_mask_can_be_stricter_than_protected_sector():
    spec = DLAProtectedSubspaceSpec(n_logical=4, code_distance=1, target_parity=0)
    protected = protected_memory_mask(spec)
    sync = sync_memory_mask(spec)

    assert protected.sum() == 8
    assert sync.sum() == 2
    assert sync[0b0000]
    assert sync[0b1111]
    assert not sync[0b0011]


def test_memory_prototype_prepares_only_valid_target_parity_words():
    spec = DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0)
    prototype = build_dla_protected_memory_prototype(spec, logical_word=(1, 1))

    assert prototype.basis_index == 0b111_111
    assert prototype.circuit.num_qubits == 6
    assert prototype.circuit.size() == 6
    with pytest.raises(ValueError, match="target DLA parity"):
        build_dla_protected_memory_prototype(spec, logical_word=(1, 0))


def test_witness_passes_for_synchronised_protected_memory_state():
    spec = DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0)
    probabilities = np.zeros(spec.hilbert_dim, dtype=np.float64)
    probabilities[0b111_111] = 1.0

    result = evaluate_dla_protected_memory(probabilities, spec=spec)

    assert result.passes is True
    assert result.protected_weight == pytest.approx(1.0)
    assert result.sync_weight == pytest.approx(1.0)
    assert result.logical_sync_order == pytest.approx(1.0)
    assert result.parity_leakage == pytest.approx(0.0)
    assert result.code_leakage == pytest.approx(0.0)


def test_witness_reports_code_and_parity_failure_modes():
    spec = DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0)
    probabilities = np.zeros(spec.hilbert_dim, dtype=np.float64)
    probabilities[0b000_000] = 0.82
    probabilities[0b000_111] = 0.08
    probabilities[0b010_111] = 0.10

    result = evaluate_dla_protected_memory(probabilities, spec=spec)

    assert result.passes is False
    assert result.protected_weight == pytest.approx(0.82)
    assert result.parity_leakage == pytest.approx(0.08)
    assert result.code_leakage == pytest.approx(0.10)
    assert "protected_weight_below_threshold" in result.failure_reasons
    assert "parity_leakage_above_threshold" in result.failure_reasons


def test_counts_witness_uses_reversed_qiskit_bit_order_for_block_layout():
    spec = DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0)
    result = DLAProtectedLogicalSyncWitness(spec)(
        counts={
            "000000": 60,
            "111111": 40,
        }
    )

    assert result.passes is True
    assert result.protected_weight == pytest.approx(1.0)
    assert result.sync_weight == pytest.approx(1.0)


@pytest.mark.parametrize(
    "counts",
    [
        {"000000": 1.5, "111111": 2},
        {"000000": True, "111111": 2},
        {"000000": -1, "111111": 2},
    ],
)
def test_counts_witness_rejects_non_integral_or_negative_shots(counts):
    spec = DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0)

    with pytest.raises(ValueError, match="counts"):
        evaluate_dla_protected_memory(counts=counts, spec=spec)


def test_legacy_logical_sync_witness_returns_concrete_metrics():
    spec = DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0)
    report = LogicalSyncWitness(spec)(
        counts={
            "000000": 9,
            "000111": 1,
        }
    )

    assert report["logical_fidelity"] == pytest.approx(0.9)
    assert report["parity_leakage"] == pytest.approx(0.1)
    assert report["passes"] is False


def test_numpy_metrics_match_expected_weights():
    spec = DLAProtectedSubspaceSpec(n_logical=2, code_distance=3, target_parity=0)
    probabilities = np.zeros(spec.hilbert_dim, dtype=np.float64)
    probabilities[0b000_000] = 0.5
    probabilities[0b111_111] = 0.25
    probabilities[0b000_111] = 0.15
    probabilities[0b010_111] = 0.10

    protected, code, target, opposite, total = _memory_metrics_numpy(probabilities, spec)

    assert protected == pytest.approx(0.75)
    assert code == pytest.approx(0.90)
    assert target == pytest.approx(0.85)
    assert opposite == pytest.approx(0.15)
    assert total == pytest.approx(1.0)


def test_spec_validation_rejects_invalid_dense_and_threshold_inputs():
    with pytest.raises(ValueError, match="odd"):
        DLAProtectedSubspaceSpec(n_logical=2, code_distance=2)
    with pytest.raises(ValueError, match="target_parity"):
        DLAProtectedSubspaceSpec(n_logical=2, target_parity=3)
    with pytest.raises(ValueError, match="min_sync_weight"):
        DLAProtectedSubspaceSpec(n_logical=2, min_sync_weight=1.2)
    with pytest.raises(ValueError, match="limited to 24"):
        _ = DLAProtectedSubspaceSpec(n_logical=9, code_distance=3).hilbert_dim
