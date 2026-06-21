# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the DLA-protected subspace memory
"""Serialisation, validation, and fallback branch tests for DLA-protected memory.

Covers the certificate, prototype, and witness serialisers and leakage
properties, the probability and counts resolution guards, the failure-reason
thresholds, the default sync-word selection across parity sectors, the spec
guard, and the native/numpy mask and metric fallbacks.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.qec.dla_protected_subspace import (
    DLAProtectedLogicalSyncWitness,
    DLAProtectedSubspaceSpec,
    _default_sync_word,
    _failure_reasons,
    _logical_agreement,
    _memory_metrics,
    _memory_metrics_numpy,
    _protected_memory_mask_numpy,
    _validate_logical_word,
    build_dla_protected_memory_prototype,
    certify_dla_protected_subspace,
    evaluate_dla_protected_memory,
    protected_memory_mask,
)


def test_certificate_serialises_and_reports_provability() -> None:
    """The analytic certificate serialises and reports its provability."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    certificate = certify_dla_protected_subspace(spec)
    payload = certificate.to_dict()
    assert payload["n_logical"] == 2
    assert payload["is_provable"] == certificate.is_provable


def test_prototype_default_word_and_serialisation() -> None:
    """A default all-zero word skips excitations and the prototype serialises."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    prototype = build_dla_protected_memory_prototype(spec)
    payload = prototype.to_dict()
    assert payload["logical_word"] == [0, 0]
    assert payload["n_qubits"] == spec.n_physical


def test_prototype_excited_word_applies_blocks() -> None:
    """A parity-matching excited word flips its repetition-code blocks."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    prototype = build_dla_protected_memory_prototype(spec, logical_word=(1, 1))
    assert prototype.logical_word == (1, 1)
    assert prototype.circuit.depth() > 0


def test_witness_result_serialises_and_exposes_leakage() -> None:
    """The witness result exposes leakage properties and serialises."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    probs = np.zeros(spec.hilbert_dim, dtype=np.float64)
    probs[0] = 1.0
    result = evaluate_dla_protected_memory(probabilities=probs, spec=spec)
    payload = result.to_dict()
    assert payload["protected_leakage"] == result.protected_leakage
    assert result.code_leakage >= 0.0
    assert result.parity_leakage == result.opposite_parity_weight
    assert result.passes == (not result.failure_reasons)


def test_resolve_rejects_both_probabilities_and_counts() -> None:
    """Supplying both probabilities and counts is rejected."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    probs = np.zeros(spec.hilbert_dim, dtype=np.float64)
    with pytest.raises(ValueError, match="not both"):
        evaluate_dla_protected_memory(
            probabilities=probs, counts={"0" * spec.n_physical: 1}, spec=spec
        )


def test_resolve_requires_probabilities_or_counts() -> None:
    """Supplying neither probabilities nor counts is rejected."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    with pytest.raises(ValueError, match="must be provided"):
        evaluate_dla_protected_memory(spec=spec)


def test_counts_reject_malformed_bitstrings() -> None:
    """Bitstrings of the wrong length or alphabet are rejected."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    with pytest.raises(ValueError, match="length"):
        evaluate_dla_protected_memory(counts={"010": 5}, spec=spec)


def test_counts_reject_zero_total() -> None:
    """A zero shot total is rejected."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    with pytest.raises(ValueError, match="positive shot total"):
        evaluate_dla_protected_memory(counts={"0" * spec.n_physical: 0}, spec=spec)


def test_counts_path_produces_normalised_probabilities() -> None:
    """A valid counts mapping evaluates through the counts resolution path."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    result = DLAProtectedLogicalSyncWitness(spec)(counts={"0" * spec.n_physical: 8})
    assert result.total_weight == pytest.approx(1.0)


def test_validate_probabilities_rejects_wrong_shape() -> None:
    """A probability vector of the wrong shape is rejected."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    with pytest.raises(ValueError, match="must have shape"):
        evaluate_dla_protected_memory(probabilities=np.zeros(3, dtype=np.float64), spec=spec)


def test_validate_probabilities_rejects_negative_values() -> None:
    """Negative probability entries are rejected."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    bad = np.zeros(spec.hilbert_dim, dtype=np.float64)
    bad[0] = -1.0
    with pytest.raises(ValueError, match="finite non-negative"):
        evaluate_dla_protected_memory(probabilities=bad, spec=spec)


def test_failure_reasons_flag_all_thresholds() -> None:
    """Each violated witness threshold contributes its failure reason."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    reasons = _failure_reasons(
        spec,
        protected_weight=0.0,
        sync_weight=0.0,
        parity_leakage=1.0,
        code_leakage=1.0,
    )
    assert set(reasons) == {
        "protected_weight_below_threshold",
        "sync_weight_below_threshold",
        "parity_leakage_above_threshold",
        "code_leakage_above_threshold",
    }


def test_weighted_sync_order_zero_when_no_protected_weight() -> None:
    """With no probability mass in protected words the sync order is zero."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    zeros = np.zeros(spec.hilbert_dim, dtype=np.float64)
    result = evaluate_dla_protected_memory(probabilities=zeros, spec=spec)
    assert result.logical_sync_order == 0.0


def test_logical_agreement_single_bit_word_is_unity() -> None:
    """A single-logical word trivially agrees with itself."""
    assert _logical_agreement((1,)) == 1.0


def test_default_sync_word_zero_parity_sector() -> None:
    """The all-zero word satisfies the even parity sector."""
    assert _default_sync_word(DLAProtectedSubspaceSpec(n_logical=2, target_parity=0)) == (0, 0)


def test_default_sync_word_all_ones_parity_sector() -> None:
    """The all-ones word satisfies an odd parity sector with one logical bit."""
    assert _default_sync_word(DLAProtectedSubspaceSpec(n_logical=1, target_parity=1)) == (1,)


def test_default_sync_word_falls_back_to_first_protected_word() -> None:
    """When neither extreme matches, the first protected word is used."""
    word = _default_sync_word(DLAProtectedSubspaceSpec(n_logical=2, target_parity=1))
    assert sum(word) % 2 == 1


def test_spec_rejects_non_positive_logical_count() -> None:
    """A non-positive logical-qubit count is rejected."""
    with pytest.raises(ValueError, match="n_logical must be a positive integer"):
        DLAProtectedSubspaceSpec(n_logical=0)


def test_validate_logical_word_rejects_wrong_length() -> None:
    """A logical word of the wrong length is rejected."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    with pytest.raises(ValueError, match="must contain 2 bits"):
        _validate_logical_word((1,), spec)


def test_validate_logical_word_rejects_non_binary_bits() -> None:
    """A logical word containing a non-binary bit is rejected."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    with pytest.raises(ValueError, match="only 0/1 bits"):
        _validate_logical_word((2, 0), spec)


def test_protected_memory_mask_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the native mask kernel errors, the numpy mask is used."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "dla_protected_memory_mask"):

            def _raise(*_args: object, **_kwargs: object) -> object:
                raise ValueError("forced numpy fallback")

            monkeypatch.setattr(engine, "dla_protected_memory_mask", _raise)
    except ImportError:
        pass
    np.testing.assert_array_equal(protected_memory_mask(spec), _protected_memory_mask_numpy(spec))


def test_memory_metrics_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the native metrics kernel errors, the numpy metrics are used."""
    spec = DLAProtectedSubspaceSpec(n_logical=2)
    probs = np.zeros(spec.hilbert_dim, dtype=np.float64)
    probs[0] = 1.0
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "dla_protected_memory_metrics"):

            def _raise(*_args: object, **_kwargs: object) -> object:
                raise ValueError("forced numpy fallback")

            monkeypatch.setattr(engine, "dla_protected_memory_metrics", _raise)
    except ImportError:
        pass
    assert _memory_metrics(probs, spec) == _memory_metrics_numpy(probs, spec)
