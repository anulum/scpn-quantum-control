# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sparse Pauli-operator budget guard tests
"""Tests for sparse Pauli-operator construction budget guards."""

from __future__ import annotations

import pytest

import scpn_quantum_control.compile_budget as compile_budget_mod
from scpn_quantum_control.compile_budget import (
    DEFAULT_PAULI_BUDGET_CAP_GIB,
    DEFAULT_PAULI_BUDGET_ENV,
    PAULI_TERM_OVERHEAD_BYTES,
    PauliOperatorBudgetError,
    estimate_pauli_operator,
    pauli_budget_bytes,
    pauli_term_upper_bound,
    require_pauli_operator_budget,
)
from scpn_quantum_control.dense_budget import GIB


def test_term_upper_bound_counts_onsite_and_pairs() -> None:
    """Four qubits give four Z terms plus two terms for each of six pairs."""
    assert pauli_term_upper_bound(4) == 4 + 2 * 6


def test_term_upper_bound_adds_zz_term_per_pair() -> None:
    """The ZZ anisotropy adds one extra term per pair."""
    assert pauli_term_upper_bound(4, include_zz=True) == 4 + 3 * 6


def test_term_upper_bound_single_qubit_has_no_pairs() -> None:
    """A single qubit contributes one Z term and no coupling terms."""
    assert pauli_term_upper_bound(1) == 1


def test_term_upper_bound_rejects_non_integer() -> None:
    """A non-integer qubit count is a type error."""
    with pytest.raises(TypeError, match="n_qubits must be an integer"):
        pauli_term_upper_bound(4.0)  # type: ignore[arg-type]


def test_term_upper_bound_rejects_below_one() -> None:
    """A qubit count below one is a value error."""
    with pytest.raises(ValueError, match="n_qubits must be >= 1"):
        pauli_term_upper_bound(0)


def test_estimate_reports_terms_bytes_and_width() -> None:
    """The estimate reports the worst-case term count, label width, and bytes."""
    estimate = estimate_pauli_operator(4, max_gib=1.0, label="probe")

    assert estimate.term_count == 16
    assert estimate.label_chars == 4
    assert estimate.bytes_required == 16 * (4 + PAULI_TERM_OVERHEAD_BYTES)
    assert estimate.budget_gib == pytest.approx(1.0)
    assert estimate.gib_required == pytest.approx(estimate.bytes_required / GIB)
    assert estimate.label == "probe"


def test_estimate_include_zz_increases_terms() -> None:
    """Enabling the ZZ anisotropy increases the estimated term count."""
    plain = estimate_pauli_operator(8)
    anisotropic = estimate_pauli_operator(8, include_zz=True)
    assert anisotropic.term_count > plain.term_count


def test_budget_bytes_honours_explicit_max_gib() -> None:
    """An explicit budget converts directly to bytes."""
    assert pauli_budget_bytes(max_gib=3.0) == int(3.0 * GIB)


def test_budget_bytes_rejects_non_positive_max_gib() -> None:
    """An explicit non-positive budget is rejected."""
    with pytest.raises(ValueError, match="max_gib must be positive"):
        pauli_budget_bytes(max_gib=0.0)


def test_budget_bytes_reads_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid environment override sets the budget directly."""
    monkeypatch.setenv(DEFAULT_PAULI_BUDGET_ENV, "2")
    assert pauli_budget_bytes() == int(2 * GIB)


def test_budget_bytes_rejects_non_numeric_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric environment override is rejected."""
    monkeypatch.setenv(DEFAULT_PAULI_BUDGET_ENV, "not-a-number")
    with pytest.raises(ValueError, match="must be a positive number"):
        pauli_budget_bytes()


def test_budget_bytes_rejects_non_positive_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-positive environment override is rejected."""
    monkeypatch.setenv(DEFAULT_PAULI_BUDGET_ENV, "0")
    with pytest.raises(ValueError, match="must be positive"):
        pauli_budget_bytes()


def test_budget_bytes_falls_back_to_cap_without_memory_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When host memory is undiscoverable, the conservative cap applies."""
    monkeypatch.delenv(DEFAULT_PAULI_BUDGET_ENV, raising=False)
    monkeypatch.setattr(compile_budget_mod, "available_memory_bytes", lambda: None)
    assert pauli_budget_bytes() == int(DEFAULT_PAULI_BUDGET_CAP_GIB * GIB)


def test_budget_bytes_uses_memory_fraction_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discoverable, tiny host memory caps the budget by its fraction."""
    monkeypatch.delenv(DEFAULT_PAULI_BUDGET_ENV, raising=False)
    monkeypatch.setattr(compile_budget_mod, "available_memory_bytes", lambda: GIB)
    assert pauli_budget_bytes() < int(DEFAULT_PAULI_BUDGET_CAP_GIB * GIB)


def test_require_allows_small_operator() -> None:
    """A small operator under a generous budget returns its estimate."""
    estimate = require_pauli_operator_budget(4, max_gib=1.0, label="small operator")
    assert estimate.term_count == 16
    assert estimate.bytes_required <= estimate.budget_bytes


def test_require_fails_closed_above_budget() -> None:
    """A pathological qubit count under a tiny budget fails closed."""
    with pytest.raises(PauliOperatorBudgetError, match="Pauli terms of width"):
        require_pauli_operator_budget(64, max_gib=1e-6, label="huge operator")


def test_require_error_names_the_environment_override() -> None:
    """The failure message points at the environment override knob."""
    with pytest.raises(PauliOperatorBudgetError, match=DEFAULT_PAULI_BUDGET_ENV):
        require_pauli_operator_budget(64, max_gib=1e-6)


def test_require_propagates_invalid_qubit_count() -> None:
    """An invalid qubit count surfaces before any budget comparison."""
    with pytest.raises(ValueError, match="n_qubits must be >= 1"):
        require_pauli_operator_budget(0, max_gib=1.0)
