# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sparse Pauli-operator construction budget guards
"""Resource guards for sparse Pauli-operator construction in the Kuramoto compiler.

The dense Hilbert-space guard in :mod:`scpn_quantum_control.dense_budget`
protects paths that materialise a ``2**n`` object. The XY/XXZ Kuramoto compiler
also builds a sparse :class:`~qiskit.quantum_info.SparsePauliOp` whose term list
grows as ``O(n**2)``: ``n`` on-site ``Z`` terms plus two (``XX``, ``YY``), or
three with the ``ZZ`` anisotropy, terms for each coupled pair. Each term carries
an ``n``-character Pauli label, so the pure-Python construction cost grows as
``O(n**3)`` bytes and an adversarial ``n`` can exhaust memory before any quantum
simulation runs — a denial-of-service vector for untrusted ``K_nm``/``omega``.

This module bounds that construction against a configurable budget, mirroring the
dense-budget guard, so the compiler fails closed on pathological ``n`` with a
clear, typed error instead of silently allocating. The estimate uses the
worst-case (fully dense coupling) term count, so the bound is deterministic in
``n`` and does not depend on the coupling values; a sparse coupling matrix only
ever builds fewer terms than the bound.

The guard is ``O(1)`` integer arithmetic with no compute hot path, so it has no
Rust/polyglot counterpart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from .dense_budget import GIB, available_memory_bytes

DEFAULT_PAULI_BUDGET_ENV: Final = "SCPN_MAX_PAULI_GIB"
DEFAULT_PAULI_RAM_FRACTION: Final = 0.30
DEFAULT_PAULI_BUDGET_CAP_GIB: Final = 2.0
# Conservative per-term memory floor in addition to the ``n``-character label:
# the CPython ``str`` header, the coefficient, and the enclosing tuple/list slot
# built transiently while assembling the Pauli list.
PAULI_TERM_OVERHEAD_BYTES: Final = 96


class PauliOperatorBudgetError(MemoryError):
    """Raised before an unsafe sparse Pauli-operator construction is attempted."""


@dataclass(frozen=True)
class PauliOperatorEstimate:
    """Estimated memory for a worst-case sparse Pauli-operator construction."""

    n_qubits: int
    term_count: int
    label_chars: int
    bytes_required: int
    budget_bytes: int
    label: str

    @property
    def gib_required(self) -> float:
        """Memory required in GiB."""
        return self.bytes_required / GIB

    @property
    def budget_gib(self) -> float:
        """Budget in GiB."""
        return self.budget_bytes / GIB


def pauli_term_upper_bound(n_qubits: int, *, include_zz: bool = False) -> int:
    """Return the worst-case Pauli-term count for a dense ``n``-qubit operator.

    Parameters
    ----------
    n_qubits:
        Number of qubits/oscillators; must be a positive integer.
    include_zz:
        Whether the ``ZZ`` anisotropy term is emitted per pair (``delta != 0``).

    Returns
    -------
    int
        ``n`` on-site ``Z`` terms plus two (``XX``, ``YY``) or three (adding
        ``ZZ``) terms for each of the ``n*(n-1)/2`` ordered pairs.
    """
    if not isinstance(n_qubits, int):
        raise TypeError("n_qubits must be an integer")
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    pairs = n_qubits * (n_qubits - 1) // 2
    per_pair = 3 if include_zz else 2
    return n_qubits + per_pair * pairs


def pauli_budget_bytes(max_gib: float | None = None) -> int:
    """Return the sparse Pauli-operator construction budget in bytes.

    An explicit ``max_gib`` wins; otherwise the ``SCPN_MAX_PAULI_GIB``
    environment override applies, then a host-memory fraction capped at
    :data:`DEFAULT_PAULI_BUDGET_CAP_GIB`.
    """
    if max_gib is not None:
        if max_gib <= 0:
            raise ValueError("max_gib must be positive")
        return int(max_gib * GIB)

    env_value = os.environ.get(DEFAULT_PAULI_BUDGET_ENV)
    if env_value:
        try:
            parsed_gib = float(env_value)
        except ValueError as exc:
            raise ValueError(f"{DEFAULT_PAULI_BUDGET_ENV} must be a positive number") from exc
        if parsed_gib <= 0:
            raise ValueError(f"{DEFAULT_PAULI_BUDGET_ENV} must be positive")
        return int(parsed_gib * GIB)

    available = available_memory_bytes()
    if available is None:
        return int(DEFAULT_PAULI_BUDGET_CAP_GIB * GIB)
    return int(min(DEFAULT_PAULI_BUDGET_CAP_GIB * GIB, available * DEFAULT_PAULI_RAM_FRACTION))


def estimate_pauli_operator(
    n_qubits: int,
    *,
    include_zz: bool = False,
    max_gib: float | None = None,
    label: str = "sparse Pauli operator",
) -> PauliOperatorEstimate:
    """Estimate the worst-case memory of a sparse Pauli-operator construction."""
    term_count = pauli_term_upper_bound(n_qubits, include_zz=include_zz)
    bytes_required = term_count * (n_qubits + PAULI_TERM_OVERHEAD_BYTES)
    return PauliOperatorEstimate(
        n_qubits=n_qubits,
        term_count=term_count,
        label_chars=n_qubits,
        bytes_required=bytes_required,
        budget_bytes=pauli_budget_bytes(max_gib),
        label=label,
    )


def require_pauli_operator_budget(
    n_qubits: int,
    *,
    include_zz: bool = False,
    max_gib: float | None = None,
    label: str = "sparse Pauli operator",
) -> PauliOperatorEstimate:
    """Raise before a sparse Pauli-operator construction exceeds the budget.

    Raises
    ------
    PauliOperatorBudgetError
        When the worst-case construction for ``n_qubits`` exceeds the active
        budget.
    ValueError
        When ``n_qubits`` is below one or ``max_gib`` is non-positive.
    TypeError
        When ``n_qubits`` is not an integer.
    """
    estimate = estimate_pauli_operator(
        n_qubits, include_zz=include_zz, max_gib=max_gib, label=label
    )
    if estimate.bytes_required > estimate.budget_bytes:
        raise PauliOperatorBudgetError(
            f"{label} for n={estimate.n_qubits} would build "
            f"{estimate.term_count} Pauli terms of width {estimate.label_chars} "
            f"(~{estimate.gib_required:.2f} GiB), above the active sparse-operator "
            f"budget {estimate.budget_gib:.2f} GiB. Reduce n, raise "
            f"{DEFAULT_PAULI_BUDGET_ENV}, or use a hardware/tensor-network path "
            f"instead of explicit operator construction."
        )
    return estimate
