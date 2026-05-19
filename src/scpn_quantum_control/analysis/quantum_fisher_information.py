# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved.
"""Quantum Fisher Information observable wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from .qfi import compute_qfi


class QuantumFisherInformation:
    """
    Computes Quantum Fisher Information for metrological gain.

    Production QFI requires an explicit coupling matrix and natural-frequency
    vector so the observable can route through the spectral QFI engine. The
    legacy sync-order/DLA estimate remains available only as an explicitly
    labelled diagnostic proxy.
    """

    def __call__(self, counts: Mapping[str, int] | None = None, **kwargs: Any) -> dict[str, float]:
        coupling_matrix = kwargs.get("coupling_matrix")
        natural_frequencies = kwargs.get("natural_frequencies")
        if coupling_matrix is not None or natural_frequencies is not None:
            if coupling_matrix is None or natural_frequencies is None:
                raise ValueError(
                    "Quantum Fisher Information requires both coupling_matrix and "
                    "natural_frequencies for production evaluation."
                )
            return self._compute_production_qfi(
                coupling_matrix=coupling_matrix,
                natural_frequencies=natural_frequencies,
                coupling_pairs=kwargs.get("coupling_pairs"),
                n_measurements=kwargs.get("n_measurements", 10000),
            )

        if not bool(kwargs.get("allow_proxy_estimate", False)):
            raise NotImplementedError(
                "Quantum Fisher Information requires coupling_matrix and "
                "natural_frequencies. Set allow_proxy_estimate=True only for the "
                "labelled sync-order/DLA diagnostic proxy."
            )

        sync_order = kwargs.get("sync_order")
        dla_asym = kwargs.get("dla_asymmetry")

        if counts and (sync_order is None or dla_asym is None):
            from .dla_parity_witness import DLAParityWitness
            from .sync_order_parameter import SyncOrderParameter

            if sync_order is None:
                sync_order = SyncOrderParameter()(counts=counts)["sync_order"]
            if dla_asym is None:
                dla_asym = DLAParityWitness()(counts=counts)["dla_asymmetry"]

        if sync_order is None or dla_asym is None:
            raise ValueError(
                "The labelled QFI proxy requires sync_order and dla_asymmetry, "
                "or counts from which both diagnostics can be derived."
            )

        sync_order = float(sync_order)
        dla_asym = float(dla_asym)
        qfi_proxy = 4.0 * sync_order * (1.0 + abs(dla_asym) / 100.0)
        return {
            "qfi_available": 0.0,
            "qfi_proxy": float(qfi_proxy),
            "is_quantum_fisher_information": 0.0,
            "sync_order_input": sync_order,
            "dla_asymmetry_input": dla_asym,
        }

    @staticmethod
    def _compute_production_qfi(
        *,
        coupling_matrix: Any,
        natural_frequencies: Any,
        coupling_pairs: Any,
        n_measurements: Any,
    ) -> dict[str, float]:
        K = np.asarray(coupling_matrix, dtype=float)
        omega = np.asarray(natural_frequencies, dtype=float)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("coupling_matrix must be a square two-dimensional array.")
        if omega.ndim != 1 or omega.shape[0] != K.shape[0]:
            raise ValueError(
                "natural_frequencies must be a one-dimensional vector matching coupling_matrix."
            )
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(omega)):
            raise ValueError("coupling_matrix and natural_frequencies must contain finite values.")
        if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12):
            raise ValueError("coupling_matrix must be symmetric.")

        measurement_budget = QuantumFisherInformation._validate_measurement_budget(n_measurements)

        pairs = (
            None
            if coupling_pairs is None
            else QuantumFisherInformation._validate_coupling_pairs(coupling_pairs, K.shape[0])
        )
        result = compute_qfi(K, omega, pairs=pairs)
        diagonal = np.diag(result.qfi_matrix)
        finite_single_shot_bounds = result.precision_bounds[np.isfinite(result.precision_bounds)]
        precision_bound = (
            float(np.min(finite_single_shot_bounds) / measurement_budget)
            if len(finite_single_shot_bounds) > 0
            else float("inf")
        )

        return {
            "qfi_available": 1.0,
            "qfi": float(np.max(diagonal)) if len(diagonal) else 0.0,
            "qfi_max_diagonal": float(np.max(diagonal)) if len(diagonal) else 0.0,
            "qfi_trace": float(np.trace(result.qfi_matrix)),
            "qfi_matrix_shape_0": float(result.qfi_matrix.shape[0]),
            "qfi_matrix_shape_1": float(result.qfi_matrix.shape[1]),
            "precision_bound_for_measurement_budget": precision_bound,
            "spectral_gap": float(result.spectral_gap),
            "coupling_pair_count": float(len(result.coupling_pairs)),
            "n_qubits": float(result.n_qubits),
            "n_measurements": float(measurement_budget),
            "is_quantum_fisher_information": 1.0,
        }

    @staticmethod
    def _validate_measurement_budget(n_measurements: Any) -> int:
        if isinstance(n_measurements, bool) or not isinstance(n_measurements, Integral):
            raise ValueError("n_measurements must be a positive integer.")
        measurement_budget = int(n_measurements)
        if measurement_budget <= 0:
            raise ValueError("n_measurements must be a positive integer.")
        return measurement_budget

    @staticmethod
    def _validate_coupling_pairs(coupling_pairs: Any, n_qubits: int) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        try:
            iterator = iter(coupling_pairs)
        except TypeError as exc:
            raise ValueError(
                "coupling_pairs must be an iterable of two distinct qubit indices."
            ) from exc

        for raw_pair in iterator:
            try:
                pair = tuple(raw_pair)
            except TypeError as exc:
                raise ValueError(
                    "coupling_pairs must be an iterable of two distinct qubit indices."
                ) from exc
            if len(pair) != 2:
                raise ValueError(
                    "coupling_pairs must contain exactly two distinct qubit indices per pair."
                )
            i_raw, j_raw = pair
            if (
                isinstance(i_raw, bool)
                or isinstance(j_raw, bool)
                or not isinstance(i_raw, Integral)
                or not isinstance(j_raw, Integral)
            ):
                raise ValueError("coupling_pairs indices must be integers.")
            i = int(i_raw)
            j = int(j_raw)
            if i == j or i < 0 or j < 0 or i >= n_qubits or j >= n_qubits:
                raise ValueError(
                    "coupling_pairs indices must be distinct and within coupling_matrix bounds."
                )
            pairs.append((min(i, j), max(i, j)))
        return pairs
