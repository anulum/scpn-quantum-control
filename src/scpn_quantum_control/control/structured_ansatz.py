# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — structured ansatz module
"""Structured Kuramoto-XY ansatz construction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit


class StructuredAnsatz:
    """Physically-informed ansatz for heterogeneous Kuramoto-XY model."""

    def __init__(self) -> None:
        self.circuit: QuantumCircuit | None = None
        self.params: dict[str, Any] = {}

    @staticmethod
    def from_kuramoto(
        K_nm: NDArray[np.float64],
        omega: NDArray[np.float64] | None = None,
        trotter_depth: int = 6,
        time_step: float = 0.1,
        lambda_fim: float = 0.0,
        coupling_scale: float = 2.0,
        **kwargs: Any,
    ) -> StructuredAnsatz:
        """
        Builds a Trotterised Kuramoto-XY circuit from coupling matrix K_nm and
        natural frequencies omega.

        Args:
            K_nm:           N×N symmetric coupling matrix (diagonal ignored).
            omega:          Length-N natural frequencies. None → all zero.
            trotter_depth:  Number of Trotter steps.
            time_step:      dt per Trotter step.
            lambda_fim:     FIM feedback angle per step (float, not a Parameter).
            coupling_scale: Multiplicative scaling applied to K_nm before circuit
                            construction. Default 2.0 doubles coupling strength
                            relative to the raw matrix, pushing the system toward
                            the Kuramoto synchronisation transition.
        """
        K_nm = np.asarray(K_nm, dtype=np.float64)
        if K_nm.ndim != 2 or K_nm.shape[0] != K_nm.shape[1]:
            raise ValueError(f"K_nm must be a square matrix, got shape {K_nm.shape}")
        if not np.all(np.isfinite(K_nm)):
            raise ValueError("K_nm must contain only finite values")
        if omega is not None:
            omega = np.asarray(omega, dtype=np.float64)
            if omega.shape != (K_nm.shape[0],):
                raise ValueError(f"omega shape must be ({K_nm.shape[0]},), got {omega.shape}")
            if not np.all(np.isfinite(omega)):
                raise ValueError("omega must contain only finite values")

        N = K_nm.shape[0]
        ansatz = StructuredAnsatz()
        ansatz.params = {
            "N": N,
            "trotter_depth": trotter_depth,
            "time_step": time_step,
            "lambda_fim": lambda_fim,
            "coupling_scale": coupling_scale,
        }

        # Scale coupling strength before circuit construction
        K_scaled = K_nm * coupling_scale

        qc = QuantumCircuit(N)
        qc.h(range(N))  # initial uniform phase superposition

        dt = time_step
        for _ in range(trotter_depth):
            # 1. Frequency term (single-qubit Z rotations)
            if omega is not None:
                for i in range(N):
                    qc.rz(2 * omega[i] * dt, i)

            # 2. XY interactions from scaled K_nm
            for i in range(N):
                for j in range(i + 1, N):
                    if abs(K_scaled[i, j]) > 1e-8:
                        theta = 2 * K_scaled[i, j] * dt
                        qc.rzz(theta, i, j)

            # 3. FIM feedback — lambda_fim is a concrete float (no Parameter),
            #    avoiding Qiskit ≥2.x parameter name-collision errors.
            if lambda_fim > 0:
                fim_angle = lambda_fim * dt
                for i in range(N):
                    qc.rz(fim_angle, i)

        ansatz.circuit = qc
        return ansatz

    def build_circuit(self) -> QuantumCircuit:
        """Returns a copy of the built Qiskit circuit for submission."""
        if self.circuit is None:
            raise ValueError("Call from_kuramoto() first.")
        return self.circuit.copy()

    def __repr__(self) -> str:
        return (
            f"StructuredAnsatz(N={self.params.get('N')}, "
            f"trotter_depth={self.params.get('trotter_depth')}, "
            f"coupling_scale={self.params.get('coupling_scale')})"
        )
