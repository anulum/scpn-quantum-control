# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qaoa Mpc
"""QAOA for MPC trajectory optimisation.

Discretises the MPC action space to binary (coil on/off per timestep),
maps the quadratic tracking cost to an Ising Hamiltonian, then solves via QAOA.
The cost keeps the vector residual ``u_t * (B @ ones) - target``; collapsing it
to norms would discard the target's sign and its direction relative to ``B``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from ..dense_budget import require_dense_allocation


class QAOA_MPC:
    """QAOA-based model predictive controller.

    Cost: ``C = sum_t ||u_t * (B @ ones) - target||^2`` with binary
    ``u_t in {0, 1}``, one coil on/off decision per timestep. The residual stays
    a vector, so a sign flip or rotation of ``target`` changes the optimum. This
    quadratic-in-binary form is equivalent to an Ising Hamiltonian.
    """

    def __init__(
        self,
        B_matrix: NDArray[np.float64],
        target_state: NDArray[np.float64],
        horizon: int,
        p_layers: int = 2,
    ) -> None:
        """Initialize the binary model-predictive controller.

        Parameters
        ----------
        B_matrix
            Linear map from binary actions to the controlled state.
        target_state
            State vector used to construct the quadratic tracking cost.
        horizon
            Positive number of binary control timesteps and circuit qubits.
        p_layers
            Positive number of alternating QAOA cost and mixer layers.

        Raises
        ------
        ValueError
            If ``horizon`` or ``p_layers`` is not positive.

        """
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if p_layers <= 0:
            raise ValueError(f"p_layers must be positive, got {p_layers}")
        self.B = np.asarray(B_matrix, dtype=np.float64)
        self.target = np.asarray(target_state, dtype=np.float64)
        self.horizon = horizon
        self.p = p_layers
        self.n_qubits = horizon
        self._cost_ham: SparsePauliOp | None = None

    def build_cost_hamiltonian(self) -> SparsePauliOp:
        """Map per-timestep quadratic binary cost to Ising Hamiltonian.

        The tracking cost is ``C(u) = sum_t ||u_t * v - r||^2`` with
        ``v = B @ ones`` and ``r`` the target. Using ``u_t^2 = u_t`` for binary
        ``u_t`` and ``u_t = (1 - Z_t)/2``::

            q  = ||v||^2 - 2 * (v . r)
            C  = H*q/2 + H*||r||^2 - (q/2) * sum_t Z_t

        so ``h_z = -q/2``. There are no ZZ terms because the timesteps are
        independent; the coupling this mapping preserves is the ``v . r``
        cross-term inside the norm, which a norm-only surrogate discards along
        with the target's sign.

        Returns
        -------
        qiskit.quantum_info.SparsePauliOp
            Diagonal identity-and-Z cost Hamiltonian for the control horizon.

        """
        actuation = self.B.sum(axis=1)
        q = float(actuation @ actuation) - 2.0 * float(actuation @ self.target)
        h_z = -q / 2.0
        c0 = q * self.horizon / 2.0 + self.horizon * float(self.target @ self.target)

        pauli_list = [("I" * self.n_qubits, c0)]
        for t in range(self.horizon):
            z_str = ["I"] * self.n_qubits
            z_str[t] = "Z"
            pauli_list.append(("".join(reversed(z_str)), h_z))

        labels, coeffs = zip(*pauli_list)
        self._cost_ham = SparsePauliOp(list(labels), list(coeffs)).simplify()
        return self._cost_ham

    def _build_qaoa_circuit(
        self, gamma: NDArray[np.float64], beta: NDArray[np.float64]
    ) -> QuantumCircuit:
        """Build the layered QAOA circuit for supplied variational angles.

        Parameters
        ----------
        gamma
            Cost-unitary angles, one per QAOA layer.
        beta
            Mixer angles, one per QAOA layer.

        Returns
        -------
        qiskit.QuantumCircuit
            Initial plus state followed by alternating cost and mixer gates.

        Raises
        ------
        RuntimeError
            If lazy cost-Hamiltonian construction does not produce an operator.

        """
        if self._cost_ham is None:
            self.build_cost_hamiltonian()
        if self._cost_ham is None:
            raise RuntimeError("cost Hamiltonian construction failed")

        qc = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            qc.h(q)

        for layer in range(self.p):
            # Cost unitary: exp(-i*gamma*C)
            for term, coeff in zip(self._cost_ham.paulis, self._cost_ham.coeffs):
                label = str(term)
                z_qubits = [i for i, c in enumerate(reversed(label)) if c == "Z"]
                angle = 2.0 * gamma[layer] * float(coeff.real)

                if len(z_qubits) == 1:
                    qc.rz(angle, z_qubits[0])

            # Mixer unitary: exp(-i*beta*X)
            for q in range(self.n_qubits):
                qc.rx(2.0 * beta[layer], q)

        return qc

    def optimize(
        self, seed: int | None = None, *, max_dense_gib: float | None = None
    ) -> NDArray[np.int64]:
        """Run QAOA optimization, return binary action sequence.

        The circuit is simulated densely, so the horizon sets a ``2**horizon``
        statevector. Admission is checked once here, before the optimiser runs,
        rather than inside the cost function that the optimiser calls up to two
        hundred times: a budget that cannot hold the state should refuse before
        any allocator is entered, not on the first iteration.

        Two objects of the statevector's size are accounted for. The state
        itself is live throughout, and the expectation value against the cost
        Hamiltonian, and later ``probabilities()``, each need a second array of
        the same dimension while the first is still held.

        Parameters
        ----------
        seed
            Optional seed for the variational parameter initialization.
        max_dense_gib
            Optional dense-allocation budget in GiB. ``None`` uses the active
            process budget.

        Returns
        -------
        numpy.ndarray
            Integer array of zero/one actions shaped ``(horizon,)``.

        Raises
        ------
        DenseAllocationError
            If the statevector for ``horizon`` qubits exceeds the budget.
        RuntimeError
            If the cost Hamiltonian could not be constructed.

        """
        require_dense_allocation(
            self.n_qubits,
            dtype=np.complex128,
            rank=1,
            object_count=2,
            max_gib=max_dense_gib,
            label="QAOA-MPC statevector",
        )
        if self._cost_ham is None:
            self.build_cost_hamiltonian()
        if self._cost_ham is None:
            raise RuntimeError("cost Hamiltonian construction failed")

        def cost_fn(params: NDArray[np.float64]) -> float:
            gamma = params[: self.p]
            beta = params[self.p :]
            qc = self._build_qaoa_circuit(gamma, beta)
            sv = Statevector.from_instruction(qc)
            return float(sv.expectation_value(self._cost_ham).real)

        x0 = np.random.default_rng(seed).uniform(0, np.pi, 2 * self.p)
        result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": 200})

        gamma_opt = result.x[: self.p]
        beta_opt = result.x[self.p :]
        qc = self._build_qaoa_circuit(gamma_opt, beta_opt)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        best_bitstring = format(int(np.argmax(probs)), f"0{self.n_qubits}b")
        actions: NDArray[np.int64] = np.array(
            [int(b) for b in reversed(best_bitstring)], dtype=np.int64
        )
        return actions
