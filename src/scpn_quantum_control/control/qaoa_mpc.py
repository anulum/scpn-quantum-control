"""QAOA for MPC trajectory optimization.

Discretizes the MPC action space to binary (coil on/off per timestep),
maps the quadratic cost to an Ising Hamiltonian, then solves via QAOA.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


class QAOA_MPC:
    """QAOA-based model predictive controller.

    Cost: C = sum_t ||B*u(t) - target||^2  discretized to binary u_t in {0,1}.
    This quadratic-in-binary is equivalent to an Ising Hamiltonian.
    """

    def __init__(
        self,
        B_matrix: np.ndarray,
        target_state: np.ndarray,
        horizon: int,
        p_layers: int = 2,
    ):
        self.B = np.asarray(B_matrix, dtype=np.float64)
        self.target = np.asarray(target_state, dtype=np.float64)
        self.horizon = horizon
        self.p = p_layers
        self.n_qubits = horizon
        self._cost_ham: SparsePauliOp | None = None

    def build_cost_hamiltonian(self) -> SparsePauliOp:
        """Map per-timestep quadratic binary cost to Ising Hamiltonian.

        C(u) = sum_t (a*u_t - b)^2,  a=||B||, b=||target||/H.
        Expanding with u_t^2=u_t and u_t=(1-Z_t)/2:
            C = const + h_z * sum_t Z_t
        where h_z = -(a^2 - 2ab)/2.  No ZZ terms (timesteps are independent).
        """
        a = float(np.linalg.norm(self.B))
        b = float(np.linalg.norm(self.target)) / self.horizon
        h_z = -(a**2 - 2.0 * a * b) / 2.0
        c0 = (a**2 - 2.0 * a * b) * self.horizon / 2.0 + self.horizon * b**2

        pauli_list = [("I" * self.n_qubits, c0)]
        for t in range(self.horizon):
            z_str = ["I"] * self.n_qubits
            z_str[t] = "Z"
            pauli_list.append(("".join(reversed(z_str)), h_z))

        labels, coeffs = zip(*pauli_list)
        self._cost_ham = SparsePauliOp(list(labels), list(coeffs)).simplify()
        return self._cost_ham

    def _build_qaoa_circuit(self, gamma: np.ndarray, beta: np.ndarray) -> QuantumCircuit:
        """Build p-layer QAOA circuit: initial |+>, alternating cost/mixer."""
        if self._cost_ham is None:
            self.build_cost_hamiltonian()

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
                elif len(z_qubits) == 2:
                    qc.cx(z_qubits[0], z_qubits[1])
                    qc.rz(angle, z_qubits[1])
                    qc.cx(z_qubits[0], z_qubits[1])

            # Mixer unitary: exp(-i*beta*X)
            for q in range(self.n_qubits):
                qc.rx(2.0 * beta[layer], q)

        return qc

    def optimize(self) -> np.ndarray:
        """Run QAOA optimization, return binary action sequence.

        Returns:
            shape (horizon,) array of 0/1 actions.
        """
        if self._cost_ham is None:
            self.build_cost_hamiltonian()

        def cost_fn(params):
            gamma = params[: self.p]
            beta = params[self.p :]
            qc = self._build_qaoa_circuit(gamma, beta)
            sv = Statevector.from_instruction(qc)
            return float(sv.expectation_value(self._cost_ham).real)

        x0 = np.random.default_rng().uniform(0, np.pi, 2 * self.p)
        result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": 200})

        gamma_opt = result.x[: self.p]
        beta_opt = result.x[self.p :]
        qc = self._build_qaoa_circuit(gamma_opt, beta_opt)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        best_bitstring = format(int(np.argmax(probs)), f"0{self.n_qubits}b")
        return np.array([int(b) for b in reversed(best_bitstring)])
