# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Out-of-time-order correlator (OTOC) for quantum chaos detection.

The OTOC measures information scrambling in the quantum Kuramoto system:

    F(t) = <W†(t) V† W(t) V>

where W(t) = e^{iHt} W e^{-iHt} is a Heisenberg-evolved operator.

For the XY Hamiltonian, the OTOC decay rate gives the quantum
Lyapunov exponent λ_Q, bounded by (Maldacena, Shenker, Stanford 2016):

    λ_Q ≤ 2π T / ℏ

At the synchronisation transition, scrambling should peak — information
spreads fastest at criticality. This connects to the L16 Lyapunov
monitoring in the SCPN framework.

The OTOC is computed via exact statevector simulation:
    1. Prepare initial state |ψ>
    2. Apply V
    3. Forward evolve: e^{-iHt}
    4. Apply W
    5. Backward evolve: e^{iHt}
    6. Apply V†
    7. Overlap with W†|ψ>

Equivalently: F(t) = Tr(ρ W†(t) V† W(t) V) for thermal ρ.
For pure state at T=0: F(t) = |<ψ|W†(t) V† W(t) V|ψ>|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_hamiltonian
from ..hardware.gpu_accel import expm


@dataclass
class OTOCResult:
    """OTOC measurement result."""

    times: np.ndarray
    otoc_values: np.ndarray  # F(t) = Re(<W†(t)V†W(t)V>)
    lyapunov_estimate: float | None  # λ_Q from exponential fit
    scrambling_time: float | None  # t* where F drops to 1/e
    n_qubits: int
    operator_w: str
    operator_v: str


def _pauli_matrix(label: str, qubit: int, n: int) -> np.ndarray:
    """Build n-qubit Pauli matrix for single-qubit operator on given qubit."""
    I2 = np.eye(2, dtype=complex)
    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    P = paulis[label]
    result = np.eye(1, dtype=complex)
    for i in range(n):
        result = np.kron(result, P if i == qubit else I2)
    out: np.ndarray = result
    return out


def compute_otoc(
    K: np.ndarray,
    omega: np.ndarray,
    times: np.ndarray | None = None,
    w_qubit: int = 0,
    v_qubit: int | None = None,
    w_pauli: str = "Z",
    v_pauli: str = "X",
) -> OTOCResult:
    """Compute OTOC F(t) for the Kuramoto-XY Hamiltonian.

    F(t) = Re(<ψ|W†(t) V† W(t) V|ψ>) where |ψ> = |0...0>.

    Args:
        K: coupling matrix
        omega: natural frequencies
        times: time points for OTOC evaluation
        w_qubit: qubit index for W operator
        v_qubit: qubit index for V operator (default: w_qubit + 1)
        w_pauli: Pauli label for W ("X", "Y", or "Z")
        v_pauli: Pauli label for V
    """
    n = K.shape[0]
    if times is None:
        times = np.linspace(0, 2.0, 30)
    if v_qubit is None:
        v_qubit = min(w_qubit + 1, n - 1)

    H_op = knm_to_hamiltonian(K, omega)
    H_raw = H_op.to_matrix()
    H_mat = H_raw.toarray() if hasattr(H_raw, "toarray") else np.array(H_raw)

    W = _pauli_matrix(w_pauli, w_qubit, n)
    V = _pauli_matrix(v_pauli, v_qubit, n)

    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0

    otoc_vals = np.zeros(len(times))

    for idx, t in enumerate(times):
        # U(t) = exp(-iHt), U†(t) = exp(iHt)
        U = expm(-1j * H_mat * t)
        U_dag = expm(1j * H_mat * t)

        # W(t) = U† W U
        W_t = U_dag @ W @ U

        # F(t) = <ψ| W†(t) V† W(t) V |ψ>
        state = V @ psi
        state = W_t @ state
        state = V.conj().T @ state
        state = W_t.conj().T @ state
        otoc_vals[idx] = float(np.real(psi.conj() @ state))

    # Estimate Lyapunov exponent from initial decay
    lyapunov = _estimate_lyapunov(times, otoc_vals)
    scrambling = _estimate_scrambling_time(times, otoc_vals)

    return OTOCResult(
        times=times,
        otoc_values=otoc_vals,
        lyapunov_estimate=lyapunov,
        scrambling_time=scrambling,
        n_qubits=n,
        operator_w=f"{w_pauli}_{w_qubit}",
        operator_v=f"{v_pauli}_{v_qubit}",
    )


def _estimate_lyapunov(times: np.ndarray, otoc: np.ndarray) -> float | None:
    """Estimate quantum Lyapunov exponent from OTOC decay.

    Fits F(t) ≈ 1 - ε × exp(λ_Q × t) in the early-time regime.
    """
    f0 = otoc[0]
    if abs(f0) < 1e-10:
        return None

    decay = 1.0 - otoc / f0
    positive = decay > 1e-10
    if np.sum(positive) < 3:
        return None

    log_decay = np.log(decay[positive])
    t_pos = times[positive]

    if len(t_pos) < 3:
        return None

    # Linear fit: log(1 - F/F0) ≈ log(ε) + λ_Q × t
    A = np.vstack([t_pos, np.ones_like(t_pos)]).T
    result = np.linalg.lstsq(A, log_decay, rcond=None)
    slope = result[0][0]
    return float(slope) if slope > 0 else None


def _estimate_scrambling_time(times: np.ndarray, otoc: np.ndarray) -> float | None:
    """Time where OTOC drops to 1/e of initial value."""
    f0 = otoc[0]
    if abs(f0) < 1e-10:
        return None

    threshold = f0 / np.e
    below = np.where(otoc < threshold)[0]
    if len(below) == 0:
        return None
    return float(times[below[0]])
