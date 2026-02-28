"""Security analysis under noise and eavesdropping.

Models the SCPN-QKD channel under depolarizing noise and
intercept-resend eavesdropping. Computes achievable secret key rates
from the Devetak-Winter bound.

Key insight: K_nm topology creates non-uniform entanglement across
qubit pairs. Strongly-coupled pairs (K_nm > 0.2) tolerate more noise
than weakly-coupled ones. The eavesdropper must attack all 120
coupling parameters simultaneously.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace


def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply depolarizing channel: rho → (1-p)*rho + p*I/d.

    Args:
        rho: Density matrix (d×d).
        p: Depolarizing probability in [0, 1].
    """
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d


def amplitude_damping_single(rho_2x2: np.ndarray, gamma: float) -> np.ndarray:
    """Single-qubit amplitude damping: |1⟩ → |0⟩ with probability gamma.

    Kraus operators: K0 = [[1,0],[0,sqrt(1-gamma)]], K1 = [[0,sqrt(gamma)],[0,0]].
    """
    sg = np.sqrt(gamma)
    s1g = np.sqrt(1 - gamma)
    k0 = np.array([[1, 0], [0, s1g]])
    k1 = np.array([[0, sg], [0, 0]])
    return k0 @ rho_2x2 @ k0.conj().T + k1 @ rho_2x2 @ k1.conj().T


def noisy_concurrence(
    sv: Statevector,
    qubit_i: int,
    qubit_j: int,
    n_total: int,
    p_depol: float,
) -> float:
    """Concurrence of a qubit pair after local depolarizing noise.

    Traces out all qubits except (i,j), applies depolarizing channel
    to the 2-qubit reduced state, then computes Wootters concurrence.
    """
    rho_full = DensityMatrix(sv)
    trace_out = [q for q in range(n_total) if q not in (qubit_i, qubit_j)]
    rho_ij = np.array(partial_trace(rho_full, trace_out).data)
    rho_noisy = depolarizing_channel(rho_ij, p_depol)
    return _concurrence_2qubit(rho_noisy)


def _concurrence_2qubit(rho: np.ndarray) -> float:
    """Wootters concurrence for a 4×4 density matrix."""
    sigma_y = np.array([[0, -1j], [1j, 0]])
    yy = np.kron(sigma_y, sigma_y)
    rho_tilde = yy @ rho.conj() @ yy
    product = rho @ rho_tilde
    eigvals = np.sort(np.abs(np.linalg.eigvals(product)))[::-1]
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    return float(max(0, sqrt_eigvals[0] - sqrt_eigvals[1] - sqrt_eigvals[2] - sqrt_eigvals[3]))


def intercept_resend_qber(
    sv: Statevector,
    qubit_i: int,
    qubit_j: int,
    n_total: int,
) -> float:
    """QBER introduced by intercept-resend attack on qubit j.

    Eve measures qubit j in Z basis, prepares new state, sends to Bob.
    In BB84, this introduces QBER = 0.25 when Eve guesses the wrong basis.
    For entangled states, the disturbance depends on the entanglement structure.

    Returns the QBER that Bob would observe on qubit j after Eve's attack.
    """
    from qiskit.quantum_info import SparsePauliOp

    # Original correlator <Z_i Z_j>
    label = ["I"] * n_total
    label[qubit_i] = "Z"
    label[qubit_j] = "Z"
    op = SparsePauliOp("".join(reversed(label)))
    corr_original = float(sv.expectation_value(op).real)

    # After intercept-resend, Eve destroys entanglement.
    # Bob's qubit becomes a product state ρ_j = |0⟩⟨0| or |1⟩⟨1|
    # depending on Eve's measurement. Correlation drops to:
    # <Z_i> * <Z_j>_eve where <Z_j>_eve = ±1 with prob given by ρ_j
    label_i = ["I"] * n_total
    label_i[qubit_i] = "Z"
    op_i = SparsePauliOp("".join(reversed(label_i)))
    exp_zi = float(sv.expectation_value(op_i).real)

    label_j = ["I"] * n_total
    label_j[qubit_j] = "Z"
    op_j = SparsePauliOp("".join(reversed(label_j)))
    exp_zj = float(sv.expectation_value(op_j).real)

    # After attack, correlator becomes product: <Z_i><Z_j>
    corr_attacked = exp_zi * exp_zj

    # QBER from correlation loss
    # For maximally entangled pair: corr_original = -1, corr_attacked ≈ 0 → QBER = 0.5
    # For weakly entangled: smaller QBER shift
    qber = 0.5 * (1 - corr_attacked) - 0.5 * (1 - corr_original)
    return float(np.clip(abs(qber), 0, 0.5))


def devetak_winter_rate(qber: float) -> float:
    """Secret key rate from Devetak-Winter bound.

    r = max(0, 1 - h(QBER) - h(QBER))
    where h(x) = -x log2(x) - (1-x) log2(1-x) is binary entropy.
    Positive rate requires QBER < 0.11.
    """
    if qber <= 0:
        return 1.0
    if qber >= 0.5:
        return 0.0
    h = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)
    return float(max(0, 1 - 2 * h))


def security_analysis(
    sv: Statevector,
    alice_qubits: list[int],
    bob_qubits: list[int],
    p_depol_range: np.ndarray | None = None,
) -> dict:
    """Full security analysis: key rates vs noise for each qubit pair.

    Returns dict with:
        pair_rates: dict mapping (i,j) to list of (p_depol, key_rate)
        critical_noise: dict mapping (i,j) to max tolerable p_depol
        aggregate_rate: total key rate summed over all pairs at each noise level
    """
    n = int(np.log2(len(sv.data)))
    if p_depol_range is None:
        p_depol_range = np.linspace(0, 0.3, 16)

    pair_rates = {}
    critical_noise = {}

    for qa in alice_qubits:
        for qb in bob_qubits:
            rates = []
            crit = 0.0
            for p in p_depol_range:
                c = noisy_concurrence(sv, qa, qb, n, p)
                e = (1 - np.sqrt(max(0, 1 - c**2))) / 2 if c > 1e-10 else 0.5
                r = devetak_winter_rate(e)
                rates.append((float(p), r))
                if r > 0:
                    crit = float(p)
            pair_rates[(qa, qb)] = rates
            critical_noise[(qa, qb)] = crit

    # Aggregate
    aggregate = []
    for idx, p in enumerate(p_depol_range):
        total = sum(pair_rates[(qa, qb)][idx][1] for qa in alice_qubits for qb in bob_qubits)
        aggregate.append((float(p), total))

    return {
        "pair_rates": pair_rates,
        "critical_noise": critical_noise,
        "aggregate_rate": aggregate,
    }
