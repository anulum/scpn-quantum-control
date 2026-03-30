# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Lindblad Master Equation Documentation

# Lindblad Master Equation Solver

`scpn_quantum_control.phase.lindblad`

Open-system dynamics for the Kuramoto-XY Hamiltonian via the Lindblad
master equation. Solves for the full density matrix under amplitude
damping and dephasing channels.

**Caveat:** Full density matrix evolution scales as $O(4^n)$ in memory.
For $n > 10$, consider the [MCWF method](open_system_hardware.md) or
[MPS/DMRG](tensor_networks.md) instead.

---

## Theory

### The Lindblad Master Equation

A closed quantum system evolves unitarily: $d|\psi\rangle/dt = -iH|\psi\rangle$.
Real systems interact with their environment. The Lindblad equation is the
most general Markovian master equation that preserves trace, Hermiticity,
and positivity of the density matrix $\rho$:

$$\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)$$

The first term generates coherent (unitary) evolution. The second term —
the *dissipator* — describes irreversible coupling to the environment
through Lindblad operators $L_k$.

### Channels in This Module

Two physical channels are implemented, parameterised per qubit:

| Channel | Lindblad operator | Rate | Physical meaning |
|---------|-------------------|------|------------------|
| Amplitude damping | $L_k = \sqrt{\gamma_\text{amp}}\, \sigma^-_k$ | $\gamma_\text{amp}$ | Energy relaxation ($T_1$ decay) |
| Pure dephasing | $L_k = \sqrt{\gamma_\text{deph}/2}\, \sigma^z_k$ | $\gamma_\text{deph}$ | Phase randomisation ($T_2$ decay) |

For the Kuramoto-XY system, amplitude damping destroys synchronisation by
relaxing excitations toward the ground state. Dephasing destroys off-diagonal
coherences without changing populations.

### The XY Hamiltonian

$$H = -\sum_{i<j} K_{ij}(X_i X_j + Y_i Y_j) - \sum_i \omega_i Z_i$$

where $K_{ij}$ is the coupling matrix (typically exponentially decaying
with distance) and $\omega_i$ are natural frequencies.

### Order Parameter and Purity

- **Kuramoto order parameter** $R$: extracted from single-qubit Pauli
  expectations $\langle X_k \rangle$, $\langle Y_k \rangle$ via the
  density matrix. Quantifies synchronisation ($R=1$ perfect sync,
  $R \to 0$ incoherent).
- **Purity** $\text{Tr}(\rho^2)$: $1$ for a pure state, $1/d$ for
  maximally mixed. Decreases under dissipation.

---

## API Reference

### `LindbladKuramotoSolver`

```python
from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver
```

#### Constructor

```python
LindbladKuramotoSolver(
    n_oscillators: int,
    K_coupling: np.ndarray,       # shape (n, n)
    omega_natural: np.ndarray,    # shape (n,)
    gamma_amp: float = 0.0,       # amplitude damping rate
    gamma_deph: float = 0.0,      # dephasing rate
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_oscillators` | `int` | Number of qubits (oscillators) |
| `K_coupling` | `ndarray (n, n)` | Coupling matrix. Must be symmetric. |
| `omega_natural` | `ndarray (n,)` | Natural frequencies |
| `gamma_amp` | `float` | Amplitude damping rate per qubit. $\gamma = 0$ → no damping. |
| `gamma_deph` | `float` | Pure dephasing rate per qubit. $\gamma = 0$ → no dephasing. |

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `build()` | `() → None` | — | Build Hamiltonian and Lindblad operators. Called automatically by `run()`. |
| `run()` | `(t_max, dt, method="RK45") → dict` | See below | Time-evolve under Lindblad dynamics. |
| `order_parameter()` | `(rho) → float` | Kuramoto $R$ | Extract $R$ from density matrix. |
| `purity()` | `(rho) → float` | $\text{Tr}(\rho^2)$ | Purity of density matrix. |

#### `run()` Return Value

```python
{
    "times": np.ndarray,      # shape (n_steps,)
    "R": np.ndarray,          # Kuramoto R at each time, shape (n_steps,)
    "purity": np.ndarray,     # Tr(ρ²) at each time, shape (n_steps,)
    "rho_final": np.ndarray,  # final density matrix, shape (dim, dim)
}
```

---

## Tutorial: Open-System Kuramoto Synchronisation

### Step 1: Set Up the System

```python
import numpy as np
from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver

# 4-oscillator chain with exponentially decaying coupling
n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)
```

### Step 2: Closed-System Baseline

```python
solver_closed = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.0, gamma_deph=0.0)
result_closed = solver_closed.run(t_max=2.0, dt=0.05)

print(f"Closed system — R: {result_closed['R'][0]:.3f} → {result_closed['R'][-1]:.3f}")
print(f"Purity: {result_closed['purity'][-1]:.6f}")  # should be 1.000000
```

### Step 3: Add Dissipation

```python
solver_open = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.05, gamma_deph=0.02)
result_open = solver_open.run(t_max=2.0, dt=0.05)

print(f"Open system — R: {result_open['R'][0]:.3f} → {result_open['R'][-1]:.3f}")
print(f"Purity: {result_open['purity'][0]:.3f} → {result_open['purity'][-1]:.3f}")
```

### Step 4: Compare

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(result_closed['times'], result_closed['R'], label='Closed')
ax1.plot(result_open['times'], result_open['R'], label='Open (γ=0.05)')
ax1.set_xlabel('Time')
ax1.set_ylabel('R')
ax1.legend()
ax1.set_title('Synchronisation Order Parameter')

ax2.plot(result_closed['times'], result_closed['purity'], label='Closed')
ax2.plot(result_open['times'], result_open['purity'], label='Open')
ax2.set_xlabel('Time')
ax2.set_ylabel('Tr(ρ²)')
ax2.legend()
ax2.set_title('Purity')

plt.tight_layout()
plt.savefig('lindblad_comparison.png', dpi=150)
```

---

## Examples

### Strong Damping Kills Synchronisation

```python
solver_strong = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.5)
result_strong = solver_strong.run(t_max=5.0, dt=0.1)
print(f"R(T=5) = {result_strong['R'][-1]:.4f}")  # → near 0
print(f"Purity(T=5) = {result_strong['purity'][-1]:.4f}")  # → near 1/2^n
```

### Dephasing Only (No Energy Relaxation)

```python
solver_deph = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.0, gamma_deph=0.1)
result_deph = solver_deph.run(t_max=2.0, dt=0.05)
# Populations unchanged, but coherences decay
```

### Verify Density Matrix Properties

```python
rho = result_open['rho_final']
assert np.allclose(np.trace(rho), 1.0), "Trace not preserved"
assert np.allclose(rho, rho.conj().T), "Not Hermitian"
eigenvalues = np.linalg.eigvalsh(rho)
assert np.all(eigenvalues >= -1e-12), "Not positive semidefinite"
```

---

## Comparison with Other Tools

| Feature | This module | QuTiP `mesolve` | MISTIQS |
|---------|-------------|-----------------|---------|
| Lindblad equation | Yes | Yes | No |
| Hamiltonian | Kuramoto-XY (built-in) | Any (user-supplied) | TFIM only |
| Coupling matrix | Arbitrary $K_{ij}$ | Any | Nearest-neighbour |
| Solver | `scipy.solve_ivp` (RK45) | Internal ODE solver | — |
| GPU | No | No (QuTiP 5: CuPy) | No |
| Output | $R(t)$, purity, $\rho$ | Arbitrary expect | — |

**When to use this module:** You want open-system dynamics for the
Kuramoto-XY Hamiltonian with the SCPN coupling matrix $K_{nm}$, integrated
with the rest of the scpn-quantum-control pipeline.

**When to use QuTiP:** You need arbitrary Hamiltonians, Floquet theory,
stochastic Schrödinger equation, or QuTiP's extensive toolbox. Our module
is not a QuTiP replacement — it is a specialised solver for one Hamiltonian.

---

## Scaling

| $n$ | Hilbert space dim | Density matrix size | Memory (complex128) |
|:---:|:-----------------:|:-------------------:|:-------------------:|
| 4 | 16 | 16 × 16 | 4 KB |
| 8 | 256 | 256 × 256 | 1 MB |
| 10 | 1,024 | 1,024 × 1,024 | 16 MB |
| 12 | 4,096 | 4,096 × 4,096 | 256 MB |
| 14 | 16,384 | 16,384 × 16,384 | 4 GB |

Beyond $n = 12$, wall-time becomes the bottleneck (the RHS evaluation at
each time step is $O(d^2)$ where $d = 2^n$). For larger systems, use
the MCWF method (`phase/tensor_jump.py`) which evolves state vectors
instead of density matrices.

---

## References

1. Lindblad, G. "On the generators of quantum dynamical semigroups."
   *Commun. Math. Phys.* **48**, 119–130 (1976).
2. Gorini, V., Kossakowski, A. & Sudarshan, E. C. G. "Completely positive
   dynamical semigroups of N-level systems." *J. Math. Phys.* **17**, 821 (1976).
3. Ameri, V. *et al.* "Mutual information as an order parameter for quantum
   synchronization." *PRA* **91**, 012301 (2015).
4. Giorgi, G. L. *et al.* "Quantum correlations and mutual synchronization."
   *PRA* **85**, 052101 (2012).

---

## See Also

- [Open-System Hardware Circuits](open_system_hardware.md) — ancilla Lindblad
  and MCWF for hardware execution
- [Tensor Networks](tensor_networks.md) — MPS/DMRG for $n > 16$
- [Symmetry Sectors](symmetry.md) — reduce Hilbert space before Lindblad
