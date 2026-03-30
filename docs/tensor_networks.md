# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tensor Network Documentation

# Tensor Network Methods

Two modules for tensor-network-based simulation of the Kuramoto-XY system:

1. **MPS/DMRG/TEBD** (`phase/mps_evolution.py`) — ground state and time
   evolution via quimb
2. **Contraction optimiser** (`phase/contraction_optimiser.py`) — optimal
   einsum paths via cotengra

These extend the package beyond exact diagonalisation limits to $n = 32$–64
qubits (system-size-dependent on bond dimension and entanglement).

**Caveat:** Both modules require optional dependencies (`quimb`, `cotengra`).
Install via `pip install scpn-quantum-control[quimb]` or
`pip install scpn-quantum-control[cotengra]`.

---

## Part 1: MPS Evolution

`scpn_quantum_control.phase.mps_evolution`

### Theory

#### Matrix Product States

An MPS represents an $n$-qubit state as a chain of tensors:

$$|\psi\rangle = \sum_{s_1 \ldots s_n} A^{s_1} A^{s_2} \cdots A^{s_n} |s_1 \ldots s_n\rangle$$

where each $A^{s_i}$ is a matrix of dimension $\chi \times \chi$ ($\chi$ =
bond dimension). Memory scales as $O(n \chi^2 d)$ instead of $O(d^n)$
for exact states.

MPS is efficient for states with bounded entanglement (area-law states).
Ground states of 1D gapped Hamiltonians satisfy this. Highly entangled
states (e.g., volume-law after long time evolution) require large $\chi$.

#### DMRG

Density Matrix Renormalisation Group finds the ground state by variationally
optimising the MPS tensors site-by-site. Converges to the true ground state
for 1D systems with moderate entanglement.

#### TEBD

Time-Evolving Block Decimation propagates $|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle$
by Trotterising the evolution into nearest-neighbour gates and applying them
to the MPS, truncating the bond dimension after each gate.

### Limitations

- **Nearest-neighbour only:** quimb's `SpinHam1D` supports NN couplings.
  Longer-range terms from the exponentially decaying $K_{nm}$ are dropped.
  For the standard SCPN coupling matrix, NN terms dominate.
- **Bond dimension:** Higher $\chi$ → more accurate but slower. Typical
  values: $\chi = 32$–128 for ground states, $\chi = 64$–256 for dynamics.
- **Entanglement growth:** TEBD accuracy degrades at long times as
  entanglement grows beyond what the bond dimension can capture.

### API Reference

```python
from scpn_quantum_control.phase.mps_evolution import (
    is_quimb_available,
    dmrg_ground_state,
    tebd_evolution,
)
```

#### `dmrg_ground_state`

```python
result = dmrg_ground_state(
    K: np.ndarray,          # (n, n) coupling matrix
    omega: np.ndarray,      # (n,) natural frequencies
    bond_dim: int = 64,     # maximum bond dimension
    cutoff: float = 1e-10,  # SVD truncation cutoff
    max_sweeps: int = 20,   # DMRG sweep limit
) -> dict
```

**Returns:**

```python
{
    "energy": float,         # ground state energy
    "mps": quimb.MatrixProductState,  # ground state MPS
    "converged": bool,       # True if energy converged
    "bond_dims": list[int],  # bond dimensions along the chain
}
```

#### `tebd_evolution`

```python
result = tebd_evolution(
    K: np.ndarray,
    omega: np.ndarray,
    t_max: float = 1.0,     # total evolution time
    dt: float = 0.05,       # Trotter step size
    bond_dim: int = 64,     # maximum bond dimension
    cutoff: float = 1e-10,  # SVD truncation
    order: int = 2,         # Trotter order (2 or 4)
) -> dict
```

**Returns:**

```python
{
    "times": np.ndarray,          # time points
    "R": np.ndarray,              # order parameter R(t)
    "bond_dims_final": list[int], # bond dimensions at final time
    "mps_final": quimb.MatrixProductState,  # final MPS
}
```

#### `is_quimb_available`

```python
is_quimb_available() -> bool
```

Returns `True` if quimb is installed.

### Tutorial

```python
import numpy as np
from scpn_quantum_control.phase.mps_evolution import (
    dmrg_ground_state, tebd_evolution, is_quimb_available
)

if not is_quimb_available():
    raise ImportError("Install quimb: pip install quimb")

n = 16
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# Ground state
gs = dmrg_ground_state(K, omega, bond_dim=64, max_sweeps=20)
print(f"DMRG energy: {gs['energy']:.6f}")
print(f"Converged: {gs['converged']}")
print(f"Bond dimensions: {gs['bond_dims']}")

# Time evolution
dyn = tebd_evolution(K, omega, t_max=2.0, dt=0.05, bond_dim=64, order=2)
print(f"R(0) = {dyn['R'][0]:.3f}")
print(f"R(T) = {dyn['R'][-1]:.3f}")
```

### Bond Dimension Selection

| System | Recommended $\chi$ | Memory per MPS |
|--------|:-------------------:|:--------------:|
| $n \leq 16$ ground state | 32–64 | < 1 MB |
| $n = 32$ ground state | 64–128 | ~10 MB |
| $n = 64$ ground state | 128–256 | ~100 MB |
| Short-time dynamics ($t < 1$) | 64–128 | ~10 MB |
| Long-time dynamics ($t > 5$) | 256+ | ~1 GB |

---

## Part 2: Contraction Optimiser

`scpn_quantum_control.phase.contraction_optimiser`

### Theory

Tensor contraction order affects computational cost dramatically.
For a chain of matrix multiplications $A \cdot B \cdot C$, the parenthesisation
matters: $(AB)C$ vs $A(BC)$ can differ by orders of magnitude in flops.

`cotengra` (Gray & Kourtis, 2021) finds near-optimal contraction paths
for arbitrary tensor networks using hyper-optimisation over graph partitioning
algorithms.

This module provides a drop-in `np.einsum` replacement that uses cotengra
when available, with numpy fallback.

### API Reference

```python
from scpn_quantum_control.phase.contraction_optimiser import (
    is_cotengra_available,
    optimal_contraction_path,
    contract,
    benchmark_contraction,
)
```

#### `contract`

```python
result = contract(
    subscripts: str,        # einsum subscripts, e.g. "ij,jk->ik"
    *operands: np.ndarray,  # input tensors
    optimiser: str = "auto",  # "auto", "greedy", "optimal"
) -> np.ndarray
```

Drop-in replacement for `np.einsum`. Uses cotengra path optimisation
when available.

#### `optimal_contraction_path`

```python
path, info = optimal_contraction_path(
    subscripts: str,
    *operands: np.ndarray,
    optimiser: str = "auto",
) -> tuple[list[tuple[int, ...]], dict]
```

Returns the contraction path and metadata (flops, memory estimates)
without performing the contraction.

#### `benchmark_contraction`

```python
result = benchmark_contraction(
    subscripts: str,
    *operands: np.ndarray,
    n_repeats: int = 10,
) -> dict
```

**Returns:**

```python
{
    "naive_ms": float,       # wall time with default path (ms)
    "optimised_ms": float,   # wall time with cotengra path (ms)
    "speedup": float,        # naive / optimised
}
```

### Example

```python
import numpy as np
from scpn_quantum_control.phase.contraction_optimiser import (
    contract, benchmark_contraction, is_cotengra_available
)

print(f"cotengra available: {is_cotengra_available()}")

# Simple matrix chain
A = np.random.randn(100, 200)
B = np.random.randn(200, 300)
C = np.random.randn(300, 50)

# Drop-in replacement for np.einsum
result = contract("ij,jk,kl->il", A, B, C)
print(f"Result shape: {result.shape}")  # (100, 50)

# Benchmark
bench = benchmark_contraction("ij,jk,kl->il", A, B, C, n_repeats=20)
print(f"Naive: {bench['naive_ms']:.1f} ms")
print(f"Optimised: {bench['optimised_ms']:.1f} ms")
print(f"Speedup: {bench['speedup']:.1f}×")
```

---

## Comparison

| Feature | This module (quimb) | quimb standalone | ITensor | TeNPy |
|---------|---------------------|------------------|---------|-------|
| DMRG | Yes | Yes | Yes | Yes |
| TEBD | Yes | Yes | Yes | Yes |
| 2D PEPS | No | Yes | Yes | No |
| Hamiltonian | Kuramoto-XY (built-in) | Any | Any | Any |
| Language | Python | Python | Julia/C++ | Python |
| cotengra | Yes (via contraction_optimiser) | Yes | No | No |

Our modules wrap quimb with Kuramoto-XY-specific Hamiltonian construction
and order parameter extraction. They are not a quimb replacement.

---

## References

1. White, S. R. "Density matrix formulation for quantum renormalization
   groups." *PRL* **69**, 2863 (1992). (DMRG)
2. Vidal, G. "Efficient simulation of one-dimensional quantum many-body
   systems." *PRL* **93**, 040502 (2004). (TEBD)
3. Gray, J. & Kourtis, S. "Hyper-optimized tensor network contraction."
   *Quantum* **5**, 410 (2021). (cotengra)
4. Gray, J. "quimb: A Python library for quantum information and many-body
   calculations." *JOSS* **3**, 819 (2018).

---

## See Also

- [Sparse Hamiltonian](sparse.md) — sparse ED for intermediate system sizes
- [Symmetry Sectors](symmetry.md) — reduce Hilbert space before MPS
- [Neural Quantum States](neural_quantum_states.md) — alternative variational ansatz
