# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Symmetry Sectors Documentation

# Symmetry-Aware Exact Diagonalisation

Three modules exploit symmetries of the XY Hamiltonian to reduce the
Hilbert space dimension for exact diagonalisation (ED):

| Module | Symmetry | Reduction factor |
|--------|----------|:----------------:|
| `analysis/symmetry_sectors.py` | Z₂ parity | 2× |
| `analysis/magnetisation_sectors.py` | U(1) total magnetisation | up to 13× (n=16) |
| `analysis/translation_symmetry.py` | Cyclic translation | up to 160× (n=16, combined) |

These can be combined: Z₂ ⊂ U(1) ⊂ U(1) × Translation.

**Caveat:** Translation symmetry requires homogeneous frequencies
($\omega_i = \omega$) and circulant coupling ($K_{ij} = K(|i-j| \bmod N)$).
The SCPN case has heterogeneous $\omega$, so translation is typically
broken — use U(1) sectors instead.

---

## Theory

### Z₂ Parity

The XY Hamiltonian commutes with the global parity operator
$P = Z_1 \otimes Z_2 \otimes \cdots \otimes Z_N$. Every eigenstate has
definite parity (even or odd number of excitations). This splits the
$2^N$-dimensional space into two sectors of $2^{N-1}$ each.

Computing the level-spacing ratio $\bar{r}$ *within* a single parity
sector is essential: overlaying two independent spectra always gives
Poisson statistics, masking the true spectral statistics (GOE vs Poisson).

### U(1) Magnetisation

The XY interaction $X_i X_j + Y_i Y_j = 2(\sigma^+_i \sigma^-_j + \sigma^-_i \sigma^+_j)$
is a flip-flop — it swaps excitations but never creates or destroys them.
Total magnetisation $M = \sum_i Z_i$ is therefore conserved: $[H, M] = 0$.

The Hilbert space decomposes into $N+1$ sectors labelled by
$M \in \{-N, -N+2, \ldots, N\}$. Sector $M$ has dimension $\binom{N}{k}$
where $k = (N+M)/2$ is the excitation count.

| N | Full dim | Z₂ sector | U(1) largest ($M=0$) | U(1) reduction |
|:-:|:--------:|:---------:|:--------------------:|:--------------:|
| 12 | 4,096 | 2,048 | 924 | 4.4× |
| 16 | 65,536 | 32,768 | 12,870 | 5.1× |
| 18 | 262,144 | 131,072 | 48,620 | 5.4× |
| 20 | 1,048,576 | 524,288 | 184,756 | 5.7× |

### Translation Symmetry

When the system is translationally invariant (uniform $\omega$, circulant $K$),
the cyclic shift operator $T: |b_0 b_1 \ldots b_{N-1}\rangle \to |b_{N-1} b_0 \ldots b_{N-2}\rangle$
commutes with $H$. Eigenstates carry definite crystal momentum
$k = 2\pi m / N$ ($m = 0, \ldots, N-1$).

Combined with U(1), the space splits into $N \times (N+1)$ sectors.
For $n=16$: the $k=0$ sector of $M=0$ contains ~805 states — a 80×
reduction from the full 65,536.

---

## Part 1: Z₂ Parity Sectors

`scpn_quantum_control.analysis.symmetry_sectors`

### API Reference

```python
from scpn_quantum_control.analysis.symmetry_sectors import (
    basis_indices_by_parity,
    project_hamiltonian,
    build_sector_hamiltonian,
    eigh_by_sector,
    level_spacing_by_sector,
    memory_estimate_mb,
)
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `basis_indices_by_parity(n)` | `int → (ndarray, ndarray)` | `(even_indices, odd_indices)` — sorted basis state indices per sector |
| `project_hamiltonian(H, sector_indices)` | `(ndarray, ndarray) → ndarray` | Projected Hamiltonian $H_\text{sector}$ |
| `build_sector_hamiltonian(K, omega, parity=0)` | `(ndarray, ndarray, int) → (ndarray, ndarray)` | `(H_sector, sector_indices)` |
| `eigh_by_sector(K, omega)` | `(ndarray, ndarray) → dict` | Both sectors diagonalised |
| `level_spacing_by_sector(K, omega)` | `(ndarray, ndarray) → dict` | Level-spacing ratio $\bar{r}$ per sector |
| `memory_estimate_mb(n, use_sectors=True)` | `(int, bool) → float` | Memory estimate in MB |

#### `eigh_by_sector` Return Value

```python
{
    "eigvals_even": np.ndarray,     # eigenvalues of even sector
    "eigvecs_even": np.ndarray,     # eigenvectors of even sector
    "indices_even": np.ndarray,     # basis indices in even sector
    "eigvals_odd": np.ndarray,
    "eigvecs_odd": np.ndarray,
    "indices_odd": np.ndarray,
    "eigvals_all": np.ndarray,      # all eigenvalues sorted
    "ground_energy": float,
    "ground_parity": int,           # 0 (even) or 1 (odd)
}
```

#### `level_spacing_by_sector` Return Value

```python
{
    "r_bar_even": float,    # mean level-spacing ratio, even sector
    "r_bar_odd": float,     # mean level-spacing ratio, odd sector
    "r_bar_combined": float,  # combined (for reference, but biased)
}
```

### Example

```python
import numpy as np
from scpn_quantum_control.analysis.symmetry_sectors import (
    eigh_by_sector, level_spacing_by_sector, memory_estimate_mb
)

n = 8
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

result = eigh_by_sector(K, omega)
print(f"Ground energy: {result['ground_energy']:.6f}")
print(f"Ground parity: {'even' if result['ground_parity'] == 0 else 'odd'}")
print(f"Even sector: {len(result['eigvals_even'])} states")
print(f"Odd sector: {len(result['eigvals_odd'])} states")

ls = level_spacing_by_sector(K, omega)
print(f"r̄(even) = {ls['r_bar_even']:.3f}")
print(f"r̄(odd) = {ls['r_bar_odd']:.3f}")
# GOE ~ 0.530, Poisson ~ 0.386

print(f"Memory (full): {memory_estimate_mb(16, False):.0f} MB")
print(f"Memory (sector): {memory_estimate_mb(16, True):.0f} MB")
```

---

## Part 2: U(1) Magnetisation Sectors

`scpn_quantum_control.analysis.magnetisation_sectors`

### Rust Acceleration

`basis_by_magnetisation` uses the Rust function `magnetisation_labels`
for popcount computation when the engine is installed.
Measured speedup: **97×** at $n=8$ (0.001 ms vs 0.11 ms).

### API Reference

```python
from scpn_quantum_control.analysis.magnetisation_sectors import (
    basis_by_magnetisation,
    sector_dimensions,
    largest_sector_dim,
    project_to_sector,
    build_sector_hamiltonian,
    eigh_by_magnetisation,
    level_spacing_by_magnetisation,
    memory_estimate,
)
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `basis_by_magnetisation(n)` | `int → dict[int, ndarray]` | Maps $M$ → basis state indices |
| `sector_dimensions(n)` | `int → dict[int, int]` | Maps $M$ → sector dimension |
| `largest_sector_dim(n)` | `int → int` | $\binom{N}{N/2}$ for even $N$ |
| `project_to_sector(H_full, sector_indices)` | `(ndarray, ndarray) → ndarray` | Projected Hamiltonian |
| `build_sector_hamiltonian(K, omega, M)` | `(ndarray, ndarray, int) → (ndarray, ndarray)` | `(H_sector, sector_indices)` |
| `eigh_by_magnetisation(K, omega, sectors=None)` | `(ndarray, ndarray, list[int] | None) → dict` | Diagonalise specified sectors |
| `level_spacing_by_magnetisation(K, omega, M=None)` | `(ndarray, ndarray, int | None) → dict` | $\bar{r}$ within sector $M$ |
| `memory_estimate(n)` | `int → dict` | Memory comparison: full vs Z₂ vs U(1) |

#### `eigh_by_magnetisation` Return Value

```python
{
    "results": {
        M: {
            "eigvals": np.ndarray,
            "eigvecs": np.ndarray,
            "indices": np.ndarray,
            "dim": int,
        }
        for M in sectors
    },
    "eigvals_all": np.ndarray,      # all eigenvalues sorted
    "ground_energy": float,
    "ground_sector": int,           # M value of ground state
    "n_sectors_computed": int,
}
```

#### `memory_estimate` Return Value

```python
{
    "full_ed_mb": float,
    "z2_sector_mb": float,
    "u1_largest_mb": float,
    "u1_m0_mb": float,
}
```

### Example

```python
import numpy as np
from scpn_quantum_control.analysis.magnetisation_sectors import (
    eigh_by_magnetisation, level_spacing_by_magnetisation,
    memory_estimate, sector_dimensions
)

n = 8
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# Sector dimensions
dims = sector_dimensions(n)
for M, d in sorted(dims.items()):
    print(f"M={M:+d}: {d} states")

# Diagonalise all sectors
result = eigh_by_magnetisation(K, omega)
print(f"Ground: E={result['ground_energy']:.4f}, M={result['ground_sector']}")

# Diagonalise only M=0 (largest sector)
result_m0 = eigh_by_magnetisation(K, omega, sectors=[0])
print(f"M=0 ground: {result_m0['results'][0]['eigvals'][0]:.4f}")

# Level spacing within M=0 (sector-resolved, no artefacts)
ls = level_spacing_by_magnetisation(K, omega, M=0)
print(f"r̄(M=0) = {ls['r_bar']:.3f}")

# Memory estimates for N=16
est = memory_estimate(16)
print(f"Full ED: {est['full_ed_mb']:.0f} MB")
print(f"U(1) M=0: {est['u1_m0_mb']:.0f} MB")
```

---

## Part 3: Translation Symmetry

`scpn_quantum_control.analysis.translation_symmetry`

### API Reference

```python
from scpn_quantum_control.analysis.translation_symmetry import (
    is_translation_invariant,
    momentum_sectors,
    momentum_sector_dimensions,
    eigh_with_translation,
)
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `is_translation_invariant(K, omega, tol=1e-10)` | `(ndarray, ndarray, float) → bool` | Check for cyclic symmetry |
| `momentum_sectors(n)` | `int → dict[int, list[int]]` | Maps momentum $m$ → representative basis states |
| `momentum_sector_dimensions(n)` | `int → dict[int, int]` | Maps $m$ → sector dimension |
| `eigh_with_translation(K, omega, momentum=0)` | `(ndarray, ndarray, int) → dict` | Diagonalise in momentum sector |

#### `eigh_with_translation` Return Value

```python
{
    "eigvals": np.ndarray,    # eigenvalues in this momentum sector
    "dim": int,                # sector dimension
    "momentum": int,           # momentum quantum number m
    "is_ti": bool,             # True if system is translation-invariant
}
```

**Raises:** `ValueError` if the system is not translation-invariant.

### Example

```python
import numpy as np
from scpn_quantum_control.analysis.translation_symmetry import (
    is_translation_invariant, eigh_with_translation,
    momentum_sector_dimensions
)

n = 8
# Circulant ring coupling (translation-invariant)
K_ring = np.zeros((n, n))
for i in range(n):
    K_ring[i, (i+1) % n] = 1.0
    K_ring[(i+1) % n, i] = 1.0

omega_uniform = np.ones(n) * 1.0

print(is_translation_invariant(K_ring, omega_uniform))  # True

# Sector dimensions
dims = momentum_sector_dimensions(n)
for m, d in sorted(dims.items()):
    print(f"k=2π·{m}/{n}: {d} states")

# Diagonalise k=0 sector
result = eigh_with_translation(K_ring, omega_uniform, momentum=0)
print(f"k=0 sector: {result['dim']} states")
print(f"Ground energy (k=0): {result['eigvals'][0]:.4f}")
```

### When Translation Is Broken

```python
omega_hetero = np.linspace(0.8, 1.2, n)
print(is_translation_invariant(K_ring, omega_hetero))  # False

# This will raise ValueError:
# eigh_with_translation(K_ring, omega_hetero, momentum=0)
```

For heterogeneous $\omega$, use U(1) magnetisation sectors instead.

---

## Tutorial: Scaling to N=20 with Symmetry

### Pipeline: Z₂ → U(1) → Sparse

```python
import numpy as np
from scpn_quantum_control.analysis.magnetisation_sectors import (
    memory_estimate, eigh_by_magnetisation
)
from scpn_quantum_control.bridge.sparse_hamiltonian import sparse_eigsh

n = 12  # start manageable

K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# Step 1: Check memory
est = memory_estimate(n)
print(f"Full ED: {est['full_ed_mb']:.0f} MB")
print(f"U(1) M=0: {est['u1_m0_mb']:.0f} MB")

# Step 2: U(1) sector ED (exact, fast for n<=14)
result = eigh_by_magnetisation(K, omega, sectors=[0])
print(f"M=0 ground: {result['results'][0]['eigvals'][0]:.6f}")

# Step 3: For n>14, switch to sparse eigsh within U(1) sector
result_sparse = sparse_eigsh(K, omega, k=10, M=0)
print(f"Sparse ground (M=0): {result_sparse['eigvals'][0]:.6f}")
```

---

## Comparison with QuSpin

| Feature | This module | QuSpin |
|---------|-------------|--------|
| Z₂ parity | Yes | Yes |
| U(1) magnetisation | Yes | Yes |
| Translation | Yes (cyclic only) | Yes (general) |
| Reflection | No | Yes |
| Custom symmetries | No | Yes (via `user_basis`) |
| Sparse Hamiltonian | Via `sparse_hamiltonian.py` | Built-in |
| Hamiltonian | Kuramoto-XY only | Any spin model |
| ED + symmetry API | Separate modules | Unified `hamiltonian` class |

QuSpin is more general. Our modules are specialised for the XY Hamiltonian
and integrate with the scpn-quantum-control pipeline (coupling matrix
$K_{nm}$, quantum circuits, Rust acceleration).

---

## References

1. Weinberg, P. & Bukov, M. "QuSpin: a Python package for dynamics and
   exact diagonalisation of quantum many body systems."
   *SciPost Phys.* **2**, 003 (2017).
2. Oganesyan, V. & Huse, D. A. "Localization of interacting fermions at
   high temperature." *PRA* **75**, 155111 (2007).
   (Level-spacing ratio $\bar{r}$ for MBL diagnostics.)

---

## See Also

- [Sparse Hamiltonian](sparse.md) — sparse CSC construction + ARPACK eigsh
- [Lindblad Solver](lindblad.md) — open-system dynamics within symmetry sectors
- [Tensor Networks](tensor_networks.md) — MPS/DMRG beyond ED limits
