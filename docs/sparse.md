# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sparse Hamiltonian Documentation

# Sparse Hamiltonian Construction

`scpn_quantum_control.bridge.sparse_hamiltonian`

The XY Hamiltonian is sparse: for each basis state $|k\rangle$, only
$O(n^2)$ off-diagonal elements are non-zero (one per qubit pair with
differing bits). This gives $O(n^2 \cdot 2^n)$ non-zeros in a
$2^n \times 2^n$ matrix — less than 1% fill for $n \geq 10$.

Sparse storage (scipy CSC) + iterative eigensolver (ARPACK `eigsh`)
enables exact diagonalisation at scales impossible with dense matrices.

---

## Theory

### Sparse Structure of the XY Hamiltonian

$$H = -\sum_{i<j} K_{ij}(X_i X_j + Y_i Y_j) - \sum_i \omega_i Z_i$$

In the computational basis:

- **Diagonal:** $H_{kk} = -\sum_i \omega_i (1 - 2b_i(k))$ where $b_i(k)$
  is the $i$-th bit of $k$
- **Off-diagonal:** $H_{k, k \oplus \text{mask}_{ij}} = -2K_{ij}$ when
  bits $i$ and $j$ differ in state $|k\rangle$

The XOR operation $k \oplus \text{mask}_{ij}$ flips bits $i$ and $j$
simultaneously — the flip-flop interaction.

### Memory Comparison

| $n$ | Dense (MB) | Sparse (MB) | Reduction |
|:---:|:----------:|:-----------:|:---------:|
| 12 | 134 | 12 | 11× |
| 14 | 2,147 | 50 | 43× |
| 16 | 32,768 | 200 | 164× |
| 18 | 524,288 | 800 | 655× |

### Combined with U(1) Sectors

Sparse construction within a single magnetisation sector reduces both
dimension *and* non-zero count. For $n=20$, $M=0$: dim = 184,756 with
~$10^6$ non-zeros → feasible on a 32 GB workstation.

---

## Rust Acceleration

The function `build_sparse_hamiltonian` uses the Rust function
`build_sparse_xy_hamiltonian` when the engine is installed. Returns COO
triplets (rows, cols, vals) that are assembled into a scipy CSC matrix.

Measured speedup: **80×** at $n=8$ (0.024 ms vs 1.9 ms).

---

## API Reference

```python
from scpn_quantum_control.bridge.sparse_hamiltonian import (
    build_sparse_hamiltonian,
    build_sparse_sector_hamiltonian,
    sparse_eigsh,
    sparsity_stats,
)
```

### `build_sparse_hamiltonian`

```python
H = build_sparse_hamiltonian(
    K: np.ndarray,       # (n, n) coupling matrix
    omega: np.ndarray,   # (n,) natural frequencies
) -> scipy.sparse.csc_matrix
```

Returns the full $2^n \times 2^n$ XY Hamiltonian as a sparse CSC matrix.

### `build_sparse_sector_hamiltonian`

```python
H_sector, indices = build_sparse_sector_hamiltonian(
    K: np.ndarray,
    omega: np.ndarray,
    M: int,              # target magnetisation
) -> tuple[scipy.sparse.csc_matrix, np.ndarray]
```

Combines sparse construction with U(1) symmetry. Returns the projected
Hamiltonian within magnetisation sector $M$ and the corresponding basis
state indices.

**Raises:** `ValueError` if $M$ is not a valid magnetisation value.

### `sparse_eigsh`

```python
result = sparse_eigsh(
    K: np.ndarray,
    omega: np.ndarray,
    k: int = 10,          # number of eigenvalues
    which: str = "SA",    # "SA" (smallest algebraic), "LA", "SM"
    M: int | None = None, # if set, compute within magnetisation sector
) -> dict
```

**Returns:**

```python
{
    "eigvals": np.ndarray,    # k eigenvalues
    "eigvecs": np.ndarray,    # corresponding eigenvectors
    "nnz": int,                # number of non-zeros
    "dim": int,                # matrix dimension
    "sector": int | None,      # magnetisation sector (if used)
}
```

Uses ARPACK (`scipy.sparse.linalg.eigsh`) for the $k$ extremal
eigenvalues. Complexity: $O(k \cdot \text{nnz} \cdot \text{iterations})$
— much faster than full diagonalisation when $k \ll \text{dim}$.

### `sparsity_stats`

```python
stats = sparsity_stats(
    n: int,
    K: np.ndarray,
) -> dict
```

**Returns:**

```python
{
    "dim": int,              # 2^n
    "nnz_estimate": int,     # estimated non-zero count
    "fill_pct": float,       # fill percentage
    "memory_sparse_mb": float,
    "memory_dense_mb": float,
}
```

Estimates without building the full matrix — useful for checking
feasibility before committing resources.

---

## Tutorial

### Basic Usage

```python
import numpy as np
from scpn_quantum_control.bridge.sparse_hamiltonian import (
    build_sparse_hamiltonian, sparse_eigsh, sparsity_stats
)

n = 10
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# Check feasibility first
stats = sparsity_stats(n, K)
print(f"Dimension: {stats['dim']}")
print(f"NNZ estimate: {stats['nnz_estimate']}")
print(f"Fill: {stats['fill_pct']:.4f}%")
print(f"Sparse: {stats['memory_sparse_mb']:.1f} MB")
print(f"Dense: {stats['memory_dense_mb']:.1f} MB")
```

### Sparse Eigenvalues

```python
# 10 smallest eigenvalues
result = sparse_eigsh(K, omega, k=10)
print(f"Ground energy: {result['eigvals'][0]:.6f}")
print(f"Gap: {result['eigvals'][1] - result['eigvals'][0]:.6f}")
```

### Combined with U(1) Sectors

```python
# Ground state within M=0 sector
result_m0 = sparse_eigsh(K, omega, k=5, M=0)
print(f"M=0 ground: {result_m0['eigvals'][0]:.6f}")
print(f"Sector dim: {result_m0['dim']}")
```

### Verify Against Dense (Small Systems)

```python
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

n_small = 6
K6 = K[:n_small, :n_small]
omega6 = omega[:n_small]

H_dense = knm_to_dense_matrix(K6, omega6)
H_sparse = build_sparse_hamiltonian(K6, omega6)

# Compare eigenvalues
eigvals_dense = np.sort(np.linalg.eigvalsh(H_dense))
result_sparse = sparse_eigsh(K6, omega6, k=len(eigvals_dense))
eigvals_sparse = np.sort(result_sparse['eigvals'])

print(f"Max error: {np.max(np.abs(eigvals_dense - eigvals_sparse)):.2e}")
```

---

## Comparison

| Feature | This module | QuSpin | SciPy `eigsh` directly |
|---------|-------------|--------|------------------------|
| XY Hamiltonian | Built-in | User constructs | User constructs |
| Sparse format | CSC (automatic) | CSR | User choice |
| U(1) sector support | Built-in | Built-in | Manual |
| Rust acceleration | 80× (construction) | No | No |
| Arbitrary spin models | No (XY only) | Yes | N/A |

---

## References

1. Lehoucq, R. B., Sorensen, D. C. & Yang, C. "ARPACK Users' Guide."
   SIAM (1998).
2. Weinberg, P. & Bukov, M. "QuSpin." *SciPost Phys.* **2**, 003 (2017).

---

## See Also

- [Symmetry Sectors](symmetry.md) — Z₂, U(1), and translation reduction
- [Tensor Networks](tensor_networks.md) — MPS/DMRG beyond sparse ED limits
- [Rust Engine](rust_engine.md) — `build_sparse_xy_hamiltonian` benchmarks
