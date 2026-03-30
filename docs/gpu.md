# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — GPU Batch VQE Documentation

# GPU Batch VQE

`scpn_quantum_control.phase.gpu_batch_vqe`

Parallel evaluation of multiple VQE parameter sets using GPU (PyTorch)
or CPU (numpy). Useful for landscape scanning, hyperparameter search,
or initialisation strategies.

**Caveat:** This evaluates a simple Ry-layer ansatz for demonstration.
For production VQE with hardware-efficient ansätze, use Qiskit's `VQE`
class or PennyLane.

**Requires:** `pip install torch` for GPU acceleration.

---

## Theory

### Batch Evaluation

A VQE with $P$ parameters needs one forward pass (wavefunction → energy)
per parameter set. With $B$ parameter sets:

| Method | Cost |
|--------|------|
| CPU sequential | $B \times T_\text{single}$ |
| GPU batched | $T_\text{single} + \text{overhead}$ (amortised across $B$) |

The GPU advantage grows with batch size. For $B \geq 100$ and $n \geq 6$,
GPU batching typically provides 5–20× speedup over sequential CPU.

### Ry-Layer Ansatz

The built-in ansatz applies $R_y(\theta_i)$ to each qubit:

$$|\psi(\theta)\rangle = \prod_i R_y(\theta_i) |0\rangle^{\otimes n}$$

This is a minimal ansatz — sufficient for landscape scanning but not
for accurate ground state preparation. The `ansatz_fn` parameter allows
custom ansätze.

---

## API Reference

```python
from scpn_quantum_control.phase.gpu_batch_vqe import (
    batch_energy_numpy,
    batch_energy_torch,
    batch_vqe_scan,
)
```

### `batch_energy_numpy`

```python
energies = batch_energy_numpy(
    H: np.ndarray,                    # (dim, dim) Hamiltonian
    param_sets: np.ndarray,           # (batch, n_params) parameter vectors
    ansatz_fn: Callable,              # params → statevector (dim,)
) -> np.ndarray  # (batch,) energies
```

CPU baseline. Evaluates each parameter set sequentially.

### `batch_energy_torch`

```python
energies = batch_energy_torch(
    H: np.ndarray,
    param_sets: np.ndarray,
    ansatz_fn: Callable,              # torch.Tensor → torch.Tensor
    device: str = "cuda",             # "cuda" or "cpu"
) -> np.ndarray  # (batch,)
```

GPU-accelerated batch evaluation via PyTorch.

### `batch_vqe_scan`

```python
result = batch_vqe_scan(
    K: np.ndarray,                    # (n, n) coupling matrix
    omega: np.ndarray,                # (n,) frequencies
    n_samples: int = 100,             # parameter sets to evaluate
    n_params: int | None = None,      # parameters per set (default: n)
    seed: int = 42,
    use_gpu: bool = False,            # use PyTorch GPU if available
) -> dict
```

**Returns:**

```python
{
    "energies": np.ndarray,       # (n_samples,) all energies
    "params": np.ndarray,         # (n_samples, n_params) all parameter sets
    "best_energy": float,          # minimum energy found
    "best_params": np.ndarray,    # parameters that achieved best energy
}
```

---

## Tutorial

### Random Parameter Scan

```python
import numpy as np
from scpn_quantum_control.phase.gpu_batch_vqe import batch_vqe_scan

n = 6
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

result = batch_vqe_scan(K, omega, n_samples=200, seed=42)
print(f"Best energy: {result['best_energy']:.6f}")
print(f"Best params: {result['best_params']}")

# Compare with exact
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
H = knm_to_dense_matrix(K, omega)
E_exact = np.linalg.eigvalsh(H)[0]
print(f"Exact ground: {E_exact:.6f}")
print(f"Gap: {result['best_energy'] - E_exact:.4f}")
```

### GPU Acceleration

```python
import torch

if torch.cuda.is_available():
    result_gpu = batch_vqe_scan(K, omega, n_samples=1000,
                                 use_gpu=True, seed=42)
    print(f"GPU best energy: {result_gpu['best_energy']:.6f}")
else:
    print("No CUDA GPU available — using CPU")
```

### Custom Ansatz

```python
from scpn_quantum_control.phase.gpu_batch_vqe import batch_energy_numpy
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

H = knm_to_dense_matrix(K, omega)
dim = 2**n

def custom_ansatz(params):
    """Two-layer Ry ansatz."""
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    # Layer 1
    for i in range(n):
        c, s = np.cos(params[i]/2), np.sin(params[i]/2)
        new_psi = np.zeros_like(psi)
        for k in range(dim):
            bit = (k >> i) & 1
            k_flip = k ^ (1 << i)
            if bit == 0:
                new_psi[k] += c * psi[k]
                new_psi[k_flip] += s * psi[k]
            else:
                new_psi[k] += c * psi[k]
                new_psi[k_flip] -= s * psi[k]
        psi = new_psi
    # Layer 2 (with params[n:2n])
    for i in range(n):
        c, s = np.cos(params[n+i]/2), np.sin(params[n+i]/2)
        new_psi = np.zeros_like(psi)
        for k in range(dim):
            bit = (k >> i) & 1
            k_flip = k ^ (1 << i)
            if bit == 0:
                new_psi[k] += c * psi[k]
                new_psi[k_flip] += s * psi[k]
            else:
                new_psi[k] += c * psi[k]
                new_psi[k_flip] -= s * psi[k]
        psi = new_psi
    return psi

param_sets = np.random.randn(50, 2*n)
energies = batch_energy_numpy(H, param_sets, custom_ansatz)
print(f"Best of 50: {energies.min():.6f}")
```

---

## Comparison

| Feature | This module | TorchQuantum | PennyLane |
|---------|-------------|--------------|-----------|
| GPU batch | Yes (PyTorch) | Yes | Yes |
| Ansatz | Simple Ry / custom | Hardware-efficient | Wide variety |
| Auto-diff | No | Yes | Yes |
| Optimiser | Scan only | Full VQE | Full VQE |
| Backend | numpy / PyTorch | PyTorch | Multiple |

This module is a landscape scanning tool. For full VQE optimisation
with gradient descent, use `param_shift.py` or PennyLane.

---

## References

1. Wang, H. *et al.* "QuantumNAS: Noise-adaptive search for robust
   quantum circuits." *HPCA* (2022). (TorchQuantum)

---

## See Also

- [Variational Methods](variational.md) — parameter-shift gradient VQE
- [Neural Quantum States](neural_quantum_states.md) — RBM variational ansatz
- [Backends & Dispatch](backends.md) — numpy/JAX/torch switching
