# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Neural Quantum States Documentation

# Neural Quantum States

Two modules for variational ground state search using neural network
wavefunctions:

1. **RBM ansatz** (`phase/nqs_ansatz.py`) — Restricted Boltzmann Machine
   with numpy, exact mode for $n \leq 12$
2. **JAX-accelerated NQS** (`phase/jax_nqs.py`) — same RBM with JAX
   auto-differentiation, ~100× faster gradients

**Caveat:** These are pedagogical/research implementations for the
Kuramoto-XY system. For production NQS at scale, use
[NetKet](https://www.netket.org/) (Vicentini et al., 2022).

---

## Theory

### Restricted Boltzmann Machine Wavefunction

The RBM ansatz parameterises the wavefunction as:

$$\log \psi(\sigma) = \sum_i a_i \sigma_i + \sum_j \log\cosh\left(\sum_i W_{ji} \sigma_i + b_j\right)$$

where $\sigma \in \{+1, -1\}^n$ is a spin configuration, $a_i$ are visible
biases, $b_j$ are hidden biases, and $W_{ji}$ are weights connecting
$n_\text{visible}$ spins to $n_\text{hidden}$ hidden units.

Total parameters: $n + n_h + n \cdot n_h$ (typically $n_h = 2n$).

### Variational Monte Carlo (VMC)

The energy expectation is:

$$E = \frac{\sum_\sigma |\psi(\sigma)|^2 \langle\sigma|H|\sigma'\rangle \psi(\sigma')/\psi(\sigma)}{\sum_\sigma |\psi(\sigma)|^2}$$

For small systems ($n \leq 12$), we evaluate the sum over all $2^n$
configurations exactly. The gradient is computed via the parameter-shift
rule (numpy) or automatic differentiation (JAX).

### Why RBM?

Carleo & Troyer (Science, 2017) showed that RBMs can represent ground
states of 1D and 2D spin models with accuracy competitive with DMRG.
The universal approximation theorem guarantees that sufficiently wide
RBMs can represent any quantum state.

In practice, convergence depends on the problem. For the XY model with
moderate coupling, RBMs with $n_h = 2n$ typically converge within
100–500 iterations.

---

## Part 1: RBM Ansatz (NumPy)

`scpn_quantum_control.phase.nqs_ansatz`

### API Reference

#### `RBMWavefunction` Class

```python
from scpn_quantum_control.phase.nqs_ansatz import RBMWavefunction

rbm = RBMWavefunction(
    n_visible: int,              # number of spins
    n_hidden: int | None = None, # hidden units (default: 2 * n_visible)
    seed: int | None = None,     # RNG seed
)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `log_psi(sigma)` | `ndarray → complex` | $\log\psi(\sigma)$ | Log-amplitude for configuration $\sigma \in \{+1,-1\}^n$ |
| `psi(sigma)` | `ndarray → complex` | $\psi(\sigma)$ | Amplitude (exponentiated) |
| `all_amplitudes()` | `() → ndarray` | Shape $(2^n,)$ | All amplitudes (exact, for $n \leq 12$) |
| `n_params()` | `() → int` | Total parameter count | $n + n_h + n \cdot n_h$ |

#### `vmc_ground_state`

```python
from scpn_quantum_control.phase.nqs_ansatz import vmc_ground_state

result = vmc_ground_state(
    K: np.ndarray,                # (n, n) coupling matrix
    omega: np.ndarray,            # (n,) frequencies
    n_hidden: int | None = None,  # hidden units (default: 2n)
    learning_rate: float = 0.01,  # gradient descent step
    n_iterations: int = 200,      # optimisation steps
    n_samples: int = 500,         # ignored in exact mode
    seed: int | None = None,
) -> dict
```

**Returns:**

```python
{
    "energy": float,                # final variational energy
    "energy_history": list[float],  # energy at each iteration
    "wavefunction": RBMWavefunction,  # trained RBM
    "n_params": int,                # total parameters
}
```

**Note:** For $n > 12$, raises `ValueError` (exact summation is $O(2^n)$).
MCMC sampling is not implemented — use NetKet for larger systems.

### Example

```python
import numpy as np
from scpn_quantum_control.phase.nqs_ansatz import vmc_ground_state

n = 6
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

result = vmc_ground_state(K, omega, n_iterations=300, seed=42)
print(f"VMC energy: {result['energy']:.6f}")
print(f"Parameters: {result['n_params']}")

# Compare with exact diagonalisation
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
H = knm_to_dense_matrix(K, omega)
E_exact = np.linalg.eigvalsh(H)[0]
print(f"Exact energy: {E_exact:.6f}")
print(f"Relative error: {abs(result['energy'] - E_exact) / abs(E_exact):.2e}")
```

---

## Part 2: JAX-Accelerated NQS

`scpn_quantum_control.phase.jax_nqs`

### Why JAX?

The numpy VMC uses finite-difference gradients: $2 \times n_\text{params}$
function evaluations per iteration. JAX replaces this with automatic
differentiation via `jax.grad` — a single forward + backward pass.

Measured speedup: ~100× for gradient computation at $n=8$.

### API Reference

```python
from scpn_quantum_control.phase.jax_nqs import (
    is_jax_available,
    jax_rbm_energy,
    jax_vmc_ground_state,
)
```

#### `jax_rbm_energy`

```python
energy = jax_rbm_energy(
    params: dict[str, Any],  # {'a': visible, 'b': hidden, 'W': weights}
    H: jax.Array,            # dense Hamiltonian
    n: int,                  # number of qubits
) -> jax.Array
```

Differentiable energy expectation. Use with `jax.grad` for gradients.

#### `jax_vmc_ground_state`

```python
result = jax_vmc_ground_state(
    K: np.ndarray,
    omega: np.ndarray,
    n_hidden: int | None = None,
    learning_rate: float = 0.01,
    n_iterations: int = 200,
    seed: int = 42,
) -> dict
```

**Returns:**

```python
{
    "energy": float,
    "energy_history": list[float],
    "params": dict,          # JAX parameter dict
    "n_params": int,
}
```

### Example

```python
import numpy as np
from scpn_quantum_control.phase.jax_nqs import (
    jax_vmc_ground_state, is_jax_available
)

if not is_jax_available():
    raise ImportError("Install JAX: pip install jax jaxlib")

n = 8
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

result = jax_vmc_ground_state(K, omega, n_iterations=300, seed=42)
print(f"JAX VMC energy: {result['energy']:.6f}")
print(f"Parameters: {result['n_params']}")
```

---

## Tutorial: Comparing NQS Methods

```python
import numpy as np
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

n = 6
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# Exact reference
H = knm_to_dense_matrix(K, omega)
E_exact = np.linalg.eigvalsh(H)[0]

# NumPy RBM
from scpn_quantum_control.phase.nqs_ansatz import vmc_ground_state
result_np = vmc_ground_state(K, omega, n_iterations=300, seed=42)

# JAX RBM (if available)
from scpn_quantum_control.phase.jax_nqs import jax_vmc_ground_state, is_jax_available
if is_jax_available():
    result_jax = jax_vmc_ground_state(K, omega, n_iterations=300, seed=42)
    jax_energy = result_jax['energy']
else:
    jax_energy = float('nan')

print(f"Exact:     {E_exact:.6f}")
print(f"NumPy VMC: {result_np['energy']:.6f} (error: {abs(result_np['energy'] - E_exact):.2e})")
print(f"JAX VMC:   {jax_energy:.6f}")
```

---

## Comparison

| Feature | This module (numpy) | This module (JAX) | NetKet | PennyLane |
|---------|--------------------|--------------------|--------|-----------|
| RBM ansatz | Yes | Yes | Yes | No |
| Other ansätze | No | No | Many (RNN, GCN, etc.) | VQE circuits |
| Gradient method | Finite difference | Auto-diff (JAX) | Auto-diff (JAX) | Auto-diff |
| MCMC sampling | No ($n \leq 12$) | No ($n \leq 12$) | Yes | N/A |
| Max system size | 12 | 12 | 1000+ | Circuit-limited |
| GPU | No | Yes (JAX) | Yes (JAX) | Yes |
| Hamiltonian | XY (built-in) | XY (built-in) | Any | Any |

---

## References

1. Carleo, G. & Troyer, M. "Solving the quantum many-body problem with
   artificial neural networks." *Science* **355**, 602 (2017).
2. Vicentini, F. *et al.* "NetKet 3: Machine learning toolbox for
   many-body quantum systems." *SoftwareX* **17**, 100933 (2022).
3. Bradbury, J. *et al.* "JAX: composable transformations of Python+NumPy
   programs." (2018). http://github.com/google/jax

---

## See Also

- [Variational Methods](variational.md) — parameter-shift gradient rule
- [GPU Batch VQE](gpu.md) — parallel parameter scanning
- [Tensor Networks](tensor_networks.md) — MPS/DMRG alternative
