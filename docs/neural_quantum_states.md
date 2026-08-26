# Neural Quantum States

Two modules for variational ground state search using neural network
wavefunctions:

1. **RBM ansatz** (`phase/nqs_ansatz.py`) — Restricted Boltzmann Machine
   with numpy, exact mode for $n \leq 12$
2. **JAX-accelerated NQS** (`phase/jax_nqs.py`) — same RBM with JAX
   automatic differentiation and JIT compilation

**Caveat:** These are pedagogical/research implementations for the
Kuramoto-XY system. Larger or sampled studies require a dedicated framework
such as [NetKet](https://www.netket.org/) (Vicentini et al., 2022); this
repository does not certify external-framework parity or deployment readiness.

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
configurations exactly. The gradient is computed by central finite differences
(NumPy) or automatic differentiation (JAX).

### Why RBM?

Carleo & Troyer (Science, 2017) introduced an RBM variational representation
and reported strong results for selected 1D and 2D spin models. That result does
not guarantee accuracy or convergence for an arbitrary Hamiltonian, network
width, optimiser, or real-valued parameterisation. This repository therefore
compares only bounded small-system runs with exact diagonalisation and reports
the observed gap.

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
    n_samples: None = None,       # sampling unsupported in this path
    seed: int | None = None,
    max_dense_gib: float | None = None,
) -> dict
```

`max_dense_gib` gates the exact dense Hamiltonian and full-configuration
statevector workspace before allocation.

**Returns:**

```python
{
    "energy": float,                # final variational energy
    "energy_history": list[float],  # energy at each iteration
    "wavefunction": RBMWavefunction,  # trained RBM
    "n_params": int,                # total parameters
    "sampling_mode": "exact_enumeration",
    "n_samples_used": 2**n,
    "gradient_method": "central_finite_difference",
}
```

**Note:** For $n > 12$, raises `ValueError` (exact summation is $O(2^n)$).
Passing `n_samples` raises `ValueError`; this NumPy path is exact
enumeration with central finite-difference gradients, not sampled VMC.
Use NetKet for larger systems or MCMC sampling.

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

No performance claim is attached to this route. JIT compilation has an upfront
cost, backend and precision affect results, and the repository has no committed
isolated NumPy-versus-JAX benchmark artefact for this runner.

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
    max_dense_gib: float | None = None,
) -> dict
```

The JAX path uses the same exact-enumeration Hilbert-space boundary as the
NumPy RBM path; `max_dense_gib` gates the dense Hamiltonian and configuration
workspace before transferring arrays to JAX.

There is no automatic NumPy fallback. If JAX is unavailable, the public JAX
functions raise `ImportError`.

**Returns:**

```python
{
    "energy": float,
    "energy_history": list[float],
    "params": dict,          # JAX parameter dict
    "n_params": int,
}
```

### JAX NQS exact-reference baseline

Use the claim-bounded product for a validated `2 <= N <= 6` run with exact
diagonalisation, variational-gap diagnostics, JAX environment provenance, a
canonical evidence digest, and the default no-advantage certificate:

```python
from scpn_quantum_control.jax_nqs_baseline_product import (
    JAXNQSBaselineSpec,
    run_jax_nqs_baseline,
)

spec = JAXNQSBaselineSpec.from_arrays(
    K,
    omega,
    n_hidden=2 * len(omega),
    n_iterations=200,
    seed=42,
)
evidence = run_jax_nqs_baseline(spec)
print(evidence.comparison.relative_error)
print(evidence.evidence_sha256)
```

See [JAX NQS baseline product](jax_nqs_baseline.md) for the complete contract.

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
| Local API cap | 12 (exponential) | 12 (exponential) | Framework-dependent | Circuit-dependent |
| Accelerator route | No | Optional JAX backend; no speed claim | JAX-dependent | Framework-dependent |
| Hamiltonian | XY (built-in) | XY (built-in) | Any | Any |

---

## References

1. Carleo, G. & Troyer, M. "Solving the quantum many-body problem with
   artificial neural networks." *Science* **355**, 602 (2017).
2. Vicentini, F. *et al.* "NetKet 3: Machine learning toolbox for
   many-body quantum systems." *SoftwareX* **17**, 100933 (2022).
3. JAX authors. [Automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html),
   [JIT compilation](https://docs.jax.dev/en/latest/jit-compilation.html), and
   [default dtypes / X64](https://docs.jax.dev/en/latest/default_dtypes.html).

---

## See Also

- [Variational Methods](variational.md) — circuit parameter-shift methods
- [GPU Batch VQE](gpu.md) — parallel parameter scanning
- [Tensor Networks](tensor_networks.md) — MPS/DMRG alternative
