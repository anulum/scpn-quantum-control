# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Variational Methods Documentation

# Variational Methods: Parameter-Shift Gradient Rule

`scpn_quantum_control.phase.param_shift`

Analytic gradient computation for variational quantum eigensolvers (VQE).
The parameter-shift rule computes exact gradients using only two circuit
evaluations per parameter — no finite-difference error, works on real
quantum hardware.

---

## Theory

### The Parameter-Shift Rule

For a cost function $C(\theta) = \langle 0 | U^\dagger(\theta) H U(\theta) | 0\rangle$
where $U(\theta)$ contains gates of the form $e^{-i\theta G/2}$ with
$G^2 = I$ (standard Pauli rotations), the gradient is:

$$\frac{\partial C}{\partial \theta_k} = \frac{1}{2}\left[C(\theta_k + \pi/2) - C(\theta_k - \pi/2)\right]$$

This is exact — not an approximation. It requires $2P$ circuit evaluations
for $P$ parameters (compared to $2P$ for central finite differences, but
without truncation error).

### Gradient Descent VQE

The module provides a basic gradient-descent VQE loop:

1. Initialise random parameters $\theta^{(0)}$
2. At each iteration: compute $\nabla C$ via parameter-shift
3. Update: $\theta^{(t+1)} = \theta^{(t)} - \eta \nabla C$
4. Repeat until convergence or iteration limit

This is a pedagogical implementation. For production VQE, use
SciPy optimisers (L-BFGS-B, COBYLA) or Qiskit's `VQE` class.

---

## API Reference

```python
from scpn_quantum_control.phase.param_shift import (
    parameter_shift_gradient,
    vqe_with_param_shift,
)
```

### `parameter_shift_gradient`

```python
grad = parameter_shift_gradient(
    cost_fn: Callable[[np.ndarray], float],  # params → energy
    params: np.ndarray,                       # shape (n_params,)
    shift: float = np.pi / 2,                # shift amount
) -> np.ndarray  # shape (n_params,)
```

Computes the gradient vector. Each component requires two evaluations
of `cost_fn`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cost_fn` | `callable` | Maps parameter vector → scalar expectation value |
| `params` | `ndarray` | Current parameter values |
| `shift` | `float` | Shift amount. $\pi/2$ for standard Pauli rotation gates. |

### `vqe_with_param_shift`

```python
result = vqe_with_param_shift(
    cost_fn: Callable[[np.ndarray], float],
    n_params: int,
    learning_rate: float = 0.1,
    n_iterations: int = 100,
    seed: int | None = None,
) -> dict
```

**Returns:**

```python
{
    "optimal_params": np.ndarray,    # best parameters found
    "energy": float,                  # final energy
    "energy_history": list[float],   # energy at each iteration
    "grad_norms": list[float],       # gradient norm at each iteration
}
```

---

## Tutorial

### Gradient of a Simple Function

```python
import numpy as np
from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

# Quadratic cost: C(θ) = θ₀² + 2θ₁²
def cost(params):
    return params[0]**2 + 2 * params[1]**2

params = np.array([1.0, 1.0])
grad = parameter_shift_gradient(cost, params)
print(f"Gradient: {grad}")
# Analytic: [2.0, 4.0] — parameter-shift gives exact result for sinusoidal
# functions, approximate for polynomials
```

### VQE for the XY Hamiltonian

```python
import numpy as np
from scpn_quantum_control.phase.param_shift import vqe_with_param_shift
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)
H = knm_to_dense_matrix(K, omega)
dim = 2**n

def vqe_cost(params):
    """Simple Ry-layer ansatz."""
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    # Apply Ry rotations
    for i in range(n):
        c, s = np.cos(params[i]/2), np.sin(params[i]/2)
        # Single-qubit Ry on qubit i
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
    return np.real(psi.conj() @ H @ psi)

result = vqe_with_param_shift(vqe_cost, n_params=n,
                               learning_rate=0.1, n_iterations=200, seed=42)
print(f"VQE energy: {result['energy']:.6f}")
print(f"Final gradient norm: {result['grad_norms'][-1]:.2e}")

E_exact = np.linalg.eigvalsh(H)[0]
print(f"Exact energy: {E_exact:.6f}")
```

---

## Comparison

| Feature | This module | Qiskit `VQE` | PennyLane `qml.grad` |
|---------|-------------|-------------|----------------------|
| Gradient method | Parameter-shift | SPSA, COBYLA, etc. | Parameter-shift, backprop |
| Optimiser | Vanilla GD | SciPy, custom | Built-in (Adam, GD, etc.) |
| Hardware-compatible | Yes (2 evals/param) | Yes | Yes |
| Ansatz | User-supplied `cost_fn` | Qiskit circuits | PennyLane circuits |
| Complexity | $2P$ evaluations | Method-dependent | $2P$ or $O(1)$ (backprop) |

This module is a building block. For production VQE with adaptive ansätze
and advanced optimisers, use Qiskit or PennyLane.

---

## References

1. Mitarai, K. *et al.* "Quantum circuit learning." *PRA* **98**, 032309 (2018).
2. Schuld, M. *et al.* "Evaluating analytic gradients on quantum hardware."
   *PRA* **99**, 032331 (2019).

---

## See Also

- [Neural Quantum States](neural_quantum_states.md) — RBM variational ansatz
- [GPU Batch VQE](gpu.md) — parallel parameter scanning
- [XY Compiler](xy_compiler.md) — depth-optimised circuits for VQE
