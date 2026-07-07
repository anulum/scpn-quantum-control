# Variational Methods: Parameter-Shift Gradient Rule

`scpn_quantum_control.phase.param_shift`

Analytic gradient computation for variational quantum eigensolvers (VQE).
The parameter-shift rule computes exact gradients using only two circuit
evaluations per parameter — no finite-difference error, works on real
quantum expectation routes once the backend policy supports the required
shifted evaluations. The implemented route is local simulator-first.

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
`PhaseVQE.solve(gradient_method="parameter_shift")` or another solver with
registered gradient semantics for the target backend.

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
    objective_or_k: Callable[[np.ndarray], float] | np.ndarray,
    omega: np.ndarray | None = None,
    n_params: int | None = None,
    learning_rate: float = 0.1,
    steps: int = 100,
    seed: int | None = None,
) -> ParamShiftVQEResult
```

**Returns:**

```python
result.best_params       # best parameters found
result.best_energy       # best energy observed
result.energies          # accepted energy history
result.gradient_norms    # gradient norm history
result.to_dict()         # mapping form for legacy notebooks
```

---

## Tutorial

### Gradient of a Simple Function

```python
import numpy as np
from scpn_quantum_control.phase.param_shift import parameter_shift_gradient

# Sinusoidal expectation-style cost: C(theta) = cos(theta_0) + sin(theta_1)
def cost(params):
    return np.cos(params[0]) + np.sin(params[1])

params = np.array([1.0, 1.0], dtype=float)
grad = parameter_shift_gradient(cost, params)
print(f"Gradient: {grad}")
# Analytic: [-sin(theta_0), cos(theta_1)]
```

### VQE for the XY Hamiltonian

```python
import numpy as np
from scpn_quantum_control.phase import PhaseVQE

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)
vqe = PhaseVQE(K, omega, ansatz_reps=1)
result = vqe.solve(maxiter=80, seed=42, gradient_method="parameter_shift")
print(f"VQE energy: {result['ground_energy']:.6f}")
print(f"Gradient norm: {result['gradient_norm']:.2e}")
print(f"Exact energy: {result['exact_energy']:.6f}")
```

---

## Comparison

| Feature | This module | Qiskit `VQE` | PennyLane `qml.grad` |
|---------|-------------|-------------|----------------------|
| Gradient method | Parameter-shift | SPSA, COBYLA, etc. | Parameter-shift, backprop |
| Optimiser | Vanilla GD | SciPy, custom | Built-in (Adam, GD, etc.) |
| Hardware-compatible | Policy-gated | Yes | Yes |
| Ansatz | User-supplied `cost_fn` | Qiskit circuits | PennyLane circuits |
| Complexity | $2P$ evaluations | Method-dependent | $2P$ or $O(1)$ (backprop) |

This module is a building block. Use the local route for deterministic
gradient validation, then promote backends only after shot policy,
uncertainty reporting, and backend capability checks are registered.

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
