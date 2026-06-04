# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quantum Gradients

# Quantum Gradients

Quantum gradients are the first differentiable-programming surface that most quantum-ML users look for. The current public route starts with parameter-shift gradients and expands toward backend-aware gradient planning, stochastic finite-shot gradients, adjoint simulator gradients, and framework adapters.

## Parameter-shift rule

For a Pauli-rotation expectation objective with generator spectrum compatible with the standard shift rule, the derivative is

$$
\frac{\partial C}{\partial \theta_k} = \frac{1}{2}\left[C(\theta_k + \pi/2) - C(\theta_k - \pi/2)\right].
$$

This rule avoids finite-difference truncation error for supported quantum expectation objectives. It still requires two objective evaluations per trainable parameter.

## Current API

```python
import numpy as np

from scpn_quantum_control.phase.param_shift import parameter_shift_gradient


def objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))


params = np.array([0.2, -0.4], dtype=float)
grad = parameter_shift_gradient(objective, params)
```

For VQE-style examples, see [Variational Methods](variational.md). For the wider differentiable namespace, see [Differentiable API](differentiable_api.md).

## Kuramoto-XY VQE route

`PhaseVQE` exposes a direct parameter-shift path for its K_nm-informed
`ry/rz/cz` ansatz:

```python
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import PhaseVQE

K = build_knm_paper27(L=2)
omega = OMEGA_N_16[:2]

vqe = PhaseVQE(K, omega, ansatz_reps=1)
result = vqe.solve(maxiter=40, seed=0, gradient_method="parameter_shift")
print(result["ground_energy"], result["gradient_norm"])
```

The implementation is local-simulator backed. Hardware gradients remain
fail-closed until backend policy, shot allocation, and uncertainty reporting are
registered for the target provider.

## Backend gradient planner

The phase namespace includes a fail-closed planner for quantum-gradient method
selection:

```python
from scpn_quantum_control.phase import plan_quantum_gradient_backend

plan = plan_quantum_gradient_backend("qasm_simulator", n_params=4, shots=4096)
print(plan.method, plan.evaluations, plan.shots)
```

Current planner behaviour:

| Backend family | Default method | Status |
|---|---|---|
| `statevector_simulator` | `parameter_shift` | Supported for deterministic local expectations. |
| `finite_shot_simulator` | `stochastic_parameter_shift` | Supported with explicit shots and uncertainty metadata. |
| `hardware_qpu` | `unsupported` | Fails closed unless a later hardware policy explicitly enables execution. |
| Unknown backend | `unsupported` | Fails closed and suggests local simulator alternatives. |

Finite-shot uncertainty can be propagated from plus/minus expectation variances:

```python
import numpy as np

from scpn_quantum_control.phase import parameter_shift_gradient_with_uncertainty

result = parameter_shift_gradient_with_uncertainty(
    plus_values=np.array([1.2, -0.3]),
    minus_values=np.array([0.8, -0.7]),
    plus_variances=np.array([0.04, 0.09]),
    minus_variances=np.array([0.04, 0.09]),
    shots=4096,
)
print(result.gradient, result.standard_error)
```

## Gradient tape MVP

For local simulator workflows, `gradient_tape` records deterministic and
finite-shot parameter-shift evaluations with backend-plan provenance:

```python
import numpy as np

from scpn_quantum_control.phase import gradient_tape


def objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


with gradient_tape(backend="statevector") as tape:
    record = tape.record_parameter_shift("single_rotation", objective, np.array([0.3]))

print(record.gradient, record.plan.method, record.evaluations)
```

The MVP is intentionally bounded. It is not a full programme-IR tape, does not
capture arbitrary Python side effects, and does not enable hardware gradients
without explicit policy approval.

## Verification requirements

Before a new quantum-gradient path is promoted, it needs visible evidence:

| Evidence | Purpose |
|---|---|
| Analytic small-circuit references | Proves exact formula on cases with closed-form expectations. |
| Finite-difference checks | Detects sign, scale, parameter-index, and broadcasting mistakes. |
| Convergence tests | Shows that gradients improve optimisation, not only local derivatives. |
| Cross-framework agreement | Compares against JAX, PennyLane, Qiskit, PyTorch, or TensorFlow where applicable. |
| Unsupported-operation tests | Confirms fail-closed behaviour for gates, backends, and observables without valid gradient rules. |

## Backend gradient methods

The backend planner classifies each execution path as one of:

- analytic parameter-shift;
- generalized parameter-shift;
- adjoint simulator gradient;
- stochastic finite-shot gradient;
- finite-difference diagnostic fallback;
- SPSA-style fallback;
- unsupported fail-closed mode.

Each gradient plan reports the selected method, backend, shots, seed, estimator
uncertainty policy, unsupported alternatives, and fail-closed reasons.

## Suitable and unsuitable scenarios

| Scenario | Status |
|---|---|
| Small Pauli-rotation expectation objective | Suitable for parameter-shift. |
| Gradient-trained Kuramoto-XY VQE | Current implementation route; convergence evidence must be attached. |
| Noisy finite-shot backend | Supported for uncertainty propagation when plus/minus variances and shots are supplied. |
| Hardware execution | Must remain disabled by default until a hardware-safe gradient policy exists. |
| Gate without registered generator spectrum | Unsupported; fail closed. |
| Dynamic circuit topology or parameter count | Unsupported unless the trace records stable parameter identity. |
| Roadmap adapters | JAX, PyTorch, TensorFlow, PennyLane, and Qiskit require parity tests before production claims. |
