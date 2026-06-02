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

## Verification requirements

Before a new quantum-gradient path is promoted, it needs visible evidence:

| Evidence | Purpose |
|---|---|
| Analytic small-circuit references | Proves exact formula on cases with closed-form expectations. |
| Finite-difference checks | Detects sign, scale, parameter-index, and broadcasting mistakes. |
| Convergence tests | Shows that gradients improve optimisation, not only local derivatives. |
| Cross-framework agreement | Compares against JAX, PennyLane, Qiskit, PyTorch, or TensorFlow where applicable. |
| Unsupported-operation tests | Confirms fail-closed behaviour for gates, backends, and observables without valid gradient rules. |

## Planned backend gradient planner

The backend planner should eventually classify each execution path as one of:

- analytic parameter-shift;
- generalized parameter-shift;
- adjoint simulator gradient;
- stochastic finite-shot gradient;
- finite-difference diagnostic fallback;
- SPSA-style fallback;
- unsupported fail-closed mode.

Each gradient result should report the selected method, backend, shots, seed, estimator uncertainty, unsupported alternatives, and transformation provenance.

## Suitable and unsuitable scenarios

| Scenario | Status |
|---|---|
| Small Pauli-rotation expectation objective | Suitable for parameter-shift. |
| Gradient-trained Kuramoto-XY VQE | Current implementation route; convergence evidence must be attached. |
| Noisy finite-shot backend | Requires variance estimates and explicit shot policy. |
| Hardware execution | Must remain disabled by default until a hardware-safe gradient policy exists. |
| Gate without registered generator spectrum | Unsupported; fail closed. |
| Dynamic circuit topology or parameter count | Unsupported unless the trace records stable parameter identity. |
| Roadmap adapters | JAX, PyTorch, TensorFlow, PennyLane, and Qiskit require parity tests before production claims. |
