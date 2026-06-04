# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable API

# Differentiable API

This page maps the public differentiable-programming namespace and the related quantum-gradient entry points. It is an API guide, not a proof that every exported symbol is production-ready for every backend. Always pair an API call with the support matrix and tests for the target primitive, backend, shape, dtype, and transform.

## Public namespaces

| Namespace | Role |
|---|---|
| `scpn_quantum_control.differentiable` | AD data structures, primitive registry contracts, optimisation helpers, program-AD metadata, and support reports. |
| `scpn_quantum_control.phase.param_shift` | Parameter-shift gradient helper and gradient-descent VQE example. |
| `scpn_quantum_control.phase.gradient_backend` | Backend gradient capability declarations, fail-closed planner, shot policy, and hardware-safe defaults. |
| `scpn_quantum_control.phase.gradient_tape` | Context-managed recording of supported deterministic and finite-shot quantum-gradient evaluations. |
| `scpn_quantum_control.compiler.mlir` | Compiler/program AD lowering, native executable kernel helpers, and support-profile reports. |

## Common objects

| Object family | Examples | Use |
|---|---|---|
| Primitive identity and rules | `PrimitiveIdentity`, `PrimitiveContract`, `CustomDerivativeRule`, `CustomDerivativeRegistry` | Bind derivative, batching, lowering, shape, dtype, and nondifferentiability rules to supported primitives. |
| Forward and reverse AD results | `GradientResult`, `JacobianResult`, `HessianResult`, `JVPResult`, `HVPResult`, `ProgramADAdjointResult` | Return structured derivative outputs and diagnostics. |
| Optimisation helpers | `DifferentiableOptimizer`, `NaturalGradientOptimizer`, `LevenbergMarquardtOptimizer` | Drive supported differentiable objectives. |
| Compiler-backed kernels | `compile_*_ad_to_native_llvm_jit`, `compile_whole_program_ad_trace_to_native_llvm_jit` | Execute bounded native AD kernels where support reports allow it. |
| Backend and shot planning | `QuantumGradientPlan`, `QuantumGradientBackendCapability`, `ShotAllocationResult`, support-profile records | Select supported local gradient methods, propagate finite-shot uncertainty, and fail closed for unsafe hardware routes. |

## Minimal parameter-shift call

```python
import numpy as np

from scpn_quantum_control.phase.param_shift import parameter_shift_gradient


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


grad = parameter_shift_gradient(cost, np.array([0.4], dtype=float))
```

## Minimal Kuramoto-XY VQE gradient call

```python
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import PhaseVQE

K = build_knm_paper27(L=2)
omega = OMEGA_N_16[:2]

vqe = PhaseVQE(K, omega, ansatz_reps=1)
result = vqe.solve(maxiter=40, seed=0, gradient_method="parameter_shift")
print(result["gradient_method"], result["n_grad_evals"])
```

The solver switches derivative-free defaults to a gradient-aware local
optimiser for this mode and returns gradient evaluation counts plus the final
gradient norm.

## Minimal backend gradient plan

```python
from scpn_quantum_control.phase import plan_quantum_gradient_backend

plan = plan_quantum_gradient_backend("statevector", n_params=4)
assert plan.method == "parameter_shift"
```

For finite-shot simulator diagnostics:

```python
plan = plan_quantum_gradient_backend("qasm_simulator", n_params=4, shots=4096)
assert plan.method == "stochastic_parameter_shift"
```

Hardware backends intentionally return an unsupported plan by default. That is a
safety boundary, not a missing exception.

## Minimal gradient tape

```python
import numpy as np

from scpn_quantum_control.phase import gradient_tape


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


with gradient_tape(backend="statevector") as tape:
    record = tape.record_parameter_shift("one_angle", cost, np.array([0.4]))

print(record.gradient, record.plan.method)
```

The tape records only supported phase-gradient evaluations. Unsupported
hardware routes fail closed through the same backend planner.

## Minimal custom primitive route

```python
from scpn_quantum_control import CustomDerivativeRule

rule = CustomDerivativeRule(
    name="square",
    value=lambda values: values[0] ** 2,
    derivative=lambda values, tangent: 2.0 * values[0] * tangent[0],
)
```

Production use should add primitive identity, shape, dtype, lowering, batching, nondifferentiability, and fail-closed tests before the primitive is advertised as supported.

## API contract checklist

Every new differentiable API must document:

- input shapes and dtype rules;
- scalar, vector, matrix, batch, and backend support;
- exact versus approximate derivative semantics;
- unsupported gates, transforms, backends, and control flow;
- finite-shot variance or numerical tolerance where relevant;
- reproducibility metadata;
- benchmark or convergence evidence before promotion.
