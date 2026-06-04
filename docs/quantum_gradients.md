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

## Gradient verification certificate

Small smooth objectives can now emit a reusable finite-difference agreement
certificate. This makes gradient correctness visible outside private test code:

```python
import numpy as np

from scpn_quantum_control.phase import verify_parameter_shift_gradient


def objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))


certificate = verify_parameter_shift_gradient(
    objective,
    np.array([0.2, -0.4]),
)

print(certificate.passed, certificate.max_abs_error)
```

For Kuramoto-XY VQE objects, use `verify_vqe_parameter_shift_gradient(vqe,
params)`. The certificate records the analytic parameter-shift gradient, the
central finite-difference gradient, absolute and relative error maxima, pass/fail
tolerances, and objective-evaluation accounting. Finite differences are used
only as an independent diagnostic for small smooth objectives; they are not
advertised as a scalable hardware-gradient method.

Second-order curvature evidence is available through
`parameter_shift_hessian(...)`, `verify_parameter_shift_hessian(...)`, and
`verify_vqe_parameter_shift_hessian(...)` for the same standard
shift-compatible objective class. Diagonal entries use the sinusoidal
second-derivative shift identity, mixed entries compose first-order shifts, and
the verification certificate compares the result against central finite
differences:

```python
from scpn_quantum_control.phase import verify_parameter_shift_hessian

certificate = verify_parameter_shift_hessian(
    objective,
    np.array([0.2, -0.4]),
)

print(certificate.passed, certificate.max_abs_error)
```

This is a local curvature diagnostic for small supported objectives. It is not a
claim of universal quantum Fisher information, arbitrary-circuit adjoint
curvature, or hardware-efficient Hessian estimation.

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

## Provider callback execution

`execute_provider_parameter_shift_gradient(...)` is the first provider-safe
execution contract for parameter-shift gradients. It does not submit hardware
jobs by itself. Instead, callers provide a strict expectation-sampling callback
that can be backed by a local simulator, a managed provider adapter, or a dry-run
fixture:

```python
import numpy as np

from scpn_quantum_control.phase import (
    ProviderExpectationSample,
    execute_provider_parameter_shift_gradient,
)


def sampler(params: np.ndarray, shots: int | None) -> ProviderExpectationSample:
    value = float(np.cos(params[0]))
    return ProviderExpectationSample(value=value, variance=0.04, shots=shots)


result = execute_provider_parameter_shift_gradient(
    sampler,
    np.array([0.4]),
    backend="qasm_simulator",
    shots=4096,
)

print(result.gradient, result.standard_error, result.total_shots)
```

The result records backend plan provenance, every plus/minus shifted parameter
vector, sample values, sample variances, shot counts, propagated standard
errors, confidence radii, and a claim boundary. Hardware aliases still fail
closed unless an explicit hardware policy enables them through the backend
planner.

## Qiskit shifted-circuit generation

For Qiskit-native circuits, the phase namespace can generate fully bound
plus/minus shifted circuits and execute a local Statevector parameter-shift
gradient:

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.phase import execute_qiskit_statevector_parameter_shift

theta = Parameter("theta")
circuit = QuantumCircuit(1)
circuit.ry(theta, 0)
observable = SparsePauliOp.from_list([("Z", 1.0)])

result = execute_qiskit_statevector_parameter_shift(
    circuit,
    observable,
    (theta,),
    np.array([0.4]),
)

print(result.value, result.gradient)
```

This closes the automatic shifted-circuit generation gap for local Qiskit
Statevector checks. It is not hardware execution or finite-shot provider
submission; use the provider callback contract above when the expectation
source is an adapter or provider runtime.

For finite-shot planning and uncertainty accounting without submitting provider
jobs, use `execute_qiskit_finite_shot_parameter_shift(...)`. It binds the same
Qiskit plus/minus shifted circuits, evaluates expectations and observable
variance with a local Statevector surrogate, and routes the samples through
`execute_provider_parameter_shift_gradient(...)`:

```python
result = execute_qiskit_finite_shot_parameter_shift(
    circuit,
    observable,
    (theta,),
    np.array([0.4]),
    shots=4096,
)

print(result.gradient, result.standard_error, result.total_shots)
```

This is useful for provider-contract tests, notebooks, and dry-run cost
modelling because it produces the same serialisable shifted-sample records,
standard errors, confidence radii, and shot totals as the provider callback
route. It is still not hardware job submission; hardware aliases continue to
fail closed until an explicit backend policy enables them.

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

## Convergence evidence

`vqe_with_param_shift` now returns auditable convergence metadata in addition to
the final parameters. Use `validate_param_shift_convergence` to turn a training
run into a machine-checkable certificate:

```python
import numpy as np

from scpn_quantum_control.phase import (
    validate_param_shift_convergence,
    vqe_with_param_shift,
)


def objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]) + np.sin(params[1]))


result = vqe_with_param_shift(
    objective,
    n_params=2,
    initial_params=np.array([2.7, -0.4]),
    learning_rate=0.35,
    steps=28,
)
diagnostics = validate_param_shift_convergence(
    result,
    gradient_tolerance=0.08,
)

print(diagnostics.best_energy, diagnostics.parameter_shift_evaluations)
```

The diagnostics report monotone accepted energy history, accepted and rejected
line-search steps, backtracking counts, final gradient norm, exact-energy gap
when a Kuramoto-XY Hamiltonian reference is available, and optional pass/fail
flags for requested gap or gradient tolerances.

## Optional JAX bridge

`phase.jax_bridge` provides a bounded JAX-facing adapter for supported
parameter-shift calls:

```python
import numpy as np

from scpn_quantum_control.phase import jax_parameter_shift_value_and_grad


def objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


result = jax_parameter_shift_value_and_grad(
    objective,
    np.array([0.4]),
    jit=True,
)
print(result.gradient, result.jitted, result.host_callback)
```

The bridge is optional and fail-closed. It imports JAX only when called. JIT
mode uses `jax.pure_callback` around host-side parameter-shift evaluation and
therefore does not claim native JAX differentiation of a quantum kernel.

For parity checks against a caller-owned JAX objective, use
`check_jax_parameter_shift_agreement(...)` with a JAX-derived gradient callable:

```python
from scpn_quantum_control.phase import check_jax_parameter_shift_agreement

agreement = check_jax_parameter_shift_agreement(
    objective,
    jax.grad(jax_objective),
    np.array([0.4]),
)
print(agreement.passed, agreement.max_abs_error)
```

This produces a pass/fail agreement certificate between SCPN's manual
parameter-shift gradient and the supplied JAX gradient. It is still a bounded
interop verifier, not automatic native JAX compilation of arbitrary quantum
kernels.

## Optional PennyLane agreement check

For PennyLane/QNode parity work, use
`check_pennylane_parameter_shift_agreement` with a caller-supplied
PennyLane-derived gradient callable:

```python
import numpy as np

from scpn_quantum_control.phase import check_pennylane_parameter_shift_agreement


def scpn_objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


def pennylane_gradient(params: np.ndarray) -> np.ndarray:
    # Replace with qml.grad(qnode)(params) in a real PennyLane round trip.
    return np.array([-np.sin(params[0])], dtype=float)


agreement = check_pennylane_parameter_shift_agreement(
    scpn_objective,
    pennylane_gradient,
    np.array([0.4]),
)
print(agreement.passed, agreement.max_abs_error)
```

This is an agreement verifier, not an automatic PennyLane QNode generator. It
fails closed when PennyLane is not importable and reports explicit gradient
error metrics when the external gradient disagrees.

For full adapter smoke tests, `check_pennylane_qnode_round_trip(...)` compares
both value and gradient parity:

```python
from scpn_quantum_control.phase import check_pennylane_qnode_round_trip

round_trip = check_pennylane_qnode_round_trip(
    scpn_objective,
    pennylane_qnode,
    qml.grad(pennylane_qnode),
    np.array([0.4]),
)
print(round_trip.passed, round_trip.value_abs_error)
```

This still stays fail-closed and caller-supplied: SCPN does not claim automatic
translation of every internal ansatz into a PennyLane QNode.

## Optional PyTorch and TensorFlow tensor bridges

For ML pipelines that need framework tensors, the phase namespace exposes
host-boundary PyTorch and TensorFlow adapters:

```python
import numpy as np

from scpn_quantum_control.phase import (
    tensorflow_parameter_shift_value_and_grad,
    torch_parameter_shift_value_and_grad,
)


def objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


torch_result = torch_parameter_shift_value_and_grad(objective, np.array([0.4]))
tf_result = tensorflow_parameter_shift_value_and_grad(objective, np.array([0.4]))

print(torch_result.torch_gradient, torch_result.host_boundary)
print(tf_result.tensorflow_gradient, tf_result.host_boundary)
```

Both adapters import the optional framework only when called, run SCPN's
deterministic parameter-shift rule on the host, and return NumPy plus framework
tensor payloads. They are not native autograd-through-simulator kernels.

## Verification requirements

Before a new quantum-gradient path is promoted, it needs visible evidence:

| Evidence | Purpose |
|---|---|
| Analytic small-circuit references | Proves exact formula on cases with closed-form expectations. |
| Finite-difference checks | Detects sign, scale, parameter-index, and broadcasting mistakes. |
| Convergence tests | Shows that gradients improve optimisation, not only local derivatives. The parameter-shift route includes explicit convergence certificates. |
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
