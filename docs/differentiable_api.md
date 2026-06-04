# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable API

# Differentiable API

This page maps the public differentiable-programming namespace and the related quantum-gradient entry points. It is an API guide, not a proof that every exported symbol is production-ready for every backend. Always pair an API call with the support matrix and tests for the target primitive, backend, shape, dtype, and transform.

The API contract is deliberately fail-closed: a supported route should return a
structured plan, result, certificate, or diagnostic; an unsupported route should
return a blocked plan or raise a targeted error instead of silently substituting
finite differences or pretending that a hardware/provider gradient exists.

## Public namespaces

| Namespace | Role |
|---|---|
| `scpn_quantum_control.differentiable` | AD data structures, primitive registry contracts, optimisation helpers, program-AD metadata, and support reports. |
| `scpn_quantum_control.phase.param_shift` | Parameter-shift gradient helper and gradient-descent VQE example. |
| `scpn_quantum_control.phase.coupling_learning` | Differentiable coupling inference from observation models with convergence and finite-difference agreement certificates. |
| `scpn_quantum_control.phase.gradient_descent` | Generic parameter-shift gradient descent with line-search traces and convergence certificates. |
| `scpn_quantum_control.phase.natural_gradient` | Metric-aware parameter-shift descent with damped solves, metric validation, line-search traces, and convergence certificates. |
| `scpn_quantum_control.phase.optimizer_audit` | Multi-start optimizer comparison evidence for parameter-shift descent and natural-gradient descent. |
| `scpn_quantum_control.phase.objectives` | Composable differentiable phase-control objectives with term-wise gradients and fail-closed parameter-shift compatibility. |
| `scpn_quantum_control.phase.objective_audit` | Finite-difference agreement, compatibility-gate, and training evidence for composed objectives. |
| `scpn_quantum_control.phase.objective_planner` | Fail-closed execution planning for pure parameter-shift, hybrid term-gradient, hardware, and unsupported composed-objective routes. |
| `scpn_quantum_control.qsnn.training` | QSNN parameter-shift gradients, full-batch descent, and training convergence evidence. |
| `scpn_quantum_control.phase.gradient_backend` | Backend gradient capability declarations, fail-closed planner, shot policy, and hardware-safe defaults. |
| `scpn_quantum_control.phase.gradient_support_matrix` | Executable support planning for gates, observables, backends, transforms, and ML/provider adapters. |
| `scpn_quantum_control.phase.transform_nesting` | Fail-closed transform-nesting planner for local, tape, ML-adapter, vectorized, and hardware gradient routes. |
| `scpn_quantum_control.phase.provider_gradient_audit` | Executable provider-gradient readiness audit for deterministic, finite-shot, multi-frequency, hardware-blocked, unknown-backend, and malformed-sample routes. |
| `scpn_quantum_control.phase.gradient_tape` | Context-managed recording of supported deterministic and finite-shot quantum-gradient evaluations. |
| `scpn_quantum_control.phase.jax_bridge` | Optional JAX host-callback adapter for supported phase parameter-shift value-and-gradient calls. |
| `scpn_quantum_control.phase.pennylane_bridge` | Optional PennyLane gradient-agreement checker for caller-supplied PennyLane/QNode gradient functions. |
| `scpn_quantum_control.phase.torch_bridge` | Optional PyTorch tensor bridge for supported phase parameter-shift value-and-gradient calls. |
| `scpn_quantum_control.phase.tensorflow_bridge` | Optional TensorFlow tensor bridge for supported phase parameter-shift value-and-gradient calls. |
| `scpn_quantum_control.compiler.mlir` | Compiler/program AD lowering, native executable kernel helpers, and support-profile reports. |

## Common objects

| Object family | Examples | Use |
|---|---|---|
| Primitive identity and rules | `PrimitiveIdentity`, `PrimitiveContract`, `CustomDerivativeRule`, `CustomDerivativeRegistry` | Bind derivative, batching, lowering, shape, dtype, and nondifferentiability rules to supported primitives. |
| Forward and reverse AD results | `GradientResult`, `JacobianResult`, `HessianResult`, `JVPResult`, `HVPResult`, `ProgramADAdjointResult` | Return structured derivative outputs and diagnostics. |
| Optimisation helpers | `DifferentiableOptimizer`, `NaturalGradientOptimizer`, `LevenbergMarquardtOptimizer` | Drive supported differentiable objectives. |
| Compiler-backed kernels | `compile_*_ad_to_native_llvm_jit`, `compile_whole_program_ad_trace_to_native_llvm_jit` | Execute bounded native AD kernels where support reports allow it. |
| Backend and shot planning | `QuantumGradientPlan`, `QuantumGradientBackendCapability`, `ShotAllocationResult`, support-profile records | Select supported local gradient methods, propagate finite-shot uncertainty, and fail closed for unsafe hardware routes. |
| Gradient support matrix | `GradientSupportCapability`, `GradientSupportPlan`, `GradientSupportMatrixAuditResult`, `gradient_support_capability`, `list_gradient_support_capabilities`, `plan_gradient_support`, `assert_gradient_support`, `run_gradient_support_matrix_audit` | Decide whether a gate, observable, backend, transform, and adapter combination is supported before execution; blocked combinations carry reasons and alternatives. |
| Transform nesting | `GradientTransformNestingPlan`, `GradientTransformNestingAuditResult`, `plan_gradient_transform_nesting`, `assert_gradient_transform_nesting_supported`, `run_gradient_transform_nesting_audit` | Decide whether transform stacks such as `grad`, `value_and_grad`, `hessian`, `grad` of `grad`, tape, `vmap`, JVP/VJP, adapter bridges, or hardware routes are safe before execution. |
| Gradient audit evidence | `DifferentiableQuantumAuditReport`, `DifferentiableWorkflowAuditSuiteResult`, `FiniteShotGradientAuditResult`, `MLFrameworkGradientAuditSuiteResult`, `ParameterShiftAnalyticAgreement`, `PhaseGradientBenchmarkSuiteResult`, `ProviderGradientReadinessAuditResult`, `run_differentiable_workflow_audit_suite`, `run_finite_shot_gradient_uncertainty_audit`, `run_ml_framework_gradient_audit`, `run_known_phase_gradient_audit`, `run_parameter_shift_audit_suite`, `run_phase_gradient_benchmark_suite`, `run_provider_gradient_readiness_audit` | Bundle finite-difference agreement, finite-shot uncertainty containment, optional ML-framework parity, analytic-gradient agreement, convergence evidence, coupling-learning checks, provider-readiness checks, and multi-case phase-gradient conformance into reviewer-facing reports. |
| Gradient-training evidence | `ParameterShiftTrainingResult`, `ParameterShiftTrainingCertificate`, `ParameterShiftNaturalGradientResult`, `ParameterShiftNaturalGradientCertificate`, `OptimizerComparisonSuiteResult`, `OptimizerConvergenceRecord`, `ParamShiftVQEResult`, `ParamShiftConvergenceDiagnostics` | Certify accepted value descent, metric-aware descent, optimizer comparison evidence, line-search behaviour, exact-gap metadata, and parameter-shift evaluation counts. |
| Objective composition | `ComposedPhaseObjective`, `ObjectiveTerm`, `ObjectiveGradientEvaluation`, `ComposedObjectiveTrainingResult`, `ComposedObjectiveGradientAgreement`, `ComposedObjectiveAuditSuiteResult`, `ComposedObjectiveExecutionPlan`, `ComposedObjectivePlannerAuditResult`, `build_phase_control_objective`, `train_composed_phase_objective`, `verify_composed_objective_gradient`, `run_composed_objective_audit_suite`, `plan_composed_objective_execution`, `run_composed_objective_planner_audit` | Combine energy, fidelity, periodic regularization, symmetry, and smooth safety penalties without misclassifying analytic classical penalties as parameter-shift quantum terms. |
| Coupling-learning evidence | `CouplingLearningResult`, `CouplingGradientVerificationResult`, `learn_couplings_from_observations`, `verify_coupling_parameter_shift_gradient` | Learn symmetric oscillator couplings from parameter-shift-compatible observation models and independently check small smooth gradients against central finite differences. |
| QSNN training evidence | `QSNNTrainingRun`, `QSNNParameterShiftDescentRun` | Attach parameter-shift traces and certificates to quantum neural network training loops. |
| Optional JAX bridge | `PhaseJAXParameterShiftResult`, `jax_parameter_shift_value_and_grad`, `is_phase_jax_available` | Expose phase parameter-shift value-and-gradient calls to JAX workflows through an explicit host-callback boundary. |
| Optional PennyLane agreement | `PennyLaneGradientAgreementResult`, `check_pennylane_parameter_shift_agreement`, `is_phase_pennylane_available` | Compare SCPN parameter-shift gradients against a caller-supplied PennyLane gradient callable. |
| Optional PyTorch bridge | `PhaseTorchParameterShiftResult`, `torch_parameter_shift_value_and_grad`, `is_phase_torch_available` | Convert supported phase parameter-shift value-and-gradient outputs into PyTorch tensors while preserving NumPy evidence. |
| Optional TensorFlow bridge | `PhaseTensorFlowParameterShiftResult`, `tensorflow_parameter_shift_value_and_grad`, `is_phase_tensorflow_available` | Convert supported phase parameter-shift value-and-gradient outputs into TensorFlow tensors while preserving NumPy evidence. |

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

## Minimal convergence certificate

```python
import numpy as np

from scpn_quantum_control.phase import (
    parameter_shift_gradient_descent,
    parameter_shift_natural_gradient_descent,
    run_parameter_shift_optimizer_comparison,
    validate_param_shift_convergence,
    validate_natural_gradient_training,
    validate_parameter_shift_training,
    vqe_with_param_shift,
)


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]) + np.sin(params[1]))


run = vqe_with_param_shift(
    cost,
    n_params=2,
    initial_params=np.array([2.7, -0.4]),
    steps=28,
    learning_rate=0.35,
)
certificate = validate_param_shift_convergence(run, gradient_tolerance=0.08)
assert certificate.monotone_energy

generic_run = parameter_shift_gradient_descent(
    cost,
    np.array([2.7, -0.4]),
    max_steps=28,
    learning_rate=0.35,
)
generic_certificate = validate_parameter_shift_training(
    generic_run,
    gradient_tolerance=0.08,
)
assert generic_certificate.monotone_accepted_values

natural_run = parameter_shift_natural_gradient_descent(
    cost,
    np.array([2.7, -0.4]),
    metric_tensor=np.eye(2),
    max_steps=28,
    learning_rate=0.35,
)
natural_certificate = validate_natural_gradient_training(
    natural_run,
    gradient_tolerance=0.08,
)
assert natural_certificate.monotone_accepted_values

comparison = run_parameter_shift_optimizer_comparison(max_steps=8)
assert comparison.passed
assert comparison.natural_gradient_not_worse_count == comparison.start_count
```

Natural-gradient training accepts an explicit metric tensor or callable metric
and validates shape, symmetry, finite values, conditioning, and descent
direction before applying a damped solve. The identity metric path is recorded
as a preconditioner baseline; it is not promoted as a quantum Fisher extraction
or arbitrary-circuit natural-gradient method.

The optimizer comparison audit is a local functional evidence tool. It runs
multiple starts through ordinary parameter-shift descent and natural-gradient
descent, records certificates for every route, and checks whether the metric
route is no worse than the baseline under the declared tolerance. It is not a
hardware benchmark, throughput result, or proof of global optimality.

## Minimal gradient support matrix

```python
from scpn_quantum_control.phase import (
    assert_gradient_support,
    plan_gradient_support,
    run_gradient_support_matrix_audit,
)


plan = plan_gradient_support(
    gate="ry",
    observable="pauli_expectation",
    backend="statevector",
    transform="grad",
    adapter="native",
    n_params=2,
)
assert_gradient_support(plan)

audit = run_gradient_support_matrix_audit()
assert audit.passed
```

The support matrix answers a broader question than the backend planner: it
combines gate, observable, backend, transform, and adapter policy into one
fail-closed decision. Built-in supported plans include local parameter-shift,
finite-shot parameter-shift with variance requirements, JAX host-callback
value-and-gradient, and Qiskit shifted-circuit gradients. Built-in blocked
plans include unregistered gates, unregistered observables, hardware without
policy approval, unsupported transform algebra, and finite-shot Hessian
requests.

## Minimal transform-nesting plan

```python
from scpn_quantum_control.phase import (
    assert_gradient_transform_nesting_supported,
    plan_gradient_transform_nesting,
    run_gradient_transform_nesting_audit,
)


plan = plan_gradient_transform_nesting(("grad", "grad"), n_params=2)
assert_gradient_transform_nesting_supported(plan)

audit = run_gradient_transform_nesting_audit()
assert audit.passed
```

Supported nesting is intentionally bounded. Local `grad`, `value_and_grad`,
deterministic local `hessian`, `grad` of `grad`, single-adapter
`value_and_grad`, and phase `gradient_tape` routes are planned as supported.
`vmap`, JVP/VJP, `jacfwd`, `jacrev`, nested ML/provider adapters, finite-shot
curvature, nested tape transforms, and hardware nesting fail closed with
reasons and alternatives.

## Minimal composed objective

```python
import numpy as np

from scpn_quantum_control.phase import (
    build_phase_control_objective,
    plan_composed_objective_execution,
    run_composed_objective_audit_suite,
    run_composed_objective_planner_audit,
    train_composed_phase_objective,
    validate_composed_objective_training,
)


objective = build_phase_control_objective(
    2,
    energy_weight=1.0,
    fidelity_target=np.zeros(2),
    fidelity_weight=0.2,
    safety_bounds=(-1.0, 1.0),
    safety_weight=0.1,
)

evaluation = objective.evaluate(np.array([0.8, -0.7]))
run = train_composed_phase_objective(objective, np.array([0.8, -0.7]))
certificate = validate_composed_objective_training(run, min_decrease=0.1)

print(evaluation.value, certificate.monotone_accepted_values)

audit = run_composed_objective_audit_suite()
assert audit.passed
assert audit.hybrid_parameter_shift_gate_failed

plan = plan_composed_objective_execution(objective)
planner_audit = run_composed_objective_planner_audit()
assert plan.recommended_entrypoint == "train_composed_phase_objective"
assert planner_audit.passed
```

The objective reports which terms are parameter-shift compatible. Periodic
energy, fidelity, regularization, and symmetry terms are compatible; the smooth
box-safety penalty is analytic-only and makes
`require_parameter_shift_compatible()` fail closed.
`run_composed_objective_audit_suite()` makes that boundary executable: it
checks finite-difference gradient agreement for pure and hybrid objectives,
confirms the pure objective passes the parameter-shift gate, confirms the
hybrid safety objective fails that gate, and records local training
certificates for both.
`plan_composed_objective_execution()` is the fail-closed routing layer: pure
periodic objectives route to local parameter-shift training, hybrid objectives
route to exact term-gradient training, and hardware, unknown backends, or forced
pure parameter-shift on analytic safety terms return unsupported plans.

## Minimal provider-gradient readiness audit

```python
from scpn_quantum_control.phase import run_provider_gradient_readiness_audit


audit = run_provider_gradient_readiness_audit()
assert audit.passed
print(len(audit.supported_records), len(audit.blocked_records))
```

The audit executes three supported local callback routes and three deliberate
failure routes. Supported records cover deterministic statevector
parameter-shift, finite-shot parameter-shift with variance metadata, and
multi-frequency finite-shot parameter-shift. Blocked records cover hardware
without explicit policy approval, unknown provider families, and finite-shot
callbacks that omit sample variance. This is provider-readiness evidence, not a
live hardware-gradient claim.

## Minimal QSNN descent certificate

```python
import numpy as np

from scpn_quantum_control.qsnn import QuantumDenseLayer, QSNNTrainer

layer = QuantumDenseLayer(1, 1, seed=42)
trainer = QSNNTrainer(layer, lr=0.4)
run = trainer.train_with_parameter_shift_descent(
    np.array([[1.0]]),
    np.array([[0.0]]),
    max_steps=40,
    min_loss_decrease=1e-4,
)

assert run.certificate.monotone_accepted_values
```

This route is full-batch local-simulator training. Hardware backends remain
disabled by default through the same fail-closed backend planner used by phase
gradients.

## Reviewer-facing gradient audit report

```python
import numpy as np

from scpn_quantum_control.phase import run_known_phase_gradient_audit


report = run_known_phase_gradient_audit(np.array([0.8, -0.5, 0.3]))

print(report.passed)
print(report.finite_difference.max_abs_error)
print(report.analytic.max_abs_error)
print(report.training_certificate.to_dict())
```

The audit report is intended for visible correctness evidence. It combines
parameter-shift versus finite-difference agreement, parameter-shift versus an
analytic gradient, and a deterministic gradient-descent convergence
certificate. The built-in benchmark is a smooth phase-rotation objective,
`mean(1 - cos(theta_i))`, with exact gradient `sin(theta_i) / n`.
Discontinuous objectives, stochastic hardware shots, arbitrary regressors, and
undeclared generator spectra are explicitly outside this report boundary.

For a wider built-in conformance pass, use the benchmark suite:

```python
from scpn_quantum_control.phase import run_phase_gradient_benchmark_suite


suite = run_phase_gradient_benchmark_suite()

print(suite.passed)
print(suite.benchmark_names)
print(suite.worst_gradient_error)
print(suite.unsupported_scenarios)
```

The suite currently covers single-frequency phase rotations, multi-frequency
phase rotations using a declared shift rule, and a coupled pair phase loss.
This is the recommended CI and paper-table entry point when users need visible
evidence that the differentiable-programming surface handles more than one
toy gradient.

For the full supported workflow audit:

```python
from scpn_quantum_control.phase import run_differentiable_workflow_audit_suite


workflow = run_differentiable_workflow_audit_suite()

print(workflow.passed)
print(workflow.workflow_names)
print(workflow.worst_gradient_error)
print(workflow.best_training_values)
print(workflow.unsupported_scenarios)
```

This single report aggregates phase-gradient conformance, finite-shot
uncertainty containment, coupling-gradient verification, and coupling-learning
training evidence. It is the best current release-note and reviewer-facing
entry point for the supported differentiable-programming surface. It does not
claim arbitrary Python reverse-mode AD, live provider calibration, dynamic
circuit topology, classical regressors without generator spectra, or
mutation-heavy program IR semantics.

For optional ML-framework parity status:

```python
from scpn_quantum_control.phase import run_ml_framework_gradient_audit


ml = run_ml_framework_gradient_audit()

print(ml.audit_passed)
print(ml.executed_frameworks)
print(ml.unavailable_frameworks)
print(ml.blocked_frameworks)
print(ml.failed_frameworks)
```

This report checks JAX, PyTorch, TensorFlow, and PennyLane routes without
requiring those optional dependencies in the base installation. Importable
adapters are compared against the native parameter-shift reference; missing
dependencies are recorded as fail-closed unavailable. PennyLane remains blocked
unless the caller supplies a QNode gradient callable, because a meaningful
round-trip requires caller-owned circuit semantics.

For finite-shot uncertainty evidence:

```python
import numpy as np

from scpn_quantum_control.phase import run_finite_shot_gradient_uncertainty_audit


def objective(theta: np.ndarray) -> float:
    return float(np.mean(1.0 - np.cos(theta)))


finite_shot = run_finite_shot_gradient_uncertainty_audit(
    objective,
    np.array([0.7, -0.4, 0.2]),
    target_standard_error=0.02,
)

print(finite_shot.passed)
print(finite_shot.max_standard_error)
print(finite_shot.within_confidence)
```

This path verifies uncertainty propagation, shot allocation, and confidence
containment for declared shifted-expectation variances. It is not a live
hardware-sampling, detector-drift, or queue-calibration certificate.

## Minimal differentiable coupling learning

```python
import numpy as np

from scpn_quantum_control.phase import (
    learn_couplings_from_observations,
    multi_frequency_parameter_shift_rule,
    verify_coupling_parameter_shift_gradient,
)


def observations(K: np.ndarray) -> np.ndarray:
    return np.array([np.sin(K[0, 1])])


run = learn_couplings_from_observations(
    observations,
    target_observations=np.array([0.0]),
    initial_couplings=np.array([[0.0, 0.8], [0.8, 0.0]]),
    rule=multi_frequency_parameter_shift_rule([2.0]),
    max_steps=80,
)

certificate = verify_coupling_parameter_shift_gradient(
    observations,
    target_observations=np.array([0.0]),
    couplings=np.array([[0.0, 0.8], [0.8, 0.0]]),
    rule=multi_frequency_parameter_shift_rule([2.0]),
)

print(run.learned_coupling_matrix, run.certificate.monotone_accepted_values)
print(certificate.passed, certificate.max_abs_error)
```

This is a bounded differentiable-programming route for sinusoidal or quantum
expectation observation models. The verifier is a small-model diagnostic that
records parameter-shift and finite-difference gradients, absolute errors,
evaluation counts, and edge provenance. It is not an arbitrary
classical-regression, discontinuous-model, shot-noisy hardware, or
production-scale finite-difference claim; hardware backends remain disabled
unless an explicit policy enables them.

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

## Minimal JAX host-callback bridge

```python
import numpy as np

from scpn_quantum_control.phase import jax_parameter_shift_value_and_grad


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


result = jax_parameter_shift_value_and_grad(
    cost,
    np.array([0.4]),
    jit=True,
)
print(result.gradient, result.host_callback)
```

This is an optional interop adapter. It imports JAX only when called and reports
`host_callback=True` for JIT-wrapped execution. Native JAX-differentiated
quantum kernels remain a separate roadmap item.

## Minimal PennyLane agreement check

```python
import numpy as np

from scpn_quantum_control.phase import check_pennylane_parameter_shift_agreement


def scpn_cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


def pennylane_grad(params: np.ndarray) -> np.ndarray:
    # Usually qml.grad(qnode)(params); written explicitly for compact docs.
    return np.array([-np.sin(params[0])], dtype=float)


agreement = check_pennylane_parameter_shift_agreement(
    scpn_cost,
    pennylane_grad,
    np.array([0.4]),
)
assert agreement.passed
```

The bridge validates agreement. It does not claim automatic QNode generation
for every SCPN circuit yet.

## Minimal PyTorch and TensorFlow tensor bridges

```python
import numpy as np

from scpn_quantum_control.phase import (
    tensorflow_parameter_shift_value_and_grad,
    torch_parameter_shift_value_and_grad,
)


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


torch_result = torch_parameter_shift_value_and_grad(cost, np.array([0.4]))
tf_result = tensorflow_parameter_shift_value_and_grad(cost, np.array([0.4]))

print(torch_result.torch_gradient, torch_result.host_boundary)
print(tf_result.tensorflow_gradient, tf_result.host_boundary)
```

These bridges are optional tensor-conversion boundaries. They are useful for
framework pipelines that need gradient payloads, but they do not claim native
PyTorch or TensorFlow autodiff through a quantum simulator.

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
