# Quantum Gradients

Quantum gradients are the first differentiable-programming surface that most quantum-ML users look for. The current public route starts with parameter-shift gradients and expands toward backend-aware gradient planning, stochastic finite-shot gradients, adjoint simulator gradients, and framework adapters.

## Parameter-shift rule

For a Pauli-rotation expectation objective with generator spectrum compatible with the standard shift rule, the derivative is

$$
\frac{\partial C}{\partial \theta_k} = \frac{1}{2}\left[C(\theta_k + \pi/2) - C(\theta_k - \pi/2)\right].
$$

This rule avoids finite-difference truncation error for supported quantum expectation objectives. Opaque `ScalarObjective` callables still require independent plus/minus evaluations for every trainable parameter and shift term because the callable does not expose gate generators or commutators.

Registered `PhaseQNodeCircuit` declarations expose more structure. Use
`plan_phase_qnode_parameter_shift_evaluations(...)` before executing gradients
when circuits reuse logical parameters across multiple gates:

```python
import numpy as np

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    parameter_shift_phase_qnode_gradient,
    plan_generic_parameter_shift_evaluations,
    plan_phase_qnode_parameter_shift_evaluations,
)

phase_qnode = PhaseQNodeCircuit(
    n_qubits=1,
    operations=(("h", (0,)), ("rz", (0,), 0), ("rz", (0,), 0)),
    observable=PauliTerm(1.0, ((0, "x"),)),
)
params = np.array([0.37], dtype=float)

generic_plan = plan_generic_parameter_shift_evaluations(params)
gate_plan = plan_phase_qnode_parameter_shift_evaluations(phase_qnode, params)
gradient = parameter_shift_phase_qnode_gradient(phase_qnode, params)
print(generic_plan.evaluations, gate_plan.planned_shifted_evaluations, gradient.gradient)
```

The registered planner groups by logical parameter, detects when a tied
commuting generator group collapses to a valid single frequency, and switches
repeated non-collapsible parameters to a multi-frequency shift rule instead of
silently applying the single-gate `pi/2` rule. It does not claim that distinct
logical parameters can be recovered from one simultaneous shift; that would be a
directional derivative, not the independent gradient vector.

### Controlled-rotation shift rule

The two-term `pi/2` rule is exact only for a single-Pauli generator, whose
eigenvalues are `{+1/2, -1/2}` and whose only positive spectral gap is `1`. A
controlled rotation (`crx`, `cry`, `crz`) acts only in the control-on subspace,
so its generator eigenvalues are `{0, 0, +1/2, -1/2}` and it carries two
distinct gaps, `{1/2, 1}`. The planner therefore assigns controlled rotations
the four-term rule (frequencies `{1/2, 1}`); a tied group of `m` identical
controlled rotations scales to `{m/2, m}`. The two-term rule is wrong for a
controlled rotation whenever the observable couples the control-on and
control-off sectors — for example a Pauli `X` or `Y` on the control qubit — and
the planner no longer applies it there.

### U3 and arbitrary single-qubit unitaries

A U3 gate, and any `2x2` unitary, is supported through an exact Euler ZYZ
decomposition into three registered single-Pauli rotations, each carrying the
canonical two-term rule:

```python
from scpn_quantum_control.phase import build_u3_operations, su2_zyz_angles

phi, theta, lam = su2_zyz_angles(target_unitary)   # U ∝ RZ(phi) RY(theta) RZ(lam)
operations = build_u3_operations(qubit=0, parameter_indices=(0, 1, 2))
```

`su2_zyz_angles` discards the global phase and returns the ZYZ angles;
`build_u3_operations` emits the `RZ·RY·RZ` operations in circuit order so the
three angles differentiate analytically. This keeps general-unitary coverage
exact and fail-closed without enlarging the differentiable-gate primitive set.

For generators with several positive frequency gaps, build an explicit
multi-frequency rule:

```python
from scpn_quantum_control.phase import (
    multi_frequency_parameter_shift_rule,
    parameter_shift_gradient,
)

rule = multi_frequency_parameter_shift_rule([1.0, 2.0, 3.0])
grad = parameter_shift_gradient(objective, params, rule=rule)
```

The helper solves the exact sine-system for symmetric plus/minus shifts and
records the declared frequency set on the rule. This broadens deterministic
simulator and dry-run gradient coverage beyond the legacy two-point case.
Finite-shot uncertainty and shot-allocation helpers accept explicit
`(n_terms, n_parameters)` plus/minus means, variances, and shot-count records
for multi-term rules. This keeps provider data honest: every shift term must
carry its own sample variance instead of being collapsed into a single opaque
parameter estimate.

## Registered Training-Suite Audit

The registered medium training evidence suite covers seeded local QNN, QGNN,
QSNN, Kuramoto-XY, open-system-control, and inverse-coupling-recovery cases.
The open-system case uses a noise-aware damped residual with damping/dephasing
parameters and an analytic gradient checked against finite differences. The
inverse-coupling case uses a full-rank synthetic observation design for
identifiable `K_nm` edge recovery and checks the analytic gradient against
finite differences. Use the training-suite audit when the question is not "did
those cases pass?" but "which requested training lanes can be closed from
current evidence?":

```python
from scpn_quantum_control.phase import run_registered_differentiable_training_suite_audit


audit = run_registered_differentiable_training_suite_audit()

print(audit.passed_model_families)
print(audit.blocked_model_families)
print(audit.ready_for_training_suite_promotion)
```

The current audit closes the registered local training-suite lanes while still
separating that evidence from arbitrary architecture support, provider hardware
execution, and production benchmark claims.

## Evidence Checklist

Before promoting a quantum-gradient result, record:

| Evidence | Why it is required |
|---|---|
| Objective definition and parameter order | Replays and shifted evaluations must address the same trainable parameters. |
| Rule family | Standard parameter-shift, multi-frequency parameter-shift, finite-shot stochastic shift, or unsupported route. |
| Shifted samples | Plus/minus expectation values, shot counts, and finite-shot variances when stochastic. |
| Agreement check | Finite-difference or independent-framework comparison on a small smooth case. |
| Optimisation trace | Accepted value descent, gradient norm, failure reason, or convergence certificate. |
| Backend plan | Statevector, finite-shot simulator, hardware-blocked, or provider-specific policy. |

This makes gradient evidence reviewable by researchers who expect PennyLane-like
visibility while preserving SCPN-specific claim boundaries.

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

Finite-difference result objects in `scpn_quantum_control.differentiable` also
carry the `FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY` provenance string via
their `claim_boundary` field. Treat those artefacts as diagnostic-only checks:
they are not analytic gradients, parameter-shift gradients, native-framework
autodiff, whole-program AD, provider execution, hardware execution, or production
benchmark evidence.

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

## Bounded QNN framework-gradient agreement

The bounded phase-QNN route now has a dedicated framework-agreement surface:

```python
from scpn_quantum_control.phase import run_bounded_qnn_framework_bridge_matrix

matrix = run_bounded_qnn_framework_bridge_matrix()
print(matrix.supported_count, matrix.fail_closed_count)
```

The bridge matrix declares the bounded routes that are implemented today:
native JAX `value_and_grad` for the bounded phase-QNN classifier, PyTorch
tensor-gradient evidence, and TensorFlow tensor-gradient evidence. It also
records explicit fail-closed rows for arbitrary framework autodiff through
simulator kernels and live provider hardware-gradient execution. Treat this
matrix as the route selector before promoting any framework-gradient result.

```python
import numpy as np

from scpn_quantum_control.phase import (
    parameter_shift_qnn_classifier_gradient,
    verify_parameter_shift_qnn_framework_agreement,
)

features = np.array([[0.0], [np.pi]], dtype=float)
labels = np.array([0.0, 1.0], dtype=float)
params = np.array([0.45], dtype=float)
expected = parameter_shift_qnn_classifier_gradient(features, labels, params)

agreement = verify_parameter_shift_qnn_framework_agreement(
    features,
    labels,
    params,
    framework_gradients={"jax": lambda _values: expected.copy()},
)
print(agreement.passed, agreement.framework_count)
```

This is reviewer-facing agreement evidence for named, caller-supplied
framework-style gradients. Each agreement record carries a machine-readable
`source_class`, a per-source `native_framework_autodiff` flag, and the same
claim-boundary text as the suite result, so downstream reports can distinguish
deterministic manual references from native framework autodiff-through-simulator
evidence. The default suite labels its built-in rows as deterministic manual
references; caller-provided rows remain caller-supplied evidence unless a
separate native adapter surface supplies the row. The suite also emits a
`conformance_table` field for every case. The table records the same circuit,
parameter vector, and observable across `scpn`, `jax`, `pytorch`, `tensorflow`,
`pennylane`, and `qiskit` rows, marks exact-state reference agreement as
`passed`, and marks finite-shot, provider-plan, and hardware-execution rows as
`blocked` until shot records, no-submit provider plans, live-ticket metadata,
raw-count replay, calibration snapshots, and hardware approval exist. The table
is a claim-boundary artefact; it does not promote deterministic manual
references into native autodiff-through-simulator evidence. The bounded
phase-QNN model
also exposes
`jax_native_qnn_value_and_grad(...)`, which expresses that model directly in JAX
operations and verifies JAX `value_and_grad` against the SCPN parameter-shift
reference, plus `torch_bounded_qnn_value_and_grad(...)`, which returns PyTorch
tensors from the analytic bounded-model gradient,
`torch_autograd_qnn_value_and_grad(...)`, which wraps the bounded model in a
custom `torch.autograd.Function`,
`run_torch_autograd_function_audit(...)`, which records direct
`Tensor.backward()` gradient parity and `torch.optim.SGD` integration for that
custom backward route,
`run_torch_func_compatibility_audit(...)`, which checks bounded
`torch.func.grad`, `torch.func.vmap`, and `torch.func.jacrev` compatibility,
`run_torch_compile_compatibility_audit(...)`, which checks bounded
`torch.compile` gradient compatibility,
`torch_bounded_qnn_module(...)` / `torch_bounded_qnn_layer(...)` plus
`run_torch_module_wrapper_audit(...)`, which check bounded PyTorch module/layer
wrapper compatibility,
`run_torch_module_state_audit(...)`, which checks strict bounded-module
`state_dict` replay and Adam optimizer-state replay for that same wrapper
surface,
`run_torch_module_device_state_audit(...)`, which checks CPU module-device
state replay and classifies CUDA replay only after a real CUDA smoke succeeds,
`run_torch_module_checkpoint_audit(...)`, which writes a real `torch.save`
checkpoint and reloads it on CPU with `weights_only=True` before strict module
plus Adam optimizer-state replay,
`run_torch_long_lived_checkpoint_matrix(...)`, which records a versioned
checkpoint schema, tensor metadata manifest, runtime fingerprint, and repeated
local CPU weights-only loads for that checkpoint route,
`run_torch_module_export_audit(...)`, which exports the same bounded module with
`torch.export.export(...)`, persists it with `torch.export.save(...)`, reloads it
with `torch.export.load(...)`, and replays the local CPU value route through
`ExportedProgram.module()`,
`run_torch_export_shape_matrix(...)`, which records separate one- and
two-parameter static export artifacts,
`run_torch_dynamic_shape_export_audit(...)`, which exports one input-driven
bounded module with symbolic batch constraints and replays multiple concrete
batch sizes after save/load,
`run_torch_aot_autograd_export_audit(...)`, which captures self-produced local
AOTAutograd forward/backward FX graphs and replays the loaded backward graph
against the SCPN parameter-shift gradient reference,
and
`tensorflow_bounded_qnn_value_and_grad(...)`, which returns TensorFlow tensors
from the analytic bounded-model gradient. Each route checks the same
parameter-shift reference. These are intentionally narrow bridge promotions:
arbitrary autodiff-through-simulator kernels, unrestricted QNN architectures,
incompatible CUDA/device placement guarantees, cross-runtime AOTAutograd
execution, dynamic-shape AOTAutograd export, dynamic feature-width export
promotion, cross-runtime checkpoint/export
portability, external checkpoint-corpus promotion, and live provider gradients
remain outside the promoted surface.

## Bounded QNN convergence evidence

`run_parameter_shift_qnn_convergence_suite(...)` packages deterministic
phase-flip training cases with explicit loss-drop, accuracy, and evaluation
accounting:

```python
from scpn_quantum_control.phase import run_parameter_shift_qnn_convergence_suite

suite = run_parameter_shift_qnn_convergence_suite()
print(suite.passed, suite.total_parameter_shift_evaluations)
```

The suite is local deterministic evidence only. It is not hardware evidence,
not finite-shot noisy training, and not a claim that arbitrary QNN/QGNN/QSNN
architectures converge.

`run_parameter_shift_qnn_multi_seed_convergence_suite(...)` extends the same
bounded cases across deterministic initial-parameter perturbations:

```python
from scpn_quantum_control.phase import run_parameter_shift_qnn_multi_seed_convergence_suite

suite = run_parameter_shift_qnn_multi_seed_convergence_suite(seeds=(11, 17, 23))
case = suite.case_by_name("single_feature_phase_flip")
print(case.worst_best_loss, case.best_loss_std, suite.total_run_count)
```

The multi-seed envelope records every seed, seeded initial parameters, per-run
pass/fail status, worst best-loss, worst accuracy, loss standard deviation, and
parameter-shift evaluation totals. It remains local deterministic evidence with
seeded initial-condition perturbations; it is not finite-shot stochastic
training, hardware execution, isolated benchmark evidence, or a broad
QNN/QGNN/QSNN convergence claim.

`run_parameter_shift_qnn_loss_landscape_suite(...)` samples the same bounded
phase-QNN objectives on local parameter grids:

```python
from scpn_quantum_control.phase import run_parameter_shift_qnn_loss_landscape_suite

suite = run_parameter_shift_qnn_loss_landscape_suite(
    case_names=("single_feature_phase_flip",),
    grid_radius=0.2,
    points_per_axis=5,
)
case = suite.case_by_name("single_feature_phase_flip")
print(case.min_loss, case.max_loss, case.max_gradient_norm)
```

The landscape evidence records grid axes, every sampled parameter vector,
losses, analytic parameter-shift gradients, gradient norms, sampled argmin,
loss span, and pass/fail thresholds. It is a local diagnostic for bounded
training surfaces, not hardware execution, finite-shot evidence, isolated
benchmark evidence, or a claim about arbitrary QNN/QGNN/QSNN landscapes.

## Bounded QNN finite-shot evidence

Seeded finite-shot simulator evidence is available for the bounded phase-QNN
classifier:

```python
import numpy as np

from scpn_quantum_control.phase import estimate_parameter_shift_qnn_finite_shot_gradient

features = np.array([[0.0], [np.pi]], dtype=float)
labels = np.array([0.0, 1.0], dtype=float)
params = np.array([0.45], dtype=float)

result = estimate_parameter_shift_qnn_finite_shot_gradient(
    features,
    labels,
    params,
    shots_per_sample=8192,
    seed=17,
)
print(result.passed, result.max_confidence_radius)
```

`run_parameter_shift_qnn_finite_shot_convergence_suite(...)` extends this to
seeded noisy-gradient training cases. The evidence records replay seeds, shot
counts, shifted-loss probes, confidence radii, and total shot use. It remains a
local simulator surface: provider jobs, unseeded stochastic training,
low-shot promotion, and arbitrary QNN architectures remain fail-closed or
staged.

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
from scpn_quantum_control.phase import (
    explain_quantum_gradient_method,
    plan_quantum_gradient_backend,
)

plan = plan_quantum_gradient_backend("qasm_simulator", n_params=4, shots=4096)
print(plan.method, plan.evaluations, plan.shots)

explanation = explain_quantum_gradient_method("qasm_simulator", n_params=4)
print(explanation.selected_method, explanation.shot_policy.planned_shots)
```

Current planner behaviour:

| Backend family | Default method | Status |
|---|---|---|
| `statevector_simulator` | `parameter_shift` | Supported for deterministic local expectations. |
| `finite_shot_simulator` | `stochastic_parameter_shift` | Supported with explicit shots and uncertainty metadata. |
| `hardware_qpu` | `unsupported` | Fails closed unless a later hardware policy explicitly enables execution. |
| Unknown backend | `unsupported` | Fails closed and suggests local simulator alternatives. |

`explain_quantum_gradient_method(...)` wraps the same planner decision in a
deterministic explanation object. It returns the selected `QuantumGradientPlan`,
ordered rejected methods with reasons, the shot policy after backend defaults,
and the fallback path for unsupported or degraded routes. The object is planner
metadata only; it does not submit hardware jobs or promote benchmark evidence.

## Gradient support matrix

`plan_gradient_support(...)` combines the backend plan with registered gate,
observable, transform, and adapter contracts:

```python
from scpn_quantum_control.phase import plan_gradient_support

plan = plan_gradient_support(
    gate="ry",
    observable="pauli_expectation",
    backend="statevector",
    transform="grad",
    adapter="native",
    n_params=2,
)
print(plan.supported, plan.recommended_method)
```

The matrix currently supports bounded local and host-bridge routes:

| Component family | Supported examples | Blocked examples |
|---|---|---|
| Gates | `rx`, `ry`, `rz`, `phase_rotation`, `controlled_phase`, fixed `cz` topology | `arbitrary_unitary` |
| Observables | `pauli_expectation`, `sparse_pauli_sum`, `kuramoto_xy_energy` | `arbitrary_povm` |
| Backends | `statevector`, `qasm_simulator` with shots and variance metadata | `hardware` without explicit policy, unknown provider families |
| Transforms | `grad`, `value_and_grad`, deterministic local `hessian`, `gradient_tape` | `vmap`, finite-shot `hessian`, unregistered transform nesting |
| Adapters | `native`, `jax`, `pytorch`, `tensorflow`, `pennylane`, `qiskit` on their declared bridge surfaces | unregistered ML/provider adapters |

Use `run_gradient_support_matrix_audit()` for a built-in executable support
matrix. It checks four supported combinations and five blocked combinations,
then returns JSON-ready plans with blocked reasons, warnings, alternatives, and
claim boundaries.

The Studio planner view is generated from the same audit through:

```bash
python -m scpn_quantum_control.gradient_plan_explanation_artifact --check
```

The committed artefacts are
`data/differentiable_phase_qnode/gradient_plan_explanations_20260709.json`
and
`data/differentiable_phase_qnode/gradient_plan_explanations_20260709.md`.
They explain each gate/observable/backend/transform/adapter cell, the selected
method family, the backend evaluation mode, and the fail-closed boundaries. The
artefact is planner evidence only; it does not run browser differentiation or
promote live hardware-gradient support.

## Transform nesting governance

`plan_gradient_transform_nesting(...)` adds a second planning layer for nested
transforms:

```python
from scpn_quantum_control.phase import plan_gradient_transform_nesting

plan = plan_gradient_transform_nesting(("grad", "grad"), n_params=2)
print(plan.supported, plan.strategy)
```

Current behaviour:

| Transform stack | Status |
|---|---|
| `grad` | Supported on registered local or declared single-adapter routes. |
| `value_and_grad` | Supported on local and declared single-adapter bridge routes. |
| `hessian` | Supported as deterministic local curvature diagnostic only. |
| `grad` then `grad` | Supported as deterministic native nested-gradient Hessian route. |
| `gradient_tape` | Supported as phase-gradient record/replay evidence. |
| `vmap`, `jvp`, `vjp`, `jacfwd`, `jacrev` | Fail closed until executable transform algebra is implemented. |
| nested ML/provider adapters | Fail closed; adapters expose only declared single-transform bridge surfaces. |
| finite-shot curvature or hardware curvature | Fail closed; second-order routes require deterministic local expectations. |

Use `run_gradient_transform_nesting_audit()` to produce JSON-ready evidence for
six supported routes and seven blocked routes.

Finite-shot uncertainty can be propagated from plus/minus expectation variances:

```python
from pathlib import Path

import numpy as np

from scpn_quantum_control.phase import parameter_shift_gradient_with_uncertainty

result = parameter_shift_gradient_with_uncertainty(
    plus_values=np.array([1.2, -0.3]),
    minus_values=np.array([0.8, -0.7]),
    plus_variances=np.array([0.04, 0.09]),
    minus_variances=np.array([0.04, 0.09]),
    shots=4096,
    sample_provenance={
        "sample_seed": "finite-shot-local-seed",
        "shot_batch_id": "finite-shot-local-batch",
        "source_class": "caller_supplied",
    },
)
print(result.gradient, result.standard_error)
```

The optional Rust extension exposes
`parameter_shift_gradient_uncertainty_rust(...)` for the same materialised
finite-shot arithmetic: shifted plus/minus means, shifted variances, plus/minus
shot counts, rule coefficients, and trainable masks are validated at the FFI
boundary before Rust returns the gradient, standard error, diagonal covariance,
and confidence radius. This is a parity kernel for uncertainty propagation; it
does not call provider samplers, submit hardware jobs, or replace the Python
evidence records.

The Python `StochasticGradientResult` now keeps those evidence records directly:
each `ParameterShiftSampleRecord` names the parameter, term index, shift,
coefficient, plus/minus values, plus/minus variances, plus/minus shot counts,
trainable mask, sample seed, shot-batch ID, source class, gradient contribution,
and variance contribution. Generic finite-shot replay fails closed when
`sample_provenance` is absent, empty, or outside the accepted source-class
allowlist, so caller-supplied tensors cannot be promoted without replay
identity. Frozen parameters are recorded with zero contribution so reviewers can
reconstruct why they did not enter the stochastic gradient. The result also
carries the `STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY` string,
`hardware_execution=False`, the confidence interval, the failure-policy status,
and failure reasons.

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
    return ProviderExpectationSample(
        value=value,
        variance=0.04,
        shots=shots,
        metadata={
            "sample_seed": "finite-shot-example-seed",
            "shot_batch_id": "finite-shot-example-batch",
            "source_class": "synthetic_fixture",
        },
    )


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
errors, confidence radii, and a claim boundary. Finite-shot samples must also
carry `sample_seed`, `shot_batch_id`, and `source_class` metadata; accepted
source classes are caller-supplied arrays, local simulator, provider replay,
provider runtime, and synthetic fixture. The executor then stamps each
plus/minus sample with the parameter index, shift index, direction, shift,
coefficient, and `shifted_parameter_digest`, so callback-supplied sample
provenance and executor-owned shifted-sample provenance remain distinct.
Hardware aliases still fail closed unless an explicit hardware policy enables
them through the backend planner.

For hardware preparation, use
`prepare_provider_hardware_parameter_shift_gradient(...)` instead of the
sampler-execution function. It returns provider/backend, policy decision,
shifted-evaluation, estimated-shot, evidence-ID, and claim-boundary metadata
without invoking a sampler and without submitting a QPU job.

Multi-frequency rules are provider-safe on the same callback contract. The
executor plans `2 * n_terms * n_parameters` samples, stores each
`(parameter_index, shift_index)` record with the exact shift and coefficient,
and aggregates independent term variances into the reported per-parameter
standard error. Finite-shot callbacks must therefore return value, variance,
and shot metadata for every shifted term instead of supplying a collapsed
gradient estimate.

### Provider-gradient readiness audit

`run_provider_gradient_readiness_audit()` turns the provider callback contract
into executable evidence:

```python
from scpn_quantum_control.phase import run_provider_gradient_readiness_audit

audit = run_provider_gradient_readiness_audit()
assert audit.passed
print([record.scenario.name for record in audit.blocked_records])
```

The built-in audit records:

| Scenario | Expected outcome |
|---|---|
| `statevector_parameter_shift` | Executes deterministic local parameter-shift and matches the analytic gradient. |
| `finite_shot_parameter_shift` | Executes finite-shot callback gradients with sample variance, sample provenance, and confidence radii. |
| `multi_frequency_finite_shot` | Executes multi-frequency parameter-shift with per-term shot and shifted-sample provenance. |
| `hardware_without_policy` | Fails closed before execution because hardware gradients require an explicit policy gate. |
| `unknown_backend` | Fails closed and suggests local simulator alternatives. |
| `finite_shot_missing_variance` | Fails during execution because finite-shot gradients require sample variance. |

This support matrix is intentionally executable rather than a static checklist.
It lets reviewers distinguish ready callback paths from blocked hardware or
malformed-sample paths without submitting jobs to a provider. Finite-shot routes
also fail closed when sample provenance is absent or uses an unknown source
class.

## Hardware-gradient policy readiness

`evaluate_hardware_gradient_policy(...)` adds an explicit gate between provider
callback readiness and real hardware-gradient preparation. It checks the
provider/backend allowlists, `allow_hardware=True`, parameter count, shift-term
count, shots per shifted expectation, total estimated shots, required evidence
IDs, and live-execution ticket status:

```python
from scpn_quantum_control.phase import (
    HardwareGradientRequest,
    evaluate_hardware_gradient_policy,
)

decision = evaluate_hardware_gradient_policy(
    HardwareGradientRequest(
        provider="ibm_quantum",
        backend="ibm_quantum",
        n_params=2,
        shots=512,
        allow_hardware=True,
        evidence_ids={
            "backend_calibration_id": "calibration-snapshot-id",
            "no_qpu_gate_id": "no-qpu-gate-id",
            "claim_boundary_id": "claim-boundary-id",
            "cost_budget_id": "budget-approval-id",
        },
    )
)
```

`run_hardware_gradient_policy_readiness_suite()` returns JSON-ready evidence
covering a bounded dry-run approval and blocked routes for missing hardware
approval, unknown provider/backend aliases, excessive shot budgets, missing
evidence IDs, and live execution without a ticket. Dry-run approval means a
provider job can be prepared under policy; it is not live QPU execution and not
a hardware-gradient result.

To bind that policy to a provider-preparation record:

```python
from scpn_quantum_control.phase import prepare_provider_hardware_parameter_shift_gradient

preparation = prepare_provider_hardware_parameter_shift_gradient(
    [0.2, -0.4],
    provider="ibm_quantum",
    backend="ibm_quantum",
    shots=512,
    evidence_ids={
        "backend_calibration_id": "calibration-snapshot-id",
        "no_qpu_gate_id": "no-qpu-gate-id",
        "claim_boundary_id": "claim-boundary-id",
        "cost_budget_id": "budget-approval-id",
    },
)
```

`preparation.gradient_available` and `preparation.hardware_execution` are both
false. The record is readiness evidence only.

No-submit hardware-gradient campaign specs build on the same boundary:

```python
from scpn_quantum_control.phase import (
    default_hardware_gradient_campaign_specs,
    run_hardware_gradient_campaign_readiness_suite,
)

specs = default_hardware_gradient_campaign_specs()
suite = run_hardware_gradient_campaign_readiness_suite(specs)
print(suite.passed, suite.hardware_execution_count)
```

The default campaign specs cover two no-submit XY-gradient preparation lanes:
parameter-shift VQE gradients on named Heron r2 backends and seeded SPSA
gradients with perturbation records. Each `HardwareGradientCampaignSpec` carries
the backend allowlist, evidence IDs, shot/evaluation budget, calibration
snapshot requirement, raw-count replay schema, statevector reference-gradient
requirement, and publication-claim boundary. `HardwareGradientCampaignPlan`
evaluates the spec through the same hardware-gradient policy used by provider
preparation. The suite never calls a provider sampler, never submits a QPU job,
and never returns a hardware-gradient value; it is campaign-readiness metadata
only until a live ticket and raw-count artefacts exist.

The campaign publication package scaffold is also available as structured
metadata:

```python
from scpn_quantum_control.phase import build_hardware_gradient_publication_package

package = build_hardware_gradient_publication_package()
print(package.submission_ready)
print(package.to_markdown())
```

`build_hardware_gradient_publication_package()` produces the preregistration,
methods sections, raw artefact map, draft claim-ledger rows, and comparison
benchmark slots for "Hardware-Validated Quantum Gradients for XY Hamiltonians".
The generated package is deliberately not submission-ready: claim rows are not
promoted, benchmark slots are marked as not executed, artefact IDs are empty,
and live hardware results are rejected if they are injected into the scaffold.
Use it as the publication control surface for the approved future hardware run,
not as evidence that the run has already happened.

The provider hardware-preparation audit packages the preparation boundary checks
into a one-call support matrix:

```python
from scpn_quantum_control.phase import run_provider_hardware_gradient_preparation_audit

audit = run_provider_hardware_gradient_preparation_audit()
assert audit.passed
print(audit.approved_count, audit.blocked_count, audit.hardware_execution_count)
```

The built-in scenarios cover bounded dry-run preparation, ticketed
live-preparation, missing evidence, excessive shot budgets, unknown
provider/backend aliases, and live preparation without a ticket. A passing audit
means all records preserved `hardware_execution == False` and
`gradient_available == False`; it is not a live QPU result.

The aggregate provider/hardware safety gate combines every differentiable
provider-facing safety surface:

```python
from scpn_quantum_control.phase import (
    DifferentiableProviderHardwareEvidenceChain,
    run_differentiable_provider_hardware_safety_audit,
)

safety = run_differentiable_provider_hardware_safety_audit()
assert safety.passed
assert safety.ready_for_hardware_gradient_promotion is False
print(safety.promotion_blockers)

chain = DifferentiableProviderHardwareEvidenceChain(
    live_execution_ticket="LIVE-2026-06-16-001",
    provider_name="ibm_quantum",
    backend_id="ibm_kingston",
    job_id="job-20260616-001",
    circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
    provider_allowlist_id="allowlist-heron-r2-20260616",
    shot_budget_id="shot-budget-4096-20260616",
    raw_count_replay_artifact_id="raw-counts-001",
    raw_count_replay_digest="sha256:" + "a" * 64,
    raw_count_shots=4096,
    calibration_snapshot_artifact_id="calibration-001",
    calibration_snapshot_digest="sha256:" + "b" * 64,
    statevector_comparison_artifact_id="statevector-001",
    statevector_comparison_digest="sha256:" + "c" * 64,
    isolated_benchmark_artifact_id="isolated-001",
    captured_at_utc="2026-06-16T00:00:00Z",
    valid_until_utc="2026-07-20T00:00:00Z",
)
ready = run_differentiable_provider_hardware_safety_audit(evidence_chain=chain)
assert ready.evidence_chain_ready
```

`run_differentiable_provider_hardware_safety_audit()` aggregates provider
gradient readiness, provider hardware-gradient preparation, provider QNode
transforms, QNode tape records, and no-submit hardware-gradient campaign
readiness. The result is the promotion guard for hardware-gradient claims:
every local safety surface must preserve zero hardware execution and zero
hardware-gradient production. Promotion remains blocked until a single
freshness-bounded `DifferentiableProviderHardwareEvidenceChain` binds the live
ticket, provider/backend/job/circuit metadata, provider allowlist, shot budget,
raw-count replay digest, calibration snapshot digest, statevector comparison
digest, and `isolated_affinity` benchmark artefact. Detached legacy IDs are
serialized for compatibility, but they no longer make the audit promotion-ready
without the validated chain.

For the full differentiable-programming lane, `run_differentiable_readiness_audit()`
aggregates the gradient support matrix, transform nesting, QNode tape and
transform suites, provider-gradient readiness, hardware policy, and provider
hardware-preparation audit into one JSON-ready ledger. This is the fastest way
to show a reviewer which routes are supported, which routes are blocked, and
that hardware execution remains closed.

## Qiskit shifted-circuit generation

For Qiskit-native circuits, the phase namespace can generate fully bound
plus/minus shifted circuits and execute a local Statevector parameter-shift
gradient:

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.phase import (
    build_qiskit_provider_gradient_workflow_artifact,
    build_qiskit_runtime_qpu_execution_artifact,
    build_qiskit_runtime_qpu_provider_evidence_bundle,
    execute_qiskit_statevector_parameter_shift,
    run_qiskit_maturity_audit,
)

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

The same Qiskit bridge accepts `multi_frequency_parameter_shift_rule(...)`.
Circuit generation emits one fully bound plus/minus pair per
`(parameter, shift_term)`, and local Statevector execution aggregates the term
coefficients into the final gradient. This covers symbolic circuits where the
same trainable parameter appears with several generator frequencies, for
example `theta` and `2 * theta` rotations in one ansatz.

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

For reviewer-facing provider maturity evidence, use
`run_qiskit_maturity_audit(...)`. It aggregates fully bound shifted-circuit
generation, deterministic local Statevector reference gradients, finite-shot
surrogate uncertainty, optional Runtime primitive execution artefacts, and the
no-submit provider hardware-gradient preparation audit:

```python
maturity = run_qiskit_maturity_audit(
    circuit,
    observable,
    (theta,),
    np.array([0.4]),
    shots=4096,
)
print(maturity.local_gradient_ready, maturity.ready_for_provider_exceedance)
```

`QiskitRuntimePrimitiveExecutionArtifact` validates Runtime/primitive execution
metadata with non-empty provider, primitive, backend, job, circuit, and
observable identities, optional session and replay IDs, positive shots when
present, SHA-256 parameter/result/metadata digests, and
`hardware_execution=False`. Attaching it marks only
`runtime_primitive_execution_evidence` as passed.
`QiskitRuntimeQPUExecutionArtifact` validates live Runtime QPU execution
metadata for EstimatorV2 and SamplerV2 routes. It requires a live execution
ticket, backend allowlist, shot budget, ISA/transpiled-circuit digest, Runtime
result and metadata digests, positive shots, hardware execution, and a live
QPU session mode; EstimatorV2 evidence must include an observable fingerprint,
while SamplerV2 evidence must not. Attaching it marks only
`live_qpu_execution_ticket` as passed.
`build_qiskit_runtime_qpu_execution_artifact(...)` is the no-submit helper for
turning captured Runtime job metadata and SHA-256 artefact digests into that
validated evidence object; it does not create or submit provider jobs.
`QiskitRawCountReplayArtifact` validates raw-count capture and replay metadata
with live-ticket ID, hardware-execution citation, shot count, measured-qubit
count, expectation value, standard error, and SHA-256 count/replay digests.
`QiskitCalibrationStatevectorComparisonArtifact` validates a live-backend
calibration snapshot against a statevector reference with SHA-256 calibration
and comparison digests, finite non-negative error, positive tolerance, and
hardware-execution citation. Raw-count replay must match the Runtime QPU
provider, backend, job, circuit fingerprint, live ticket, and shot count;
calibration/statevector comparison must match the Runtime QPU provider,
backend, circuit fingerprint, and live ticket. These artefacts can clear only
their corresponding Runtime/QPU, raw-count replay, and calibration/statevector
comparison gates. The audit keeps `ready_for_provider_exceedance=False` until
isolated benchmark evidence exists.
`QiskitRuntimeQPUProviderEvidenceBundle` validates that whole attachable chain
as one no-submit audit input, requires explicit `captured_at_utc` and
`valid_until_utc` metadata, rejects inverted freshness windows, and can carry
an isolated benchmark artefact ID. The Qiskit maturity audit rejects expired
provider bundles before provider-exceedance readiness is evaluated. Without an
isolated benchmark ID, benchmark promotion remains blocked even when the
Runtime QPU, raw-count, and calibration comparison artefacts match.
`QiskitProviderGradientWorkflowArtifact` covers captured Runtime
provider-gradient workflow evidence for parameter-shift, finite-difference,
LCU, SPSA, QGT, and QFI methods. Build those artefacts with
`build_qiskit_provider_gradient_workflow_artifact(...)`; the maturity audit
keeps `provider_gradient_workflow_evidence` blocked until all six methods are
attached and matched to the same Runtime QPU evidence chain. Workflow artefacts
must have unique IDs and must retain the Runtime QPU provider, backend, job,
circuit, primitive family, observable fingerprint, parameter digest,
live-ticket, and shot-count chain. Any drift in those fields fails closed
before the audit can clear the provider-gradient workflow gate.

Each workflow artefact also carries method-specific provenance metadata. The
builder rejects missing, ambiguous, or cross-method metadata: parameter-shift
requires a shift-rule ID and shifted-circuit count, finite-difference requires
the stencil and positive step size, LCU requires a generator digest and term
count, SPSA requires perturbation seed/count metadata, and QGT/QFI require
matrix digests plus a matrix dimension matching the parameter count.

## Gradient Tape Boundary

For local simulator workflows, `gradient_tape` records deterministic and
finite-shot parameter-shift evaluations with backend-plan provenance. The tape
now preserves the active parameter-shift rule identity through `record.method`,
the backend replay plan through `record.plan_method`, and the per-parameter
shift-term count through `record.shift_terms` and `record.to_dict()`:

```python
import numpy as np

from scpn_quantum_control.phase import gradient_tape


def objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


with gradient_tape(backend="statevector") as tape:
    record = tape.record_parameter_shift("single_rotation", objective, np.array([0.3]))

print(record.gradient, record.method, record.shift_terms, record.evaluations)
```

Multi-frequency generators are supported when callers pass the same
`ParameterShiftRule` used by the native gradient engine. Deterministic replay
plans `2 * shift_terms * n_params` objective evaluations. Finite-shot replay
uses the same rule but requires `plus_values`, `minus_values`, `plus_variances`,
and `minus_variances` to be shaped as `(shift_terms, n_params)` whenever
`shift_terms > 1`; flat arrays fail closed because they cannot encode which
sample belongs to which Fourier term.

The current tape remains intentionally bounded. It is not a full programme-IR
tape, does not capture arbitrary Python side effects, and does not enable
hardware gradients without explicit policy approval. Deterministic records copy
parameter inputs, stamp `parameter_fingerprint` and `replay_fingerprint`, reject
objective functions that mutate replay arrays, and fail closed when identical
parameters do not replay to the same scalar value. `run_gradient_tape_contract_audit()`
exercises the DP-003 hardening cycle: independent nested tapes, same-tape
re-entry rejection, persistent clear/reuse, external alias snapshots,
objective-mutation rejection, and control-flow replay stability.

## Parameter-shift gradient descent

For training and onboarding evidence, the phase namespace exposes a bounded
gradient-descent loop over native parameter-shift gradients:

```python
import numpy as np

from scpn_quantum_control.phase import (
    parameter_shift_gradient_descent,
    validate_parameter_shift_training,
)


def objective(params: np.ndarray) -> float:
    return float(1.0 - np.cos(params[0]))


run = parameter_shift_gradient_descent(
    objective,
    np.array([0.8]),
    learning_rate=0.5,
    max_steps=80,
    gradient_tolerance=1e-7,
)
certificate = validate_parameter_shift_training(
    run,
    gradient_tolerance=1e-7,
    target_value=0.0,
    target_value_tolerance=1e-10,
)

print(run.best_value, certificate.monotone_accepted_values)
```

`ParameterShiftTrainingResult` records the initial/final/best objective value,
initial/final/best parameter vectors, final gradient, accepted and rejected
step counts, total objective evaluations, backend plan, native method, shift
term count, and every line-search step. `validate_parameter_shift_training`
turns that trace into a machine-checkable certificate for monotone accepted
values, target-value tolerance, minimum decrease, and final-gradient tolerance.
Multi-frequency rules preserve their `shift_terms` through every step.

Unsuitable scenarios remain explicit: hardware backends fail closed unless a
future policy enables them, non-finite objectives are rejected, and line-search
failure is recorded as `reason="line_search_failed"` instead of being promoted
as convergence.

## Bounded phase-QNN classifier

For users who need a small trainable quantum neural classifier rather than a
VQE objective, the phase namespace includes a deterministic data-reuploading
classifier:

```python
import numpy as np

from scpn_quantum_control.phase import (
    run_parameter_shift_qnn_conformance_suite,
    run_parameter_shift_qnn_optimizer_benchmark_suite,
    train_parameter_shift_qnn_classifier,
    verify_parameter_shift_qnn_classifier_gradient,
)


run = train_parameter_shift_qnn_classifier(
    np.array([[0.0], [np.pi]], dtype=float),
    np.array([0.0, 1.0], dtype=float),
    initial_params=np.array([0.8], dtype=float),
    learning_rate=0.7,
    max_steps=80,
    target_loss=0.0,
    target_loss_tolerance=1e-4,
)

print(run.prediction.probabilities, run.prediction.accuracy)

verification = verify_parameter_shift_qnn_classifier_gradient(
    np.array([[0.0], [np.pi]], dtype=float),
    np.array([0.0, 1.0], dtype=float),
    run.best_params,
)
print(verification.passed, verification.max_abs_error)

suite = run_parameter_shift_qnn_conformance_suite()
print(suite.passed, suite.case_count, suite.unsuitable_scenario_count)

optimizer_suite = run_parameter_shift_qnn_optimizer_benchmark_suite()
print(optimizer_suite.passed, optimizer_suite.evidence_class)
print(optimizer_suite.optimizer_names)
```

The route is intentionally narrow and auditable: one trainable phase per
feature column, full-batch binary MSE, deterministic local execution, explicit
multi-frequency parameter-shift descent, finite-difference gradient replay, and
fail-closed hardware policy. Optional external-gradient callables can be named
in the verification report to record JAX, PennyLane, or other adapter agreement
without claiming automatic framework conversion. The conformance suite
propagates external agreement source classes and native-autodiff flags into its
case payloads, packages the supported route, and records unsuitable scenarios
such as hardware backend promotion, arbitrary architectures, non-finite data,
native framework autodiff, finite-shot gradients without uncertainty,
feature/parameter contract mismatches, unregistered feature maps or observables,
and external gradients without provenance as staged or fail-closed. Each
scenario includes the evidence required before that route can be promoted. The
optimizer benchmark suite compares
parameter-shift training against finite-difference gradient descent, full-batch
SGD, Adam, SciPy L-BFGS-B with parameter-shift Jacobians, a diagonal-Fisher
natural-gradient baseline, seeded SPSA, and deterministic derivative-free grid
candidates. Each `QNNOptimizerBaselineResult` records best loss, accuracy,
evaluation count, step count, convergence flag, method label, and wall-clock
runtime, but the suite labels every row as non-isolated functional evidence. It
is the current production QNN foothold, not a throughput benchmark, hardware
performance claim, or unrestricted arbitrary QNN/QGNN/QSNN training route.

Synthetic exact-answer datasets for this route are available through
`load_differentiable_domain_benchmark_datasets()` and validated by
`run_differentiable_domain_benchmark_dataset_validation()`. The QNN rows carry
exact bounded phase-response probabilities, full-batch MSE losses, and
parameter-shift gradients; the Kuramoto-XY row carries an exact two-oscillator
order parameter, mean phase, XY energy, and phase-energy gradient. These rows
are conformance fixtures, not performance or hardware evidence.
The same API also exposes published public-domain artefact rows via
`load_differentiable_published_domain_benchmark_cases()` and
`run_differentiable_published_domain_benchmark_validation()`, validating that
the committed EEG, ITER-style MHD, IEEE 5-bus, and FEP artefacts remain
publication-safe, preserve their source-equation IDs and formula strings, and
round-trip through the Kuramoto conversion path.

## Parameter-shift natural gradient

For metric-aware training, the phase namespace exposes a bounded natural
gradient route. It computes native parameter-shift gradients, validates an
explicit metric tensor or callable metric, regularises the metric according to
an explicit damping/eigenvalue-floor/condition-number policy, solves the
regularised system, and then applies the same fail-closed Armijo line-search
discipline used by the ordinary descent route:

```python
import numpy as np

from scpn_quantum_control.phase import (
    parameter_shift_natural_gradient_descent,
    phase_qnode_natural_gradient_metric,
    phase_qnode_computational_basis_fisher_information,
    PhaseQNodeCircuit,
    validate_natural_gradient_training,
)


def objective(params: np.ndarray) -> float:
    return float((1.0 - np.cos(params[0])) + 0.05 * (1.0 - np.cos(params[1])))


metric = np.diag(np.array([1.0, 0.05]))
run = parameter_shift_natural_gradient_descent(
    objective,
    np.array([0.8, 0.8]),
    metric_tensor=metric,
    learning_rate=0.4,
    max_steps=20,
)
certificate = validate_natural_gradient_training(
    run,
    min_decrease=0.1,
)

print(run.best_value, certificate.monotone_accepted_values)
```

`ParameterShiftNaturalGradientResult` records metric source, damping,
eigenvalue floor, degeneracy tolerance, condition-number limit, final metric
rank/nullity, regularisation reason, accepted/rejected steps, backend plan,
shift-term count, final Euclidean and natural-gradient norms, and a claim
boundary. Each step also records the diagonal shift used for the metric solve.
The identity metric is allowed as a reproducible preconditioner baseline, but it
is labelled as such and is not a claim of quantum Fisher extraction.
Positive-semidefinite singular metrics are allowed only when the policy adds a
positive diagonal shift through damping, an eigenvalue floor, or a
condition-number cap. Indefinite metrics, non-symmetric metrics, wrong shapes,
non-finite entries, singular unregularised systems, unsafe hardware backends,
and non-descent metrics fail closed.

For registered local Phase-QNode circuits, the phase namespace also exposes an
exact pure-state metric provider.  It propagates analytic state derivatives
through the registered statevector gate family, computes the Fubini-Study
metric and the quantum Fisher information relation `QFI = 4 * metric`, and
returns the Fubini-Study metric in the shape expected by natural-gradient
training:

```python
circuit = PhaseQNodeCircuit(
    n_qubits=1,
    operations=(("ry", (0,), 0),),
    observable="pauli_z",
)
qnode_metric = phase_qnode_natural_gradient_metric(circuit)
classical_fisher = phase_qnode_computational_basis_fisher_information(
    circuit,
    np.array([0.8]),
)

run = parameter_shift_natural_gradient_descent(
    lambda params: float(1.0 - np.cos(params[0])),
    np.array([0.8]),
    metric_tensor=qnode_metric,
    learning_rate=0.4,
    max_steps=20,
)
```

This route is bounded to pure-state local Phase-QNode statevectors. It is not a
finite-shot classical Fisher estimator, density-matrix metric, noisy-channel
metric, provider metric, or hardware metric claim.

Degenerate QFI metrics are common when a parameter does not change the
projective state, for example an `rz` parameter on an initial computational
basis state. In that case the QFI/Fubini-Study metric can have non-zero nullity.
Use an explicit `eigenvalue_floor` or damping policy before using the metric as
a natural-gradient preconditioner, and inspect `final_metric_nullity`,
`final_regularization_reason`, and each step's `diagonal_shift` in the returned
training record.

## Trainability diagnostics and shot dry-runs

`run_barren_plateau_trainability_report(...)` samples parameter-shift gradients
across a caller-supplied parameter matrix, computes empirical gradient mean and
variance, classifies flat low-variance landscapes, and runs the variance-aware
finite-shot allocator without executing a backend:

```python
import numpy as np

from scpn_quantum_control.differentiable import Parameter
from scpn_quantum_control.phase import run_barren_plateau_trainability_report


def objective(params: np.ndarray) -> float:
    return float((1.0 - np.cos(params[0])) + 0.2 * (1.0 - np.cos(params[1])))


report = run_barren_plateau_trainability_report(
    objective,
    np.array([[0.2, -0.3], [0.7, 0.4], [-0.5, 0.6]]),
    parameters=(Parameter("theta0"), Parameter("theta1")),
    plus_variances=np.array([0.04, 0.09]),
    minus_variances=np.array([0.05, 0.10]),
    target_standard_error=0.02,
    min_shots=16,
    cost_per_shot=0.001,
    cost_unit="credits",
)

print(report.status, report.shot_dry_run.estimated_quantum_shots)
```

The dry-run record contains the backend gradient plan, shifted-evaluation count,
allocated plus/minus shots, predicted standard errors, cap warnings, estimated
cost, cost unit, and `hardware_execution=False`. Hardware and provider backends
fail closed because this route never requests hardware policy approval. When
the caller omits plus/minus variances, the report derives a conservative
variance tensor from sampled gradient variance and the configured variance
floor. The external comparison suite also records a Catalyst boundary row for
this BL-14 surface: adaptive finite-shot trainability dry-runs are not promoted
as Catalyst broadcast/vmap evidence.

`phase_qnode_computational_basis_fisher_information(circuit, params)` computes
the exact classical Fisher matrix for computational-basis statevector
probabilities using the same analytic state derivatives. It fails closed at
zero-probability outcomes because the probability-space Fisher expression is
singular there. The result keeps this exact matrix as the reference even when
finite-shot evidence is requested.

Passing `shot_count=...` adds a multinomial delta-method uncertainty model for
the exact computational-basis probability distribution:

```python
fisher = phase_qnode_computational_basis_fisher_information(
    circuit,
    params,
    shot_count=4096,
)

print(fisher.classical_fisher_information)
print(fisher.fisher_standard_error)
print(fisher.fisher_confidence_radius)
```

Passing `observed_counts=...` replays a strictly positive raw-count record as a
plug-in finite-shot Fisher estimate while preserving
`classical_fisher_information` as the exact statevector reference. The returned
evidence records `shot_count`, `count_record`, `empirical_probabilities`,
`finite_shot_classical_fisher_information`, confidence metadata, and a
non-promotional claim boundary. Hardware sampling, backend calibration,
adaptive measurements, provider runtime, and optimal measurement selection
remain outside this local route.

The optional Rust extension exposes parity kernels for the materialised metric
evidence as `phase_qnode_fubini_study_metric_rust(...)` and
`phase_qnode_computational_basis_fisher_rust(...)`. These functions take split
real/imaginary state amplitudes and split real/imaginary derivative rows; they
mirror the algebra used by the Python Phase-QNode API but do not execute the
circuit family themselves.

For reviewer-facing optimizer evidence, run the multi-start comparison audit:

```python
from scpn_quantum_control.phase import (
    run_ground_state_optimizer_convergence_suite,
    run_parameter_shift_optimizer_comparison,
)


suite = run_parameter_shift_optimizer_comparison(max_steps=8)

print(suite.passed)
print(suite.best_optimizer)
print(suite.natural_gradient_not_worse_count, suite.start_count)

ground_suite = run_ground_state_optimizer_convergence_suite()
print(ground_suite.passed, ground_suite.optimizer_names)
print(ground_suite.boundary_rows[0].failure_class)
```

The suite executes ordinary parameter-shift descent and natural-gradient descent
from several starts, records certificates for every route, and checks whether
the metric-aware route is no worse than the baseline under the declared
tolerance. Its claim boundary is intentionally narrow: local smooth phase
objectives, functional convergence evidence, no hardware execution, no
throughput statement, and no global optimality proof.

`run_ground_state_optimizer_convergence_suite()` is the BL-15 companion for
known small ground states. It compares natural-gradient, Adam, L-BFGS-B, seeded
SPSA, and COBYLA on deterministic exact-energy objectives, emits one
certificate per executable row, and writes local benchmark evidence through
`scripts/export_ground_state_optimizer_convergence.py`. The artefact rows are
classified as `functional_non_isolated`; they are convergence evidence, not
isolated timing, hardware execution, or a global optimality statement. The
QNG-QJIT-class baseline is represented as a hard-gap boundary row until a
compiler-owned metric-fusion route exists.

For bounded open-system objectives, use the Lindblad/MCWF evidence suite:

```python
from scpn_quantum_control.phase import run_open_system_objective_suite


open_suite = run_open_system_objective_suite()
print(open_suite.passed, open_suite.backend_names)
print(open_suite.records[0].invariant_certificate)
print(open_suite.records[1].reproducibility_certificate)
```

`run_open_system_objective_suite()` evaluates small Kuramoto-XY open-system
objectives through the production `LindbladKuramotoSolver` density-matrix path
and the production `mcwf_ensemble` trajectory-batch path. It records central
finite-difference gradients over coupling and damping scales, density-matrix
trace/Hermiticity/positivity certificates, and same-seed MCWF replay
certificates. The companion artifact command is:

```bash
PYTHONPATH=src:. python scripts/export_open_system_objective_evidence.py
```

The committed evidence lives at
`data/differentiable_phase_qnode/open_system_objective_evidence_20260709.json`
and `.md`. These rows are local finite-difference objective evidence only:
adjoint Lindblad sensitivities, unbiased stochastic-gradient estimators,
provider execution, hardware gradients, and isolated timing promotion remain
hard-gap boundary rows.

For bounded coupling recovery from time series, use the BL-17 recovery suite:

```python
from scpn_quantum_control.phase import run_coupling_recovery_suite


suite = run_coupling_recovery_suite()
print(suite.passed)
print(suite.records[0].learned_couplings)
print(suite.boundary_rows[0].reason)
```

`run_coupling_recovery_suite()` generates deterministic Kuramoto phase
trajectories with known coupling matrices, recovers Kuramoto couplings from
phase time series, and recovers XY couplings from edge-resolved pair-energy
observations. The default cases cover clean, noisy, and missing-data inputs.
The companion artifact command is:

```bash
PYTHONPATH=src:. python scripts/export_coupling_recovery_evidence.py
```

The committed evidence lives at
`data/differentiable_phase_qnode/coupling_recovery_evidence_20260709.json`
and `.md`. These rows are local synthetic known-ground-truth recovery evidence
only: hardware Hamiltonian learning, provider execution, isolated timing, and
arbitrary partial-observation inference remain fail-closed boundary rows.

For bounded synchronisation witnesses over phase clouds, use the BL-18
sync-witness suite:

```python
from scpn_quantum_control.phase import run_sync_witness_suite


suite = run_sync_witness_suite()
print(suite.passed)
print(suite.records[0].betti1_curve)
print(suite.boundary_rows[0].reason)
```

`run_sync_witness_suite()` builds deterministic synchronised, desynchronised,
and clustered phase clouds and certifies each with harmonic Kuramoto order
parameters, bootstrap order-parameter uncertainty, and exact Vietoris-Rips
persistent homology (Betti-0/1 curves and dimension-0/1 persistence diagrams)
over geodesic phase distances. The companion artifact command is:

```bash
PYTHONPATH=src:. python scripts/export_sync_witness_evidence.py
```

The committed evidence lives at
`data/differentiable_phase_qnode/sync_witness_evidence_20260709.json`
and `.md`. These rows are local synthetic reference-regime witness evidence
only: hardware phase tomography, provider execution, isolated timing, and
high-dimensional manifold inference remain fail-closed boundary rows.

## Composed differentiable objectives

Real control objectives usually mix energy, fidelity, regularization,
symmetry, and safety terms. `ComposedPhaseObjective` makes that explicit:

```python
import numpy as np

from scpn_quantum_control.phase import build_phase_control_objective


objective = build_phase_control_objective(
    2,
    energy_weight=1.0,
    fidelity_target=np.zeros(2),
    fidelity_weight=0.2,
    safety_bounds=(-1.0, 1.0),
    safety_weight=0.1,
)
evaluation = objective.evaluate(np.array([0.8, -0.7]))

print(evaluation.value, evaluation.parameter_shift_compatible)
```

Periodic energy, target-fidelity, regularization, and symmetry terms remain
parameter-shift compatible. Smooth box-safety penalties are analytic classical
terms; the objective can train them with exact term-wise gradients, but it
fails closed if a caller requires a pure parameter-shift objective. This keeps
hybrid quantum-control objectives useful without overstating the gradient mode.

Use the built-in audit when documentation, notebooks, or reviews need visible
evidence:

```python
from scpn_quantum_control.phase import run_composed_objective_audit_suite


audit = run_composed_objective_audit_suite()

print(audit.passed)
print(audit.pure_gradient.max_abs_error)
print(audit.hybrid_parameter_shift_gate_failed)
```

The audit verifies exact term-wise gradients against finite differences for a
pure periodic objective and a hybrid objective with an analytic safety term,
then trains both with exact term gradients. It also records why the hybrid
objective is unsuitable for pure parameter-shift execution.

Before training, use the planner to select the safe route:

```python
from scpn_quantum_control.phase import plan_composed_objective_execution


plan = plan_composed_objective_execution(objective)
print(plan.supported, plan.mode, plan.recommended_entrypoint)
```

Pure periodic objectives route to local parameter-shift descent. Hybrid
objectives with analytic safety penalties route to exact term-gradient descent.
Hardware aliases, unknown backends, and forced pure parameter-shift execution on
hybrid objectives return unsupported plans with explicit blocked reasons.

The QSNN trainer composes with the same route for full-batch quantum neural
network training:

```python
from scpn_quantum_control.qsnn import QuantumDenseLayer, QSNNTrainer

layer = QuantumDenseLayer(1, 1, seed=42)
trainer = QSNNTrainer(layer, lr=0.4)
run = trainer.train_with_parameter_shift_descent(X, y, max_steps=40)

print(run.best_loss, run.certificate.monotone_accepted_values)
```

This gives QSNN notebooks and experiments the same convergence evidence as
phase objectives: backend plan, line-search trace, parameter-shift evaluation
count, final-gradient norm, best loss, and fail-closed hardware boundaries.

For reviewer-facing correctness evidence, run the bundled gradient audit:

```python
import numpy as np

from scpn_quantum_control.phase import (
    run_differentiable_workflow_audit_suite,
    run_finite_shot_gradient_uncertainty_audit,
    run_known_phase_gradient_audit,
    run_ml_framework_gradient_audit,
    run_phase_gradient_benchmark_suite,
)


report = run_known_phase_gradient_audit(np.array([0.8, -0.5, 0.3]))
suite = run_phase_gradient_benchmark_suite()
finite_shot = run_finite_shot_gradient_uncertainty_audit(
    lambda theta: float(np.mean(1.0 - np.cos(theta))),
    np.array([0.7, -0.4, 0.2]),
    target_standard_error=0.02,
)
workflow = run_differentiable_workflow_audit_suite()
ml = run_ml_framework_gradient_audit()

print(report.passed, report.max_gradient_error, report.best_value)
print(suite.passed, suite.benchmark_names, suite.worst_gradient_error)
print(finite_shot.passed, finite_shot.max_standard_error)
print(workflow.passed, workflow.workflow_names)
print(ml.audit_passed, ml.executed_frameworks, ml.unavailable_frameworks)
```

The report combines three independent checks: parameter-shift versus central
finite-difference agreement, parameter-shift versus an analytic closed-form
gradient, and deterministic parameter-shift gradient-descent convergence. It is
designed for local smooth phase-rotation objectives and CI/paper evidence
tables. It does not certify discontinuous losses, shot-noisy hardware
gradients, arbitrary regression models, or undeclared generator spectra.
The suite extends the same report format across single-frequency,
multi-frequency, and coupled-pair phase objectives so evidence tables can show
broader coverage without claiming full arbitrary-program AD or hardware
gradient completeness.
The finite-shot audit adds confidence-containment evidence for declared
plus/minus variances and planned shot budgets. It validates stochastic
uncertainty propagation and shot accounting, while live hardware sampling,
detector drift, and queue calibration remain separate provider-validation
tasks.
The workflow audit is the broadest current supported evidence path: it combines
phase conformance, finite-shot uncertainty, coupling-gradient verification, and
coupling-learning training certificates in one serialisable report.
The ML-framework audit records JAX, PyTorch, TensorFlow, and PennyLane parity
status against the native parameter-shift reference. It executes available
adapters, records missing optional dependencies as unavailable, and blocks
PennyLane execution until a caller supplies a QNode gradient callable.

The shared Phase-QNode circuit, observable, support, execution, gradient, metric, and Fisher record
classes plus registered vocabulary and constructor validation live in the NumPy/stdlib-only
`phase.qnode_circuit_contracts` leaf. The executable `phase.qnode_circuit` facade and phase package
re-export the same public class objects while retaining the established functional API.

For the registered local Phase-QNode subset, use the executable circuit and
parity suite when a concrete circuit family is required:

```python
import numpy as np

from scpn_quantum_control.phase import (
    build_phase_qnode_template,
    build_registered_phase_qnode_circuit,
    build_sparse_ising_chain_hamiltonian,
    decompose_phase_qnode_controlled_gate,
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeDensityCircuit,
    PhaseQNodeNoiseChannel,
    execute_phase_qnode_circuit,
    execute_phase_qnode_density_matrix,
    parameter_shift_phase_qnode_gradient,
    plan_phase_qnode_parameter_shift_evaluations,
    phase_qnode_computational_basis_fisher_support_report,
    phase_qnode_gradient_support_report,
    phase_qnode_metric_support_report,
    run_phase_qnode_framework_parity_suite,
)

circuit = PhaseQNodeCircuit(
    n_qubits=2,
    operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rzz", (0, 1), 1)),
    observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
)
params = np.array([0.2, -0.3], dtype=float)

value = execute_phase_qnode_circuit(circuit, params)
plan = plan_phase_qnode_parameter_shift_evaluations(circuit, params)
gradient = parameter_shift_phase_qnode_gradient(circuit, params)
parity = run_phase_qnode_framework_parity_suite(
    scenario="registered_two_qubit_entangling_statevector"
)
print(value.value, plan.planned_shifted_evaluations, gradient.gradient, parity.frameworks)
```

The registered gate set includes controlled-H/S/T, Toffoli (`ccnot`), CCZ, and
Fredkin (`cswap`) gates. Use `decompose_phase_qnode_controlled_gate(...)` when a
toolchain needs an exact registered operation-list expansion for Toffoli or
Fredkin gates before execution or export:

```python
controlled_ops = decompose_phase_qnode_controlled_gate(("cswap", (0, 1, 2)))
controlled_circuit = PhaseQNodeCircuit(
    n_qubits=3,
    operations=(("x", (0,)), ("x", (1,)), *controlled_ops),
    observable=PauliTerm(1.0, ((2, "z"),)),
)
print(execute_phase_qnode_circuit(controlled_circuit, np.array([])).value)
```

Use `build_sparse_ising_chain_hamiltonian(...)` for larger sparse Pauli
Hamiltonian observables without hand-authoring every term. Scalar coefficients
are broadcast across sites or edges; arrays must match the site or edge count:

```python
sparse_observable = build_sparse_ising_chain_hamiltonian(
    n_qubits=6,
    x_field=np.linspace(0.05, 0.15, 6),
    z_field=np.linspace(0.2, 0.7, 6),
    zz_coupling=np.linspace(0.4, 0.9, 6),
    periodic=True,
)
sparse_circuit = PhaseQNodeCircuit(
    n_qubits=6,
    operations=tuple(("ry", (index,), index) for index in range(6)),
    observable=sparse_observable,
)
sparse_gradient = parameter_shift_phase_qnode_gradient(
    sparse_circuit,
    np.linspace(0.11, 0.61, 6),
)
print(sparse_gradient.gradient)
```

For local mixed-state evidence, use `PhaseQNodeDensityCircuit` and
`execute_phase_qnode_density_matrix(...)`. The density route supports the same
registered unitary gates plus bounded single-qubit Kraus channels:
`bit_flip`, `phase_flip`, `depolarizing`, and `amplitude_damping`. It returns
the density matrix, trace, purity, support report, and claim boundary:

```python
noisy_circuit = PhaseQNodeDensityCircuit(
    n_qubits=1,
    operations=(
        ("x", (0,)),
        PhaseQNodeNoiseChannel("amplitude_damping", (0,), 0.25),
    ),
    observable=PauliTerm(1.0, ((0, "z"),)),
)
noisy = execute_phase_qnode_density_matrix(noisy_circuit, np.array([]))
print(noisy.value, noisy.trace, noisy.purity)
```

Noisy density execution is deterministic local simulator evidence. It is not a
parameter-shift gradient route, not a pure-state Fubini-Study/QFI route, not a
finite-shot estimator, and not provider or hardware evidence.

Preflight pure-state gradient, metric, and exact computational-basis Fisher
routes before execution when circuits may contain density/noise operations or
boundary probabilities:

```python
gradient_report = phase_qnode_gradient_support_report(noisy_circuit, np.array([]))
metric_report = phase_qnode_metric_support_report(noisy_circuit, np.array([]))
fisher_report = phase_qnode_computational_basis_fisher_support_report(
    noisy_circuit,
    np.array([]),
)
print(gradient_report.supported, gradient_report.failure_reason)
print(metric_report.supported, metric_report.failure_reason)
print(fisher_report.supported, fisher_report.failure_reason)
```

The corresponding execution APIs raise `PhaseQNodeSupportError` with the same
report. Classical Fisher support reports also block singular computational-basis
probabilities, so callers can distinguish an unsupported route from a valid
pure-state circuit sitting exactly on a zero-probability boundary.

`parameter_shift_phase_qnode_gradient(...)` returns the same gate-aware
evaluation plan in `PhaseQNodeGradientResult.evaluation_plan`. The
`planned_shifted_evaluations` count excludes the baseline value evaluation,
matching the historical `parameter_shift_evaluations` field; Qiskit
Statevector parity tests count the baseline separately as
`1 + planned_shifted_evaluations`.

Use `build_phase_qnode_template(...)` when the circuit should come from a
registered multi-qubit ansatz rather than hand-authored operations. The current
template registry contains `ghz_chain`, `hardware_efficient_ry`, and
`hardware_efficient_ryrz`; the hardware-efficient templates support chain or
ring CNOT entanglers, return `PhaseQNodeTemplateSpec` metadata with an explicit
claim boundary, and execute through the same `PhaseQNodeCircuit` support report
path:

```python
template = build_phase_qnode_template(
    "hardware_efficient_ryrz",
    n_qubits=3,
    n_layers=2,
    entangler="ring",
)
params = np.linspace(0.2, 0.8, template.parameter_count)
gradient = parameter_shift_phase_qnode_gradient(template.circuit(), params)
```

Use `build_registered_phase_qnode_circuit(...)` when a hand-authored registered
operation sequence needs explicit depth and operation-budget validation. The
returned `PhaseQNodeRegisteredCircuitSpec` carries a `PhaseQNodeDepthProfile`
with ordered qubit-conflict depth, gate counts, parameter count, entangling
pairs, and a claim boundary:

```python
spec = build_registered_phase_qnode_circuit(
    n_qubits=3,
    operations=(
        ("ry", (0,), 0),
        ("rx", (1,), 1),
        ("cnot", (0, 1)),
        ("rz", (2,), 2),
        ("rzz", (1, 2), 3),
    ),
    observable=PauliTerm(1.0, ((0, "z"), (1, "z"), (2, "x"))),
    max_depth=4,
    max_operations=5,
)
```

For dense local observables, use `DenseHermitianObservable(matrix)`.  The matrix
must be finite, square, Hermitian, power-of-two dimensional, and sized to the
declared `n_qubits`; invalid matrices fail during circuit construction instead
of being accepted as opaque simulator inputs.

For covariance objectives, use `PauliCovarianceObservable(left, right)`.  The
local statevector path evaluates the symmetrised covariance
`0.5 <AB + BA> - <A><B>` and differentiates it with the product rule over the
shifted component expectations, rather than applying a naive two-point shift to
the nonlinear covariance scalar.

The circuit family is intentionally bounded: unsupported gates, unregistered
observables, dynamic provider paths, and hardware-backed gradients raise or
record support reports rather than falling back to an interpreter success.

Coupling learning uses the same optimizer for inverse oscillator problems:

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
    np.array([0.0]),
    np.array([[0.0, 0.8], [0.8, 0.0]]),
    rule=multi_frequency_parameter_shift_rule([2.0]),
)

certificate = verify_coupling_parameter_shift_gradient(
    observations,
    np.array([0.0]),
    np.array([[0.0, 0.8], [0.8, 0.0]]),
    rule=multi_frequency_parameter_shift_rule([2.0]),
)

print(run.best_loss, run.max_abs_residual)
print(certificate.passed, certificate.max_abs_error)
```

The result records the initial and learned symmetric coupling matrices,
trainable edge list, target and predicted observations, residuals, optimizer
trace, convergence certificate, and a claim boundary. The supported claim is
sinusoidal or quantum-expectation observation models with declared shift rules;
arbitrary classical regressors and hardware execution remain fail-closed.
The verification certificate records the parameter-shift gradient, independent
central finite-difference gradient, absolute-error vector, maximum error,
evaluation counts, and edge provenance. It is intended for small smooth
diagnostic objectives; discontinuities, shot-noisy hardware runs, and arbitrary
regression models remain outside the certified boundary.

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
parameter-shift calls. Its immutable evidence records live in the dependency-free
`phase.jax_bridge_contracts` leaf; bounded parameter-shift, native-QNN, and custom-VJP QNN
implementations live in `phase.jax_gradients`. Registered local Phase-QNode statevector,
flat/PyTree transform, PMAP-sharding, and AOT/export execution lives in
`phase.jax_qnode_transforms`; bounded-QNN JIT/VMAP/PMAP/PyTree and nested-transform audits live in
`phase.jax_compatibility`; lowering declarations, cloud planning, and maturity aggregation live in
`phase.jax_maturity`. The shallow facade re-exports the records, retains the established public
function signatures, and injects its optional-JAX loader under the same fail-closed boundary:

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

For the bounded phase-QNN classifier, the bridge also exposes a native
custom-VJP route plus audited JIT, VMAP, and PMAP/sharding compatibility
reports, plus bounded PyTree parameter support:

```python
import jax

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    SparsePauliHamiltonian,
    jax_custom_vjp_qnn_value_and_grad,
    jax_phase_qnode_native_transform_audit,
    jax_phase_qnode_pytree_transform_audit,
    jax_phase_qnode_sharding_transform_audit,
    jax_phase_qnode_value_and_grad,
    run_jax_jit_compatibility_audit,
    run_jax_maturity_audit,
    run_jax_nested_transform_algebra_audit,
    run_jax_phase_qnode_lowering_matrix,
    run_jax_pytree_compatibility_audit,
    run_jax_sharding_compatibility_audit,
    run_jax_vmap_compatibility_audit,
)

custom_vjp = jax_custom_vjp_qnn_value_and_grad(
    features,
    labels,
    params,
    jit=True,
)
print(custom_vjp.passed, custom_vjp.custom_vjp, custom_vjp.host_callback)

jit_audit = run_jax_jit_compatibility_audit(
    features=features,
    labels=labels,
    params=params,
)
print(jit_audit.passed, jit_audit.unsupported_native_routes)

vmap_audit = run_jax_vmap_compatibility_audit(
    features=features,
    labels=labels,
    params_batch=np.array([[0.25], [0.45], [0.65]], dtype=float),
)
print(vmap_audit.passed, vmap_audit.batch_size)

sharding_audit = run_jax_sharding_compatibility_audit(
    features=features,
    labels=labels,
    params_batch=np.linspace(
        0.25,
        0.65,
        int(jax.local_device_count()),
        dtype=float,
    ).reshape(int(jax.local_device_count()), 1),
)
print(sharding_audit.passed, sharding_audit.sharding_mode)

pytree_audit = run_jax_pytree_compatibility_audit(
    features=np.array([[0.0, 0.2, 0.4], [np.pi, np.pi + 0.2, np.pi + 0.4]]),
    labels=labels,
    params_pytree={
        "encoder": np.array([0.25, 0.45], dtype=float),
        "readout": {"phase": np.array([0.65], dtype=float)},
    },
)
print(pytree_audit.passed, pytree_audit.leaf_shapes)

nested_audit = run_jax_nested_transform_algebra_audit(
    features=features,
    labels=labels,
    params_batch=np.array([[0.25], [0.45]], dtype=float),
    params_pytree={"phase": params},
)
print(nested_audit.passed, nested_audit.ready_for_provider_exceedance)

lowering_matrix = run_jax_phase_qnode_lowering_matrix()
print(lowering_matrix.route_status("registered_phase_qnode_statevector_lowering"))
print(lowering_matrix.route_status("registered_phase_qnode_native_transform_lowering"))
print(lowering_matrix.route_status("registered_phase_qnode_pytree_transform_lowering"))
print(lowering_matrix.route_status("registered_phase_qnode_pmap_sharding_lowering"))

registered_circuit = PhaseQNodeCircuit(
    n_qubits=2,
    operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rz", (1,), 1)),
    observable=SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"), (1, "z"))),)),
)
registered_jax = jax_phase_qnode_value_and_grad(
    registered_circuit,
    np.array([0.17, -0.23], dtype=float),
    jit=True,
)
print(registered_jax.passed, registered_jax.host_callback, registered_jax.jitted)

registered_transforms = jax_phase_qnode_native_transform_audit(
    registered_circuit,
    np.array([0.17, -0.23], dtype=float),
)
print(registered_transforms.passed, registered_transforms.transform_names)

registered_pytree_transforms = jax_phase_qnode_pytree_transform_audit(
    registered_circuit,
    {
        "parameter_0": np.array([0.17], dtype=float),
        "parameter_1": (np.array([-0.23], dtype=float),),
    },
)
print(registered_pytree_transforms.passed, registered_pytree_transforms.transform_names)

registered_sharding_transforms = jax_phase_qnode_sharding_transform_audit(
    registered_circuit,
    np.tile(
        np.array([[0.17, -0.23]], dtype=float),
        (int(jax.local_device_count()), 1),
    ),
)
print(registered_sharding_transforms.passed, registered_sharding_transforms.sharding_mode)

maturity = run_jax_maturity_audit(
    features=features,
    labels=labels,
    params=params,
    params_batch=np.array([[0.25], [0.45]], dtype=float),
    params_pytree={"phase": params},
)
print(maturity.bounded_model_ready, maturity.ready_for_provider_exceedance)
```

`jax_custom_vjp_qnn_value_and_grad(...)` registers a JAX `custom_vjp` for the
bounded classifier loss and checks the returned gradient against the canonical
multi-frequency SCPN parameter-shift gradient.
`run_jax_jit_compatibility_audit(...)` JITs the bounded native JAX and
custom-VJP routes, requires both to stay no-host-callback routes, and records the
generic parameter-shift bridge as host-callback interop through
`unsupported_native_routes`. `run_jax_vmap_compatibility_audit(...)` VMAPs the
same bounded native and custom-VJP routes over parameter batches, verifies each
row against SCPN parameter-shift references, and records those references as a
host-side loop rather than native VMAP.
`run_jax_sharding_compatibility_audit(...)` uses `jax.pmap` with one parameter
row per local JAX device and records whether the evidence is single-device or
multi-device. `run_jax_pytree_compatibility_audit(...)` accepts nested numeric
parameter PyTrees, flattens them into the bounded phase-QNN parameter vector,
and restores gradients to the same tree structure. These remain bounded model
routes: they are not arbitrary simulator autodiff, not provider-backed
execution, not hardware gradients, and not claims that every quantum objective
can be lowered into JAX JIT, VMAP, distributed sharding, or arbitrary PyTree
programs.
`run_jax_nested_transform_algebra_audit(...)` verifies bounded
`JIT(value_and_grad)` under VMAP, `JIT(VMAP(value_and_grad))`, and
JIT-wrapped PyTree value-and-gradient routes against SCPN parameter-shift
references. The same record keeps arbitrary Phase-QNode JAX lowering,
arbitrary `jacfwd`/`jacrev`, arbitrary Hessian algebra,
hardware/provider-callback transform safety, and isolated-benchmark promotion
blocked until dedicated artefacts exist.
`jax_phase_qnode_value_and_grad(...)` lowers registered deterministic local
`PhaseQNodeCircuit` statevector execution into native JAX `value_and_grad`,
enables JAX x64 for complex statevector fidelity, optionally wraps the route in
`jax.jit`, and reports `host_callback=False`. It validates the JAX value and
gradient against the canonical SCPN statevector value and gate-aware
parameter-shift gradient. `jax_phase_qnode_native_transform_audit(...)` runs the
same registered local statevector value function through native JAX `grad`,
`value_and_grad`, `jacfwd`, `jacrev`, `hessian`, `jvp`, `vjp`, `vmap`, and
`jit`, compares first-order and batched gradients against SCPN parameter-shift
references, checks JVP/VJP contractions and Hessian symmetry, and reports
`host_callback=False`. `jax_phase_qnode_pytree_transform_audit(...)` accepts
nested numeric PyTree parameters for the same registered local circuit family,
checks native JAX `grad`, `value_and_grad`, `jacfwd`, `jacrev`, `jvp`, `vjp`,
`hessian`, `vmap`, and `jit` against SCPN parameter-shift references, restores
gradients to the caller's PyTree structure, records flattened Hessian symmetry
evidence, and reports `host_callback=False`.
`jax_phase_qnode_sharding_transform_audit(...)` maps one registered local
statevector value-and-gradient row per local JAX device through `jax.pmap`,
checks every row against SCPN parameter-shift references, labels single-device
CPU runs as pmap smoke evidence, and reports `host_callback=False`.
`run_jax_phase_qnode_lowering_matrix(...)` makes the
native-lowering boundary explicit: bounded QNN native, custom-VJP, JIT, VMAP,
PyTree, registered deterministic statevector, registered deterministic
native-transform, and registered deterministic pmap/sharding routes are listed
as no-host-callback passes, while finite-shot, provider, hardware, and
dynamic-circuit JAX lowering remain blocked until their own policy and parity
artefacts exist.
`run_jax_maturity_audit(...)` is the provider-parity gate for JAX: it aggregates
the bounded passes and emits explicit blockers for full arbitrary
`jacfwd`/`jacrev`/Hessian transform algebra, finite-shot/provider/hardware
routes, hardware/provider callback transform safety, and promotion-grade
isolated benchmark evidence.

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
kernels. Multi-frequency rules are supported through the same host-boundary
path, and the JAX result reports the native method and shift-term count.

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

This caller-supplied path fails closed when PennyLane is not importable and
reports explicit gradient error metrics when the external gradient disagrees.
When a multi-frequency rule is supplied, the report records the native SCPN
method and shift-term count so the comparison cannot be mistaken for the legacy
two-point rule.

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

For registered local `PhaseQNodeCircuit` declarations, SCPN can also generate a
bounded PennyLane QNode and verify the generated value/gradient route:

```python
from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    build_pennylane_qnode_from_phase_qnode,
    check_pennylane_phase_qnode_round_trip,
    run_pennylane_maturity_audit,
)

phase_qnode = PhaseQNodeCircuit(
    1,
    (("ry", (0,), 0), ("rx", (0,), 1)),
    PauliTerm(1.0, ((0, "z"),)),
)
conversion = build_pennylane_qnode_from_phase_qnode(
    phase_qnode,
    device_name="default.qubit",
    shots=None,
    diff_method="parameter-shift",
)
generated = check_pennylane_phase_qnode_round_trip(
    phase_qnode,
    np.array([0.4, -0.2]),
)
maturity = run_pennylane_maturity_audit(
    objective=scpn_objective,
    pennylane_objective=scpn_objective,
    pennylane_gradient=pennylane_gradient,
    values=np.array([0.4]),
    circuit=phase_qnode,
    phase_qnode_values=np.array([0.4, -0.2]),
)
print(generated.passed, conversion.device_name, conversion.shots)
print(maturity.identical_circuit_ready, maturity.ready_for_provider_exceedance)
```

The generated route records device, shot policy, interface, diff method, gates,
observable family, and differentiable parameter indices. Its claim boundary is
limited to registered static gates and direct expectation observables with
PennyLane equivalents, including CH, Toffoli, Fredkin, and controlled phase
equivalents for CS/CT/CCZ where the optional dependency is installed. Provider
submission, hardware execution, dynamic circuits, noise models, and covariance
observable conversion remain explicit non-claims.
Device metadata is trimmed and rejected when empty or when it contains control
characters before `qml.device(...)` or QNode construction is reached. Interface
and diff-method metadata is constrained to canonical PennyLane interfaces
(`auto`, `autograd`, `jax`, `tf`, `torch`) and documented QNode diff methods
(`adjoint`, `backprop`, `best`, `device`, `finite-diff`, `hadamard`,
`parameter-shift`, `spsa`) before QNode construction.
`scpn_quantum_control.phase.pennylane_provider_plugin` owns the provider-plugin
artefact types and fail-closed plugin matrix; `pennylane_bridge` re-exports the
same objects for compatibility with older imports.
`PennyLaneProviderPluginExecutionArtifact` validates provider-plugin execution
metadata with non-empty plugin/provider/device/backend identities, positive
shots when present, SHA-256 result and metadata digests, optional replay
metadata, explicit PennyLane `interface`, `diff_method`, and `shot_policy`
metadata, canonical interface values (`auto`, `autograd`, `jax`, `tf`, `torch`)
and documented QNode diff methods (`adjoint`, `backprop`, `best`, `device`,
`finite-diff`, `hadamard`, `parameter-shift`, `spsa`) instead of undocumented
aliases, `shot_policy="analytic"` with `shots=None` or
`shot_policy="finite_shot"` with a positive shot count,
non-hardware execution mode, and `hardware_execution=False`.
`run_pennylane_plugin_matrix(...)` records local `default.qubit` exact-state,
shot-policy metadata, generated Phase-QNode export, and supported tape-import
routes as passed. Passing a validated provider execution artefact marks
`provider_plugin_execution` as passed, and a matching
`PennyLaneProviderGradientParityArtifact` marks provider-gradient parity as
passed only when provider identity, circuit fingerprint, PennyLane interface,
diff method, and shot policy match. Hardware-plugin execution remains blocked
until its own ticketed artefact is attached, and isolated-benchmark promotion
remains blocked with required artefacts listed per route.
Passing a `PennyLaneProviderEvidenceBundle` keeps provider execution,
provider-gradient parity, and optional ticketed hardware execution in one
exclusive attachment. The bundle requires explicit UTC capture/expiry metadata,
rejects inverted freshness windows, rejects hardware evidence whose provider,
circuit fingerprint, or shot count no longer matches the provider execution
chain, rejects provider-gradient parity whose interface, diff method, or shot
policy drifts from the provider execution chain, and fails closed when the
bundle has expired at the review cutoff.
Passing a validated `PennyLaneHardwarePluginExecutionArtifact` marks only
`hardware_plugin_execution` as passed; it must carry ticket, allowlist,
shot-budget, hardware evidence, raw-count, calibration digest,
calibration capture/expiry timestamps, and metadata provenance before the route
opens. Stale calibration metadata fails closed at the review cutoff before the
hardware-plugin route can pass.
`run_pennylane_maturity_audit(...)` combines caller-supplied gradient agreement,
caller-supplied QNode round-trip parity, generated Phase-QNode export parity,
optional PennyLane tape import parity, device metadata, shot policy, diff
method, grouped registered Phase-QNode parameter-shift evaluation counts,
optional provider execution, provider-gradient parity, and hardware execution
artefacts, and the plugin matrix. The audit can mark
`identical_circuit_ready=True` only when a
PennyLane import tape is supplied and every bounded route passes. It keeps
provider exceedance blocked until promotion-grade isolated benchmark artefacts
exist.

### Importing a PennyLane tape

The inverse direction reads a `pennylane.tape.QuantumScript` and builds the
equivalent registered `PhaseQNodeCircuit`. Every gate parameter becomes a
Phase-QNode parameter in tape order, and a single Pauli-word expectation maps to
the registered observable family.

```python
import pennylane as qml
from scpn_quantum_control.phase import (
    import_phase_qnode_from_pennylane,
    check_pennylane_phase_qnode_import_round_trip,
)

tape = qml.tape.QuantumScript(
    [qml.Hadamard(0), qml.RX(0.6, wires=0), qml.CNOT([0, 1]), qml.CRY(0.9, wires=[0, 1])],
    [qml.expval(qml.PauliZ(0) @ qml.PauliX(1))],
)
imported = import_phase_qnode_from_pennylane(tape)        # -> PhaseQNodeCircuit + parameters
verdict = check_pennylane_phase_qnode_import_round_trip(tape)
print(verdict.value_match, verdict.gradient_match)
```

`check_pennylane_phase_qnode_import_round_trip` executes the source tape and the
imported circuit and compares both the expectation value and the parameter-shift
gradient — the gradient comparison restricts the PennyLane tape to its gate
parameters so observable (Hamiltonian) coefficients are not differentiated, and
it independently confirms the four-term controlled-rotation rule against
PennyLane's own gradient. Import is fail-closed: gates outside the registered
set, multi-parameter gates (for example `Rot`), non-integer or non-contiguous
wires, multiple or non-expectation measurements, and non-Pauli or identity
observables are rejected. Non-finite imported gate parameters and invalid
value/gradient tolerances are rejected before the round-trip comparison runs.
Mid-circuit measurement, channel, template, provider, and hardware import
remain explicit non-claims.

## Optional framework tensor bridges and cloud validation plans

The 19 immutable Torch result, route, evidence, matrix, and cloud-plan records live in the
NumPy/stdlib-only `phase.torch_bridge_contracts` leaf. The executable bridge and phase package
re-export the same class objects. Optional Torch loading, numeric/tensor validation, host
parameter-shift conversion, analytic bounded-QNN gradients, and custom-autograd bounded-QNN
gradients live in `phase.torch_gradients`; the facade retains the established public signatures
and injects its active loader. Deterministic registered Phase-QNode statevector execution,
`torch.func` transforms, `torch.compile` diagnostics, and compiler-boundary routes live in the
one-way `phase.torch_qnode_transforms` leaf under the same facade boundary. Bounded phase-QNN
`torch.func`/`torch.compile` compatibility, module/layer construction and auditing, and the
deterministic compiled training loop live in the one-way `phase.torch_compatibility` leaf.
Lowering declarations, CUDA/ecosystem diagnostics, cloud planning, live-overlay validation, and
maturity aggregation live in the one-way `phase.torch_maturity` orchestration leaf.

For ML pipelines that need framework tensors, the phase namespace exposes
host-boundary adapters, deterministic registered statevector routes, and
non-promotional cloud-validation plans for accelerator routes blocked by local
hardware:

```python
import numpy as np

from scpn_quantum_control.phase import (
    plan_jax_cloud_validation_batch,
    run_tensorflow_function_compatibility_audit,
    run_tensorflow_gradient_tape_compatibility_audit,
    run_tensorflow_keras_layer_wrapper_audit,
    run_tensorflow_maturity_audit,
    run_tensorflow_xla_compatibility_audit,
    plan_torch_cloud_validation_batch,
    run_torch_ecosystem_maturity_audit,
    run_torch_maturity_audit,
    run_torch_phase_qnode_lowering_matrix,
    run_torch_autograd_function_audit,
    run_torch_export_shape_matrix,
    run_torch_dynamic_shape_export_audit,
    run_torch_aot_autograd_export_audit,
    run_torch_training_loop_audit,
    run_torch_training_loop_matrix,
    tensorflow_bounded_qnn_keras_layer,
    tensorflow_parameter_shift_value_and_grad,
    torch_phase_qnode_compile_audit,
    torch_phase_qnode_transform_audit,
    torch_phase_qnode_value_and_grad,
    torch_parameter_shift_value_and_grad,
)


def objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


torch_result = torch_parameter_shift_value_and_grad(objective, np.array([0.4]))
tf_result = tensorflow_parameter_shift_value_and_grad(objective, np.array([0.4]))
tf_tape_audit = run_tensorflow_gradient_tape_compatibility_audit(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    params=np.array([0.45], dtype=float),
)
tf_function_audit = run_tensorflow_function_compatibility_audit(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    params=np.array([0.45], dtype=float),
)
tf_xla_audit = run_tensorflow_xla_compatibility_audit(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    params=np.array([0.45], dtype=float),
)
tf_keras_layer = tensorflow_bounded_qnn_keras_layer(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    initial_params=np.array([0.45], dtype=float),
)
tf_keras_audit = run_tensorflow_keras_layer_wrapper_audit(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    initial_params=np.array([0.45], dtype=float),
)
tf_maturity = run_tensorflow_maturity_audit(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    params=np.array([0.45], dtype=float),
)
torch_maturity = run_torch_maturity_audit(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    params=np.array([0.45], dtype=float),
    params_batch=np.array([[0.25], [0.45], [0.65]], dtype=float),
)
torch_training_loop = run_torch_training_loop_audit(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    initial_params=np.array([0.45], dtype=float),
)
torch_autograd_function_audit = run_torch_autograd_function_audit(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    initial_params=np.array([0.45], dtype=float),
)
torch_training_loop_matrix = run_torch_training_loop_matrix()
torch_export_shape_matrix = run_torch_export_shape_matrix(
    export_dir=Path("bounded_phase_qnn_export_shapes"),
)
torch_dynamic_shape_export = run_torch_dynamic_shape_export_audit(
    export_path=Path("bounded_phase_qnn_dynamic_shape.pt2"),
)
torch_aot_autograd_export = run_torch_aot_autograd_export_audit(
    features=np.array([[0.0], [np.pi]], dtype=float),
    labels=np.array([0.0, 1.0], dtype=float),
    initial_params=np.array([0.45], dtype=float),
    artifact_dir=Path("bounded_phase_qnn_aot_autograd"),
)
torch_ecosystem = run_torch_ecosystem_maturity_audit()
torch_lowering = run_torch_phase_qnode_lowering_matrix()
jax_cloud_batch = plan_jax_cloud_validation_batch(runner="jarvislabs")
torch_cloud_batch = plan_torch_cloud_validation_batch(runner="jarvislabs")

print(torch_result.torch_gradient, torch_result.host_boundary)
print(torch_training_loop.final_loss, torch_training_loop.passed)
print(torch_autograd_function_audit.gradient_shape, torch_autograd_function_audit.open_gaps)
print(torch_training_loop_matrix.scenario_count, torch_training_loop_matrix.passed)
print(torch_export_shape_matrix.scenario_count, torch_export_shape_matrix.open_gaps)
print(torch_dynamic_shape_export.batch_sizes, torch_dynamic_shape_export.open_gaps)
print(torch_aot_autograd_export.gradient_shape, torch_aot_autograd_export.open_gaps)
print(torch_maturity.bounded_model_ready, torch_maturity.ready_for_provider_exceedance)
print(torch_ecosystem.route_status("cuda_accelerator_device"))
print(torch_lowering.route_status("registered_phase_qnode_statevector_lowering"))
print(torch_lowering.route_status("registered_phase_qnode_torch_func_transform_lowering"))
print(jax_cloud_batch.local_execution_status, jax_cloud_batch.required_artifacts)
print(torch_cloud_batch.local_execution_status, torch_cloud_batch.required_artifacts)
print(tf_maturity.bounded_model_ready, tf_maturity.ready_for_provider_exceedance)
print(tf_result.tensorflow_gradient, tf_result.host_boundary)
```

Both parameter-shift adapters import the optional framework only when called,
run SCPN's deterministic parameter-shift rule on the host, and return NumPy plus
framework tensor payloads. Multi-frequency rules preserve the native method and
shift-term count in the adapter result. The separate
`torch_autograd_qnn_value_and_grad(...)` route is native PyTorch autograd only
for the bounded phase-QNN model. The separate
`run_torch_autograd_function_audit(...)` route checks the promoted custom
`torch.autograd.Function` loss through direct `Tensor.backward()` and
`torch.optim.SGD` integration, with higher-order autograd, CUDA, provider,
hardware, arbitrary-simulator, isolated-benchmark, and performance routes kept
blocked. The separate
`torch_phase_qnode_value_and_grad(...)` route lowers deterministic registered
local Phase-QNode statevector execution into native PyTorch autograd without
host callbacks and checks the value and gradient against the SCPN
parameter-shift reference. It is not finite-shot, provider-backed, dynamic, or
hardware lowering evidence. The separate `torch_phase_qnode_transform_audit(...)`
route runs registered local Phase-QNode statevector execution through
`torch.func.grad`, `torch.func.jacrev`, and `torch.func.vmap`, checking single
and batched gradients against SCPN parameter-shift references without host
callbacks. The separate
`torch_phase_qnode_compile_audit(...)` route compiles both the registered local
Phase-QNode statevector value function and its `torch.func.grad` gradient
function through non-fullgraph `torch.compile` on CPU, then checks value and
gradient against SCPN parameter-shift references. Fullgraph `torch.compile`,
CUDA/device compile, provider, finite-shot, hardware, and performance claims
remain outside that route. The separate
`run_torch_func_compatibility_audit(...)` route verifies `torch.func.grad`,
`vmap`, and `jacrev` only for the same bounded model. The separate
`run_torch_compile_compatibility_audit(...)` route verifies compiled bounded-loss
gradients only for that same model. The separate
`torch_bounded_qnn_module(...)` / `torch_bounded_qnn_layer(...)` wrapper route
verifies a bounded PyTorch `nn.Module`/layer loss and gradient only for that same
model. `run_torch_training_loop_audit(...)` compiles that bounded module loss,
uses `torch.func.grad` for deterministic gradient-descent updates, records loss
and gradient histories, and checks every gradient route against SCPN
parameter-shift references. It is local training-loop correctness evidence, not
CUDA, provider, finite-shot, hardware, isolated benchmark, or performance
promotion evidence. `run_torch_training_loop_matrix(...)` expands that route
into deterministic bounded one- and two-parameter scenarios, records loss
descent, parameter-update norm, compile-mode coverage, and gradient parity, and
keeps CUDA, provider/hardware, arbitrary-architecture, isolated benchmark, and
performance routes blocked. `run_torch_module_state_audit(...)` separately validates
strict module `state_dict` replay and Adam optimizer-state replay on local
CPU-compatible tensors, while `validate_torch_bounded_qnn_state_dict(...)`
checks candidate keys, shapes, and dtypes without loading them.
`run_torch_module_device_state_audit(...)` checks CPU `module.to(...)` state
replay and attempts CUDA `module.to(...)` state replay only after the installed
PyTorch runtime passes a real CUDA smoke. `run_torch_module_checkpoint_audit(...)`
writes a real `torch.save` checkpoint and reloads it on CPU with
`weights_only=True` before strict module plus Adam optimizer-state replay.
`run_torch_long_lived_checkpoint_matrix(...)` records the checkpoint schema,
tensor metadata manifest, runtime fingerprint, and repeated local CPU
weights-only loads without promoting cross-runtime, CUDA, or external
checkpoint-corpus replay.
`run_torch_module_export_audit(...)` exports the same bounded module with
`torch.export.export(...)`, saves and reloads the `ExportedProgram`, and replays
the local CPU value route through `ExportedProgram.module()`. Incompatible CUDA,
gradient export for this `torch.export` route, dynamic feature-width export
promotion, and cross-runtime checkpoint/export portability remain blocked until
dedicated artefacts exist. `run_torch_export_shape_matrix(...)` records multiple
static feature shapes as separate export artifacts.
`run_torch_dynamic_shape_export_audit(...)` adds the dedicated input-driven
dynamic-batch value route while keeping CUDA, cross-runtime export portability,
provider, hardware, isolated-benchmark, and performance promotion blocked.
`run_torch_aot_autograd_export_audit(...)` captures local AOTAutograd
forward/backward FX graphs, saves and reloads the self-produced PyTorch
artifacts, and replays the loaded backward graph against the SCPN
parameter-shift gradient; cross-runtime execution, CUDA replay, dynamic-shape
AOTAutograd export, isolated-benchmark, and performance promotion remain
blocked. The separate `run_torch_ecosystem_maturity_audit(...)` route records
installed `nn.Module`/`Parameter`, `torch.func`, `torch.compile`, and CUDA-device
capability state. A visible CUDA device is still blocked if the installed
PyTorch wheel cannot execute a tensor smoke on that hardware. Registered
Phase-QNode non-fullgraph `torch.compile` lowering is available for
deterministic local statevector circuits, while fullgraph lowering remains
blocked on PyTorch Dynamo symbolic-shape extraction. The separate
`run_torch_maturity_audit(...)` route aggregates those bounded PyTorch passes
and ecosystem routes into a provider-maturity record. When called with
`live_overlay_artifact_path`, it validates the external-comparison JSON and only
marks live overlay execution passed when that artefact contains a successful
PyTorch row with dependency version, value/gradient error, runtime, memory, and
claim-boundary metadata. Provider exceedance still remains blocked until
registered Phase-QNode fullgraph `torch.compile` lowering, compatible CUDA/device
evidence, finite-shot/provider/hardware Phase-QNode Torch lowering, full
compiler/autograd integration, and promotion-grade isolated benchmark artefacts
exist. The aggregate audit also includes the
`plan_torch_cloud_validation_batch(...)` run spec, which records local
CUDA/device skip reasons, blocked PyTorch Phase-QNode routes, required
JarvisLabs/cloud artefacts, and reproduction commands for the deferred
accelerator validation batch. The separate
`plan_jax_cloud_validation_batch(...)` run spec records local JAX device
metadata, GTX 1060 or single-device skip reasons, blocked JAX accelerator and
PMAP routes, required CUDA/XLA/pmap and isolated-benchmark artefacts, and
reproduction commands for the deferred JarvisLabs/cloud validation batch. Both
cloud plans are scheduling/runbook evidence only; they do not submit network or
hardware jobs and do not promote GPU, multi-device, hardware, or performance
claims without returned artefacts. The separate
`run_torch_phase_qnode_lowering_matrix(...)` route makes that boundary explicit:
bounded QNN tensor, custom-autograd, `torch.func`, `torch.compile`, and
module/layer routes plus deterministic registered statevector, `torch.func`
transform, and non-fullgraph `torch.compile` Phase-QNode lowering are marked
passed, while registered Phase-QNode fullgraph `torch.compile` lowering,
CUDA/device lowering, finite-shot lowering, provider callbacks, hardware
lowering, dynamic-circuit lowering, and isolated-benchmark promotion remain
blocked with the required artefacts listed in the returned route metadata. The
separate
`run_tensorflow_gradient_tape_compatibility_audit(...)`
route verifies TensorFlow `GradientTape` only for the same bounded classifier
loss and checks the returned gradient against the SCPN parameter-shift
reference. The separate `run_tensorflow_function_compatibility_audit(...)`
route traces only that same bounded loss through `tf.function` and checks its
`GradientTape` gradient against the same reference. The separate
`run_tensorflow_xla_compatibility_audit(...)` route requests
`tf.function(jit_compile=True)` only for that same bounded loss. The separate
`tensorflow_bounded_qnn_keras_layer(...)` and
`run_tensorflow_keras_layer_wrapper_audit(...)` routes expose only that same
bounded loss through a Keras `Layer` and checks its `GradientTape` gradient
against the same reference. `run_tensorflow_phase_qnode_lowering_matrix(...)`
then records the route-level boundary: bounded tensor, `GradientTape`,
`tf.function`, XLA, and Keras-layer routes are marked passed, while arbitrary
registered Phase-QNode statevector lowering, graph autodiff, finite-shot
lowering, provider callbacks, hardware lowering, dynamic circuits, and
isolated-benchmark promotion remain blocked with their required artefacts
listed in the returned route metadata. `run_tensorflow_maturity_audit(...)`
aggregates the bounded TensorFlow evidence plus that matrix into a
provider-maturity record while keeping arbitrary Phase-QNode TensorFlow
lowering, full graph autodiff-through-simulator, provider callbacks, hardware
gradients, and isolated benchmark promotion blocked.
`run_tensorflow_maintenance_decision()` records the adopted maintenance
strategy: TensorFlow stays maintained for bounded compatibility routes only.
Broad Graph/XLA parity, arbitrary TensorFlow simulator autodiff, provider
callbacks, hardware gradients, unrestricted Keras training-loop coverage, and
performance promotion remain blocked until their route-specific artefacts
exist.

## Enzyme/MLIR compiler maturity audit

The compiler lane is audited with
`run_enzyme_mlir_maturity_audit(...)`. The result is JSON-ready and separates
four evidence classes:

- verified SCPN MLIR-runtime execution for a registered local Phase-QNode value
  and parameter-shift gradient;
- the bounded in-process native LLVM/JIT surface already available for
  supported compiler/program AD kernels;
- local `enzyme`, `opt`, `mlir-opt`, and `clang` command/version metadata when
  the compiler stack is installed;
- attached native Enzyme execution evidence as either a successful correctness
  row or a named runtime hard gap;
- attached MLIR/LLVM correctness evidence for the bounded SCPN MLIR-runtime and
  native LLVM/JIT support snapshot;
- attached compiler-AD breadth evidence covering scalar forward/reverse mode,
  vector JVP/VJP, matrix JVP/VJP, loop activity, alias activity, MLIR lowering,
  LLVM IR generation, native Enzyme execution, and matching isolated benchmark
  metadata;
- attached raw `EnzymeMLIRCompilerADBreadthArtifact` case rows covering exactly
  the required scalar, vector, matrix, loop/activity, MLIR, LLVM, and native
  Enzyme routes before the derived breadth evidence can be accepted;
- `build_enzyme_mlir_compiler_ad_breadth_gap_artifact(...)` for partial raw
  captures, which fills every absent route as a named `hard_gap` row and exposes
  `failed_case_ids` instead of allowing sliced evidence to look complete;
- attached `EnzymeMLIRBenchmarkAttachment` built from
  `PhaseQNodeAffinityArtifactValidation`, so the isolated benchmark gate is
  satisfied only by promotion-ready `isolated_affinity` evidence with raw timing
  rows and complete host metadata;
- hard gaps for missing toolchains when they are absent, failed native Enzyme
  execution, missing derived compiler-AD breadth evidence, missing isolated
  benchmark artefacts, raw breadth case failures, and missing validated
  isolated benchmark attachments.

```python
from scpn_quantum_control import run_enzyme_mlir_maturity_audit

audit = run_enzyme_mlir_maturity_audit()
assert audit.ready_for_provider_exceedance is False
```

This audit is intentionally stricter than the diagnostic external-comparison
row. A local SCPN MLIR-runtime pass is executable compiler evidence, but it is
not an Enzyme parity claim. The committed artefact
`data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json`
attaches the current MLIR/LLVM correctness snapshot and a bounded native LLVM
Enzyme scalar derivative probe plus
`data/differentiable_phase_qnode/enzyme_mlir_compiler_ad_breadth_artifact_20260706.json`,
which records a complete 11-case raw breadth artifact with explicit hard-gap
rows for missing raw cases. The separate Enzyme-JAX external-comparison row
remains a runtime hard gap. Provider-exceedance remains blocked until
`isolated_affinity` benchmark artefacts are validated through
`EnzymeMLIRBenchmarkAttachment` with correctness, native Enzyme execution, and
raw-plus-derived compiler-AD breadth evidence. Partial breadth captures are
represented as complete artifacts with explicit case hard gaps. A string
benchmark ID without the validated attachment is still a hard gap.

The maturity audit resolves live `enzyme`, `opt`, `mlir-opt`, and `clang`
commands to absolute executable files before it captures version metadata.
Relative or non-executable tool paths are recorded as unavailable toolchain
evidence instead of being passed to a subprocess. Synthetic `toolchain_probe`
and `version_probe` callbacks remain available for deterministic tests and
pre-collected evidence rows where no subprocess is launched.

The real Enzyme/LLVM execution runner resolves `clang` and `opt` to executable
absolute paths before any compiler subprocess is started, and it requires
`SCPN_ENZYME_PLUGIN`, when set, to point at an absolute existing
`LLVMEnzyme-*.so` file. Invalid overrides fail closed instead of falling back to
another local plugin. The generated native executable is also validated before
it is run, and every subprocess is launched without a shell. Missing or invalid
tooling is recorded as `hard_gap` evidence, not as an execution success.

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
- materialised score-function likelihood-ratio estimator;
- unsupported fail-closed mode.

Each gradient plan reports the selected method, backend, shots, seed, estimator
uncertainty policy, unsupported alternatives, and fail-closed reasons. Use
`explain_quantum_gradient_method(...)` when integration code needs the full
deterministic explanation object: selected plan, rejected methods with reasons,
shot policy, and safe fallback path from one API call.
Stochastic estimator results additionally carry a
`StochasticGradientConfidenceInterval` and `failure_policy_status`; the helper
`gradient_confidence_interval(...)` can evaluate the same fail-closed policy
against materialised gradients and standard errors without rerunning an
objective.

The implemented SPSA diagnostic route is available as
`spsa_gradient_estimate(...)` in the differentiable module. It draws seeded
Bernoulli perturbations, evaluates caller-provided plus/minus objective probes,
records every probe pair, and returns gradient, standard-error, diagonal
covariance, confidence-radius, evaluation-count, shot-count, and claim-boundary
metadata plus confidence-interval and failure-policy status. When `shots` is supplied, the objective must return
`SPSAObjectiveSample` values with finite variances and positive shot counts so
the estimator can propagate finite-shot uncertainty. The optional Rust parity
kernel `spsa_gradient_rust(...)` validates and reproduces the same uncertainty
calculation from materialised SPSA records. `gradient_confidence_interval_rust(...)`
reproduces the interval and failure-policy calculation from materialised
gradient and standard-error arrays; neither kernel executes objectives,
providers, or hardware jobs.

The implemented score-function route is available as
`score_function_gradient_estimate(...)`. It applies the likelihood-ratio
identity only when finite scalar rewards and finite score vectors are already
materialised. The result records each weighted score sample, the explicit
baseline, empirical covariance, standard errors, confidence radii, trainable
mask, confidence-interval status, failure-policy reasons, and claim boundary.
The optional Rust parity kernel
`score_function_gradient_rust(...)` validates the same materialised rewards and
score vectors and reproduces the Python uncertainty calculation. This is not
sampler autodiff and not an arbitrary discrete-program gradient.

## Suitable and unsuitable scenarios

| Scenario | Status |
|---|---|
| Small Pauli-rotation expectation objective | Suitable for parameter-shift. |
| Gradient-trained Kuramoto-XY VQE | Current implementation route; convergence evidence must be attached. |
| Noisy finite-shot backend | Supported for uncertainty propagation when plus/minus variances and shots are supplied. |
| Seeded local SPSA diagnostic | Supported for caller-supplied objectives when perturbation radius, repetitions, seed, sample variances, and shot counts satisfy the estimator contract. |
| Materialised score-function diagnostic | Supported when finite rewards and score vectors are supplied by a mathematically valid likelihood-ratio model. |
| Confidence-policy gate | Supported for materialised stochastic-gradient standard errors with active trainable parameters and positive thresholds. |
| Hardware execution | Must remain disabled by default until a hardware-safe gradient policy exists. |
| Gate without registered generator spectrum | Unsupported; fail closed. |
| Dynamic circuit topology or parameter count | Unsupported unless the trace records stable parameter identity. |
| Roadmap adapters | JAX, PyTorch, TensorFlow, PennyLane, and Qiskit require parity tests before production claims. |

## Phase QNode tape readiness

`phase_qnode_tape(...)` is the supported QNode-style record layer for phase
objectives. It wraps existing parameter-shift and finite-shot tape contracts and
adds the metadata users expect from a trainable quantum node: QNode name,
observable, backend plan, parameter-shift evaluation count, shot budget, replay
seed, confidence radii, provider name, requested job identifier, and a claim
boundary.

The readiness helper `run_phase_qnode_tape_readiness_suite()` records three
representative routes: deterministic local parameter shift, seeded finite-shot
replay, and a hardware/provider boundary that fails closed before submission.
This is differentiable execution evidence for supported phase objectives, not a
claim of arbitrary QNode autodiff, native framework tracing through simulator
kernels, or unrestricted provider-backed hardware gradients.

## Vector QNode directional transforms, Jacobians, and native manual vmap

`execute_phase_qnode_vector_jacobian(...)` extends the deterministic local QNode
transform route to one-dimensional vector outputs. It evaluates each output
component through the same parameter-shift rule used by scalar objectives and
returns a `(output_dim, n_params)` Jacobian with explicit evaluation accounting.

```python
import numpy as np
from scpn_quantum_control.phase import (
    execute_phase_qnode_vector_jacobian,
    execute_phase_qnode_vector_jvp,
    execute_phase_qnode_vector_vjp,
)


def vector_objective(params: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.cos(params[0]) + 0.1 * np.sin(params[1]),
            np.sin(params[0]) - 0.25 * np.cos(params[1]),
        ]
    )


result = execute_phase_qnode_vector_jacobian(
    "jacfwd",
    vector_objective,
    np.array([0.2, -0.4]),
)
print(result.values, result.jacobian, result.parameter_shift_evaluations)

jvp = execute_phase_qnode_vector_jvp(
    vector_objective,
    np.array([0.2, -0.4]),
    np.array([0.5, -1.25]),
)
vjp = execute_phase_qnode_vector_vjp(
    vector_objective,
    np.array([0.2, -0.4]),
    np.array([2.0, -0.75]),
)
print(jvp.jvp, vjp.vjp)
```

The vector JVP and VJP routes are computed as `jacobian @ tangent` and
`jacobian.T @ cotangent` from the same parameter-shift Jacobian evidence. They
validate tangent/cotangent shapes and fail closed for finite-shot, hardware,
provider, and framework-native adapter routes.

When `scpn_quantum_engine` is installed, the matching PyO3 parity kernels are
`phase_qnode_vector_jvp_rust(jacobian, tangent)` and
`phase_qnode_vector_vjp_rust(jacobian, cotangent)`. They are dense contraction
kernels over already materialised Jacobian evidence and preserve the same
real-valued input boundary.

`execute_phase_qnode_vector_hessian(...)` computes deterministic local
vector-output Hessian tensors by materialising one parameter-shift Hessian per
output component. The result tensor has shape `(output_dim, n_params,
n_params)`, is checked against real finite vector-output objectives, and fails
closed for finite-shot, hardware, provider, and framework-native adapter
routes.

The Rust parity surface for materialised vector Hessian tensors is
`phase_qnode_vector_hessian_tensor_rust(hessian_tensor)`. It validates finite
real tensor entries, square component Hessians, and Hessian symmetry before
returning the symmetrised tensor.

`execute_phase_qnode_vmap_grad(...)` implements the first native vectorized
gradient surface as a deterministic host-side manual loop over scalar
parameter-shift gradients:

```python
from scpn_quantum_control.phase import execute_phase_qnode_vmap_grad


def scalar_objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))


batched = np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.6]])
result = execute_phase_qnode_vmap_grad(scalar_objective, batched)
print(result.batched_values, result.batched_gradients)
```

The readiness helper `run_phase_qnode_vector_transform_readiness_suite()`
records supported `jacfwd`, `jacrev`, vector `jvp`, vector `vjp`, vector
`hessian`, and `vmap.grad` routes plus fail-closed hardware, adapter, and finite-shot
scenarios. The implementation deliberately does not claim provider
vectorization, framework-native `vmap`, finite-shot batched-gradient
statistics, or hardware transform execution.

## Provider-callback QNode transforms

`execute_provider_qnode_transform(...)` binds the provider callback gradient
contract to scalar QNode transform evidence. The caller supplies a sampler that
returns `ProviderExpectationSample` records for shifted parameter vectors. The
executor supports scalar `grad`, `value_and_grad`, `jvp`, `vjp`,
`jacfwd`/`jacrev`, and reports the shifted samples, shot totals, standard error,
and confidence radii carried by `execute_provider_parameter_shift_gradient(...)`.

```python
import numpy as np
from scpn_quantum_control.phase import (
    ProviderExpectationSample,
    execute_provider_qnode_transform,
)


def sampler(params: np.ndarray, shots: int | None) -> ProviderExpectationSample:
    return ProviderExpectationSample(
        value=float(np.cos(params[0]) + 0.25 * np.sin(params[1])),
        variance=None if shots is None else 0.04,
        shots=shots,
    )


result = execute_provider_qnode_transform(
    "value_and_grad",
    sampler,
    np.array([0.2, -0.4]),
    backend="qasm_simulator",
    shots=400,
)
print(result.value, result.gradient, result.standard_error, result.total_shots)
```

`execute_provider_qnode_vmap_grad(...)` provides a host-side manual
`vmap(grad)` route over provider callback gradients. It records one provider
gradient result per batch row and fails closed if any row lacks required
finite-shot variance metadata.

The readiness helper `run_provider_qnode_transform_readiness_suite()` records
supported deterministic, finite-shot, directional, scalar-Jacobian, and manual
batch routes alongside blocked hardware, curvature, and malformed finite-shot
scenarios. This is provider-callback execution evidence, not live provider job
submission or unrestricted hardware-gradient execution.

## Scalar QNode transform execution

`execute_phase_qnode_transform(...)` executes scalar local phase-QNode transforms
when the transform-nesting planner declares the route supported. Current
executable routes are `grad`, `value_and_grad`, deterministic local `hessian`,
deterministic local `hessian_vector_product`, scalar `jvp`, scalar `vjp`,
scalar `jacfwd`, and scalar `jacrev`. Directional and Jacobian routes are
implemented through parameter-shift gradients for scalar objectives: JVP returns
the gradient-tangent contraction, VJP returns the scalar-cotangent pullback, and
`jacfwd`/`jacrev` return a one-row Jacobian.

`execute_phase_qnode_hessian_vector_product(...)` materialises the deterministic
local parameter-shift Hessian and returns `H @ vector` with the Hessian evidence
and vector provenance. It is a bounded second-order local diagnostic, not a
finite-shot HVP, hardware HVP, sparse implicit HVP, or arbitrary-program
second-order AD claim.

The Rust parity surface for this contraction is
`phase_qnode_hessian_vector_product_rust(hessian, vector)`. It validates finite
real inputs, square Hessian shape, and vector width before returning `H @
vector`.

Complex and Wirtinger derivatives are an explicit fail-closed boundary on the
Phase-QNode transform APIs. `phase_qnode_complex_derivative_contract()` returns
the machine-readable contract: parameters, tangents, cotangents, HVP vectors,
vector outputs, and batched parameter matrices must be real-valued finite
arrays. Complex-valued objectives, holomorphic derivatives, Wirtinger partials,
and complex tangent/cotangent algebra are not silently coerced; callers must
split complex controls into real and imaginary real-valued controls before using
these transform surfaces.

The optional extension mirrors this boundary through
`phase_qnode_complex_derivative_contract_rust()` so Rust/PyO3 consumers can
inspect the same real-only contract without implying holomorphic or Wirtinger
support.

The readiness helper `run_phase_qnode_transform_readiness_suite()` records both
supported scalar routes and fail-closed hardware, finite-shot curvature, and
scalar-executor vectorized-transform routes. This closes the scalar local QNode
transform gap; vector-output Jacobians and native manual `vmap(grad)` are handled
by the vector transform API above, while arbitrary program AD, framework-native
nested transforms, and hardware transform execution remain outside this claim
boundary.
