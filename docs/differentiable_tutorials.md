# Differentiable Tutorials

This page is the practical differentiable-programming path. It connects the
physics object, support diagnostics, framework boundaries, compiler report, and
training evidence without requiring provider credentials or hardware access.

Run the complete script:

```bash
python examples/23_differentiable_api_workflow.py
```

Run the local benchmark evidence reproduction script:

```bash
python examples/24_differentiable_benchmark_reproduction.py
```

Run the canonical first-path namespace smoke:

```bash
python examples/30_diff_first_path.py
```

Run the bounded QFI/FSS finite-size evidence smoke:

```bash
python examples/31_qfi_fss_differentiable_report.py
```

These scripts are deliberately small. They are tutorial and integration smoke
paths, not benchmarks, not hardware claims, and not claims that arbitrary
simulator kernels are framework-native differentiable.

## What The Workflow Covers

| Step | API surface | Evidence produced |
|---|---|---|
| Minimal QNode | `PhaseQNodeCircuit`, `execute_phase_qnode_circuit(...)`, `parameter_shift_phase_qnode_gradient(...)` | Local statevector value and analytic parameter-shift gradient for a registered gate and observable. |
| Finite-spectrum shifts | `plan_generalised_parameter_shift(...)`, `value_and_generalised_parameter_shift_grad(...)`, `estimate_generalised_parameter_shift_shot_noise(...)` | Declared generator spectra, exact shifted-evaluation gradients, and materialised finite-shot confidence envelopes with no provider or hardware execution claim. |
| Controlled gates | `decompose_phase_qnode_controlled_gate(...)`, `registered_phase_qnode_decompositions()` | Exact registered Toffoli/Fredkin decompositions plus native controlled-H/S/T, Toffoli, CCZ, and Fredkin execution. |
| Sparse Ising observables | `build_sparse_ising_chain_hamiltonian(...)` | Validated nearest-neighbour sparse Pauli Hamiltonians with scalar or site/edge coefficient vectors for larger local Phase-QNode circuits. |
| Density and noise | `PhaseQNodeDensityCircuit`, `PhaseQNodeNoiseChannel`, `execute_phase_qnode_density_matrix(...)` | Local density-matrix value, trace, purity, and support report for registered unitary gates plus bounded single-qubit Kraus channels. |
| QNode route preflight | `phase_qnode_support_report(...)`, `phase_qnode_density_support_report(...)`, `phase_qnode_gradient_support_report(...)`, `phase_qnode_metric_support_report(...)`, `phase_qnode_computational_basis_fisher_support_report(...)` | Strict support reports for value, density, pure-state gradient, pure-state metric/QFI, exact computational-basis Fisher, and singular-probability boundary paths. |
| Diagnostics | `explain_differentiability(...)` | Fail-closed reasons, suggested alternatives, dependency rows, device rows, backend rows, and support payload. |
| Framework boundary | Bounded QNN bridge matrix inside the diagnostic report | Implemented JAX/PyTorch/TensorFlow bounded rows are separated from arbitrary simulator autodiff and provider hardware gaps. |
| Compiler report | `differentiable_compile_report(...)` | Primitive-level compiler-AD planning and MLIR evidence for a selected registered primitive. |
| Training evidence | `train_parameter_shift_qnn_classifier(...)`, `verify_parameter_shift_qnn_classifier_gradient(...)` | Tiny bounded phase-QNN training run plus finite-difference gradient verification. |
| Benchmark reproduction | `write_differentiable_benchmark_evidence_bundle(...)` | Temporary local benchmark evidence bundle with explicit `functional_non_isolated` classification unless run under the isolated benchmark CI contract. |
| Canonical namespace | `scpn_quantum_control.diff`, `scpn.diff`, `DifferentiableCircuit`, `jit_or_explain(...)` | No-credential first-path value/gradient execution, serializable diagnostics, shot policy, estimator provenance, and explicit fail-closed JIT metadata. |
| QFI/FSS evidence | `differentiable_qfi_fss_report()`, `differentiable_api("qfi_fss_report")` | Small local Kuramoto-XY finite-size gap scan with BKT and inverse-size residual diagnostics plus non-hardware, non-performance, and non-thermodynamic-limit claim boundaries. |

## Canonical Differentiable Namespace

Use `scpn_quantum_control.diff` for new code and `scpn.diff` when a shorter
compatibility import helps notebooks or external examples. Both expose the same
first-path transforms:

```python
import numpy as np

from scpn_quantum_control import diff


def phase_cost(params: np.ndarray) -> float:
    return float(np.sin(params[0]) + params[1] ** 2)


circuit = diff.differentiable_circuit(
    phase_cost,
    name="phase_cost_first_path",
    parameter_names=("theta", "bias"),
)
params = np.array([0.3, 0.5], dtype=np.float64)

print(circuit(params))
print(circuit.grad(params, method="finite_difference"))
print(circuit.diagnostics.to_dict())
print(diff.jit_or_explain(circuit).to_dict())
```

For older code that imports directly from `scpn_quantum_control.differentiable`
or `scpn_quantum_control.differentiable_api`, keep those imports when you need
low-level result contracts or JSON envelopes. Use `diff` when you want the
stable user-facing path with circuit metadata and fail-closed route diagnostics.

## Minimal QNode

```python
import numpy as np

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_gradient_support_report,
)

circuit = PhaseQNodeCircuit(
    n_qubits=1,
    operations=(("ry", (0,), 0),),
    observable=PauliTerm(1.0, ((0, "z"),)),
)
params = np.array([0.4], dtype=float)

value = execute_phase_qnode_circuit(circuit, params)
gradient_report = phase_qnode_gradient_support_report(circuit, params)
if not gradient_report.supported:
    raise RuntimeError(gradient_report.failure_reason)
gradient = parameter_shift_phase_qnode_gradient(circuit, params)
print(value.value)
print(gradient.gradient)
```

This is the smallest useful local quantum-gradient route: one registered
rotation, one registered observable, and one trainable parameter.

For a registered multi-qubit ansatz, build a template and pass its circuit to
the same execution and gradient calls:

```python
import numpy as np

from scpn_quantum_control.phase import (
    build_phase_qnode_template,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

template = build_phase_qnode_template(
    "hardware_efficient_ryrz",
    n_qubits=3,
    n_layers=2,
    entangler="ring",
)
params = np.linspace(0.1, 0.9, template.parameter_count)

value = execute_phase_qnode_circuit(template.circuit(), params)
gradient = parameter_shift_phase_qnode_gradient(template.circuit(), params)
print(template.to_dict()["claim_boundary"])
print(value.value)
print(gradient.gradient)
```

Template support is deliberately bounded to local statevector execution over
registered gates and observables. It does not imply dynamic circuits,
provider-backed execution, finite-shot sampling, or native framework autodiff
through a simulator.

For an arbitrary registered-depth circuit, build a validated spec and inspect
its depth profile before execution:

```python
import numpy as np

from scpn_quantum_control.phase import (
    PauliTerm,
    build_registered_phase_qnode_circuit,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

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
params = np.array([0.2, -0.3, 0.1, 0.4], dtype=float)

print(spec.depth_profile.to_dict())
value = execute_phase_qnode_circuit(spec.circuit, params)
gradient = parameter_shift_phase_qnode_gradient(spec.circuit, params)
print(value.value)
print(gradient.gradient)
```

Depth budgets are local circuit-resource gates. They are not hardware duration,
transpilation, pulse-schedule, noise, or performance evidence.

For controlled-gate export, ask the registry which exact decompositions are
available and expand only those gates before passing the circuit to another
toolchain:

```python
import numpy as np

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    decompose_phase_qnode_controlled_gate,
    execute_phase_qnode_circuit,
    registered_phase_qnode_decompositions,
)

print(registered_phase_qnode_decompositions())
ops = decompose_phase_qnode_controlled_gate(("ccnot", (0, 1, 2)))
circuit = PhaseQNodeCircuit(
    n_qubits=3,
    operations=(("x", (0,)), ("x", (1,)), *ops),
    observable=PauliTerm(1.0, ((2, "z"),)),
)

value = execute_phase_qnode_circuit(circuit, np.array([], dtype=float))
print(value.value)
```

The decomposition registry is intentionally explicit: unsupported controlled
gates, wrong qubit arity, and parameterised decomposition requests fail closed.

For mixed-state checks, build a density circuit and add registered local noise
channels. The route reports density entries, trace, and purity; gradients and
pure-state metrics remain separate fail-closed surfaces:

```python
import numpy as np

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeDensityCircuit,
    PhaseQNodeNoiseChannel,
    execute_phase_qnode_density_matrix,
    registered_phase_qnode_noise_channels,
)

print(registered_phase_qnode_noise_channels())
circuit = PhaseQNodeDensityCircuit(
    n_qubits=1,
    operations=(
        ("x", (0,)),
        PhaseQNodeNoiseChannel("amplitude_damping", (0,), 0.25),
    ),
    observable=PauliTerm(1.0, ((0, "z"),)),
)

value = execute_phase_qnode_density_matrix(circuit, np.array([], dtype=float))
print(value.value, value.trace, value.purity)
```

This is local deterministic noisy-channel evidence only. It is not finite-shot
sampling, provider execution, hardware execution, or noisy-gradient evidence.

## Why A Route Is Blocked

```python
from scpn_quantum_control import explain_differentiability

diagnostic = explain_differentiability(
    gate="arbitrary_unitary",
    observable="pauli_expectation",
    backend="hardware",
    shots=1024,
)
print(diagnostic.blocked_reasons)
print(diagnostic.suggested_alternatives)
print(diagnostic.backend_matrix)
```

Use this before execution when a route might be unsupported. The report explains
why it cannot differentiate and names safer alternatives. It also shows bounded
framework dependency rows and backend/device capability rows so integrations
can decide whether they need native, JAX, PyTorch, TensorFlow, provider-callback,
or hardware-policy work.

## Compiler Report

```python
from scpn_quantum_control import differentiable_compile_report

compile_report = differentiable_compile_report(
    primitive_identities=("scpn.program_ad.array:getitem@1",)
)
print(compile_report.payload["primitive_count"])
print(compile_report.method)
```

This is planning and interchange evidence. Treat a compiler report as executable
only when its selected primitive plan declares an executable backend.

## Training Evidence

```python
import numpy as np

from scpn_quantum_control.phase import (
    train_parameter_shift_qnn_classifier,
    verify_parameter_shift_qnn_classifier_gradient,
)

features = np.array([[0.0], [np.pi]], dtype=float)
labels = np.array([0.0, 1.0], dtype=float)

training = train_parameter_shift_qnn_classifier(
    features,
    labels,
    initial_params=np.array([0.8], dtype=float),
    learning_rate=0.7,
    max_steps=80,
    target_loss=0.0,
    target_loss_tolerance=1.0e-4,
)
verification = verify_parameter_shift_qnn_classifier_gradient(
    features,
    labels,
    training.best_params,
)
print(training.training.best_value)
print(training.prediction.accuracy)
print(verification.passed)
```

This training path is intentionally bounded to a local phase-QNN classifier. It
is useful onboarding and regression evidence, but it is not evidence for
arbitrary QNN/QGNN/QSNN architectures, unseeded stochastic training, or provider
hardware training.

## Benchmark Reproduction

```python
import json
import tempfile
from pathlib import Path

from scpn_quantum_control.benchmarks.differentiable_evidence import (
    BenchmarkIsolationMetadata,
    write_differentiable_benchmark_evidence_bundle,
)
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    run_differentiable_external_comparison_suite,
    write_differentiable_external_comparison,
)

timing_rows = tuple(row.to_dict() for row in run_differentiable_external_comparison_suite())
failure_classes = sorted(
    {
        row["failure_class"]
        for row in timing_rows
        if row["status"] == "hard_gap" and row["failure_class"] is not None
    }
)
metadata = BenchmarkIsolationMetadata.from_ci_environment(
    {},
    command=("python", "examples/24_differentiable_benchmark_reproduction.py"),
    cpu_affinity=None,
    isolation_method=None,
    load_before=None,
    load_after=None,
    governor=None,
    frequency_mhz=None,
    heavy_jobs_running=False,
)
with tempfile.TemporaryDirectory(prefix="scpn-qc-diff-bench-") as directory:
    external_artifact = write_differentiable_external_comparison(
        Path(directory) / "external_comparison.json",
        artifact_id="diff-qnode-local-external-comparison-example",
    )
    bundle = write_differentiable_benchmark_evidence_bundle(
        Path(directory),
        metadata=metadata,
        timing_rows=timing_rows,
        artifact_id="diff-qnode-local-reproduction-example",
    )
    payload = json.loads(bundle.raw_json_path.read_text(encoding="utf-8"))
    print(payload["metadata"]["classification"])
    print(external_artifact.classification)
    print(failure_classes)
```

Local tutorial runs should print `functional_non_isolated`. That is a useful
reproducibility and integration check for both benchmark and external-comparison
artefacts, but it cannot close the promotion blocker for true
`isolated_affinity` evidence.

The CI evidence writer also emits the external comparison companion artefact:

```bash
python scripts/run_differentiable_benchmark_evidence.py \
  --output-dir differentiable-benchmark-evidence
```

The output directory contains `diff-qnode-ci-evidence-schema-v1.json`,
`diff-qnode-ci-evidence-schema-v1.csv`, `diff-qnode-ci-evidence-schema-v1.md`,
and `diff-qnode-external-comparison.json`. The benchmark JSON references the
external comparison artefact ID in `evidence_artifact_ids`; the external
comparison JSON remains `functional_non_isolated`.

## Claim Boundaries

- Tutorials and examples prove that the public workflow is runnable.
- Support and diagnostic reports decide whether a route is allowed before
  derivative execution.
- Framework rows are capability declarations for bounded bridges, not universal
  simulator-autodiff claims.
- Local runs are functional evidence; production performance wording still
  requires isolated-affinity benchmark artefacts.
