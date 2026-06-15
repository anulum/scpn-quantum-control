# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable Tutorials

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

The script is deliberately small. It is a tutorial and integration smoke path,
not a benchmark, not a hardware claim, and not a claim that arbitrary simulator
kernels are framework-native differentiable.

## What The Workflow Covers

| Step | API surface | Evidence produced |
|---|---|---|
| Minimal QNode | `PhaseQNodeCircuit`, `execute_phase_qnode_circuit(...)`, `parameter_shift_phase_qnode_gradient(...)` | Local statevector value and analytic parameter-shift gradient for a registered gate and observable. |
| Diagnostics | `explain_differentiability(...)` | Fail-closed reasons, suggested alternatives, dependency rows, device rows, backend rows, and support payload. |
| Framework boundary | Bounded QNN bridge matrix inside the diagnostic report | Implemented JAX/PyTorch/TensorFlow bounded rows are separated from arbitrary simulator autodiff and provider hardware gaps. |
| Compiler report | `differentiable_compile_report(...)` | Primitive-level compiler-AD planning and MLIR evidence for a selected registered primitive. |
| Training evidence | `train_parameter_shift_qnn_classifier(...)`, `verify_parameter_shift_qnn_classifier_gradient(...)` | Tiny bounded phase-QNN training run plus finite-difference gradient verification. |
| Benchmark reproduction | `write_differentiable_benchmark_evidence_bundle(...)` | Temporary local benchmark evidence bundle with explicit `functional_non_isolated` classification unless run under the isolated benchmark CI contract. |

## Minimal QNode

```python
import numpy as np

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

circuit = PhaseQNodeCircuit(
    n_qubits=1,
    operations=(("ry", (0,), 0),),
    observable=PauliTerm(1.0, ((0, "z"),)),
)
params = np.array([0.4], dtype=float)

value = execute_phase_qnode_circuit(circuit, params)
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
    bundle = write_differentiable_benchmark_evidence_bundle(
        Path(directory),
        metadata=metadata,
        timing_rows=timing_rows,
        artifact_id="diff-qnode-local-reproduction-example",
    )
    payload = json.loads(bundle.raw_json_path.read_text(encoding="utf-8"))
    print(payload["metadata"]["classification"])
    print(failure_classes)
```

Local tutorial runs should print `functional_non_isolated`. That is a useful
reproducibility and integration check, but it cannot close the promotion blocker
for true `isolated_affinity` evidence.

## Claim Boundaries

- Tutorials and examples prove that the public workflow is runnable.
- Support and diagnostic reports decide whether a route is allowed before
  derivative execution.
- Framework rows are capability declarations for bounded bridges, not universal
  simulator-autodiff claims.
- Local runs are functional evidence; production performance wording still
  requires isolated-affinity benchmark artefacts.
