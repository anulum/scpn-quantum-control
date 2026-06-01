# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — API Reference

# API Reference

For first-path user workflows, start with the
[Stable Facades API](stable_facades_api.md). It is the mkdocstrings reference
for public facades such as `scpn_quantum_control.kuramoto_core`. The lower-level
sections below are advanced module references for maintainers, extension authors,
and researchers who need direct subsystem access.

## Reference Levels

| Level | Start here | Use when |
| --- | --- | --- |
| Stable facade | [Stable Facades API](stable_facades_api.md) | Building notebooks, tutorials, cross-repository integrations, or user-facing workflows. |
| Workflow guide | [Kuramoto Core Facade](kuramoto_core_facade.md) | Compiling arbitrary `K_nm`/`omega` problems without depending on low-level module layout. |
| Source-validation register | [Paper 0 Validation Register](paper0/paper0_validation_register.md) | Inspecting generated Paper 0 source-accounting specs, fixtures, and validation modules under their explicit claim boundary. |
| Runtime contract | [QPU Data Artifact](qpu_data_artifact.md), [Pipeline Runtime Contract](pipeline_runtime_contract.md) | Exchanging persisted QPU results or compute-unit metadata. |
| Advanced module reference | This page and [Auto-Generated Module Index](autodoc.md) | Auditing, extending, or debugging subsystem internals. |

## stable facades

### `kuramoto_core`

```python
build_kuramoto_problem(K_nm, omega, metadata=None) -> KuramotoProblem
compile_hamiltonian(problem) -> SparsePauliOp
compile_dense_hamiltonian(problem) -> np.ndarray
compile_trotter_circuit(problem, time, trotter_steps=10, trotter_order=1) -> QuantumCircuit
measure_order_parameter(problem, statevector) -> tuple[float, float]
```

Validate an arbitrary symmetric Kuramoto coupling problem, attach serialisable
metadata, and compile the common Hamiltonian/circuit objects used by simulator,
witness, and hardware workflows. See [Kuramoto Core Facade](kuramoto_core_facade.md)
for the workflow page.

### `paper0`

```python
from scpn_quantum_control.paper0 import validate_upde_fixture
from scpn_quantum_control.paper0.spec_loader import load_upde_validation_spec
```

The `paper0` package exposes the completed source-accounting register for Paper
0. Generated validation modules preserve ledger-bounded source spans, component
labels, fixture summaries, spec bundles, and claim boundaries. These modules are
documentation and regression infrastructure for source ingestion; they do not
convert Paper 0 source statements into measured hardware evidence or external
scientific validation. See [Paper 0 Validation Register](paper0/paper0_validation_register.md).

### `compiler.mlir`

```python
import numpy as np

from scpn_quantum_control import (
    CompilerADExecutableConfig,
    CompilerADTransformPlan,
    DifferentiableMLIRCompileConfig,
    MLIRCompileConfig,
    PrimitiveLoweringStatus,
    build_compiler_ad_transform_plan,
    compile_compiler_ad_transform_plan_to_mlir,
    compile_custom_derivative_rule_to_executable,
    compile_custom_derivative_rule_to_mlir,
    compile_kuramoto_to_mlir,
    compile_matrix_2x2_determinant_ad_to_native_llvm_jit,
    compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit,
    compile_matrix_2x2_inverse_ad_to_native_llvm_jit,
    compile_matrix_2x2_solve_ad_to_native_llvm_jit,
    compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit,
    compile_matrix_matrix_product_ad_to_native_llvm_jit,
    compile_matrix_quadratic_form_ad_to_native_llvm_jit,
    compile_matrix_trace_ad_to_native_llvm_jit,
    compile_matrix_vector_product_ad_to_native_llvm_jit,
    compile_scalar_binary_elementwise_ad_to_native_llvm_jit,
    compile_scalar_quadratic_ad_to_native_llvm_jit,
    compile_scalar_unary_elementwise_ad_to_native_llvm_jit,
    compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit,
    compile_vector_dot_ad_to_native_llvm_jit,
    compile_vector_squared_norm_ad_to_native_llvm_jit,
    compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit,
    make_executable_ad_kernel_batching_rule,
)

module = compile_kuramoto_to_mlir(
    problem,
    MLIRCompileConfig(time=0.4, trotter_steps=2, trotter_order=2),
)
module.text      # deterministic MLIR-style textual IR
module.sha256    # digest over module.text

diff_module = compile_custom_derivative_rule_to_mlir(
    rule,
    values,
    DifferentiableMLIRCompileConfig(),
)
kernel = compile_custom_derivative_rule_to_executable(
    rule,
    values,
    CompilerADExecutableConfig(),
)
kernel.jvp(values, tangent)

native_kernel = compile_scalar_quadratic_ad_to_native_llvm_jit(
    rule,
    quadratic=3.0,
    linear=-2.0,
    constant=0.5,
    sample_values=values,
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_kernel.gradient(values)

native_unary_kernel = compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
    rule,
    primitive="sin",
    sample_values=values,
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_unary_kernel.gradient(values)

native_binary_kernel = compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
    rule,
    primitive="multiply",
    sample_values=np.array([1.5, -2.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_binary_kernel.gradient(np.array([1.5, -2.0], dtype=np.float64))

native_dot_kernel = compile_vector_dot_ad_to_native_llvm_jit(
    rule,
    dimension=2,
    sample_values=np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_dot_kernel.gradient(np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64))

native_norm_kernel = compile_vector_squared_norm_ad_to_native_llvm_jit(
    rule,
    dimension=3,
    sample_values=np.array([1.5, -2.0, 0.25], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_norm_kernel.gradient(np.array([1.5, -2.0, 0.25], dtype=np.float64))

native_matvec_kernel = compile_matrix_vector_product_ad_to_native_llvm_jit(
    rule,
    dimension=2,
    sample_values=np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_matvec_kernel.vjp(
    np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64),
    np.array([1.0, 1.0], dtype=np.float64),
)

native_matmul_kernel = compile_matrix_matrix_product_ad_to_native_llvm_jit(
    rule,
    dimension=2,
    sample_values=np.array([1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_matmul_kernel.jvp(
    np.array([1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25], dtype=np.float64),
    np.array([0.2, -0.1, 0.3, 0.4, -0.5, 0.75, 0.25, -0.2], dtype=np.float64),
)

native_trace_kernel = compile_matrix_trace_ad_to_native_llvm_jit(
    rule,
    dimension=2,
    sample_values=np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_trace_kernel.gradient(np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64))

native_frobenius_kernel = compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit(
    rule,
    dimension=2,
    sample_values=np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_frobenius_kernel.gradient(np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64))

native_det_kernel = compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
    rule,
    sample_values=np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_det_kernel.gradient(np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64))

native_matrix_eigen_kernel = compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
    rule,
    sample_values=np.array([2.0, 0.25, 0.75, 1.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_matrix_eigen_kernel.value(np.array([2.0, 0.25, 0.75, 1.0], dtype=np.float64))

native_inverse_kernel = compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
    rule,
    sample_values=np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_inverse_kernel.value(np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64))

native_solve_kernel = compile_matrix_2x2_solve_ad_to_native_llvm_jit(
    rule,
    sample_values=np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_solve_kernel.value(np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64))

native_cholesky_kernel = compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
    rule,
    sample_values=np.array([4.0, 1.0, 3.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_cholesky_kernel.value(np.array([4.0, 1.0, 3.0], dtype=np.float64))

native_eigen_kernel = compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
    rule,
    sample_values=np.array([2.0, 0.5, 3.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_eigen_kernel.value(np.array([2.0, 0.5, 3.0], dtype=np.float64))

native_quadratic_form_kernel = compile_matrix_quadratic_form_ad_to_native_llvm_jit(
    rule,
    dimension=2,
    sample_values=np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
native_quadratic_form_kernel.gradient(
    np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64)
)

native_batching_rule = make_executable_ad_kernel_batching_rule(native_det_kernel)

ad_plan = build_compiler_ad_transform_plan(custom_rule_registry)
ad_plan_module = compile_compiler_ad_transform_plan_to_mlir(ad_plan)
```

The MLIR compiler surface emits deterministic, auditable Kuramoto-XY IR with
explicit omega terms, coupling terms, Trotter parameters, resource counts, and
claim-boundary metadata. It also lowers exact custom derivative rules into a
deterministic differentiable-primitive MLIR-style interchange artifact carrying
parameter metadata, output values, exact custom Jacobian payloads, and resource
counts. `compile_custom_derivative_rule_to_executable()` adds an executable
compiler-backed AD boundary for exact custom primitive rules: it emits
deterministic MLIR provenance plus deterministic LLVM-style scalar-gradient
provenance for verified scalar-output kernels, binds normalized value/JVP/VJP
runtime kernels, exposes verified scalar-output `gradient()` execution through
VJP cotangent-one semantics, and verifies those kernels against the source rule
before returning. `make_program_ad_linalg_matrix_power_executable_lowering_rule()`
and `make_program_ad_linalg_multi_dot_executable_lowering_rule()` bind concrete
static linalg signatures to verified MLIR-runtime value/JVP kernels derived from
the direct derivative factories.
`compile_scalar_quadratic_ad_to_native_llvm_jit()` is the native executable
LLVM MCJIT backend for scalar quadratic primitives: it emits MLIR provenance,
generates LLVM IR for value, JVP, VJP, and gradient functions, verifies native
execution against the source derivative rule, and returns an executable kernel
with `backend="native_llvm_jit"`.
`compile_scalar_unary_elementwise_ad_to_native_llvm_jit()` extends the same
native execution boundary to scalar `sin`, `cos`, and `exp` elementwise
primitives using LLVM intrinsics for value and analytic derivative kernels.
`compile_scalar_binary_elementwise_ad_to_native_llvm_jit()` covers scalar
`add`, `subtract`, and `multiply` elementwise primitives with native value,
JVP, VJP, and two-component gradient kernels.
`compile_vector_dot_ad_to_native_llvm_jit()` emits dimension-specialised native
LLVM MCJIT kernels for vector dot products over concatenated `[x, y]` inputs,
with scalar value/JVP and full `[y, x]` VJP/gradient output.
`compile_vector_squared_norm_ad_to_native_llvm_jit()` covers nonlinear vector
reductions by compiling `sum(x_i**2)` with scalar value/JVP and exact
`2*x` VJP/gradient output.
`compile_matrix_vector_product_ad_to_native_llvm_jit()` adds vector-output
native linalg by compiling `A @ x` over row-major concatenated `[A, x]`
inputs with exact JVP and VJP output; `gradient()` remains fail-closed because
the public gradient helper is scalar-output only.
`compile_matrix_matrix_product_ad_to_native_llvm_jit()` extends native linalg
coverage to square `A @ B` over concatenated `[A, B]` inputs with exact
matrix-output JVP and VJP; `gradient()` remains fail-closed for the same
scalar-output boundary.
`compile_matrix_trace_ad_to_native_llvm_jit()` covers scalar-output matrix
reductions by compiling `trace(A)` over row-major matrix inputs with exact
identity-mask JVP, VJP, and gradient output.
`compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit()` covers
nonlinear scalar-output matrix reductions by compiling `sum(A_ij**2)` over
row-major matrix inputs with exact `2*A` JVP, VJP, and gradient output.
`compile_matrix_2x2_determinant_ad_to_native_llvm_jit()` adds a fixed-size
polynomial determinant primitive for row-major 2x2 matrices with exact
adjugate-gradient JVP, VJP, and gradient output.
`compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit()` adds a bounded
real-simple nonsymmetric 2x2 spectral primitive over row-major matrix inputs
with exact closed-form eigenvalue, JVP, and VJP kernels; complex spectra,
repeated eigenvalues, and the public vector-output gradient helper remain
fail-closed.
`compile_matrix_2x2_inverse_ad_to_native_llvm_jit()` adds a bounded
nonsingular row-major 2x2 inverse primitive with exact rational value, JVP,
and VJP kernels; singular matrices and the public vector-output gradient helper
remain fail-closed.
`compile_matrix_2x2_solve_ad_to_native_llvm_jit()` adds a bounded nonsingular
row-major `A x = b` primitive over concatenated `[A, b]` inputs with exact
linear-solve value, JVP, and VJP kernels; singular matrices and the public
vector-output gradient helper remain fail-closed.
`compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit()` adds a bounded
positive-definite symmetric factorisation primitive over upper-triangle
`[a00, a01, a11]` inputs with exact lower-triangle value, JVP, and VJP
kernels; non-positive-definite matrices and the public vector-output gradient
helper remain fail-closed.
`compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit()` adds a bounded
distinct-eigenvalue symmetric spectral primitive over upper-triangle
`[a00, a01, a11]` inputs with exact closed-form value, JVP, and VJP kernels;
repeated eigenvalues and the public vector-output gradient helper remain
fail-closed.
`compile_matrix_quadratic_form_ad_to_native_llvm_jit()` extends native compiler
AD to rank-2 scalar linalg by compiling `x.T @ A @ x` over row-major
concatenated `[A, x]` inputs with exact matrix-entry gradients
`outer(x, x)` and vector gradients `(A + A.T) @ x`.
`make_scalar_quadratic_native_llvm_jit_lowering_rule()` and
`make_scalar_unary_elementwise_native_llvm_jit_lowering_rule()` and
`make_scalar_binary_elementwise_native_llvm_jit_lowering_rule()` and
`make_vector_dot_native_llvm_jit_lowering_rule()` and
`make_vector_squared_norm_native_llvm_jit_lowering_rule()` and
`make_matrix_matrix_product_native_llvm_jit_lowering_rule()` and
`make_matrix_trace_native_llvm_jit_lowering_rule()` and
`make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule()` and
`make_matrix_2x2_determinant_native_llvm_jit_lowering_rule()` and
`make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule()` and
`make_matrix_2x2_inverse_native_llvm_jit_lowering_rule()` and
`make_matrix_2x2_solve_native_llvm_jit_lowering_rule()` and
`make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule()` and
`make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule()` and
`make_matrix_vector_product_native_llvm_jit_lowering_rule()` and
`make_matrix_quadratic_form_native_llvm_jit_lowering_rule()` bind those native
backends to primitive registry lowering metadata when the primitive has the
matching static signature. Other primitive families remain fail-closed for
native LLVM/JIT until they provide their own verified lowering rule. This
surface does not claim LLVM/QIR lowering for unrelated primitives, cloud
submission, pulse compilation, or hardware execution.
`make_executable_ad_kernel_batching_rule()` binds a verified executable
compiler AD kernel into primitive-registry `vmap` dispatch. Its automatic mode
maps one-argument calls to `value` and two-argument calls to `jvp` or `vjp`
only when tangent and cotangent dimensions are unambiguous; equal input/output
dimensions require an explicit method selection and otherwise fail closed.
`build_compiler_ad_transform_plan()` converts registered primitive identities
into deterministic compiler AD transform metadata with explicit JVP/VJP/adjoint
intent, MLIR dialect operation names, primitive-specific batching-rule
presence, static-argument-rule presence, registry
lowering metadata, promoted static derivative factory/signature contracts,
promoted paired primitive nondifferentiability-policy, fail-closed boundary,
and boundary-policy contracts, executable MLIR-runtime availability when a
lowering rule is registered, and
fail-closed Rust/LLVM backend status. Its deterministic module metadata also
lists rule-coverage, JVP forward-readiness coverage, VJP reverse-readiness
coverage, batching-rule coverage, boundary-contracted policy/effect coverage,
aggregate boundary-contract primitive coverage, complete registry-contract primitive coverage,
reverse-contract primitive coverage, reverse-incomplete primitive identities,
adjoint-contract primitive coverage, adjoint-incomplete primitive identities,
transform-contract primitive coverage, transform-incomplete primitive identities,
native-backend contract coverage, native-backend incomplete primitive identities,
backend-specific Rust/LLVM/JIT contract coverage, backend-specific Rust/LLVM/JIT
incomplete primitive identities, backend-specific Rust/LLVM/JIT blocker
provenance maps, MLIR-runtime contract coverage, MLIR-runtime incomplete primitive identities,
MLIR-runtime blocker provenance maps, verified MLIR-runtime provenance coverage,
per-primitive readiness verdict metadata, aggregate readiness verdict counts,
per-primitive hard-gap queues, next hard-gap metadata, aggregate hard-gap counts,
per-hard-gap primitive reverse indexes,
global hard-gap priority ordering,
structured hard-gap frontier records,
boundary-policy coverage, MLIR-runtime lowering primitive
identities, and uncontracted primitive identities so derivative-only rules
cannot be mistaken for complete compiler contracts. Static program-AD array,
shape, elementwise, reduction, product, cumulative, and linalg contracts expose
MLIR metadata for direct derivative factories and fixed signatures; concrete
static linalg signatures can also bind optional verified MLIR-runtime lowering
rules, MLIR-runtime availability fails closed unless the primitive has a
registered lowering rule, and Rust native backend availability remains blocked
until executable differentiated backends exist. LLVM/JIT native backend
availability is recognized only for primitives that carry verified
`native_llvm_jit` lowering metadata. Rust differentiated backend claims remain
fail-closed until native Rust lowering is implemented.
`compile_compiler_ad_transform_plan_to_mlir()`
emits that plan as MLIR-style interchange; executable native LLVM/JIT
differentiated runtimes are marked executable only for verified native lowering
metadata, and native Rust differentiated runtime remains unavailable until
backed by lowering and runtime verification.

### `control.realtime_runtime`

```python
from scpn_quantum_control import RealtimeRuntimeConfig, VirtualRealtimeClock
from scpn_quantum_control import run_realtime_control_loop

clock = VirtualRealtimeClock()
config = RealtimeRuntimeConfig(sample_period_s=0.01, deadline_s=0.005)
result = run_realtime_control_loop(10, step_fn, config=config, clock=clock)
result.missed_deadlines
result.max_latency_s
```

The runtime records scheduled start, actual start, finish time, latency, jitter,
deadline-miss state, and numeric metrics for fixed-period software control
loops. It can use a monotonic wall clock or deterministic virtual clock. It is
not an intra-shot hardware-latency claim.

### `deployment.cloud_native`

```python
from scpn_quantum_control import CloudDeploymentSpec, ContainerResources
from scpn_quantum_control import generate_cloud_manifests

bundle = generate_cloud_manifests(
    CloudDeploymentSpec(
        name="scpn-qc",
        image="registry.example/scpn-quantum-control:0.9.7",
        command=("scpn-bench", "stable-core-contract-gate"),
        resources=ContainerResources(cpu="1000m", memory="1Gi"),
        env={"SCPN_EXECUTION_MODE": "offline"},
    )
)
bundle.files["deployment.yaml"]
bundle.files["docker-compose.yaml"]
```

Cloud-native deployment generation emits Kubernetes Deployment/Service and
Docker Compose manifests with resource limits, non-root execution,
read-only-root filesystem, no privilege escalation, and deterministic digests.
Secret-like environment variables are rejected. The generator does not read
credentials, create clusters, or contact cloud APIs.

### `hardware.hal`

```python
from scpn_quantum_control import (
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumWorkload,
)

hal = HardwareAbstractionLayer.with_builtin_profiles()
profile = hal.profile("local_statevector")
hal.register_backend(LocalDeterministicSimulator(profile))

job = hal.submit(
    "local_statevector",
    QuantumWorkload(
        workload_id="demo",
        ir_format="mlir",
        program="module {}",
        n_qubits=4,
        shots=1024,
    ),
)
result = hal.result(job)
```

The HAL is the provider-neutral execution contract above the older descriptor
registry. Built-in profiles are metadata-only and do not import provider SDKs,
read credentials, open network sessions, or queue jobs. Cloud routes fail closed
unless a concrete adapter is injected and an approval token is supplied at
submission time.

Built-in route families cover IBM Quantum, IonQ direct, AWS Braket QPU and
managed-simulator routes, Azure Quantum QPU, emulator, and private-preview
routes, Quantinuum, Rigetti QCS, QuEra/Bloqade, IQM, Pasqal, OQC, D-Wave Leap,
qBraid IonQ, qBraid dynamic runtime, Strangeworks Compute, Quandela photonics,
and local simulator adapters.
Provider-specific SDK credentials, queue selection, pricing, and region policy
remain adapter responsibilities.

Aggregator/provider combinations are exposed separately from executable HAL
profiles through `built_in_aggregator_provider_routes()`. This keeps broad
broker catalogues explicit without duplicating runtime adapters: direct Braket
and Azure provider rows resolve to their specific HAL profiles, the direct IBM
Quantum row resolves to `ibm_quantum`, the direct D-Wave row resolves to
`dwave_leap`, the direct IonQ row resolves to `ionq_cloud`, the direct IQM row
resolves to `iqm_cloud`, the direct OQC row resolves to `oqc_cloud`, the direct
Pasqal row resolves to `pasqal_cloud`, the direct Quandela row resolves to
`quandela_cloud`, the direct Quantinuum row resolves to `quantinuum_cloud`, the
direct QuEra/Bloqade row resolves to `quera_bloqade`, the direct Rigetti row
resolves to `rigetti_qcs`, while dynamic qBraid and Strangeworks rows resolve
to `qbraid_runtime` and `strangeworks_compute` respectively. Use
`resolve_aggregator_provider_route()` when routing code needs a single
validated row plus the executable HAL profile for a requested aggregator,
provider, and IR format.
`aggregator_provider_optional_dependency_matrix()` adds the corresponding
offline SDK-import evidence for preflight checks before authenticated provider
capability probes or live submissions.
`probe_aggregator_provider_capability()` is the provider-neutral no-submit
contract for authenticated metadata probes: it resolves the broker route,
accepts only no-submit target snapshots, and returns ready/blocked/unknown
readiness decisions before any submission path is considered.
`snapshot_from_azure_target()`, `snapshot_from_braket_device()`,
`snapshot_from_dwave_solver()`, `snapshot_from_iqm_backend()`,
`snapshot_from_ionq_backend()`, `snapshot_from_oqc_target()`,
`snapshot_from_pasqal_target()`, `snapshot_from_qiskit_runtime_backend()`,
`snapshot_from_qbraid_device()`, `snapshot_from_quandela_processor()`,
`snapshot_from_quantinuum_backend()`, `snapshot_from_quera_bloqade()`,
`snapshot_from_rigetti_qcs()`, and `snapshot_from_strangeworks_backend()`
provide concrete no-submit adapters for
injected provider or broker SDK objects: they read declared target metadata,
route-supported IR formats, queue, limit, online, simulator, calibration, and
gate metadata without invoking submission APIs.
`build_openpulse_control_readiness()` extends this no-submit surface for
pulse-level control lanes by verifying OpenPulse-compatible IR and native
feature availability, then producing a calibration workflow dossier only when
the target metadata passes readiness gates.

## Advanced Module Reference

The following sections expose lower-level modules directly. Prefer the stable
facades above unless you specifically need a bridge adapter, solver class, or
subsystem implementation detail.

## bridge

### `knm_hamiltonian`

```python
knm_to_hamiltonian(K: np.ndarray, omega: np.ndarray) -> SparsePauliOp
```
Convert Knm coupling matrix and natural frequencies to XY Hamiltonian as Qiskit `SparsePauliOp`.
H = -\sum_{i<j} K_{ij}(X_iX_j + Y_iY_j) - \sum_i \omega_i Z_i$. Use when you need the
`SparsePauliOp` for circuit construction. For dense matrix operations, use `knm_to_dense_matrix`.

```python
knm_to_dense_matrix(K: np.ndarray, omega: np.ndarray, delta: float = 0.0) -> np.ndarray
```
Build dense XY Hamiltonian matrix (shape `2^n × 2^n`, complex). Uses Rust bitwise flip-flop
construction when `scpn_quantum_engine` is installed (10-50× faster than Qiskit for n≤10),
falls back to `knm_to_hamiltonian(...).to_matrix()`. Preferred for eigendecomposition, OTOC,
entanglement entropy, Krylov complexity, and any code that needs a numpy matrix.

```python
knm_to_sparse_matrix(K: np.ndarray, omega: np.ndarray, delta: float = 0.0) -> csc_matrix
```
Build sparse XY/XXZ Hamiltonian matrix in CSC format for large-scale memory-efficient simulation.

```python
knm_to_xxz_hamiltonian(K: np.ndarray, omega: np.ndarray, delta: float = 0.0) -> SparsePauliOp
```
XXZ Hamiltonian with anisotropy $\Delta$: H = -\sum_{i<j} K_{ij}(X_iX_j + Y_iY_j + \Delta Z_iZ_j) - \sum_i \omega_i Z_i$.
At $\Delta=0$: XY model (standard Kuramoto mapping). At $\Delta=1$: isotropic Heisenberg.

```python
knm_to_ansatz(K: np.ndarray, reps: int = 2, threshold: float = 0.01) -> QuantumCircuit
```
Build physics-informed ansatz: CZ entanglement only between pairs where `K[i,j] > threshold`.

```python
build_knm_paper27(L: int = 16, K_base: float = 0.45, K_alpha: float = 0.3) -> np.ndarray
```
Canonical {nm}$ coupling matrix from Paper 27 with exponential decay, calibration anchors, and cross-hierarchy boosts.

```python
build_kuramoto_ring(n: int, coupling: float = 1.0, omega: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]
```
Nearest-neighbour ring topology. Returns `(K, omega)`.

```python
OMEGA_N_16: np.ndarray  # 16 canonical frequencies (rad/s)
```

### `phase_artifact`

```python
LockSignatureArtifact(source_layer, target_layer, plv, mean_lag)
LayerStateArtifact(R, psi, lock_signatures={})
UPDEPhaseArtifact(layers, cross_layer_alignment, stability_proxy, regime_id, metadata={})
```

Helpers:

```python
UPDEPhaseArtifact.to_dict() -> dict
UPDEPhaseArtifact.to_json(indent=2) -> str
UPDEPhaseArtifact.from_dict(payload) -> UPDEPhaseArtifact
UPDEPhaseArtifact.from_json(payload) -> UPDEPhaseArtifact
```

### `orchestrator_adapter`

```python
PhaseOrchestratorAdapter.from_orchestrator_state(state, metadata=None) -> UPDEPhaseArtifact
PhaseOrchestratorAdapter.to_orchestrator_payload(artifact) -> dict
PhaseOrchestratorAdapter.to_scpn_control_telemetry(artifact) -> dict
PhaseOrchestratorAdapter.build_knm_from_binding_spec(binding_spec, zero_diagonal=False) -> np.ndarray
PhaseOrchestratorAdapter.build_omega_from_binding_spec(binding_spec, default_omega=1.0) -> np.ndarray
```

### `control_plasma_knm`

Compatibility bridge for `scpn-control` plasma-native Knm update.

```python
build_knm_plasma(mode="baseline", L=8, K_base=0.30, zeta_uniform=0.0, ..., repo_src=None) -> np.ndarray
build_knm_plasma_spec(...) -> dict  # {K, zeta, layer_names}
build_knm_plasma_from_config(R0, a, B0, Ip, n_e, ..., repo_src=None) -> np.ndarray
plasma_omega(L=8, repo_src=None) -> np.ndarray
```

### `sc_to_quantum`

```python
probability_to_angle(p: float) -> float  # p -> 2*arcsin(sqrt(p))
angle_to_probability(theta: float) -> float  # theta -> sin^2(theta/2)
bitstream_to_statevector(bits: np.ndarray) -> np.ndarray
measurement_to_bitstream(counts: dict, length: int) -> np.ndarray
```

### `spn_to_qcircuit`

```python
spn_to_circuit(W_in, W_out, thresholds) -> QuantumCircuit
```

## qsnn

### `dynamic_coupling.DynamicCouplingEngine`

```python
DynamicCouplingEngine(n_qubits, initial_K, omega, learning_rate=0.1, decay_rate=0.05)
    .step(dt: float) -> dict  # K_updated, correlation_matrix, statevector
    .run_coevolution(steps, dt) -> list[dict]
```
Quantum Hebbian Learning: macroscopic topology evolves according to microscopic quantum correlations.

### `qlif.QuantumLIFNeuron`

```python
QuantumLIFNeuron(v_rest=-70.0, v_threshold=-55.0, tau_mem=10.0, dt=1.0, n_shots=100)
    .step(input_current: float) -> int  # 0 or 1
    .get_circuit() -> QuantumCircuit
    .reset()
```

### `qsynapse.QuantumSynapse`

```python
QuantumSynapse(weight: float, w_min: float = 0.0, w_max: float = 1.0)
    .apply(circuit, pre_qubit, post_qubit)
    .effective_weight() -> float
```

### `qstdp.QuantumSTDP`

```python
QuantumSTDP(learning_rate: float = 0.01, shift: float = pi/2)
    .update(synapse, pre_measured, post_measured)
```

### `qlayer.QuantumDenseLayer`

```python
QuantumDenseLayer(n_neurons: int, n_inputs: int, weights: np.ndarray | None = None, spike_threshold: float = 0.5)
    .forward(input_values: np.ndarray) -> np.ndarray  # spike array (0/1)
    .get_weights() -> np.ndarray  # (n_neurons, n_inputs)
```

### `training.QSNNTrainer`

```python
QSNNTrainer(layer: QuantumDenseLayer, lr: float = 0.01)
    .parameter_shift_gradient(inputs, target) -> np.ndarray
    .train_epoch(X, y) -> float  # mean loss
    .train(X, y, epochs=10) -> list[float]  # loss history
```

`QSNNTrainer.parameter_shift_gradient()` delegates to the native
`scpn_quantum_control.differentiable` parameter-shift primitive. The training
demo is therefore no longer a separate manual-gradient implementation.

## differentiable

Native scalar-objective autodiff surface. The core path has no PennyLane or JAX
runtime dependency; optional adapters expose JAX and PennyLane gradients when
those extras are installed.

```python
from scpn_quantum_control import (
    ArmijoLineSearchResult,
    CustomDerivativeCheckResult,
    CustomDerivativeRule,
    CustomDerivativeRegistry,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    DifferentiableOptimizer,
    DualNumber,
    FixedPointSensitivityResult,
    FisherConjugateGradientResult,
    FisherVectorProductResult,
    GradientCheckResult,
    HVPResult,
    HessianResult,
    ImplicitSensitivityResult,
    JVPResult,
    JacobianResult,
    LeastSquaresCovarianceResult,
    LevenbergMarquardtDampingUpdate,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtResult,
    LevenbergMarquardtStep,
    LevenbergMarquardtTrial,
    NaturalGradientOptimizationResult,
    NaturalGradientOptimizer,
    OptimizationResult,
    Parameter,
    ParameterBounds,
    ParameterShiftRule,
    PrimitiveIdentity,
    ReverseNode,
    ShotAllocationResult,
    SparseMatrixResult,
    StochasticGradientResult,
    VJPResult,
    allocate_parameter_shift_shots,
    armijo_backtracking_line_search,
    batch_complex_step_gradient,
    batch_custom_jacobian,
    batch_custom_jvp,
    batch_custom_vjp,
    batch_finite_difference_hvp,
    batch_finite_difference_jvp,
    batch_finite_difference_vjp,
    batch_value_and_parameter_shift_grad,
    batch_value_and_finite_difference_hvp,
    batch_value_and_finite_difference_jvp,
    batch_value_and_finite_difference_vjp,
    batch_value_and_complex_step_grad,
    batch_value_and_custom_jacobian,
    batch_value_and_custom_jvp,
    batch_value_and_custom_vjp,
    batch_vector_jacobian_product,
    check_custom_derivative_consistency,
    check_parameter_shift_consistency,
    complex_step_gradient,
    custom_derivative_rule_for,
    custom_gauss_newton_gradient,
    custom_jacobian,
    custom_jvp,
    custom_levenberg_marquardt_step,
    custom_vjp,
    dual_cos,
    dual_exp,
    dual_log,
    dual_sin,
    dense_to_sparse_matrix,
    empirical_fisher_conjugate_gradient,
    empirical_fisher_vector_product,
    evaluate_levenberg_marquardt_step,
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_hvp,
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
    forward_mode_gradient,
    gauss_newton_gradient,
    grad,
    hessian,
    huber_residual_weights,
    implicit_fixed_point_sensitivity,
    implicit_stationary_sensitivity,
    jacobian,
    least_squares_covariance,
    levenberg_marquardt_step,
    parameter_shift_gradient,
    parameter_shift_gradient_with_uncertainty,
    registered_custom_jacobian,
    registered_custom_jvp,
    registered_custom_vjp,
    register_custom_derivative_rule,
    reverse_cos,
    reverse_exp,
    reverse_log,
    reverse_mode_gradient,
    reverse_sin,
    soft_l1_residual_weights,
    sparse_empirical_fisher_metric,
    sparse_hessian,
    sparse_jacobian,
    update_levenberg_marquardt_damping,
    value_and_complex_step_grad,
    value_and_custom_jacobian,
    value_and_custom_jvp,
    value_and_custom_vjp,
    value_and_finite_difference_grad,
    value_and_finite_difference_hessian,
    value_and_finite_difference_hvp,
    value_and_finite_difference_jacobian,
    value_and_finite_difference_jvp,
    value_and_forward_mode_grad,
    value_and_grad,
    whole_program_grad,
    whole_program_value_and_grad,
    vmap,
    value_and_hessian,
    value_and_jacobian,
    value_and_parameter_shift_grad,
    value_and_reverse_mode_grad,
    vector_jacobian_product,
)

result = value_and_parameter_shift_grad(
    lambda theta: np.sin(theta[0]) + np.cos(theta[1]),
    [0.1, -0.2],
    parameters=[Parameter("theta"), Parameter("phi")],
)
result.value                # scalar objective value
result.gradient             # np.ndarray, same length as parameter vector
result.method               # "parameter_shift"

check = check_parameter_shift_consistency(
    lambda theta: np.sin(theta[0]),
    [0.3],
    tolerance=1e-5,
)
check.passed                # True when parameter shift matches finite difference

updated = DifferentiableOptimizer(learning_rate=0.01).step(
    [0.1, -0.2],
    result,
    bounds=[ParameterBounds(lower=-np.pi, upper=np.pi, periodic=True), ParameterBounds()],
    max_gradient_norm=10.0,
)

opt_result = DifferentiableOptimizer(learning_rate=0.05).minimize(
    lambda theta: 1.0 - np.cos(theta[0]),
    [0.3],
    gradient_method="parameter_shift",
    bounds=[ParameterBounds(lower=-np.pi, upper=np.pi, periodic=True)],
    max_gradient_norm=10.0,
    max_steps=100,
    gradient_tolerance=1e-8,
)
opt_result.reason            # "gradient_tolerance", "value_tolerance", or "max_steps"
opt_result.best_value        # lowest objective value observed

lm_result = LevenbergMarquardtOptimizer(damping=1e-3, max_steps=50).minimize(
    lambda theta: np.array([theta[0] - 1.0, 2.0 * (theta[1] + 0.5)]),
    [3.0, 1.5],
    bounds=[ParameterBounds(), ParameterBounds()],
    weight_fn=lambda residuals: soft_l1_residual_weights(residuals, scale=2.0),
)
lm_result.values             # final accepted parameter vector
lm_result.best_value         # lowest weighted residual objective observed
lm_result.accepted_history   # per-step LM trust-region acceptance flags
```

Available primitives:

```python
ArmijoLineSearchResult(values, value, step_size, direction, directional_derivative, accepted, evaluations, value_history, reason, parameter_names, trainable)
DualNumber(primal, tangent=0.0)
ReverseNode(primal)
Parameter(name: str, trainable: bool = True)
ParameterBounds(lower=None, upper=None, periodic=False)
ParameterShiftRule(shift: float = np.pi / 2, coefficient: float = 0.5)
ShotAllocationResult(shots, predicted_standard_error, covariance, target_standard_error, total_shots, method, parameter_names, trainable)
SparseMatrixResult(row_indices, column_indices, values, shape, method, parameter_names, trainable)
FisherConjugateGradientResult(solution, residual_norm_history, iterations, converged, tolerance, damping, parameter_names, trainable)
FisherVectorProductResult(value, tangent, product, residual_projection, damping, method, evaluations, parameter_names, trainable)
GradientResult(value, gradient, method, shift, coefficient, evaluations, parameter_names, trainable)
StochasticGradientResult(value, gradient, standard_error, covariance, confidence_radius, shots, confidence_level, method, shift, coefficient, evaluations, parameter_names, trainable)
OptimizationResult(values, final_gradient, value_history, steps, converged, reason, best_values=None, best_value=None)
GradientCheckResult(reference, candidate, max_abs_error, l2_error, value_delta, tolerance, passed)
CustomDerivativeCheckResult(custom_jvp, custom_vjp, reference_jvp, reference_vjp, adjoint_inner_error, jvp_l2_error, vjp_l2_error, tolerance, passed)
CustomDerivativeRule(name, value_fn, jvp_rule=None, vjp_rule=None, parameter_names=(), trainable=())
HVPResult(value, hvp, tangent, method, step, evaluations, parameter_names, trainable)
JacobianResult(value, jacobian, method, step, evaluations, parameter_names, trainable)
JVPResult(value, jvp, tangent, method, step, evaluations, parameter_names, trainable)
HessianResult(value, hessian, method, step, evaluations, parameter_names, trainable)
ImplicitSensitivityResult(sensitivity, hessian, cross_derivative, damping, condition_number, method, parameter_names, trainable, hyperparameter_names)
FixedPointSensitivityResult(sensitivity, state_jacobian, parameter_jacobian, system_matrix, damping, condition_number, method, parameter_names, trainable, hyperparameter_names)
LeastSquaresCovarianceResult(covariance, standard_errors, residual_variance, degrees_of_freedom, condition_number, parameter_names, trainable)
NaturalGradientResult(base_gradient, metric, natural_gradient, damping, condition_number)
LevenbergMarquardtDampingUpdate(trial, next_damping, action)
LevenbergMarquardtResult(values, residual, value_history, damping_history, accepted_history, steps, converged, reason, best_values, best_value)
LevenbergMarquardtStep(gauss_newton, step, candidate_values, damping, predicted_reduction)
LevenbergMarquardtTrial(step_result, candidate_residual, candidate_value, actual_reduction, reduction_ratio, accepted)
NaturalGradientOptimizationResult(values, final_gradient, final_natural_gradient, value_history, gradient_norm_history, natural_step_norm_history, steps, converged, reason, best_values, best_value)
NaturalGradientOptimizer(learning_rate=0.01, damping=0.0, rcond=1e-12, max_step_norm=None)
VJPResult(value, cotangent, vjp, method, step, evaluations, parameter_names, trainable)
WeightedGradientResult(value, gradient, components, weights, method, evaluations, parameter_names, trainable)
armijo_backtracking_line_search(objective, values, gradient_result, direction, bounds=None, initial_step=1.0, contraction=0.5, sufficient_decrease=1e-4, max_steps=20) -> ArmijoLineSearchResult
parameter_shift_gradient(objective, values, parameters=None, rule=None) -> np.ndarray
value_and_parameter_shift_grad(objective, values, parameters=None, rule=None) -> GradientResult
parameter_shift_gradient_with_uncertainty(plus_values, minus_values, plus_variances, minus_variances, plus_shots, minus_shots=None, value=0.0, parameters=None, rule=None, confidence_level=0.95, confidence_z=1.959963984540054) -> StochasticGradientResult
allocate_parameter_shift_shots(plus_variances, minus_variances, target_standard_error, parameters=None, rule=None, min_shots=1, max_shots_per_evaluation=None) -> ShotAllocationResult
forward_mode_gradient(objective, values, parameters=None) -> np.ndarray
value_and_forward_mode_grad(objective, values, parameters=None) -> GradientResult
reverse_mode_gradient(objective, values, parameters=None) -> np.ndarray
value_and_reverse_mode_grad(objective, values, parameters=None) -> GradientResult
dual_sin(value) -> DualNumber
dual_cos(value) -> DualNumber
dual_exp(value) -> DualNumber
dual_log(value) -> DualNumber
reverse_sin(value) -> ReverseNode
reverse_cos(value) -> ReverseNode
reverse_exp(value) -> ReverseNode
reverse_log(value) -> ReverseNode
grad(objective, values, parameters=None, method="parameter_shift", rule=None, step=None) -> np.ndarray
value_and_grad(objective, values, parameters=None, method="parameter_shift", rule=None, step=None) -> GradientResult
whole_program_value_and_grad(objective, values, parameters=None, trace=True) -> WholeProgramADResult
whole_program_grad(objective, values, parameters=None, trace=True) -> np.ndarray
vmap(function, in_axes=0, out_axes=0) -> callable
batch_parameter_shift_gradient(objectives, values, parameters=None, rule=None) -> np.ndarray
batch_value_and_parameter_shift_grad(objectives, values, parameters=None, rule=None) -> tuple[GradientResult, ...]
finite_difference_gradient(objective, values, parameters=None, step=1e-6) -> np.ndarray
value_and_finite_difference_grad(objective, values, parameters=None, step=1e-6) -> GradientResult
batch_value_and_finite_difference_grad(objectives, values, parameters=None, step=1e-6) -> tuple[GradientResult, ...]
complex_step_gradient(objective, values, parameters=None, step=1e-30) -> np.ndarray
custom_gauss_newton_gradient(rule, values, parameters=None, weights=None, damping=0.0, rcond=1e-12) -> NaturalGradientResult
custom_jacobian(rule, values, parameters=None) -> np.ndarray
value_and_custom_jacobian(rule, values, parameters=None) -> JacobianResult
custom_jvp(rule, values, tangent, parameters=None) -> np.ndarray
custom_vjp(rule, values, cotangent, parameters=None) -> VJPResult
value_and_complex_step_grad(objective, values, parameters=None, step=1e-30) -> GradientResult
value_and_custom_jvp(rule, values, tangent, parameters=None) -> JVPResult
value_and_custom_vjp(rule, values, cotangent, parameters=None) -> VJPResult
batch_complex_step_gradient(objectives, values, parameters=None, step=1e-30) -> np.ndarray
batch_custom_jvp(rule, values, tangents, parameters=None) -> np.ndarray
batch_value_and_custom_jvp(rule, values, tangents, parameters=None) -> tuple[JVPResult, ...]
batch_custom_vjp(rule, values, cotangents, parameters=None) -> np.ndarray
batch_value_and_custom_vjp(rule, values, cotangents, parameters=None) -> tuple[VJPResult, ...]
batch_custom_jacobian(rule, values, parameters=None) -> np.ndarray
batch_value_and_custom_jacobian(rule, values, parameters=None) -> tuple[JacobianResult, ...]
batch_value_and_complex_step_grad(objectives, values, parameters=None, step=1e-30) -> tuple[GradientResult, ...]
finite_difference_jacobian(objective, values, parameters=None, step=1e-6) -> np.ndarray
value_and_finite_difference_jacobian(objective, values, parameters=None, step=1e-6) -> JacobianResult
jacobian(objective, values, parameters=None, method="finite_difference", step=1e-6) -> np.ndarray
jacfwd(objective, values, parameters=None, method="finite_difference", step=1e-6) -> np.ndarray
jacrev(objective, values, parameters=None, method="finite_difference", step=1e-6) -> np.ndarray
value_and_jacfwd(objective, values, parameters=None, method="finite_difference", step=1e-6) -> JacobianResult
value_and_jacrev(objective, values, parameters=None, method="finite_difference", step=1e-6) -> JacobianResult
value_and_jacobian(objective, values, parameters=None, method="finite_difference", step=1e-6) -> JacobianResult
dense_to_sparse_matrix(matrix, parameter_names=None, trainable=None, method="dense_to_sparse", tolerance=0.0) -> SparseMatrixResult
sparse_jacobian(jacobian_result, tolerance=0.0) -> SparseMatrixResult
sparse_hessian(hessian_result, tolerance=0.0) -> SparseMatrixResult
sparse_empirical_fisher_metric(jacobian, weights=None, damping=0.0, tolerance=0.0) -> SparseMatrixResult
finite_difference_jvp(objective, values, tangent, parameters=None, step=1e-6) -> np.ndarray
value_and_finite_difference_jvp(objective, values, tangent, parameters=None, step=1e-6) -> JVPResult
batch_finite_difference_jvp(objective, values, tangents, parameters=None, step=1e-6) -> np.ndarray
batch_value_and_finite_difference_jvp(objective, values, tangents, parameters=None, step=1e-6) -> tuple[JVPResult, ...]
finite_difference_vjp(objective, values, cotangent, parameters=None, step=1e-6) -> VJPResult
batch_finite_difference_vjp(objective, values, cotangents, parameters=None, step=1e-6) -> np.ndarray
batch_value_and_finite_difference_vjp(objective, values, cotangents, parameters=None, step=1e-6) -> tuple[VJPResult, ...]
batch_vector_jacobian_product(jacobian, cotangents) -> tuple[VJPResult, ...]
vector_jacobian_product(jacobian, cotangent) -> VJPResult
finite_difference_hessian(objective, values, parameters=None, step=1e-4) -> np.ndarray
value_and_finite_difference_hessian(objective, values, parameters=None, step=1e-4) -> HessianResult
hessian(objective, values, parameters=None, method="finite_difference", step=1e-4) -> np.ndarray
value_and_hessian(objective, values, parameters=None, method="finite_difference", step=1e-4) -> HessianResult
implicit_stationary_sensitivity(hessian, cross_derivative, parameters=None, hyperparameter_names=None, damping=0.0, rcond=1e-12) -> ImplicitSensitivityResult
implicit_fixed_point_sensitivity(state_jacobian, parameter_jacobian, parameters=None, hyperparameter_names=None, damping=0.0, rcond=1e-12) -> FixedPointSensitivityResult
finite_difference_hvp(objective, values, tangent, parameters=None, step=1e-5) -> np.ndarray
value_and_finite_difference_hvp(objective, values, tangent, parameters=None, step=1e-5) -> HVPResult
batch_finite_difference_hvp(objective, values, tangents, parameters=None, step=1e-5) -> np.ndarray
batch_value_and_finite_difference_hvp(objective, values, tangents, parameters=None, step=1e-5) -> tuple[HVPResult, ...]
empirical_fisher_metric(jacobian, weights=None, damping=0.0) -> np.ndarray
empirical_fisher_conjugate_gradient(jacobian, rhs, weights=None, damping=1e-8, tolerance=1e-10, max_iterations=None) -> FisherConjugateGradientResult
empirical_fisher_vector_product(jacobian, tangent, weights=None, damping=0.0) -> FisherVectorProductResult
evaluate_levenberg_marquardt_step(objective, step_result, weights=None, acceptance_threshold=1e-4) -> LevenbergMarquardtTrial
gauss_newton_gradient(jacobian, weights=None, damping=0.0, rcond=1e-12) -> NaturalGradientResult
huber_residual_weights(residuals, delta=1.0, min_weight=0.0) -> np.ndarray
least_squares_covariance(jacobian, weights=None, residual_variance=None, damping=0.0, rcond=1e-12) -> LeastSquaresCovarianceResult
levenberg_marquardt_step(jacobian, values, weights=None, damping=1e-3, bounds=None, max_step_norm=None, rcond=1e-12) -> LevenbergMarquardtStep
custom_levenberg_marquardt_step(rule, values, parameters=None, weights=None, damping=1e-3, bounds=None, max_step_norm=None, rcond=1e-12) -> LevenbergMarquardtStep
soft_l1_residual_weights(residuals, scale=1.0, min_weight=0.0) -> np.ndarray
update_levenberg_marquardt_damping(trial, decrease_factor=1/3, increase_factor=2.0, min_damping=1e-12, max_damping=1e12, high_quality_ratio=0.75) -> LevenbergMarquardtDampingUpdate
check_parameter_shift_consistency(objective, values, parameters=None, rule=None, finite_difference_step=1e-6, tolerance=1e-5) -> GradientCheckResult
check_custom_derivative_consistency(rule, values, tangent, cotangent, parameters=None, finite_difference_step=1e-6, tolerance=1e-5) -> CustomDerivativeCheckResult
DifferentiableOptimizer(learning_rate=0.01).step(values, gradient_result, bounds=None, max_gradient_norm=None) -> np.ndarray
DifferentiableOptimizer(...).minimize(objective, initial_values, parameters=None, rule=None, gradient_method="parameter_shift", finite_difference_step=1e-6, bounds=None, max_gradient_norm=None, max_steps=100, gradient_tolerance=1e-8, value_tolerance=None) -> OptimizationResult
NaturalGradientOptimizer(...).minimize(objective, initial_values, metric_fn, parameters=None, rule=None, gradient_method="parameter_shift", finite_difference_step=1e-6, bounds=None, max_steps=100, gradient_tolerance=1e-8, step_tolerance=1e-8, value_tolerance=None) -> NaturalGradientOptimizationResult
LevenbergMarquardtOptimizer(...).minimize(objective, initial_values, parameters=None, bounds=None, weight_fn=None, rcond=1e-12) -> LevenbergMarquardtResult
is_jax_autodiff_available() -> bool
jax_value_and_grad(objective, values) -> tuple[float, np.ndarray]
natural_gradient(gradient_result, metric, damping=0.0, rcond=1e-12) -> NaturalGradientResult
weighted_gradient_sum(components, weights, method="weighted_sum") -> WeightedGradientResult
```

All native and optional-adapter inputs are fail-closed real-numeric boundaries:
parameter arrays, objective return values, optimiser learning rates,
parameter-shift rules, JAX objective values, and gradients reject strings,
booleans, object arrays, complex values, shape mismatches, and non-finite
numbers before training or hardware-adapter code consumes them.
`DifferentiableOptimizer.minimize()` is deliberately bounded: it records scalar
objective history, preserves non-trainable parameters, and exits only through
explicit gradient tolerance, optional value tolerance, or `max_steps`.
`armijo_backtracking_line_search()` provides a fail-closed sufficient-decrease
step selector for scalar objectives, masking frozen directions, projecting
declared bounds, and returning accepted/rejected provenance plus trial values.
The optimizer accepts `gradient_method="parameter_shift"` for gate-generator
objectives and `gradient_method="finite_difference"` for smooth diagnostic
objectives that are not parameter-shift compatible.
`ParameterBounds` applies closed-interval projection to optimizer updates and
initial values, allowing angular, hardware-calibration, or physically admissible
parameter domains to be enforced at the differentiable layer.
For quantum rotation angles, set `periodic=True` with finite `lower` and `upper`
to wrap values into the half-open interval `[lower, upper)` instead of clipping.
`max_gradient_norm` clips only trainable-gradient components before an optimizer
step, which prevents unstable updates while preserving frozen parameters.
`OptimizationResult` also records `best_values` and `best_value`, so callers can
recover the best observed iterate even when the final bounded step is not the
lowest objective point in the run.
`finite_difference_gradient()` is provided as a central-difference diagnostic
and fallback for smooth scalar objectives that are not generated by a
parameter-shift-compatible quantum gate; production gate gradients should keep
using `parameter_shift_gradient()` when the generator rule is known.
`parameter_shift_gradient_with_uncertainty()` propagates independent plus/minus
shot variances into gradient standard errors, diagonal covariance, and
confidence radii. This keeps hardware-calibration gradients honest when finite
shot budgets dominate deterministic numerical error.
`allocate_parameter_shift_shots()` turns plus/minus estimator variances and a
target gradient standard error into integer per-shift shot budgets with
predicted uncertainty. Optional per-evaluation caps deliberately surface when a
budget cannot meet the target instead of hiding residual stochastic risk.
`forward_mode_gradient()` and `value_and_forward_mode_grad()` provide true
dual-number forward-mode automatic differentiation for scalar objectives written
against the `DualNumber` arithmetic contract and supported elementary
primitives. The canonical `grad(..., method="forward_mode")` dispatches through
this exact tangent-propagation path instead of finite differences.
`reverse_mode_gradient()` and `value_and_reverse_mode_grad()` provide
dependency-free tape backpropagation for scalar objectives written against the
`ReverseNode` arithmetic contract and supported elementary primitives. The
canonical `grad(..., method="reverse_mode")` dispatches through this exact
adjoint-propagation path and evaluates the objective once.
`complex_step_gradient()` provides high-accuracy first derivatives for
real-analytic scalar objectives that can safely propagate infinitesimal complex
perturbations; the public parameter boundary remains real-valued, and objective
outputs must still be scalar and finite. The batched complex-step helpers mirror
the parameter-shift and finite-difference batch contracts so multi-objective
calibration workflows retain per-objective value, metadata, and evaluation
provenance.
`check_parameter_shift_consistency()` compares a parameter-shift candidate
against central finite differences and returns explicit error metrics, so custom
rules can be validated before being used in training loops.
`CustomDerivativeRule`, `custom_jvp()`, and `custom_vjp()` provide an exact-rule boundary for primitives with known physics derivatives. Custom rules evaluate the primitive once, enforce trainable masks, reject shape drift, and preserve JVP/VJP provenance without falling back to finite-difference steps.
`PrimitiveIdentity`, `CustomDerivativeRegistry`, and the `registered_custom_*` helpers bind exact custom derivative rules to stable primitive identities such as quantum gate, Hamiltonian, or residual-map implementations. The registry rejects malformed identities and conflicting registrations unless explicit overwrite is requested, allowing polyglot primitive surfaces to resolve derivative rules by identity instead of passing rule objects manually through every call.
`PrimitiveTransformRule` extends that identity binding with primitive-specific batching rules, executable compiler lowering rules, and lowering metadata. `primitive_contract_for()` returns the registered derivative/batching/lowering/shape/dtype/static-argument/policy/effect contract view, while `primitive_complete_contract_for()` fails closed until every compiler- and vectorization-facing facet is declared, including static-argument rules and fail-closed nondifferentiable-boundary metadata. Program AD elementwise primitives (`scpn.program_ad.elementwise:sin@1`, `cos@1`, `exp@1`, `expm1@1`, `log@1`, `log1p@1`, `sqrt@1`, `tan@1`, `tanh@1`, `arcsin@1`, `arccos@1`, `reciprocal@1`, `square@1`, `abs@1`, `negative@1`, `add@1`, `subtract@1`, `multiply@1`, `divide@1`, `power@1`, `maximum@1`, and `minimum@1`), selection primitives (`scpn.program_ad.selection:where@1` and `clip@1`), shape primitives (`scpn.program_ad.shape:reshape@1`, `ravel@1`, and `transpose@1`), reduction primitives (`scpn.program_ad.reduction:sum@1`, `prod@1`, and `mean@1`), product primitives (`scpn.program_ad.product:dot@1`, `vdot@1`, `inner@1`, `outer@1`, and `matmul@1`), cumulative primitives (`scpn.program_ad.cumulative:cumsum@1`, `cumprod@1`, and `diff@1`), array indexing primitives (`scpn.program_ad.array:getitem@1`, `take@1`, `take_along_axis@1`, `delete@1`, `pad@1`, and `insert@1`), and linalg primitives (`scpn.program_ad.linalg:det@1`, `inv@1`, `solve@1`, `trace@1`, `diag@1`, `matrix_power@1`, `multi_dot@1`, `eig@1`, `eigh@1`, `eigvals@1`, `eigvalsh@1`, `svd@1`, and `pinv@1`) are identity-gated through this contract surface before trace execution and expose primitive-specific shape/dtype/static-argument rules plus deterministic NumPy batching rules for mapped scalar/array outputs. Direct registry derivative kernels are available where the flat-vector primitive signature is unambiguous (unary elementwise `sin`, `cos`, `exp`, `expm1`, `log`, `log1p`, `sqrt`, `tan`, `tanh`, `arcsin`, `arccos`, `reciprocal`, `square`, fail-closed `abs`, and `negative`; equal-flat binary elementwise `add`, `subtract`, `multiply`, `divide`, positive-base `power`, tie-fail-closed `maximum`, and tie-fail-closed `minimum`; `sum`, zero-safe `prod`, `mean`; `dot`, `vdot`, and vector `inner` over two equal flat operands; `outer` over two equal flat operands; `matmul` over two equal square-matrix operands; `cumsum`, zero-safe `cumprod`, first-order `diff`; `det`, `inv`, square-matrix `trace`, vector-RHS `solve`, real-simple diagonalizable `eig`, and distinct-spectrum symmetric `eigh`, real-simple diagonalizable `eigvals`, distinct-spectrum symmetric `eigvalsh`, and distinct-positive singular-value `svd`, and constant-full-rank `pinv`). Direct elementwise, reduction, product, cumulative, and feasible linalg kernels expose exact JVP and VJP rules for forward- and reverse-mode registry dispatch; `program_ad_elementwise_binary_derivative_rule()` builds direct value/JVP/VJP rules for fixed broadcasted binary elementwise signatures; `program_ad_selection_where_derivative_rule()` and `program_ad_selection_clip_derivative_rule()` build direct value/JVP/VJP rules for fixed static piecewise-selection and clipping signatures; `program_ad_reduction_sum_derivative_rule()`, `program_ad_reduction_mean_derivative_rule()`, and `program_ad_reduction_prod_derivative_rule()` build direct value/JVP/VJP rules for fixed flat or axis-aware reduction signatures; `program_ad_cumulative_cumsum_derivative_rule()`, `program_ad_cumulative_cumprod_derivative_rule()`, and `program_ad_cumulative_diff_derivative_rule()` build direct value/JVP/VJP rules for fixed flat or axis-aware cumulative signatures; `program_ad_product_inner_derivative_rule()`, `program_ad_product_outer_derivative_rule()`, and `program_ad_product_matmul_derivative_rule()` build direct value/JVP/VJP rules for fixed inner-product, flattened outer-product, and rank-1/rank-2 matmul signatures, including rectangular matrix products; `program_ad_array_getitem_derivative_rule()`, `program_ad_array_take_derivative_rule()`, `program_ad_array_take_along_axis_derivative_rule()`, `program_ad_array_delete_derivative_rule()`, `program_ad_array_pad_derivative_rule()`, and `program_ad_array_insert_derivative_rule()` build direct value/JVP/VJP rules for fixed static gather-index, static-take, and static take-along-axis signatures with repeated-index scatter-add adjoints, including static integer/boolean advanced getitem selectors; `program_ad_shape_reshape_derivative_rule()`, `program_ad_shape_ravel_derivative_rule()`, and `program_ad_shape_transpose_derivative_rule()` build direct value/JVP/VJP rules for fixed shape-transform signatures while base array/shape/reduction registry dispatch remains fail-closed until static metadata is supplied; `program_ad_linalg_solve_derivative_rule()` builds direct value/JVP/VJP rules for fixed vector- or matrix-RHS linear solves, `program_ad_linalg_trace_derivative_rule()`, `program_ad_linalg_diag_derivative_rule()`, and `program_ad_linalg_diagflat_derivative_rule()` build direct diagonal scatter/gather adjoint rules for fixed trace/diagonal signatures, and `program_ad_linalg_matrix_power_derivative_rule()` and `program_ad_linalg_multi_dot_derivative_rule()` build direct value/JVP/VJP rules for fixed static powers and fixed operand-shape sequences, `program_ad_linalg_eig_derivative_rule()` builds exact eigenvalue/eigenvector JVP/VJP rules for fixed real square matrices with real simple diagonalizable eigensystems, and `program_ad_linalg_eigh_derivative_rule()` builds exact eigenvalue/eigenvector JVP/VJP rules for fixed square symmetric matrices with distinct eigenvalues, `program_ad_linalg_eigvals_derivative_rule()` builds exact eigenvalue JVP/VJP rules for fixed real square matrices with real simple diagonalizable spectra, `program_ad_linalg_eigvalsh_derivative_rule()` builds exact spectral JVP/VJP rules for fixed square symmetric matrices with distinct eigenvalues, and `program_ad_linalg_svdvals_derivative_rule()` builds exact singular-value JVP/VJP rules for fixed rank-2 matrices with distinct positive singular values, and `program_ad_linalg_pinv_derivative_rule()` builds exact Moore-Penrose pseudoinverse JVP/VJP rules for fixed rank-2 matrices that stay above a static rank cutoff. Array and shape primitive metadata records static gather-index scatter-add, static-take scatter-add, static take-along-axis scatter-add, static delete scatter-add, static constant-pad scatter-add, static constant-insert scatter-add, element-count-preserving reshape, flat-view, and static-axis-permutation boundaries. Selection primitive metadata records predicate branch and clipping-boundary policies, elementwise primitive metadata records positive-domain, endpoint-singularity, denominator, variable-power, cusp, and tie boundaries where those mathematical domains affect differentiability, reduction and cumulative metadata records static-axis, stable-shape, zero-factor, non-empty, ordered-sequence, and finite-difference spacing boundaries, product primitive metadata records dot, vdot, inner, outer, and matmul contraction-alignment boundaries, and linalg primitive metadata records the singular, diagonal-offset, flattened-diagonal, static-shape, real-simple diagonalizable eigensystem, or distinct-spectrum symmetric-eigenspace, real-simple diagonalizable eigenspace, or distinct-positive singular-value, or rank-threshold nondifferentiable boundary for each primitive; all boundary classes are marked fail-closed for compiler and registry planning. Compiler contracts expose static MLIR signature metadata and derivative-factory names for array, shape, elementwise, reduction, product, cumulative, and linalg planning surfaces, but remain intentionally incomplete until native lowering rules exist. When `vmap(..., primitive_identity=..., registry=...)` is used, the transform dispatches through the registered batching rule instead of the generic eager loop; missing batching rules fail closed. Compiler AD planning consumes the same lowering metadata and lowering-rule presence for MLIR-runtime/Rust/LLVM status reporting.
`check_custom_derivative_consistency()` audits an exact rule pair against the adjoint identity `<J t, c> = <t, J^T c>` and finite-difference JVP/VJP references before the rule is trusted in production control paths.
`custom_jacobian()` and `value_and_custom_jacobian()` materialise exact dense Jacobians from custom JVP columns or VJP rows, preserving trainable masks and `step=0.0` provenance for downstream least-squares, natural-gradient, and benchmark workflows.
The `batch_custom_*` helpers provide exact batched JVP, VJP, and Jacobian transforms over tangent, cotangent, or parameter batches so custom physics derivatives can be benchmarked and vectorised without finite-difference reconstruction.
`custom_gauss_newton_gradient()` and `custom_levenberg_marquardt_step()` feed exact custom residual Jacobians into least-squares optimisation directly, preserving damping, weights, bounds, trainable masks, and solver provenance without finite-difference Jacobian reconstruction.
The `batch_value_*` helpers return one `GradientResult` per scalar objective so
multi-objective workflows keep objective values, gradient metadata, trainable
masks, and evaluation counts instead of only a stacked gradient matrix.
The canonical transform helpers `grad()`, `value_and_grad()`, `jacobian()`, `jacfwd()`, `jacrev()`, `jvp()`, `vjp()`, and
`hessian()` provide stable user-facing names with explicit method dispatch. `jacfwd()` and `jacrev()` are explicit transform-algebra aliases over the current finite-difference Jacobian backend until true forward- and reverse-Jacobian engines land; `jvp()` and `vjp()` are canonical finite-difference directional and adjoint-product transform names over the same validated backend. Tests guarantee their composition semantics without overclaiming backend implementation. Transform nesting contracts cover `grad(vmap(f))`, `vmap(grad(f))`, JVP/VJP consistency against Jacobians, Hessian symmetry, custom derivative rules under `vmap`, and whole-program AD under `vmap`. This
`vmap()` adds a composable eager vectorization transform over selected input axes, with broadcast arguments, nested tuple/list/dict outputs, explicit `out_axes`, and fail-closed shape validation. It is deterministic NumPy execution, not a JIT claim.
`whole_program_value_and_grad()` is the exact operator-intercepted whole-program AD path for differentiable Python programs that execute through traceable scalar and array values. It emits bytecode/source IR metadata, records `WholeProgramIRNode` graph nodes, preserves trainable masks, executes Python loops, local aliasing, list and rank-1/rank-2 array mutation, supported NumPy scalar and vector ufuncs including reciprocal, `log1p`, `expm1`, `tan`, `arcsin`, and `arccos` with singular reciprocal/log1p/tan/inverse-trig boundaries rejected, axis reductions, means, static-axis stack/concatenate, `np.hstack`/`np.vstack`/`np.column_stack`/`np.dstack` convenience assembly, nested `np.block` assembly, static `np.split`/`np.array_split`/`np.hsplit`/`np.vsplit`/`np.dsplit` gather assembly, static `np.tril`/`np.triu` triangular masking with zeroed masked-entry adjoints, static `np.diagonal` offset/axis gather assembly, `np.append`, including `axis=None` concatenate flattening, reshape including static `np.broadcast_to`/`np.broadcast_arrays` broadcast assembly, one inferred `-1` dimension/ravel/repeat/tile/atleast-rank/squeeze/expand-dims/swapaxes/moveaxis/roll/rot90/flip composition, registry-gated clip/norm workflows, dot/vdot/matmul products, determinant, inverse, solve, trace, diagonal extraction/construction, flattened diagonal construction, matrix-power expansion for square rank-2 matrices, and static `multi_dot` rank-1/rank-2 matrix-chain composition, real-simple `np.linalg.eig` eigenvalue/eigenvector outputs, distinct-spectrum symmetric `np.linalg.eigh` eigenvalue/eigenvector outputs, real-simple `np.linalg.eigvals` eigenvalue spectra with reverse adjoint replay, distinct-spectrum symmetric `np.linalg.eigvalsh` eigenvalue spectra with reverse adjoint replay, distinct-positive `np.linalg.svd(..., compute_uv=False)` singular-value spectra with reverse adjoint replay, constant-full-rank `np.linalg.pinv` pseudoinverse matrices with reverse adjoint replay, transpose, registry-gated static integer/boolean getitem gather semantics, registry-gated `np.take` raise/wrap/clip gather semantics, registry-gated `take_along_axis` semantics, registry-gated static `np.delete` gather semantics, registry-gated static constant `np.pad` scatter semantics, registry-gated static constant `np.insert` scatter semantics, registry-gated piecewise `where`/`minimum`/`maximum` semantics, static-condition `np.select` folds, callable `np.piecewise` folds, and executed-branch control flow with derivative-carrying values, and rejects derivative-losing operations such as dynamic indices, `float()` conversion, raw ndarray coercion, and unsupported spectral options (`svd` with `compute_uv=True`, `pinv` outside constant-full-rank static-cutoff semantics) without explicit degeneracy/multiplicity/nondifferentiability primitive policies instead of falling back to finite differences. The differentiable-programming benchmark suite includes static concatenate/stack assembly, stack convenience assembly, nested block assembly, static split-family assembly, static triangular-mask assembly, static diagonal gather assembly, static broadcast-arrays assembly, static advanced-indexing, take raise/wrap/clip, take-along-axis, static delete, and static constant-pad, static constant-insert, and append conformance paths, a selection conformance row for `where`, `clip`, `np.select`, and callable `np.piecewise`, plus a linalg primitive conformance row for `det`, `inv`, `solve`, `trace`, `diag`, `diagflat`, `matrix_power`, `multi_dot`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `svd`, and `pinv` against closed-form analytic derivatives plus optional JAX external-reference rows for loop-heavy, linalg, and transform-nesting cases when that backend is installed. Rust and LLVM/JIT executable whole-program AD lowerings remain blocked until real polyglot interpreter or compiler backends exist.
keeps today's parameter-shift, finite-difference, and complex-step backends
compatible while leaving room for future reverse-mode, forward-mode, sparse, and
implicit-differentiation implementations behind the same public contract.
`implicit_stationary_sensitivity()` implements the implicit-function theorem for
stationary systems by solving `dx*/dalpha = -H^-1 B` on the trainable
subspace. It validates symmetric positive-definite Hessians, supports damping,
tracks condition number, and preserves hyperparameter provenance.
`implicit_fixed_point_sensitivity()` differentiates converged fixed-point maps
`x* = T(x*, alpha)` by solving `(I - dT/dx) dx*/dalpha = dT/dalpha` on the
trainable subspace. It keeps the nonsymmetric fixed-point operator explicit,
supports damping, rejects singular or ill-conditioned maps, and preserves state
and hyperparameter provenance.
`finite_difference_jacobian()` and `value_and_finite_difference_jacobian()`
support vector-valued diagnostics such as multi-observable residual maps while
requiring stable one-dimensional finite outputs across all perturbations.
`SparseMatrixResult` and the `sparse_*` helpers provide dependency-free
coordinate sparse representations for Jacobians, Hessians, and empirical Fisher
metrics. Sparse conversion validates shapes, duplicate coordinates, trainable
masks, and finite values so large structured derivatives can be stored or passed
between control layers without silently corrupting dense semantics.
`value_and_finite_difference_jvp()` computes directional forward-mode products
without materialising a full Jacobian; `vector_jacobian_product()` and
`finite_difference_vjp()` expose reverse-mode cotangent contractions with the
same trainable-parameter masking used by gradients and solvers.
Batched JVP, VJP, and HVP helpers require explicit two-dimensional tangent or
cotangent batches, returning either stacked numeric products or one provenance
object per row for transform-style workflows.
`finite_difference_hessian()` and `value_and_finite_difference_hessian()`
provide central-difference second-order curvature diagnostics for scalar losses;
non-trainable parameters produce zero Hessian rows and columns.
`value_and_finite_difference_hvp()` computes scalar-objective Hessian-vector
products by differentiating finite-difference gradients along a masked tangent,
which supports Newton-CG and trust-region diagnostics without materialising a
full Hessian.
`natural_gradient()` solves a symmetric positive-definite metric system on the
trainable parameter subspace, with optional non-negative damping and condition
number guarding for Fisher/Fubini-Study style preconditioners.
`NaturalGradientOptimizer.minimize()` composes scalar gradients with an explicit
metric callback into a bounded optimization loop, preserving trainable masks,
box or periodic bounds, natural-step clipping, and convergence provenance.
`empirical_fisher_metric()` builds a validated weighted ``J.T @ W @ J`` metric
from `JacobianResult` or raw Jacobian arrays, with optional non-negative damping
for natural-gradient preconditioning.
`empirical_fisher_vector_product()` computes the same weighted Fisher or
Gauss-Newton metric action matrix-free, applies trainable-parameter masking,
and records the residual-space projection used by conjugate-gradient or
trust-region solvers.
`empirical_fisher_conjugate_gradient()` solves trainable empirical-Fisher linear
systems with matrix-free conjugate gradients, retaining residual-norm history
and explicit convergence provenance for large natural-gradient or
Gauss-Newton solves.
`least_squares_covariance()` inverts the trainable empirical-Fisher block to
estimate residual-fit parameter covariance, standard errors, residual variance,
degrees of freedom, and condition number while zeroing frozen-parameter
uncertainty.
`gauss_newton_gradient()` converts a residual-map `JacobianResult` into the
least-squares gradient ``J.T @ W @ r`` and solves the damped Gauss-Newton
metric system on trainable parameters; subtract the returned
`natural_gradient` component from parameters for a residual-minimising update.
`huber_residual_weights()` computes robust IRLS weights for residual maps,
preserving quadratic behaviour near zero residuals while downweighting outliers
before they enter Fisher, Gauss-Newton, or Levenberg-Marquardt solves.
`soft_l1_residual_weights()` provides a smooth differentiable-friendly robust
weighting curve for residual maps, avoiding the Huber kink while retaining
bounded influence for outliers before empirical-Fisher, Gauss-Newton, or
Levenberg-Marquardt solves.
`LevenbergMarquardtOptimizer.minimize()` composes finite-difference residual
Jacobians, optional IRLS residual weights, box or periodic bounds, bounded
adaptive damping, trust-region acceptance, and explicit convergence provenance
into a complete residual-map optimization loop.
outlier influence control for weighted Gauss-Newton and LM solves.
`levenberg_marquardt_step()` turns that preconditioned residual solve into a
bounded candidate update with optional physical bounds, trainable-step norm
limiting, and predicted quadratic-model reduction for accept/reject policies.
`evaluate_levenberg_marquardt_step()` evaluates the candidate residual map,
computes actual weighted residual reduction, and reports the actual/predicted
reduction ratio used by Levenberg-Marquardt trust-region damping policies.
`update_levenberg_marquardt_damping()` converts an LM trial into a bounded
deterministic damping decision: high-quality accepted steps reduce damping,
marginal accepted steps keep it, and rejected steps increase damping for retry.
`weighted_gradient_sum()` combines compatible `GradientResult` components into a
single scalarised multi-objective gradient while preserving component
provenance, weights, evaluation counts, trainable masks, and parameter names.

PennyLane VQE bridge:

```python
from scpn_quantum_control.hardware.pennylane_adapter import PennyLaneRunner

runner = PennyLaneRunner(K, omega, device="default.qubit")
grad_result = runner.vqe_value_and_grad(params, ansatz_depth=1)
grad_result.method          # "pennylane_autodiff"
```

Claim boundary: this is a native differentiable-programming foundation for
scalar SCPN quantum objectives and QSNN training. It is not yet a full
PyTorch/JAX-style parameter-container system, hardware-shot gradient estimator,
or optimiser suite.

## phase

### `structured_ansatz`

```python
build_structured_ansatz(coupling_matrix, reps=2, entanglement_gate="cz", threshold=1e-6) -> QuantumCircuit
```
Generalized topology-informed ansatz. Places entangling gates only between physically connected qubits.

### `lindblad_engine.LindbladSyncEngine`

```python
LindbladSyncEngine(K, omega, gamma=0.1)
    .evolve(t_max, n_steps=100, method="trajectory", initial_state=None, n_traj=20, observables=None) -> dict
```
Open quantum system solver. Supports memory-efficient Monte Carlo Wavefunction (MCWF) trajectory path for N=16 simulation.
Lower-level Lindblad Kuramoto solver inputs use the same strict numeric boundary
as the gate-model XY path: coupling, frequency, and damping-rate values must be
explicit real numeric values, not strings, booleans, objects, or complex values
that NumPy could silently coerce before validation.

### `xy_kuramoto.QuantumKuramotoSolver`

```python
QuantumKuramotoSolver(
    n_oscillators,
    K_coupling,
    omega_natural,
    trotter_order=None,
    evolution_config=None,
)
    .build_hamiltonian() -> SparsePauliOp
    .evolve(time, trotter_steps=None) -> QuantumCircuit
    .measure_order_parameter(statevector) -> tuple[float, float]  # (R, psi)
    .run(t_max: float, dt: float, trotter_per_step: int | None = None) -> dict  # times, R
```

`TrotterEvolutionConfig` carries the default Trotter order and repetition
counts. `run()` uses exact labelled time boundaries: when `t_max` is not an
integer multiple of `dt`, the final interval is shortened so the state evolution
and returned `times[-1]` both end at `t_max`.

### `trotter_upde.QuantumUPDESolver`

```python
QuantumUPDESolver(n_layers=16, knm=None, omega=None)
    .build_hamiltonian() -> SparsePauliOp
    .step(dt, shots=10000) -> dict  # per_layer_R, global_R
    .run(n_steps, dt, shots=10000) -> dict
```

### `phase_vqe.PhaseVQE`

```python
PhaseVQE(K: np.ndarray, omega: np.ndarray, ansatz_reps: int = 2, threshold: float = 0.01)
    .solve(optimizer="COBYLA", maxiter=200, seed: int | None = None) -> dict
```

### `pulse_shaping` (added April 2026)

PMP-optimal ICI pulse sequences (Liu *et al.* 2023) and the unified
$(\alpha,\beta)$-hypergeometric pulse family (Ventura Meinersen *et al.*,
arXiv:2504.08031). All compute paths are Rust-accelerated:

| Function | Speedup vs Python |
|----------|------------------:|
| `hypergeometric_envelope_batch` (10k points) | 44× |
| `ici_three_level_evolution_batch` (2k points) | 1,665× |
| `ici_mixing_angle_batch` | parity-checked |

```python
from scpn_quantum_control.phase.pulse_shaping import (
    build_ici_pulse,
    build_hypergeometric_pulse,
    ici_three_level_evolution,
    hypergeometric_envelope,
    build_trotter_pulse_schedule,
)

# ICI pulse with Lindblad decay simulation
pulse = build_ici_pulse(t_total=2.0, omega_0=20.0, gamma_decay=0.05, n_points=500)
populations = ici_three_level_evolution(pulse)  # shape (500, 3) — P_g, P_e, P_s

# (α,β)-hypergeometric envelope (STIRAP-optimal: α=β=0.5)
hpulse = build_hypergeometric_pulse(t_total=1.0, omega_0=10.0, alpha=0.5, beta=0.5)

# Build a complete Trotter pulse schedule from a coupling matrix
schedule = build_trotter_pulse_schedule(n_qubits=4, k_matrix=K, t_step=0.1)
```

### `hardware.openpulse_control` (added May 2026)

OpenPulse schedule and calibration workflow surfaces for IBM pulse-level lanes.

```python
from scpn_quantum_control.hardware.openpulse_control import (
    compile_hypergeometric_openpulse_schedule,
    build_rabi_amplitude_calibration_workflow,
    estimate_rabi_pi_amplitude,
)

openpulse_schedule = compile_hypergeometric_openpulse_schedule(
    hpulse,
    qubit=1,
    dt=2.22e-10,
)
workflow = build_rabi_amplitude_calibration_workflow(
    backend_name="ibm_fez",
    qubit=1,
    amplitude_grid=[0.1, 0.2, 0.3, 0.4, 0.5],
    shots=4096,
    dt=2.22e-10,
)
fit = estimate_rabi_pi_amplitude(
    amplitudes=[0.1, 0.2, 0.3, 0.4, 0.5],
    excited_population=[0.05, 0.23, 0.61, 0.92, 0.71],
)
```
These routines are calibration-gated and no-submit by design; they create
reviewable pulse/control artefacts without contacting provider endpoints.

## hardware

### `qubit_mapper.dynq_initial_layout` (added April 2026)

Topology-agnostic qubit placement via Louvain community detection on a
calibration-weighted QPU graph. Implements the DynQ method
(Liu *et al.*, arXiv:2601.19635). Quality scoring is Rust-accelerated.

```python
from scpn_quantum_control.hardware.qubit_mapper import dynq_initial_layout

result = dynq_initial_layout(
    gate_errors={(i, j): error_rate for ...},
    circuit_width=4,
    readout_errors={i: error_rate for ...},
    seed=42,
)
result.initial_layout       # list[int] — physical qubits to use
result.selected_region      # ExecutionRegion — quality_score, connectivity, etc.
```

See [`dynq_qubit_mapping.md`](dynq_qubit_mapping.md) for the full
theory and Qiskit transpiler integration recipe.

## mitigation (highlights — see also `mitigation_api.md`)

### `symmetry_decay` — GUESS (added April 2026)

Physics-informed zero-noise extrapolation using the conserved
$\sum Z_i$ as the guide observable for the XY Hamiltonian. Used as
both the noise calibration and the headline scientific result of the
April 2026 ibm_kingston Phase 1 campaign.

```python
from scpn_quantum_control.mitigation.symmetry_decay import (
    learn_symmetry_decay,
    guess_extrapolate,
    xy_magnetisation_ideal,
)

n = 4
s_ideal = xy_magnetisation_ideal(n, "ground")
model = learn_symmetry_decay(
    ideal_symmetry_value=s_ideal,
    noisy_symmetry_values=[3.92, 3.65, 3.10],   # at noise scales 1, 3, 5
    noise_scales=[1, 3, 5],
)
result = guess_extrapolate(target_noisy_value=0.45,
                            symmetry_noisy_value=3.92,
                            decay_model=model)
result.mitigated_value     # corrected target observable
```

See [`symmetry_decay_guess.md`](symmetry_decay_guess.md) for the full
theory and a worked Phase 1 example.

## control

### `topological_optimizer.TopologicalCouplingOptimizer`

```python
TopologicalCouplingOptimizer(n_qubits, initial_K, omega, learning_rate=0.05, dt=1.0)
    .step(n_samples=5) -> dict  # K_updated, p_h1_current, gradient_norm
    .optimize(steps=10, n_samples=3) -> list[dict]
```
Rewires graph topology to minimise topological defects ($p_{h1}$)  in the quantum state.

### `hardware_topological_optimizer.HardwareTopologicalOptimizer`

Hardware-in-the-loop variant using real IBM QPU measurement counts for topological optimization.

### `qaoa_mpc.QAOA_MPC`

```python
QAOA_MPC(B_matrix, target_state, horizon, p_layers=2)
    .build_cost_hamiltonian() -> SparsePauliOp
    .optimize() -> np.ndarray  # action sequence
```

### `vqls_gs.VQLS_GradShafranov`

```python
VQLS_GradShafranov(grid_size=4, source_profile=None)
    .discretize() -> tuple[np.ndarray, np.ndarray]  # (A, b)
    .build_ansatz(n_qubits, reps=2) -> QuantumCircuit
    .solve() -> np.ndarray  # psi profile
```

### `qpetri.QuantumPetriNet`

```python
QuantumPetriNet(n_places, n_transitions, W_in, W_out, thresholds=None)
    .encode_marking(marking) -> QuantumCircuit
    .step_report(marking) -> QuantumPetriStepReport
    .step(marking, shots=1000) -> np.ndarray  # new marking
    .run_campaign(markings) -> QuantumPetriCampaignReport
```
Uses Rust kernels from `scpn_quantum_engine` for transition activity,
state-metric evaluation, and finite-shot sampling when the extension is
available; otherwise uses deterministic NumPy fallback paths.

### `q_disruption.QuantumDisruptionClassifier`

```python
QuantumDisruptionClassifier(n_features=11, n_layers=3)
    .predict(features: np.ndarray) -> float  # risk score [0, 1]
    .train(X, y, epochs=10, lr=0.01)
```

### `q_disruption_iter`

```python
ITERFeatureSpec()  # 11 ITER disruption features with min/max ranges
normalize_iter_features(raw: np.ndarray) -> np.ndarray  # min-max to [0, 1]
generate_synthetic_iter_data(n_samples, disruption_fraction=0.3, allow_synthetic=True) -> (X, y)
from_fusion_core_shot(
    shot_data: dict,
    allow_center_defaults=False,
    allow_density_proxy=False,
) -> (features, label, warnings)
DisruptionBenchmark(n_train=100, n_test=50, allow_synthetic=True).run(epochs=10) -> dict
scpn_control_bridge_dependency_contract() -> dict
validate_scpn_control_bridge_dependency_contract(payload: dict) -> dict
```

The bridge dependency contract mirrors the `scpn-control` fail-closed facade
requirements without importing that package. It records the required
`QuantumDisruptionClassifier(seed=...).predict(normalised_iter_features)` API,
the 11-feature ITER ordering, declared centre-default policy, Qiskit core
dependencies, optional provider families, CONTROL report schemas, and the
downstream non-admission policy.

## applications

### `eeg_classification`

```python
eeg_plv_to_vqe(plv_matrix, natural_frequencies, reps=2, threshold=0.1) -> EEGVQEResult
eeg_quantum_kernel(state_a, state_b) -> float
```
EEG state classification pipeline using Phase Locking Value (PLV) matrices and Structured VQE.

## mitigation

### `compound_mitigation`

```python
compound_mitigate_pipeline(target_circuit, target_counts, run_on_backend, expected_parity, ...)
```
Combines CPDR with Z2 Symmetry Verification for order-of-magnitude noise reduction.

### `zne`

```python
zne_extrapolate(noise_scales: list[int], expectation_values: list[float], order: int = 1) -> ZNEResult
```
Richardson extrapolation to zero noise.

### `dd`

```python
insert_dd_sequence(circuit: QuantumCircuit, idle_qubits: list[int], sequence: DDSequence = DDSequence.XY4) -> QuantumCircuit
```

## hardware

### `fast_classical`

```python
fast_sparse_evolution(K, omega, t_total, n_steps, initial_state=None, delta=0.0) -> dict
```
High-performance sparse evolution engine. Bypasses circuit overhead; supports N=20 systems.

### `runner.HardwareRunner`

```python
HardwareRunner(...)
    .connect()
    .transpile(circuit) -> QuantumCircuit
    .run_sampler(circuits, shots=10000, name="experiment") -> list[JobResult]
```

The default IBM instance CRN is read via `SCPNConfig.ibm_instance`
(see `config` below) when the `[config]` extra is installed, falling
back to `SCPN_IBM_CRN` and then the legacy `SCPN_IBM_INSTANCE`
environment variable otherwise.

### `backends` — plugin registry

```python
from scpn_quantum_control.hardware import (
    describe_backend, describe_hal_backend_profile, get_backend, list_backends,
    list_hal_backend_descriptors, list_quantum_backends,
    register_backend, discover_backends,
)

list_backends()                         # ['qiskit_ibm', 'pennylane', ...]
backend = get_backend("qiskit_ibm")     # BackendProtocol instance
backend.name                            # 'qiskit_ibm'
backend.is_available()                  # True iff installed + importable

descriptor = describe_backend("qiskit_ibm")
descriptor.provider                     # 'ibm_quantum'
descriptor.can_submit                   # True
descriptor.submit_requires_approval     # True
list_quantum_backends()                 # sorted QuantumBackendDescriptor list

hal_route = describe_hal_backend_profile("quera_bloqade")
hal_route.execution_mode                # 'cloud_neutral_atom_analog'
hal_route.adapter_module                # concrete HAL adapter owner
list_hal_backend_descriptors()          # every built-in HAL route profile

# Third-party plugin — register via entry_points in pyproject.toml
# [project.entry-points."scpn_quantum_control.backends"]
# acme_trapped_ion = "acme_plugin:AcmeBackend"
discover_backends()                     # scan entry_points group
```

Every registered backend exposes `name: str` and `is_available() ->
bool`. Discovery is lazy; one broken plugin never blocks the rest.
The HAL descriptor helpers are metadata-only: they do not import provider
SDKs, authenticate, inspect queues, or submit jobs.

### `hal` — provider-neutral execution contract

```python
from scpn_quantum_control.hardware import (
    BackendProfile,
    AzureQuantumHALAdapter,
    BraketLocalHALAdapter,
    CirqLocalHALAdapter,
    DWaveLeapHALAdapter,
    HardwareAbstractionLayer,
    IonQCloudHALAdapter,
    IQMHALAdapter,
    OQCHALAdapter,
    PasqalPulserHALAdapter,
    PennyLaneDeviceHALAdapter,
    QbraidRuntimeHALAdapter,
    QuandelaPercevalHALAdapter,
    QuEraBloqadeHALAdapter,
    QuantinuumCloudHALAdapter,
    RigettiQCSHALAdapter,
    QiskitAerHALAdapter,
    QuantumBackend,
    QuantumWorkload,
    azure_openqasm3_to_workload,
    braket_circuit_to_workload,
    bloqade_ahs_workload,
    built_in_backend_profiles,
    cirq_circuit_workload,
    dwave_bqm_workload,
    ionq_qis_workload,
    iqm_qiskit_workload,
    oqc_openqasm3_workload,
    pulser_sequence_workload,
    pennylane_gate_workload,
    provider_optional_dependency_matrix,
    qbraid_program_to_workload,
    quandela_perceval_workload,
    quantinuum_tket_workload,
    rigetti_quil_workload,
    qiskit_circuit_to_workload,
)

profiles = built_in_backend_profiles()
hal = HardwareAbstractionLayer(profiles)
```

`BackendProfile` records the provider, broker, modality, SDK package,
accepted IR formats, cloud/submission policy, and capability limits. The HAL
validates workload IR, qubit count, shots, backend registration, and explicit
approval before delegating to an injected `QuantumBackend`.

Qiskit adapters are concrete HAL implementations. `qiskit_circuit_to_workload`
uses QPY encoded as base64 for high-fidelity Qiskit circuit transfer, while
`qiskit_circuit_to_qasm3_workload` is available for OpenQASM 3 routes when
`qiskit-qasm3-import` is installed.

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(1, 1)
qc.x(0)
qc.measure(0, 0)

hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))
job = hal.submit("local_qiskit_aer", qiskit_circuit_to_workload(qc, workload_id="x", shots=128))
counts = hal.result(job).counts
```

Braket adapters cover the local SDK simulators and AWS Braket device submission
surface. `BraketLocalHALAdapter` runs local SV/DM simulators; `BraketAwsHALAdapter`
requires an injected device or device ARN and still passes through HAL approval
gating.

```python
from braket.circuits import Circuit

circuit = Circuit().h(0).cnot(0, 1)
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(BraketLocalHALAdapter(hal.profile("local_braket_sv")))
result = hal.result(
    hal.submit(
        "local_braket_sv",
        braket_circuit_to_workload(circuit, workload_id="bell", shots=128),
    )
)
```

Azure Quantum adapters use injected `Target` objects or explicit target
factories. They do not create workspaces or read credentials during HAL
construction.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    AzureQuantumHALAdapter(
        hal.profile("azure_quantum_ionq_simulator"),
        target=target,
    )
)
job = hal.submit(
    "azure_quantum_ionq_simulator",
    azure_openqasm3_to_workload(
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nh q[0];",
        workload_id="azure_h",
        n_qubits=1,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The local Cirq adapter layer provides `CirqLocalHALAdapter` and
`cirq_circuit_workload()`. It executes through an injected Cirq simulator or
simulator factory, normalises measurement histograms into HAL counts, and keeps
local execution on the same lifecycle surface as the cloud adapters.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    CirqLocalHALAdapter(
        hal.profile("local_cirq"),
        circuit_factory=cirq_circuit_factory,
        simulator_factory=cirq_simulator_factory,
    )
)
job = hal.submit(
    "local_cirq",
    cirq_circuit_workload(cirq_payload, workload_id="cirq_bell", n_qubits=2, shots=128),
)
```

The direct OQC adapter layer provides `OQCHALAdapter` and
`oqc_openqasm3_workload()`. It consumes OpenQASM 3 programs, validates the
program header, submits through an injected QCAAS-style client or client
factory, and normalises provider counts into HAL counts. Remote execution
remains approval-gated.

`snapshot_from_oqc_target()` provides the matching no-submit readiness path for
injected OQC target metadata or metadata JSON. It records target name, qubit
count, declared OpenQASM/QIR support, native gate set, shot and circuit limits,
queue depth, online state, simulator flag, topology, and calibration timestamp
without calling QCAAS submission APIs.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    OQCHALAdapter(
        hal.profile("oqc_cloud"),
        client_factory=calibrated_oqc_client_factory,
        target="Lucy",
    )
)
job = hal.submit(
    "oqc_cloud",
    oqc_openqasm3_workload(openqasm3_program, workload_id="oqc_bell", n_qubits=2, shots=128),
    approval_id="approved-run",
)
```

`provider_optional_dependency_matrix()` reports the optional import names and
missing import names for every built-in HAL route without importing SDKs or
touching provider networks.

The `scpn-provider-smoke` command exposes the same no-network matrix for
operator preflight checks after installing provider extras. The portable
`[providers]` extra covers routes whose SDKs resolve cleanly together; D-Wave,
IQM, and QuEra remain dedicated extras for isolated runner environments.
Use `--backend` or `--sdk-package` to turn the command into a narrow preflight
gate, and use `--plan-isolated` to print deterministic virtual-environment
lanes for `dwave_leap`, `iqm_cloud`, and `quera_bloqade`.

The direct Quandela adapter layer provides `QuandelaPercevalHALAdapter` and
`quandela_perceval_workload()`. It consumes `scpn.quandela.perceval.v1`
photonic plans, validates modes, input occupations, optical components, and
postselection bounds, and executes through an injected Perceval processor or
sampler factory. Remote execution remains approval-gated.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    QuandelaPercevalHALAdapter(
        hal.profile("quandela_cloud"),
        processor_factory=calibrated_quandela_processor_factory,
        sampler_factory=calibrated_perceval_sampler_factory,
    )
)
job = hal.submit(
    "quandela_cloud",
    quandela_perceval_workload(photonic_plan, workload_id="quandela_pair", n_modes=2, shots=128),
    approval_id="approved-run",
)
```

`snapshot_from_quandela_processor()` provides the matching no-submit readiness
path for injected Quandela processor metadata or metadata JSON. It records
processor name, mode count, declared Perceval/OpenQASM/MLIR support, optical
component set, photonic feature flags, shot and circuit limits, queue depth,
online state, simulator flag, topology, and calibration timestamp without
calling processor or sampler APIs.

The direct D-Wave Leap adapter layer provides `DWaveLeapHALAdapter` and
`dwave_bqm_workload()`. It consumes `scpn.dwave.bqm.v1`, validates binary
quadratic model structure, submits through an injected D-Wave sampler or
caller-supplied sampler factory, and normalises sample-set occurrences into HAL
counts. Remote Leap execution remains approval-gated.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    DWaveLeapHALAdapter(
        hal.profile("dwave_leap"),
        sampler_factory=calibrated_dwave_sampler_factory,
    )
)
job = hal.submit(
    "dwave_leap",
    dwave_bqm_workload(
        linear={"0": -1.0, "1": 0.5},
        quadratic={("0", "1"): -0.25},
        workload_id="dwave_pair",
        n_variables=2,
        reads=128,
    ),
    approval_id="approved-run",
)
```

`snapshot_from_dwave_solver()` provides the matching no-submit readiness path
for injected D-Wave solver metadata or metadata JSON. It records solver name,
qubit count, declared BQM/Ising/QUBO/MLIR support, annealing topology, read
limits, queue/load estimate, online state, simulator flag, category, and last
update timestamp without calling sampler APIs.

The direct IonQ adapter uses the IonQ Quantum Cloud API v0.4 job lifecycle:
`POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/results/probabilities`, and
`PUT /jobs/{id}/status/cancel`. It accepts IonQ QIS JSON payloads through
`ionq_qis_workload()`, keeps API keys outside the workload, and still requires
HAL approval before submission.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    IonQCloudHALAdapter(
        hal.profile("ionq_cloud"),
        api_key=ionq_api_key,
        backend="simulator",
    )
)
job = hal.submit(
    "ionq_cloud",
    ionq_qis_workload(
        [{"gate": "h", "target": 0}, {"gate": "cnot", "control": 0, "target": 1}],
        workload_id="ionq_bell",
        n_qubits=2,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The direct IQM adapter layer provides `IQMHALAdapter` and
`iqm_qiskit_workload()`. It uses the IQM Qiskit provider path lazily, accepts
injected backend objects for tests or calibrated execution routes, encodes
circuits as QPY-backed `qiskit_qpy` workloads, and preserves HAL approval
gating for remote submission.
`snapshot_from_iqm_backend()` provides the matching no-submit readiness path for
injected IQM backend metadata or metadata JSON. It records target name, qubit
count, declared QPY/Qiskit/OpenQASM support, native gate set, shot and circuit
limits, queue depth, online state, simulator flag, architecture name, and
calibration timestamp without running a circuit.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    IQMHALAdapter(
        hal.profile("iqm_cloud"),
        server_url=iqm_server_url,
        quantum_computer="garnet",
    )
)
job = hal.submit(
    "iqm_cloud",
    iqm_qiskit_workload(qiskit_circuit, workload_id="iqm_bell", shots=128),
    approval_id="approved-run",
)
```

The direct Pasqal adapter layer provides `PasqalPulserHALAdapter` and
`pulser_sequence_workload()`. It consumes the repository's
`pulser_sequence_plan_v1` neutral-atom export schema, validates register
coordinates, Rabi envelope points, detuning terms, interaction terms, and FIM
feedback terms, then submits through an injected Pasqal client or
caller-supplied client factory. Automatic client construction is
calibration-gated.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    PasqalPulserHALAdapter(
        hal.profile("pasqal_cloud"),
        client_factory=calibrated_pasqal_client_factory,
        target="FRESNEL",
    )
)
job = hal.submit(
    "pasqal_cloud",
    pulser_sequence_workload(pulser_plan, workload_id="pasqal_pair", n_qubits=2, shots=128),
    approval_id="approved-run",
)
```

`snapshot_from_pasqal_target()` provides the matching no-submit readiness path
for injected Pasqal target metadata or metadata JSON. It records target name,
atom count, declared Pulser/Pasqal IR/OpenQASM/MLIR support, supported bases,
channel declarations, shot and sequence limits, queue depth, online state,
simulator flag, lattice geometry, and calibration timestamp without calling
Pasqal submission APIs.

PennyLane, qBraid, and Strangeworks adapters are concrete HAL routes, not registry aliases.
`PennyLaneDeviceHALAdapter` executes strict native-gate payloads on a local
PennyLane device and fails closed on unsupported gates. `QbraidRuntimeHALAdapter`
uses injected qBraid devices or providers and still requires the HAL approval
token for cloud submission. `snapshot_from_qbraid_device()` also normalises
qBraid catalogue `program_specs` into HAL IR tokens and records the resolved
broker route without submitting work. `StrangeworksComputeHALAdapter` follows
the same dynamic-catalog contract for injected Strangeworks backends or
workspaces. `snapshot_from_strangeworks_backend()` normalises Strangeworks
catalogue program declarations such as `available_programs` into HAL IR tokens,
records the resolved broker route, and treats backend `state`/`availability`
metadata as no-submit readiness evidence.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(PennyLaneDeviceHALAdapter(hal.profile("local_pennylane")))
pl_result = hal.result(
    hal.submit(
        "local_pennylane",
        pennylane_gate_workload(
            [{"gate": "h", "wires": [0]}, {"gate": "cnot", "wires": [0, 1]}],
            workload_id="pl_bell",
            n_qubits=2,
            shots=128,
        ),
    )
)

hal.register_backend(QbraidRuntimeHALAdapter(hal.profile("qbraid_ionq"), device=qbraid_device))
qbraid_job = hal.submit(
    "qbraid_ionq",
    qbraid_program_to_workload(
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nh q[0];",
        workload_id="qbraid_h",
        ir_format="openqasm3",
        n_qubits=1,
        shots=128,
    ),
    approval_id="approved-run",
)
```

```python
hal.register_backend(
    StrangeworksComputeHALAdapter(
        hal.profile("strangeworks_compute"),
        workspace=strangeworks_workspace,
        backend_id="ionq.simulator",
    )
)
sw_job = hal.submit(
    "strangeworks_compute",
    strangeworks_program_to_workload(
        "OPENQASM 3.0;",
        workload_id="sw_openqasm",
        ir_format="openqasm3",
        n_qubits=1,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The direct QuEra/Bloqade adapter layer provides `QuEraBloqadeHALAdapter` and
`bloqade_ahs_workload()`. It consumes the repository's `bloqade_ahs_plan_v1`
neutral-atom export schema, validates atom geometry and piecewise schedules,
runs an injected Bloqade local or remote routine with `run(shots=..., name=...)`,
normalises `fetch()`/`report()` bitstrings or count mappings into HAL counts,
and cancels batches that expose `cancel()`. Automatic provider-object
construction remains calibration-gated; production callers inject the calibrated
Bloqade routine or a routine factory.
`snapshot_from_quera_bloqade()` provides the matching no-submit readiness path
for injected Bloqade routine metadata or metadata JSON. It records target name,
atom count, declared Bloqade/Braket AHS/MLIR support, native analogue
operations, shot and circuit limits, queue depth, online state, simulator flag,
lattice geometry, and calibration timestamp without running a routine.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    QuEraBloqadeHALAdapter(
        hal.profile("quera_bloqade"),
        routine=calibrated_bloqade_routine,
        routine_name="aquila-approved-route",
    )
)
job = hal.submit(
    "quera_bloqade",
    bloqade_ahs_workload(
        bloqade_ahs_plan,
        workload_id="quera_rydberg_pair",
        n_qubits=2,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The direct Quantinuum adapter layer provides `QuantinuumCloudHALAdapter` and
`quantinuum_tket_workload()`. It follows the pytket-quantinuum execution path:
construct a pytket `Circuit`, compile it with
`QuantinuumBackend.get_compiled_circuit(...)`, submit with
`process_circuit(..., n_shots=...)`, inspect `circuit_status(...)`, retrieve
`get_result(...).get_counts()`, and cancel through `QuantinuumBackend.cancel`.
The direct route is tket-native; OpenQASM 3, QIR, or MLIR must be translated
before submission.

```python
quantinuum_machine = "H1-1E"
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    QuantinuumCloudHALAdapter(
        hal.profile("quantinuum_cloud"),
        machine=quantinuum_machine,
    )
)
job = hal.submit(
    "quantinuum_cloud",
    quantinuum_tket_workload(
        {"name": "h_sample", "qubits": 1},
        workload_id="quantinuum_h",
        n_qubits=1,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The direct Rigetti adapter layer provides `RigettiQCSHALAdapter` and
`rigetti_quil_workload()`. It follows the pyQuil `QuantumComputer` execution
path: construct a Quil `Program`, wrap it in a shot loop, compile it with the
selected QCS quantum computer, run the compiled executable, and normalise the
`ro` readout register into HAL counts. The route is still approval-gated by
HAL, and direct execution accepts Quil workloads only; OpenQASM or MLIR must be
translated before submission.
`snapshot_from_rigetti_qcs()` provides the matching no-submit readiness path
for injected Rigetti QCS `QuantumComputer` metadata or metadata JSON. It records
target name, qubit count, declared Quil/OpenQASM support, native gate set, shot
and queue limits, online state, simulator flag, compiler versions, and
calibration timestamp without compiling or running a program.

```python
rigetti_qc_name = "9q-square-qvm"
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    RigettiQCSHALAdapter(
        hal.profile("rigetti_qcs"),
        quantum_computer_name=rigetti_qc_name,
    )
)
job = hal.submit(
    "rigetti_qcs",
    rigetti_quil_workload(
        "DECLARE ro BIT[1]\nH 0\nMEASURE 0 ro[0]",
        workload_id="rigetti_h",
        n_qubits=1,
        shots=128,
    ),
    approval_id="approved-run",
)
```

### `async_runner.AsyncHardwareRunner`

```python
import asyncio
from scpn_quantum_control.hardware import AsyncHardwareRunner

async def main():
    async_runner = AsyncHardwareRunner([runner_a, runner_b], max_concurrent=2)
    handles = await async_runner.submit_batch_async(
        [circuits_a, circuits_b], shots=4096, name="dla_parity",
    )
    results = await async_runner.wait_all_async(handles)

asyncio.run(main())
```

Fans out `sampler.run(...)` calls across one or more
`HardwareRunner` instances via `asyncio.to_thread`. Bounded by an
`asyncio.Semaphore`; round-robin across runners. Legacy sync
`HardwareRunner` remains unchanged.

### `provenance.capture_provenance`

```python
from scpn_quantum_control.hardware.provenance import capture_provenance

prov = capture_provenance()             # returns a JSON-serialisable dict
# prov["git"]["commit"], prov["runtime"]["python"], etc.
```

Records git commit + branch + dirty flag, Python / package /
`scpn_quantum_engine` versions, hostname (hashed when
`SCPN_ANONYMOUS_HOSTNAME=1`, also available via
`SCPNConfig.anonymous_hostname`), and UTC timestamp.

## qec

### `biological_surface_code.BiologicalSurfaceCode`

```python
BiologicalSurfaceCode(K, threshold=1e-5)
    .verify_css_commutation() -> bool
    .estimate_logical_qubits() -> int
    .code_summary() -> dict[str, int | bool]
    .x_syndrome_from_z_errors(z_errors) -> np.ndarray
    .apply_z_correction(z_errors, correction) -> np.ndarray
```
Native topological error correction code mapped directly to the hierarchical SCPN coupling graph.
`K` must be a finite square symmetric zero-diagonal coupling matrix.
`threshold` must be finite and non-negative; edges with
`abs(K[i, j]) >= threshold` are data qubits.

### `biological_surface_code.BiologicalMWPMDecoder`

```python
BiologicalMWPMDecoder(code)
    .decode_z_errors(syndrome_x) -> np.ndarray
    .decode_and_apply(z_errors) -> tuple[np.ndarray, np.ndarray]
    .last_decoder_backend -> str
```
Minimum Weight Perfect Matching decoder using biological coupling strengths as distance metrics.
`syndrome_x` must be a one-dimensional binary vector with length equal
to the number of X stabilizers. Because this graph decoder does not
model rough boundaries, every connected component must have even
syndrome parity; odd component parity raises `ValueError` instead of
silently discarding an unmatched defect.
When Rust acceleration is installed, exact MWPM is used for defect sets
up to the current exact solver limit; larger defect sets automatically
fallback to the Python NetworkX decoder and expose the selected path via
`last_decoder_backend`.

### `biological_diagnostics.BiologicalSurfaceDiagnostics`

```python
analyse_biological_surface_code(code, node_domains=None, metadata=None)
    -> BiologicalSurfaceDiagnostics
```
Biology-oriented diagnostics over the coupling graph used by the biological
surface code. Reports weighted-degree and betweenness criticality, cycle-basis
burden, community modularity, and optional inter-domain coupling aggregates.

### `biological_pipeline.BiologicalQecExecution`

```python
run_biological_qec_execution(
    K,
    z_errors,
    threshold=1e-5,
    node_domains=None,
    metadata=None,
) -> BiologicalQecExecution
```
Campaign-facing end-to-end helper that constructs the biological code,
computes diagnostics, executes decode/correction, and emits JSON-serialisable
payloads via `BiologicalQecExecution.to_payload()`.

Batch lane:
```python
run_biological_qec_batch_execution(
    K,
    z_error_matrix,
    threshold=1e-5,
    node_domains=None,
    metadata=None,
) -> BiologicalQecBatchExecution
```
Aggregates multi-pattern campaign runs with decode-backend counts, success
rate, and mean syndrome/correction/residual metrics.

CLI entry point:
```bash
scpn-biological-qec-report --k K.npy --z-errors z.npy --output report.json
```
`--z-errors` accepts either a one-dimensional vector (single run) or a
two-dimensional matrix (batch campaign).

### `control_qec.ControlQEC`

```python
ControlQEC(distance=3)
    .protect_signal(circuit) -> QuantumCircuit
    .decode_syndrome(syndrome) -> np.ndarray  # correction
```

## config — unified runtime configuration

Available via the `[config]` extra (pydantic-settings).

### `SCPNConfig`

```python
from scpn_quantum_control.config import SCPNConfig, get_config, reload_config

cfg = get_config()                  # process-wide singleton
cfg.anonymous_hostname              # bool — SCPN_ANONYMOUS_HOSTNAME
cfg.ibm_instance                    # str  — SCPN_IBM_CRN, fallback SCPN_IBM_INSTANCE
cfg.ibm_backend                     # str  — SCPN_IBM_BACKEND
cfg.ibm_channel                     # str  — "ibm_cloud" | "ibm_quantum"
cfg.ibm_shots                       # int  — default shot count
cfg.gpu_enable                      # bool — SCPN_GPU_ENABLE
cfg.jax_disable                     # bool — SCPN_JAX_DISABLE
cfg.result_dir                      # Path — results directory
cfg.figure_dir                      # Path — figure output directory
cfg.log_level                       # "DEBUG"|"INFO"|"WARNING"|"ERROR"|"CRITICAL"
cfg.log_format                      # "console" | "json"

# Tests that mutate os.environ must call reload_config():
monkeypatch.setenv("SCPN_IBM_SHOTS", "1024")
cfg = reload_config()               # discards cached instance
```

Layered sources, highest priority first:

1. Explicit kwargs: `SCPNConfig(ibm_shots=8192)`.
2. `SCPN_*`-prefixed environment variables.
3. A `.env` file in the current working directory.
4. Pydantic-declared defaults.

## logging_setup — structlog bootstrap

Available via the `[logging]` extra (structlog).

```python
from scpn_quantum_control.logging_setup import configure_logging, get_logger

configure_logging()                 # once, at application start
log = get_logger(__name__)
log.info("submitted_job", backend="ibm_kingston", shots=4096)
```

`configure_logging(level=None, format=None, force=False)`:

* `level` defaults to `SCPNConfig.log_level`.
* `format` defaults to `SCPNConfig.log_format`; non-TTY stderr
  auto-downgrades `console` → `json`.
* `force=True` reconfigures even if the same arguments have been
  applied before.

Stdlib `logging.getLogger(...).info(...)` calls route through the
same structlog processor chain — ISO-UTC timestamps, log-level tag,
stack info, context-var merging.

## accel — multi-language dispatcher

The `accel/` package exposes compute functions through a
Rust → Julia → Python chain. Rust is the default when the optional
`scpn-quantum-engine` wheel is installed. Julia kicks in via the
`[julia]` extra. Python is the always-available correctness floor.

```python
from scpn_quantum_control.accel import (
    order_parameter, last_tier_used, available_tiers,
    MultiLangDispatcher, dispatch,
)

r = order_parameter(theta)          # Rust (default) → Julia → Python
last_tier_used()                    # 'rust' | 'julia' | 'python' | None
available_tiers()                   # ['rust', 'julia'] on a full install

# Generic name-based dispatch
r = dispatch("order_parameter", theta)
```

Custom chain for a new compute function:

```python
from scpn_quantum_control.accel import MultiLangDispatcher

disp = MultiLangDispatcher([
    ("rust",   _rust_impl),         # ImportError falls through
    ("julia",  _julia_impl),        # RuntimeError falls through
    ("python", _python_impl),       # floor — MUST be last
])
result = disp(arg)
disp.last_tier                      # which tier served the call
```

See `docs/pipeline_performance.md` §"Multi-language accel chain" for
wall-time measurements and `docs/language_policy.md` for the
ordering rules.
