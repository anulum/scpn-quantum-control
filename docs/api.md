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
from scpn_quantum_control import MLIRCompileConfig, compile_kuramoto_to_mlir

module = compile_kuramoto_to_mlir(
    problem,
    MLIRCompileConfig(time=0.4, trotter_steps=2, trotter_order=2),
)
module.text      # deterministic MLIR-style textual IR
module.sha256    # digest over module.text
```

The MLIR compiler surface emits deterministic, auditable Kuramoto-XY IR with
explicit omega terms, coupling terms, Trotter parameters, resource counts, and
claim-boundary metadata. It is an interchange layer; it does not claim LLVM/QIR
lowering, cloud submission, pulse compilation, or hardware execution.

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
    DifferentiableOptimizer,
    GradientCheckResult,
    HessianResult,
    JacobianResult,
    LevenbergMarquardtStep,
    OptimizationResult,
    Parameter,
    ParameterBounds,
    ParameterShiftRule,
    batch_value_and_parameter_shift_grad,
    check_parameter_shift_consistency,
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_jacobian,
    gauss_newton_gradient,
    levenberg_marquardt_step,
    parameter_shift_gradient,
    value_and_finite_difference_grad,
    value_and_finite_difference_hessian,
    value_and_finite_difference_jacobian,
    value_and_parameter_shift_grad,
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
```

Available primitives:

```python
Parameter(name: str, trainable: bool = True)
ParameterBounds(lower=None, upper=None, periodic=False)
ParameterShiftRule(shift: float = np.pi / 2, coefficient: float = 0.5)
GradientResult(value, gradient, method, shift, coefficient, evaluations, parameter_names, trainable)
OptimizationResult(values, final_gradient, value_history, steps, converged, reason, best_values=None, best_value=None)
GradientCheckResult(reference, candidate, max_abs_error, l2_error, value_delta, tolerance, passed)
JacobianResult(value, jacobian, method, step, evaluations, parameter_names, trainable)
HessianResult(value, hessian, method, step, evaluations, parameter_names, trainable)
NaturalGradientResult(base_gradient, metric, natural_gradient, damping, condition_number)
LevenbergMarquardtStep(gauss_newton, step, candidate_values, damping, predicted_reduction)
WeightedGradientResult(value, gradient, components, weights, method, evaluations, parameter_names, trainable)
parameter_shift_gradient(objective, values, parameters=None, rule=None) -> np.ndarray
value_and_parameter_shift_grad(objective, values, parameters=None, rule=None) -> GradientResult
batch_parameter_shift_gradient(objectives, values, parameters=None, rule=None) -> np.ndarray
batch_value_and_parameter_shift_grad(objectives, values, parameters=None, rule=None) -> tuple[GradientResult, ...]
finite_difference_gradient(objective, values, parameters=None, step=1e-6) -> np.ndarray
value_and_finite_difference_grad(objective, values, parameters=None, step=1e-6) -> GradientResult
batch_value_and_finite_difference_grad(objectives, values, parameters=None, step=1e-6) -> tuple[GradientResult, ...]
finite_difference_jacobian(objective, values, parameters=None, step=1e-6) -> np.ndarray
value_and_finite_difference_jacobian(objective, values, parameters=None, step=1e-6) -> JacobianResult
finite_difference_hessian(objective, values, parameters=None, step=1e-4) -> np.ndarray
value_and_finite_difference_hessian(objective, values, parameters=None, step=1e-4) -> HessianResult
empirical_fisher_metric(jacobian, weights=None, damping=0.0) -> np.ndarray
gauss_newton_gradient(jacobian, weights=None, damping=0.0, rcond=1e-12) -> NaturalGradientResult
levenberg_marquardt_step(jacobian, values, weights=None, damping=1e-3, bounds=None, max_step_norm=None, rcond=1e-12) -> LevenbergMarquardtStep
check_parameter_shift_consistency(objective, values, parameters=None, rule=None, finite_difference_step=1e-6, tolerance=1e-5) -> GradientCheckResult
DifferentiableOptimizer(learning_rate=0.01).step(values, gradient_result, bounds=None, max_gradient_norm=None) -> np.ndarray
DifferentiableOptimizer(...).minimize(objective, initial_values, parameters=None, rule=None, gradient_method="parameter_shift", finite_difference_step=1e-6, bounds=None, max_gradient_norm=None, max_steps=100, gradient_tolerance=1e-8, value_tolerance=None) -> OptimizationResult
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
`check_parameter_shift_consistency()` compares a parameter-shift candidate
against central finite differences and returns explicit error metrics, so custom
rules can be validated before being used in training loops.
The `batch_value_*` helpers return one `GradientResult` per scalar objective so
multi-objective workflows keep objective values, gradient metadata, trainable
masks, and evaluation counts instead of only a stacked gradient matrix.
`finite_difference_jacobian()` and `value_and_finite_difference_jacobian()`
support vector-valued diagnostics such as multi-observable residual maps while
requiring stable one-dimensional finite outputs across all perturbations.
`finite_difference_hessian()` and `value_and_finite_difference_hessian()`
provide central-difference second-order curvature diagnostics for scalar losses;
non-trainable parameters produce zero Hessian rows and columns.
`natural_gradient()` solves a symmetric positive-definite metric system on the
trainable parameter subspace, with optional non-negative damping and condition
number guarding for Fisher/Fubini-Study style preconditioners.
`empirical_fisher_metric()` builds a validated weighted ``J.T @ W @ J`` metric
from `JacobianResult` or raw Jacobian arrays, with optional non-negative damping
for natural-gradient preconditioning.
`gauss_newton_gradient()` converts a residual-map `JacobianResult` into the
least-squares gradient ``J.T @ W @ r`` and solves the damped Gauss-Newton
metric system on trainable parameters; subtract the returned
`natural_gradient` component from parameters for a residual-minimising update.
`levenberg_marquardt_step()` turns that preconditioned residual solve into a
bounded candidate update with optional physical bounds, trainable-step norm
limiting, and predicted quadratic-model reduction for accept/reject policies.
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
```

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
