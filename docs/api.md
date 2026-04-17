# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — API Reference

# API Reference

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

### `xy_kuramoto.QuantumKuramotoSolver`

```python
QuantumKuramotoSolver(n_oscillators, K_coupling, omega_natural, backend=None)
    .build_hamiltonian() -> SparsePauliOp
    .evolve(time, trotter_steps=1) -> QuantumCircuit
    .measure_order_parameter(statevector) -> tuple[float, float]  # (R, psi)
    .run(t_max: float, dt: float, trotter_per_step: int = 5) -> dict  # times, R
```

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
    .step(marking, shots=1000) -> np.ndarray  # new marking
```

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
generate_synthetic_iter_data(n_samples, disruption_fraction=0.3) -> (X, y)
from_fusion_core_shot(shot_data: dict) -> (features, label, warnings)
DisruptionBenchmark(n_train=100, n_test=50).run(epochs=10) -> dict
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
back to the raw `SCPN_IBM_INSTANCE` environment variable otherwise.

### `backends` — plugin registry

```python
from scpn_quantum_control.hardware import (
    get_backend, list_backends, register_backend, discover_backends,
)

list_backends()                         # ['qiskit_ibm', 'pennylane', ...]
backend = get_backend("qiskit_ibm")     # BackendProtocol instance
backend.name                            # 'qiskit_ibm'
backend.is_available()                  # True iff installed + importable

# Third-party plugin — register via entry_points in pyproject.toml
# [project.entry-points."scpn_quantum_control.backends"]
# acme_trapped_ion = "acme_plugin:AcmeBackend"
discover_backends()                     # scan entry_points group
```

Every registered backend exposes `name: str` and `is_available() ->
bool`. Discovery is lazy; one broken plugin never blocks the rest.

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
```
Native topological error correction code mapped directly to the hierarchical SCPN coupling graph.

### `biological_surface_code.BiologicalMWPMDecoder`

```python
BiologicalMWPMDecoder(code)
    .decode_z_errors(syndrome_x) -> np.ndarray
```
Minimum Weight Perfect Matching decoder using biological coupling strengths as distance metrics.

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
cfg.ibm_instance                    # str  — SCPN_IBM_INSTANCE
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
