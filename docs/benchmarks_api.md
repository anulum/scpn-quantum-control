# Benchmarks API Reference

The `benchmarks` package measures the computational frontier: at what
system size does quantum hardware outperform the best classical methods
for simulating Kuramoto-XY dynamics? Seven modules answer this from
different angles — documented classical baselines, exact diagonalisation,
GPU statevector, MPS tensor networks, application-oriented metrics, and
differentiable-programming conformance.

7 modules, differentiable-programming conformance rows, TN/MPS baseline design,
and 3 crossover estimates.

Rust kernel execution-mode evidence is tracked separately from benchmark timing.
`tools/audit_rust_kernel_execution.py` writes static SIMD/threading inventory
artefacts such as
`data/rust_kernel_execution/rust_kernel_execution_audit_2026-06-16.json`; those
rows classify PyO3 kernels as scalar/unknown, ndarray-dot, rayon-threaded, or
explicit-SIMD evidence only. They do not make performance claims. Timing
promotion still requires the isolated benchmark metadata described below.

## Architecture

```
Classical Methods                    Comparison                     Quantum Methods
──────────────────                  ──────────                      ───────────────
SciPy ODE / QuTiP / MPS ← classical_baselines → provenance envelope
  Phase ODE, Lindblad, TEBD                           honest optional status

Exact diag + expm       ← quantum_advantage →  Trotter on statevector
  O(2^n × 2^n × 2^n)                              O(n^2 × reps × 2^n)
  Limit: n ≈ 14                                    Limit: n ≈ 23 (RAM)

GPU statevector (A100)  ← gpu_baseline →        QPU gate execution
  O(2^n × n_gates)                                 O(n_gates × gate_time)
  Memory: 2^n × 16B                                Unlimited
  Limit: n ≈ 33 (80 GB)                           Limit: decoherence

MPS tensor network      ← mps_baseline →        Quantum correlations
  O(chi^3 × n × gates)                            Native entanglement
  chi ~ 2^S (entropy)                              No truncation needed
  Fails: volume-law S

AppQSim protocol        ← appqsim_benchmark →   Application fidelity
  Exact ground truth                                VQE / Trotter output
```

## Module Reference

### 1. `classical_baselines` — Documented Reference Backends

Provides explicit baseline runs and availability reporting:

- `scipy_ode_baseline` — classical Kuramoto ODE via SciPy `solve_ivp`.
- `qutip_lindblad_baseline` — optional density-matrix open-system baseline
  via QuTiP `mesolve`.
- `mps_tebd_baseline` — optional tensor-network baseline via quimb TEBD.
- `run_documented_classical_baselines` — runs the baseline suite for one
  `K_nm`/`omega` problem.

See [Classical Baselines](classical_baselines.md) for the provenance contract
and examples.

### 1b. `tn_mps_baseline_design` — N=30-40 TN/MPS Baseline Plan

Builds the no-claim design artifact for the S2 tensor-network baseline
follow-up:

- `build_tn_mps_baseline_design` — deterministic CPU-first N=30-40 design manifest.
- `render_tn_mps_baseline_design_markdown` — human-reviewable report.

The design selects the Python/quimb CPU MPS adapter as the first execution path,
keeps the bounded native Schmidt/resource model as a deterministic scaffold, and
blocks ITensor/Julia, GPU TN, tensor-network-hardness, and broad-advantage claims
until measured rows exist.

See [TN/MPS Baseline Design](tn_mps_baseline_design.md) for the generated
artifact and regeneration command.

### 1c. `tn_mps_crossover_stage1` — QWC-5.1 Crossover Gate

Builds the QWC-5.1 stage-1 gate for larger-than-16-node N=30-40
tensor-network crossover rows:

- `build_tn_mps_crossover_stage1` — deterministic row-schema and admission
  report.
- `validate_tn_mps_crossover_rows` — future measured or skipped row validator.
- `render_tn_mps_crossover_stage1_markdown` — human-reviewable report.

The gate pins required fields for wall time, memory, max bond, discarded
weight, entropy proxy, truncation policy, omitted coupling mass, command,
machine, dependency, commit, host-load, claim-boundary, and notes. It keeps
stage-2 N=30-40 compute owner-gated and leaves `advantage_claim_allowed=false`.

See [TN/MPS Crossover Stage-1 Gate](tn_mps_crossover_stage1.md) for the
generated artifact and regeneration command.

### 1d. `josephson_magnitude_study` — Josephson K_nm Magnitude Study

Builds the QWC-5.2 no-claim preregistration artifact for the Josephson
K_nm measured-magnitude follow-up:

- `build_josephson_knm_magnitude_study_design` — deterministic design manifest
  for the N=14 rho=0.989947 topology candidate and larger N=20/30/40 targets.
- `render_josephson_knm_magnitude_study_markdown` — human-reviewable report.

The design intentionally separates topology correlation from physical coupling
magnitude validation. It blocks promotion until calibrated Josephson/transmon
coupling units, locked normalisation, uncertainty, direct magnitude, spectral
response, and null-model gates pass.

See [Josephson K_nm Magnitude Study](josephson_knm_magnitude_study.md) for the
generated artifact and regeneration command.

### 2. `quantum_advantage` — Two Classical Simulation Methods, Scaling Crossover

Measures wall-clock time for exact classical simulation (exact
diagonalisation + matrix exponential) against Trotter evolution on
the statevector simulator.

> **Estimate vs measurement, classical vs quantum — read this first.**
> Despite the module and function names, **both** `classical_benchmark` and
> `quantum_benchmark` are *classical-hardware measurements of two classical
> simulation methods*: dense exact evolution versus statevector-Trotter
> simulation. Neither is a quantum-processor timing, and the `estimate_crossover`
> point is where one *classical* method overtakes the other — **not** a
> quantum-advantage crossover. The `gpu_baseline` and `mps_baseline` timings are
> *analytic estimates*, not simulations (they are labelled as such in the tables
> below). The one surface that actually *decides* the advantage question at
> matched budget — and stays fail-closed `inconclusive` without a QPU row — is
> §2b `decisive_advantage_protocol` / §2c `decisive_run_harness`.
>
> **Why no broad advantage is expected here.** The Kuramoto-XY model sits at a
> BKT-type critical point, where the half-chain entanglement grows only
> logarithmically, `S ~ (c/3)·log n` with `c = 1`. An MPS therefore needs only a
> bond dimension `χ ~ n^{1/3}` — always tractable — so the classical
> tensor-network spoofer keeps pace across the accessible NISQ range and the
> honest expected outcome is a crossover-only or classical-wins label, never
> broad quantum advantage. (This mirrors `mps_baseline.quantum_advantage_n`,
> which only predicts intractability under *volume-law* entanglement, `S ~ n/2`,
> which this model does not exhibit.)

#### `classical_benchmark(n, t_max=1.0, dt=0.1)`

Times classical exact evolution of the XY Hamiltonian:
1. Build K_nm (Paper 27) and compile to dense matrix
2. Compute matrix exponential exp(-iHdt) for each time step
3. Evolve state by matrix-vector multiplication
4. Also performs exact diagonalisation for ground energy

For n > 14, returns `t_total_ms = inf` (2^14 = 16,384 state dimension;
matrix expm requires O(2^3n) operations, ~4.4 trillion for n=14).

Returns: `{t_total_ms, ground_energy, R_final}`

#### `quantum_benchmark(n, t_max=1.0, dt=0.1, trotter_reps=5)`

Times Trotter evolution on statevector:
1. Build Kuramoto initial state: `Ry(omega_i)|0>` per qubit
2. Compile Hamiltonian to `SparsePauliOp`
3. Construct `PauliEvolutionGate` with `LieTrotter` synthesis
4. Evolve for n_steps = t_max / dt, each with `trotter_reps` repetitions

Returns: `{t_total_ms, n_trotter_steps}`

Statevector simulation is O(2^n × n_gates) — exponential in n but polynomial
in circuit depth. For n=20, the statevector is 16 MB (feasible); for n=30
it is 16 GB (GPU territory).

Benchmark result provenance records the current git commit through an admitted
absolute-path `git` executable. Missing, non-executable, or failing `git`
commands fail closed to `unknown` without blocking benchmark execution.

#### `estimate_crossover(results)`

Fits exponential scaling `t = a * exp(b * n)` to both the dense-exact and the
statevector-Trotter timings. The crossover is where the two *classical* curves
intersect:

```
n_cross = log(a_q / a_c) / (b_c - b_q)
```

Returns `int | None`. Returns None if fewer than 3 data points or if the
statevector-Trotter method is not asymptotically cheaper than dense exact
evolution. This is a crossover *between two classical simulation methods* — it
carries no quantum-advantage claim; the advantage question is decided only by
§2b/§2c.

#### `run_scaling_benchmark(sizes=None, t_max=1.0, dt=0.1)`

Full scaling benchmark across system sizes. Default: `[4, 8, 12, 16, 20]`.

```python
from scpn_quantum_control.benchmarks import run_scaling_benchmark

results = run_scaling_benchmark(sizes=[4, 8, 12])
for r in results:
    print(f"n={r.n_qubits}: classical={r.t_classical_ms:.1f}ms, "
          f"quantum={r.t_quantum_ms:.1f}ms")
if results[0].crossover_predicted:
    print(f"Predicted crossover: n={results[0].crossover_predicted}")
```

Warns if n > 23 (statevector memory > 128 MB).

#### `AdvantageResult`

| Field | Type | Description |
|-------|------|-------------|
| `n_qubits` | int | System size |
| `t_classical_ms` | float | Classical exact evolution time (inf if infeasible) |
| `t_quantum_ms` | float | Trotter statevector evolution time |
| `errors` | dict | Error metrics (optional) |
| `crossover_predicted` | int or None | Extrapolated crossover qubit count |

---

### 2b. `decisive_advantage_protocol` — Single-Decision Advantage Gate

Where `advantage_protocol` collects the full S2 size-by-baseline matrix, this
module narrows it to the one comparison the quantum-advantage gap contract
requires to be *decided*: one observable, one system size, one accuracy target,
at a matched wall-time budget. It is the design layer for a reproducible
hardware benchmark — not an advantage claim.

The decision rule is **fail-closed**. `evaluate_decision` returns
`qpu_decides_advantage` only when a QPU row strictly beats a *present*
best-classical row at the target size within accuracy and budget; every
ambiguous or under-populated case degrades to
`exact_hilbert_space_crossover_only`, `classical_wins`, or `inconclusive`. Per
the repository physics (classical exact/MPS/ODE win through the accessible NISQ
range) the expected outcome of the default protocol is a crossover or
classical-wins label.

Every decision baseline produces the **same** dynamical observable — the
synchronisation order parameter `R` at the final evolution time — so the
comparison is like-for-like. The best-classical path is `classical_ode` and
`mps_tensor_network`; the exact-only reference is `dense_statevector_evolution`.
Ground-state eigensolvers are excluded: they answer the spectral-gap question,
not the dynamical order-parameter question a Trotterised QPU run decides, and a
sparse Krylov propagation earns its keep only above the dense memory wall
(`n ≳ 13`), outside this single decision size.

| Symbol | Purpose |
|--------|---------|
| `DecisionCriterion` | Observable, target size, accuracy target, matched budget, and the best-classical / exact baseline labels |
| `SubmissionGate` | Depth and shot ceilings a QPU submission must not exceed (`check` is fail-closed) |
| `DecisiveAdvantageProtocol` | Single-size scaling protocol + criterion + gate + conservative QPU-time estimate; `validate_rows` delegates to `validate_scaling_rows` |
| `DecisionOutcome` | The `DecisionLabel` and its justifications |
| `default_decisive_advantage_protocol()` | The preregistered order-parameter `R` decision at `n = 12` (near the ~11.6 exact-Hilbert-space crossover), 1 % accuracy, matched 60 s budget |
| `evaluate_decision(protocol, rows)` | Validate measured rows and return the fail-closed `DecisionOutcome` |

```python
from scpn_quantum_control.benchmarks.decisive_advantage_protocol import (
    default_decisive_advantage_protocol,
    evaluate_decision,
)

protocol = default_decisive_advantage_protocol()
outcome = evaluate_decision(protocol, measured_rows)
print(outcome.label)  # e.g. "classical_wins"
```

The `scripts/report_decisive_advantage_gate.py` CLI emits the preregistration
manifest (no `--rows`) or the decision (`--rows rows.json`).

---

### 2c. `decisive_run_harness` — Measured Classical-Baseline Harness

Where `decisive_advantage_protocol` declares *which* comparison to decide, this
module *runs* the classical side of it at the preregistered decision size and
emits schema-valid rows that `evaluate_decision` can score. It never fabricates a
quantum row: with no QPU credits the `qpu_hardware` column is absent, so the
honest outcome of the default protocol is `inconclusive` — the gate stays closed.

Each baseline reuses an existing solver and produces the dynamical `R`:
`dense_statevector_evolution` reuses `classical_exact_evolution` (the exact
reference, `reference_error` zero); `classical_ode` reuses `scipy_ode_baseline`;
`mps_tensor_network` reuses `mps_tebd_baseline`, degrading to an explicit
size-gated skip when the tensor-network extra is absent rather than fabricating a
value.

| Symbol | Purpose |
|--------|---------|
| `DecisiveRunConfig` | Deterministic run inputs (`t_max`, `dt`, `mps_bond_dim`, `include_mps`, `reserved_core`) |
| `DecisiveRunArtifact` | Serialisable record: rows, delegated validation, fail-closed decision, provenance, host-isolation grade |
| `run_decisive_benchmark(protocol=None, config=None, *, host_readiness=None)` | Run the classical baselines and assemble the artifact |
| `dense_reference_row` / `ode_row` / `mps_row` | Per-baseline row builders (the reference returns its final `R`) |
| `git_commit` / `command_line` / `dependency_versions` | Provenance stamps read from the live environment, never guessed |

**Measurement grade.** Wall-clock timing is *measured*; `memory_bytes` is the
documented analytic memory model shared with the other benchmark surfaces
(statevector `2ⁿ·16 B`, MPS `n·2·χ²·16 B`, ODE `n_times·n·8 B`). The artifact
records the host-isolation verdict from `capture_host_readiness`; on a shared
host the timings are labelled `advisory_shared_host`, and because the decision is
`inconclusive` without a QPU row, no advantage is ever claimed on advisory
timings.

```python
from scpn_quantum_control.benchmarks.decisive_run_harness import (
    DecisiveRunConfig,
    run_decisive_benchmark,
)

artifact = run_decisive_benchmark(config=DecisiveRunConfig(t_max=1.0, dt=0.1))
print(artifact.decision["label"])   # "inconclusive" — no QPU row present
print(artifact.timing_grade)        # e.g. "advisory_shared_host"
```

The `scripts/run_decisive_advantage_benchmark.py` CLI emits the preregistration
manifest (default) or runs the classical baselines and writes the artifact
(`--run`, with `--no-mps`, `--t-max`, `--dt`, `--reserved-core` options).

---

### 2d. `layout_method_comparison` — DynQ vs Optimiser vs SABRE

Compares three layout methods for the XY-Trotter circuit on one problem,
coupling map, and calibration set: the DynQ layout as-is, the DynQ layout
refined by the discrete Kuramoto optimiser
(`hardware/kuramoto_layout_optimiser.py`), and Qiskit's SABRE. Routed depth and
two-qubit gate counts are *measured* on the transpiled circuits (shared basis,
seeded routing — reproducible); the success probability is an *analytic model*
(product of `1 − gate_error` over every routed two-qubit gate priced at its
calibrated edge), and `R proxy = p · R_ideal` degrades the method-independent
ideal order parameter under a global depolarising model. Neither model is a
hardware measurement, and the artifact says so in its notes.

| Symbol | Purpose |
|--------|---------|
| `LayoutComparisonConfig` | Run inputs (`t`, `reps`, `order`, shared `seed`, DynQ and search settings) |
| `LayoutComparisonArtifact` | Serialisable record: rows, `r_ideal`, host verdict, provenance, honest labels; renders the Markdown table |
| `run_layout_method_comparison(gate_errors, K, omega, ...)` | Full comparison; fails closed when no DynQ region fits |
| `routed_layout_metrics` | Measured depth/2Q-count + calibration-priced success model (injectable) |
| `ideal_xy_order_parameter` | Exact statevector `R` of the compiled circuit (injectable) |
| `coupling_map_from_gate_errors` | Symmetric `CouplingMap` over the calibrated edges |

The providers (`r_provider`, `metrics_provider`, `depth_provider`) are
injectable, so the assembly logic is fully testable without transpilation; the
default depth provider is wrapped with the run seed for a reproducible
optimiser cost landscape. Measured results and the honest reading live in
`dynq_qubit_mapping.md` §8.4; the CLI is
`scripts/run_layout_method_comparison.py`.

---

### 2e. `closed_loop_publication_run` — Reproducible Software-in-the-Loop Artifact

Packages the software-in-the-loop closed-loop surfaces
(`measure_closed_loop_latency_budget`, `build_closed_loop_publication_package`,
the dynamic-circuit template builders) into one reproducible artifact with
rigorously honest labels: the latency budget is a *software budget/telemetry
surface, not a hardware measurement*; the OpenQASM 3 dynamic-circuit templates
(mid-circuit measurement + `if_test` conditionals, with content digests) are
*exportable but un-run*; the claim ledger keeps provider-prepared and live-QPU
claims `blocked`. Provenance and the host-isolation timing grade follow the
same pattern as the other run harnesses. The latency measurer and controller
factory are injectable for tests. Details in `closed_loop_control.md`; the CLI
is `scripts/run_closed_loop_publication.py`.

---

### 2f. `hls_cosimulation_evidence` — Hash-Bound HLS Co-Simulation Evidence

Compiles the generated Vivado/Vitis HLS pulse-player testbench under a host
compiler (`g++`) against the packaged non-synthesis shim
(`codegen/hls_host_shim`), executes it (bit-true `PASS <n>` verdict), and
binds every input to the verdict with SHA-256 digests: header, testbench,
XDC, shim headers, compile command, and compiler identity. The boundary is
explicit — *codegen + software co-simulation only, no synthesis, no timing
closure, no board execution* — and the handoff artifact names the
SC-NEUROCORE consumer contract (`sc-neurocore.hdl_gen.hls_ingest.v1`).
Fail-closed: a missing compiler raises; compile/run failures are recorded as
`passed=false` failure evidence. The co-simulation runner is injectable for
tests. Details in `ultrascale_hls.md`; the CLI is
`scripts/run_hls_cosimulation_evidence.py`.

---

### 3. `gpu_baseline` — GPU vs QPU Comparison

Estimates GPU resources needed for statevector simulation and compares
with QPU execution time.

#### GPU Model

NVIDIA A100 80 GB:
- 312 TFLOPS FP64
- 80 GB HBM2e

Statevector simulation:
- Memory: `2^n × 16 bytes` (complex128)
- FLOPs: `n_gates × 2^n × 10` (matrix-vector with constant factor)
- Time: FLOPs / TFLOPS

| n | Memory | GPU Time (A100) |
|---|--------|----------------|
| 16 | 1 MB | 0.003 ms |
| 24 | 256 MB | 0.8 ms |
| 30 | 16 GB | 50 s |
| 33 | 128 GB | OOM |
| 40 | 16 TB | Infeasible |

#### QPU Model

Conservative sequential gate execution:
- Gate time: 0.5 us (Heron r2 CZ)
- Time: `n_gates × 0.5 us`

For XY Trotter circuit: `n_gates = reps × (n(n-1)/2 CZ + 2n RZ)`

#### `gpu_baseline_comparison(n, trotter_reps=10)`

Returns `GPUBaselineResult` with GPU time, QPU time, and crossover.

```python
from scpn_quantum_control.benchmarks.gpu_baseline import gpu_baseline_comparison

result = gpu_baseline_comparison(n=20, trotter_reps=10)
print(f"GPU: {result.estimated_gpu_time_s:.2e}s")
print(f"QPU: {result.qpu_time_s:.2e}s")
print(f"GPU faster: {result.gpu_faster}")
print(f"Crossover: n={result.crossover_n}")
```

#### `scaling_comparison(n_values=None)`

Batch comparison across system sizes. Default: `[4, 8, 16, 24, 32, 40]`.
Returns dict with columns: n, gpu_time_s, qpu_time_s, memory_gb, gpu_faster.

#### Utility Functions

| Function | Returns |
|----------|---------|
| `statevector_memory_gb(n)` | GPU memory in GB |
| `statevector_flops(n, n_gates)` | Total FLOPs |
| `estimate_gpu_time(n, n_gates, tflops)` | Wall time in seconds |
| `estimate_qpu_time(n, n_gates, gate_time_us)` | Wall time in seconds |
| `gate_count_xy_trotter(n, reps)` | Total gate count |

#### `GPUBaselineResult`

| Field | Type | Description |
|-------|------|-------------|
| `n_qubits` | int | System size |
| `n_gates` | int | Total gate count |
| `statevector_memory_gb` | float | GPU memory requirement |
| `statevector_flops` | float | Computation cost |
| `estimated_gpu_time_s` | float | A100 wall time |
| `qpu_time_s` | float | QPU wall time |
| `gpu_faster` | bool | True if GPU is faster |
| `crossover_n` | int | n where QPU wins |

---

### 4. `mps_baseline` — Tensor Network Comparison

Matrix Product State (MPS) resource estimation for the Kuramoto-XY
system. MPS provides the classical baseline: if MPS at affordable bond
dimension matches the quantum simulation, there is no quantum advantage.

#### MPS Theory

Bond dimension chi controls MPS expressibility:
- chi = 1: product states only (zero entanglement)
- chi = 2^(n/2): exact representation (full Hilbert space)
- chi ~ poly(n): efficient classical simulation

The required chi is set by the half-chain entanglement entropy S:

```
chi >= 2^S
```

For the Kuramoto-XY system at different coupling regimes:
- Below BKT (weak coupling): S ~ log(n), chi ~ poly(n) — MPS efficient
- At BKT critical point: S ~ (c/3) log(n) with c=1, chi ~ n^(1/3) — MPS efficient
- Above BKT (strong coupling): S ~ n/2 (volume law), chi ~ 2^(n/2) — MPS fails

The quantum advantage boundary is where MPS fails: when the half-chain
entropy implies chi > chi_max (limited by available RAM).

#### `required_bond_dimension(entropy)`

Minimum chi from entanglement entropy: `chi = ceil(2^S)`.

#### `mps_memory(n, chi)`

Memory for MPS: `n × 2 × chi^2 × 16 bytes` (n tensors of shape
(chi, 2, chi) in complex128).

#### `quantum_advantage_n(chi_max=1024, entropy_per_qubit=0.5)`

Estimates system size where MPS fails under volume-law entanglement:

```
S = entropy_per_qubit × n/2
chi = 2^S > chi_max  ⟹  n > 2 × log2(chi_max) / entropy_per_qubit
```

For chi_max=1024, entropy_per_qubit=0.5: n > 40.
For chi_max=256: n > 32.

#### `mps_baseline_comparison(K, omega, chi_max=256)`

Full comparison for a specific system:

```python
from scpn_quantum_control.benchmarks.mps_baseline import mps_baseline_comparison
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

K = build_knm_paper27(L=8)
omega = OMEGA_N_16[:8]
result = mps_baseline_comparison(K, omega)
print(f"Entropy S = {result.half_chain_entropy:.3f}")
print(f"Required chi = {result.required_bond_dim}")
print(f"MPS memory: {result.mps_memory_bytes / 1e6:.1f} MB")
print(f"Exact memory: {result.exact_memory_bytes / 1e6:.1f} MB")
print(f"Compression: {result.compression_ratio:.1f}x")
print(f"MPS tractable: {result.mps_tractable}")
```

#### `MPSBaselineResult`

| Field | Type | Description |
|-------|------|-------------|
| `n_qubits` | int | System size |
| `half_chain_entropy` | float | S(n/2) from exact ground state |
| `required_bond_dim` | int | chi = ceil(2^S) |
| `mps_memory_bytes` | int | MPS storage requirement |
| `exact_memory_bytes` | int | Full statevector storage |
| `compression_ratio` | float | exact/MPS memory ratio |
| `quantum_advantage_threshold` | int | n where MPS at chi_max fails |
| `mps_tractable` | bool | chi_required <= chi_max |

---

### 5. `appqsim_protocol` — Application-Oriented Metrics

Reference: Lubinski et al., QST 8, 024003 (2023).

Measures simulation quality via application-relevant metrics, not just
circuit fidelity. For the Kuramoto-XY system:

1. **Order parameter accuracy**: `|R_quantum - R_exact|`
2. **Energy accuracy**: `|E_q - E_exact| / |E_exact| × 100%`
3. **Correlation fidelity**: `1 - ||C_q - C_exact||_F / ||C_exact||_F`

These answer the physics question: does the quantum simulation correctly
reproduce the synchronisation transition? A VQE that gets the energy
right but the correlators wrong is not useful for studying phase transitions.

#### `appqsim_benchmark(K, omega, circuit_sv=None, n_gates=0, circuit_depth=0)`

Full AppQSim evaluation:

```python
from scpn_quantum_control.benchmarks.appqsim_protocol import appqsim_benchmark
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

K = build_knm_paper27(L=4)
omega = OMEGA_N_16[:4]
metrics = appqsim_benchmark(K, omega)
print(f"R error: {metrics.order_parameter_error:.4f}")
print(f"Energy error: {metrics.energy_relative_error_pct:.2f}%")
print(f"Correlation fidelity: {metrics.correlation_fidelity:.4f}")
```

If `circuit_sv` is not provided, internally runs VQE (ansatz_reps=2,
maxiter=100) to generate the quantum state.

The correlation fidelity computes the Frobenius-norm distance between
quantum and exact `<X_i X_j + Y_i Y_j>` correlator matrices over all
qubit pairs.

#### `AppQSimMetrics`

| Field | Type | Description |
|-------|------|-------------|
| `order_parameter_error` | float | |R_q - R_exact| |
| `energy_relative_error_pct` | float | |E_q - E_exact| / |E_exact| × 100 |
| `correlation_fidelity` | float | 1 - ||C_q - C_exact||_F / ||C_exact||_F |
| `n_qubits` | int | System size |
| `n_gates` | int | Circuit gate count |
| `circuit_depth` | int | Circuit depth |

---

### 6. `differentiable_programming` — Program AD Conformance

Provides deterministic correctness rows for the native differentiable
programming surface. These rows compare implemented program AD gradients
against analytic references and explicitly avoid wall-clock, compiler, LLVM,
Rust, JIT, or hardware performance claims.

#### `run_differentiable_programming_benchmark_suite()`

Runs the committed conformance rows:

| Case | Category | Contract |
|------|----------|----------|
| `loop_heavy_scalar` | loop-heavy | Executed Python loops with scalar ufuncs |
| `program_ad_ir_roundtrip_contracts` | ir-roundtrip | Bounded `program_ad_effect_ir.v1` parser and stable serialization round-trip conformance for emitted Program AD SSA/effect/control/phi metadata; bytecode/source compiler frontend, full alias lattice, non-executed branch semantics, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `program_ad_rust_scalar_interpreter_contracts` | rust-interpreter | Optional native Rust scalar forward/value-gradient replay of opcode-bearing `program_ad_effect_ir.v1` rows with parity against Python whole-program and analytic references when `scpn_quantum_engine` is built; BL-02 dynamic-boundary fail-closed evidence covers dynamic indexing, dynamic axes, dynamic ddof/correction, dynamic q/method, dynamic trapezoid-grid metadata, and zero-variance `std` gradients through the public Rust value+gradient export. Python-only environments report explicit blocked reasons and do not promote Rust execution, reverse-mode Rust AD, general Program AD execution, LLVM/JIT, provider, hardware, or performance evidence |
| `program_ad_control_phi_metadata_contracts` | control-phi | Program AD control-join metadata conformance for supported executed runtime and source control regions, with `ProgramADPhiNode` parser round-trip, analytic gradient parity, and adjoint replay parity; non-executed branch adjoints, full compiler phi lowering, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `program_ad_registry_dispatch_contracts` | registry-dispatch | Registry-dispatched coverage for 118 declared Program AD primitives across array, shape, reduction, stencil, interpolation, assembly, signal, elementwise, selection, product, cumulative, and linalg families; the row validates derivative, batching, lowering metadata, shape, dtype, static-argument, nondifferentiability, and effect contracts only. `mirror_program_ad_registry_metadata_with_rust()` can validate the same snapshot through the optional Rust metadata mirror and report primitive-name overlap with bounded Rust scalar/static-linalg, compact interpolation, compact signal, compact stencil, compact cumulative, and elementwise/static-structural replay, while executable registry Rust, dynamic array semantics, LLVM/JIT, provider, hardware, and performance evidence remain blocked |
| `program_adjoint_replay_provenance_contracts` | reverse-adjoint | Program AD reverse adjoint generation over supported executed scalar IR, with `program_adjoint_replay_gradient()` executable step-stream parity, `ProgramADAdjointResult` gradient parity, generated `ProgramADAdjointStep` rows, finite local pullback scales, cotangent-flow rows, reverse effect-order rows, replay node/effect/runtime control/phi row bindings, and blocked non-executed phi inputs bound to `program_ad_effect_ir.v1`; full reverse-mode compiler AD, non-executed branch adjoints, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `elementwise_boundary_contracts` | elementwise-boundary | Registry-gated builtin `abs`, NumPy absolute value, positive-domain, nonzero-denominator, and inverse-trig boundary contracts with analytic gradient and adjoint parity checks; unsupported domain boundaries, derivative-losing `sign`/`heaviside` kernels, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `matrix_heavy_linear_algebra` | matrix-heavy | Dot, inner, outer, trace, tensordot, and einsum semantics |
| `selection_piecewise_contracts` | selection-heavy | Registry-gated `where`/`clip` branch and boundary contracts, strict no-tie `sort`, static selection folds with `np.select`, callable `np.piecewise`, static-selector `np.choose`, static-mask `np.compress`, and same-size static-mask `np.extract`, plus fail-closed integer-output selector contracts exposed through the dashboard selection primitive row; dynamic masks, dynamic selectors, ties, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `structured_numeric_primitive_contracts` | structured-numeric | Registry-gated product, interpolation, signal, and stencil contracts for `inner`, `outer`, `matmul`, `tensordot`, `einsum`, `interp`, `convolve`, `correlate`, and `gradient`; compact Rust value+gradient replay is covered for static-grid `interp`, static rank-1 `convolve`/`correlate`, and static scalar/coordinate-spacing `gradient` Program AD IR, while dynamic interpolation grids, dynamic signal promotion, singular stencil spacing, LLVM/JIT executable lowering, hardware, and performance promotion remain blocked |
| `cumulative_primitive_contracts` | cumulative-primitive | Registry-gated bounded `cumsum`, `cumprod`, and `diff` trace contracts with analytic gradient and adjoint parity checks; dynamic axis promotion, LLVM/JIT executable lowering, hardware, and performance promotion remain blocked; compact Rust value+gradient replay is covered by focused bridge tests |
| `assembly_primitive_contracts` | assembly-primitive | Registry-gated like-constructor and stack assembly contracts for `zeros_like`, `ones_like`, `full_like`, `hstack`, `vstack`, `column_stack`, and `dstack`; dynamic shape assembly, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `reduction_primitive_contracts` | reduction-primitive | Registry-gated bounded `sum`, `prod`, `mean`, `var`, `std`, `trapezoid`, unique `max`/`min`, `median`, scalar-`q` `quantile`, and scalar-`q` `percentile` contracts with analytic gradient and adjoint parity checks; dynamic axes, dynamic q/method metadata, dynamic ddof/correction metadata, dynamic trapezoid-grid metadata, zero-variance `std` gradients, tie boundaries, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `shape_primitive_contracts` | shape-primitive | Registry-gated bounded reshape, ravel, transpose, expand/squeeze, swap/move axis, repeat, rank-promotion, tile, roll, rot90, flip, flipud, and fliplr contracts with analytic gradient and adjoint parity checks; dynamic shape arguments, invalid axes, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `broadcast_primitive_contracts` | broadcast-primitive | Registry-gated bounded `broadcast_to`, `broadcast_arrays`, and binary elementwise rank-broadcasting contracts with analytic gradient and adjoint parity checks; dynamic output shapes, incompatible shapes, subclass propagation, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `linalg_primitive_contracts` | linalg-primitive | Registry-gated determinant, inverse, solve, trace, diagonal, flattened diagonal, matrix-power, and multi-dot contracts |
| `indexing_static_gather_contracts` | indexing-heavy | Static slicing, static-axis concatenate/stack assembly, `np.hstack`/`np.vstack`/`np.column_stack`/`np.dstack` assembly conveniences, nested `np.block` assembly, static `np.split`/`np.array_split`/`np.hsplit`/`np.vsplit`/`np.dsplit` gather assembly, static `np.tril`/`np.triu` triangular masks, static `np.diagonal` offset/axis gather assembly, static `np.broadcast_arrays` broadcast assembly, static integer/boolean advanced getitem, `np.take` raise/wrap/clip modes, `np.take_along_axis`, static `np.delete`, static constant `np.pad`, static constant `np.insert`, `np.append`, strict finite no-tie `np.sort` adjoint routing, static-grid `np.trapezoid` adjoint routing, static scalar and coordinate `np.gradient` finite-difference adjoint routing, static-grid `np.interp` piecewise-linear adjoint routing, one-dimensional `np.convolve` signal/kernel adjoint routing, one-dimensional `np.correlate` signal/reference adjoint routing, and repeated adjoint accumulation |
| `mutation_heavy_forward_only` | mutation-heavy | Static array mutation dataflow |
| `shape_view_alias_metadata_contracts` | alias-effect | Program AD alias metadata conformance for supported executed shape/view transformations; full static alias lattice, non-executed view/control paths, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `slice_mutation_alias_metadata_contracts` | alias-effect | Program AD alias metadata conformance for static rank-1 slice mutation; broader object aliases, non-executed view/control paths, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `loop_carried_state_alias_metadata_contracts` | alias-effect | Program AD source metadata conformance for loop-carried derivative state; full loop checkpointing, non-executed paths, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `program_ad_static_alias_lattice_contracts` | alias-lattice | Static alias-lattice readiness over emitted `program_ad_effect_ir.v1` components with typed view-alias provenance, typed list-alias provenance, typed loop-carried state provenance, typed control-path alias provenance, typed rebinding-alias provenance, bounded local object-attribute, explicit mutation-effect blockers, malformed view-alias blockers, malformed list-alias blockers, malformed loop-carried state blockers, malformed rebinding-alias blockers, malformed control-path alias blockers, unsupported-Python frontend diagnostic blockers, captured/global object-attribute roots/details pinned to static object-model blockers, unknown alias-edge provenance pinned to fail-closed blockers, non-executed phi blockers, and control-path alias blocker reporting; captured/global object-attribute alias sets, malformed control/view/list/loop/rebinding-alias promotion, unknown dynamic alias promotion, arbitrary dynamic-Python frontend lowering, mutation adjoints, non-executed branch adjoints, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked |
| `transform_nesting_vmap_program_grad` | transform-nesting | `vmap` over program AD gradients plus whole-program `grad(vmap(f))` over trace-aware leaves |
| `transform_nesting_whole_program_higher_order` | transform-nesting | `jacfwd` and `jacrev` over whole-program `grad(vmap(f))` checked against analytic block-diagonal curvature |

#### `run_differentiable_programming_external_reference_suite()`

Runs optional JAX reference comparisons when JAX is installed. When JAX is not
available, it returns an empty tuple rather than weakening the base dependency
contract.

#### `run_quantum_gradient_benchmark_suite()`

Runs deterministic parameter-shift correctness rows for small smooth quantum
expectation objectives. Each row records the parameter-shift gradient, central
finite-difference gradient, analytic reference gradient, finite-difference
verification pass/fail flag, objective-evaluation count, and a claim boundary.

| Case | Category | Contract |
|------|----------|----------|
| `single_rotation_parameter_shift` | quantum-gradient | One-parameter Pauli-rotation expectation with analytic `-sin(theta)` reference |
| `two_parameter_phase_expectation` | quantum-gradient | Two-parameter phase expectation with analytic mixed sine/cosine reference |
| `sparse_ising_chain_six_qubit_expectation` | quantum-gradient | Six-parameter nearest-neighbour sparse Ising-chain expectation with analytic field/coupling gradient reference |
| `jax_registered_phase_qnode_aot_export_lowering` | quantum-gradient | Optional installed-JAX AOT/export value-route diagnostic for deterministic registered local Phase-QNode circuits; gradient fields remain parameter-shift and finite-difference references while exported VJP, persistent cross-platform execution, hardware, and performance promotion remain blocked |
| `torch_registered_phase_qnode_compile_boundary_diagnostic` | quantum-gradient | Optional installed-PyTorch compile-boundary diagnostic for deterministic registered local Phase-QNode circuits; non-fullgraph correctness is compared with parameter-shift and finite-difference references while dynamic-shape, fullgraph compiled-frame, AOTAutograd/export, CUDA, provider, hardware, isolated-benchmark, and performance promotion remain blocked |

These rows are correctness/conformance benchmarks only. They do not claim
hardware execution, provider integration, framework-native autodiff, or
wall-clock performance.

#### `run_differentiable_external_comparison_suite()`

Runs optional external comparison rows for JAX, PyTorch, TensorFlow,
PennyLane, LLVM/Enzyme runner evidence, and Catalyst qjit/MLIR/QIR runner
evidence. The SCPN analytic reference remains the source of truth. Missing
optional dependencies are emitted as `hard_gap` rows instead of being omitted.

Every row carries dependency-version metadata for the backend being classified.
The suite also emits explicit unsupported-route rows for promotion-blocking
cases: unsupported batching, unsupported nested transforms, unsupported complex
dtype routes, and unsupported hardware-device routes. Those rows are hard gaps,
not skipped tests or degraded successes.
Catalyst rows additionally carry a dedicated `catalyst_comparison` payload that
names qjit/MLIR/QIR compiled-workflow scope, compiled-differentiation scope,
control-flow gaps, finite-shot limitations, unsupported provider routes, and a
claim boundary. A configured runner success is still bounded CPU correctness
evidence only; it does not promote arbitrary Catalyst workflows, finite-shot
jobs, provider submission, or performance claims.

#### `write_differentiable_external_comparison()`

Writes the external comparison rows to a JSON artefact with schema
`scpn_qc_differentiable_external_comparison_v1`. The artefact records the
row payloads, dependency versions, toolchain metadata where available, failure
classes, Python/platform metadata, and the fixed `functional_non_isolated`
classification. It is a reproducibility and correctness artefact only:
`production_eligible` and `promotion_ready` are false until the isolated
benchmark gate supplies artefact IDs and the claim ledger is updated.
The writer publishes a `row_schema.required_fields` list and rejects rows that
do not carry value error, gradient error, runtime, memory, batching support,
transform support, failure class, dependency versions, toolchain slot, and
claim-boundary fields. The schema also includes the `catalyst_comparison` slot;
it is `null` for non-Catalyst rows and required for every Catalyst row.
`scripts/run_differentiable_benchmark_evidence.py` writes this companion JSON
file as `diff-qnode-external-comparison.json` and inserts the real
external-comparison artefact ID into the benchmark evidence bundle's
`evidence_artifact_ids` list.

#### `run_identical_circuit_gradient_comparison_suite()`

Runs a stricter exact-state competitor-gradient comparison for the same
registered Phase-QNode circuit across SCPN, Qiskit, and PennyLane:

- Circuit: one-qubit `RY(theta)`.
- Parameters: `[0.4]`.
- Observable: `Z0`.
- Shot policy: exact-state mode, `shots=None`.
- Failure classes: dependency missing or runtime error per backend.

Both Qiskit and PennyLane rows must carry the same circuit fingerprint and pass
value and gradient agreement before the artefact reports
`identical_circuit_ready=True`.

#### `write_identical_circuit_gradient_comparison()`

Writes the identical-circuit rows to a JSON artefact with schema
`scpn_qc_identical_circuit_gradient_comparison_v1`. The committed local
artefact
`data/differentiable_phase_qnode/identical_circuit_gradient_comparison_20260616.json`
records two success rows, one for Qiskit and one for PennyLane, with the same
circuit, same parameters, same observable, and exact-state shot policy. It is a
correctness artefact only: `promotion_ready` remains false until separate
isolated benchmark evidence and claim-ledger promotion metadata exist.

For LLVM/Enzyme, set `SCPN_ENZYME_RUNNER` to an absolute path for an executable
file that reads a JSON request on stdin and writes JSON with:

```json
{
  "value": 0.0,
  "gradient": [0.0, 0.0],
  "toolchain": {"enzyme": "version", "llvm": "version"}
}
```

The runner row rejects relative paths, missing files, and non-executable files
before subprocess execution. It enforces a timeout
(`SCPN_ENZYME_RUNNER_TIMEOUT_SECONDS`, default `10`), validates finite
scalar/vector outputs, records toolchain metadata, and reports
`correctness_mismatch` unless value and gradient match the SCPN reference. These
rows are comparison evidence only; they do not claim provider execution, QPU
execution, GPU execution, arbitrary-program AD, or production performance. Use
the same absolute-executable rule for `SCPN_CATALYST_RUNNER`.

When Enzyme is supplied through the Enzyme-JAX package rather than a standalone
`enzyme` executable, set `ENZYME_LLVM_PLUGIN` to the installed native extension
path. The benchmark metadata records the `enzyme_ad` package version plus the
runner and plugin paths. If the package is installed but the runner fails during
lowering or execution, the row is a `runtime_error` hard gap rather than a
`dependency_missing` hard gap.

#### `run_differentiable_hardening_slice_gate()`

Returns a JSON-ready `DifferentiableHardeningSliceGateResult` for the focused
closeout checks required by every differentiable hardening slice. Callers pass
changed source paths and module-specific pytest files; the gate records the
expected Ruff, mypy, pytest, test-quality audit, and claim-ledger validation
commands and rejects bucket-wide pytest targets such as `tests`.

The result also replays benchmark-evidence classification smoke cases:
GitHub-hosted runners remain `functional_non_isolated`, incomplete
self-hosted isolated metadata remains a `hard_gap`, complete isolated-runner
metadata is the only `isolated_affinity` path, and requested accelerator
execution without visible device evidence remains `silent_accelerator_fallback`.
This API does not execute the listed commands and does not promote any
benchmark row to production evidence.

#### `run_differentiable_isolated_benchmark_plan()`

Returns a JSON-ready `DifferentiableIsolatedBenchmarkPlan` for the current
differentiable benchmark and evidence artefacts that are not yet
promotion-grade. The plan covers the committed local benchmark bundle,
Phase-QNode affinity row, identical-circuit gradient comparison, domain dataset
closure, PyTorch maturity audit, Enzyme/MLIR maturity audit, and the
compiler-promotion batch gate that remains blocked on isolated compiler
benchmark artefact IDs. Each row records source artefact paths, source
classifications, the required `self-hosted`, `linux`, and `isolated-benchmark`
runner labels, a `taskset` plus `chrt` rerun command, required host context,
expected output paths, and blockers.

The committed artefact
`data/differentiable_phase_qnode/differentiable_isolated_benchmark_plan_20260627.json`
is a batch plan, not a benchmark result. `promotion_ready` remains false until
every row has validated `isolated_affinity` output artefacts and no host or
source-classification blockers. The companion validator checks paths, rerun
commands, labels, output locations, and source classifications without executing
benchmarks or changing claim-ledger promotion status.

The CI benchmark evidence writer records accelerator metadata in every bundle.
The default is explicit CPU-only evidence. To request accelerator evidence, set
`SCPN_BENCH_ACCELERATOR_BACKEND=cuda` or `rocm` and provide visible-device
metadata through `SCPN_BENCH_ACCELERATOR_DEVICE_IDS`, `CUDA_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, or `HIP_VISIBLE_DEVICES`. CUDA requests can also use
JAX CUDA device discovery when the CUDA-enabled `jaxlib` plugin is installed.
Optional names and runtime versions can be recorded with
`SCPN_BENCH_ACCELERATOR_DEVICE_NAMES` and
`SCPN_BENCH_ACCELERATOR_RUNTIME` (`cuda=12.4,cudnn=9.1`). Requested
accelerator execution without matching visible devices is classified as
`hard_gap` / `silent_accelerator_fallback`, so a CPU fallback cannot be reused
as GPU benchmark evidence.

---

## Crossover Summary

The three crossover estimates address different classical methods:

| Method | Classical Cost | Crossover n | Bottleneck |
|--------|---------------|-------------|------------|
| Exact diag + expm | O(8^n) | ~14 | RAM + FLOPs |
| GPU statevector | O(2^n × gates) | ~33 (A100) | GPU memory |
| MPS tensor network | O(chi^3 × n × gates) | 32-40 (chi_max=256-1024) | Entanglement |

Below n=14, classical exact methods win for the benchmarked workloads.
Between 14 and 33, GPU statevector estimates set the relevant local
memory boundary. Above n=33-40, exact statevector methods exceed the
assumed GPU memory envelope and MPS estimates become entanglement- and
bond-cap dependent. This is a resource-boundary diagnostic only; no broad
quantum-advantage claim follows without a committed classical baseline,
observable tolerance, and hardware dataset for the specific workload.

## Dependencies

| Module | Internal | External |
|--------|----------|----------|
| `quantum_advantage` | bridge.knm_hamiltonian, hardware.classical | scipy (curve_fit) |
| `gpu_baseline` | — | — (pure estimates) |
| `mps_baseline` | analysis.entanglement_spectrum | numpy |
| `appqsim_protocol` | bridge.*, hardware.classical, analysis.* | qiskit |
| `differentiable_programming` | differentiable | numpy; optional jax reference rows |

The core benchmark suite runs with the base installation. Optional external
reference rows declare their backend availability instead of fabricating
comparisons.

## Testing

35 tests across 4 test files:

- `test_quantum_advantage.py` — Scaling correctness, crossover estimation, edge cases
- `test_gpu_baseline.py` — Memory estimates, time estimates, comparison logic
- `test_mps_baseline.py` — Bond dimension, memory, tractability threshold
- `test_appqsim_protocol.py` — Metric ranges, VQE fallback, correlator fidelity

## Pipeline Performance

Measured on ML350 Gen8 (128 GB RAM, Xeon E5-2620v2). The **Kind** column marks
each figure as a *measured* classical-hardware timing or an *analytic estimate*,
and names the method class. No row is a quantum-processor timing.

| Operation | System | Wall Time | Kind |
|-----------|--------|-----------|------|
| `classical_benchmark` | 4 qubits | 8 ms | measured — dense exact evolution (classical) |
| `classical_benchmark` | 8 qubits | 120 ms | measured — dense exact evolution (classical) |
| `classical_benchmark` | 12 qubits | 8,500 ms | measured — dense exact evolution (classical) |
| `classical_benchmark` | 14 qubits | ~45,000 ms | measured — dense exact evolution (classical) |
| `quantum_benchmark` | 4 qubits | 15 ms | measured — statevector-Trotter *simulation* (classical) |
| `quantum_benchmark` | 8 qubits | 45 ms | measured — statevector-Trotter *simulation* (classical) |
| `quantum_benchmark` | 12 qubits | 350 ms | measured — statevector-Trotter *simulation* (classical) |
| `quantum_benchmark` | 16 qubits | 3,200 ms | measured — statevector-Trotter *simulation* (classical) |
| `gpu_baseline_comparison` | any n | 0.01 ms | analytic estimate (no simulation) |
| `mps_baseline_comparison` | 8 qubits | 25 ms | measured — MPS tensor network (classical) |
| `appqsim_benchmark` | 4 qubits | 350 ms | measured — application-metric suite (classical) |

Dense exact evolution hits a memory/compute wall at n=14 (~45 s), because its
cost is `O(2^{3n})` per step. The statevector-Trotter *simulation* is cheaper —
its cost is `O(2^n · n_gates)`: **exponential in n** (the statevector dimension),
polynomial in circuit depth — so it overtakes dense exact evolution at the
`estimate_crossover` point. That crossover is **between two classical simulation
methods**, not a quantum speed-up; a real quantum-advantage comparison at matched
budget is decided only by §2b/§2c and, absent a QPU row, stays `inconclusive`.
The GPU baseline is a pure analytic estimate (no simulation is run).
