# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Benchmarks API Reference

# Benchmarks API Reference

The `benchmarks` package measures the computational frontier: at what
system size does quantum hardware outperform the best classical methods
for simulating Kuramoto-XY dynamics? Four modules answer this from
different angles — exact diagonalisation, GPU statevector, MPS tensor
networks, and application-oriented metrics.

4 modules, 10 public symbols, 3 crossover estimates.

## Architecture

```
Classical Methods                    Comparison                     Quantum Methods
──────────────────                  ──────────                      ───────────────
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

### 1. `quantum_advantage` — Classical vs Quantum Scaling

Measures wall-clock time for exact classical simulation (exact
diagonalisation + matrix exponential) against Trotter evolution on
the statevector simulator.

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

#### `estimate_crossover(results)`

Fits exponential scaling `t = a * exp(b * n)` to both classical and quantum
timings. The crossover is where the curves intersect:

```
n_cross = log(a_q / a_c) / (b_c - b_q)
```

Returns `int | None`. Returns None if fewer than 3 data points or if
quantum scaling is not slower than classical (unexpected).

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

### 2. `gpu_baseline` — GPU vs QPU Comparison

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

### 3. `mps_baseline` — Tensor Network Comparison

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

### 4. `appqsim_protocol` — Application-Oriented Metrics

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

## Crossover Summary

The three crossover estimates address different classical methods:

| Method | Classical Cost | Crossover n | Bottleneck |
|--------|---------------|-------------|------------|
| Exact diag + expm | O(8^n) | ~14 | RAM + FLOPs |
| GPU statevector | O(2^n × gates) | ~33 (A100) | GPU memory |
| MPS tensor network | O(chi^3 × n × gates) | 32-40 (chi_max=256-1024) | Entanglement |

Below n=14, classical exact methods win. Between 14 and 33, GPU
simulation is fastest. Above n=33-40, only quantum hardware (or
approximate classical methods with uncontrolled error) can solve the
full Kuramoto-XY dynamics.

## Dependencies

| Module | Internal | External |
|--------|----------|----------|
| `quantum_advantage` | bridge.knm_hamiltonian, hardware.classical | scipy (curve_fit) |
| `gpu_baseline` | — | — (pure estimates) |
| `mps_baseline` | analysis.entanglement_spectrum | numpy |
| `appqsim_protocol` | bridge.*, hardware.classical, analysis.* | qiskit |

No optional external dependencies. All benchmarks run with the base
installation.

## Testing

35 tests across 4 test files:

- `test_quantum_advantage.py` — Scaling correctness, crossover estimation, edge cases
- `test_gpu_baseline.py` — Memory estimates, time estimates, comparison logic
- `test_mps_baseline.py` — Bond dimension, memory, tractability threshold
- `test_appqsim_protocol.py` — Metric ranges, VQE fallback, correlator fidelity

## Pipeline Performance

Measured on ML350 Gen8 (128 GB RAM, Xeon E5-2620v2):

| Operation | System | Wall Time |
|-----------|--------|-----------|
| `classical_benchmark` | 4 qubits | 8 ms |
| `classical_benchmark` | 8 qubits | 120 ms |
| `classical_benchmark` | 12 qubits | 8,500 ms |
| `classical_benchmark` | 14 qubits | ~45,000 ms |
| `quantum_benchmark` | 4 qubits | 15 ms |
| `quantum_benchmark` | 8 qubits | 45 ms |
| `quantum_benchmark` | 12 qubits | 350 ms |
| `quantum_benchmark` | 16 qubits | 3,200 ms |
| `gpu_baseline_comparison` | any n | 0.01 ms (pure estimate) |
| `mps_baseline_comparison` | 8 qubits | 25 ms |
| `appqsim_benchmark` | 4 qubits | 350 ms |

The classical benchmark hits a wall at n=14 (45 seconds). The quantum
benchmark scales polynomially in 2^n, reaching 3.2 seconds at n=16.
GPU and MPS baselines are pure estimates (no actual simulation).
