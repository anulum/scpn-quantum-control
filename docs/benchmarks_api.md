# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Benchmarks API Reference

# Benchmarks API Reference

## Quantum Advantage Scaling (`benchmarks.quantum_advantage`)

Measures wall-clock time for classical (exact diag + matrix exp) vs
quantum (Trotter on statevector), then extrapolates the scaling crossover.

### `run_scaling_benchmark`

```python
from scpn_quantum_control.benchmarks import run_scaling_benchmark

results = run_scaling_benchmark(sizes=[4, 8, 12, 16])
for r in results:
    print(f"n={r.n_qubits}: classical={r.t_classical_ms:.1f}ms, quantum={r.t_quantum_ms:.1f}ms")
print(f"Predicted crossover: {results[0].crossover_predicted} qubits")
```

### `AdvantageResult`

```python
@dataclass
class AdvantageResult:
    n_qubits: int
    t_classical_ms: float
    t_quantum_ms: float
    errors: dict
    crossover_predicted: int | None
```

### Individual Benchmarks

```python
classical_benchmark(n, t_max=1.0, dt=0.1) -> dict  # t_total_ms, ground_energy, R_final
quantum_benchmark(n, t_max=1.0, dt=0.1, trotter_reps=5) -> dict  # t_total_ms, n_trotter_steps
estimate_crossover(results) -> int | None  # exponential fit crossover qubit count
```

Classical benchmark returns `inf` for $n > 14$ (matrix expm infeasible).

---

## GPU vs QPU Crossover (`benchmarks.gpu_baseline`)

Compares statevector GPU simulation wall-time against QPU execution time to
determine where quantum hardware becomes faster.

### `gpu_baseline_comparison`

```python
from scpn_quantum_control.benchmarks.gpu_baseline import gpu_baseline_comparison

result = gpu_baseline_comparison(n=12, trotter_reps=10)
print(f"GPU: {result.estimated_gpu_time_s:.2e}s, QPU: {result.qpu_time_s:.2e}s")
print(f"GPU faster: {result.gpu_faster}, crossover: n={result.crossover_n}")
```

### `GPUBaselineResult`

```python
@dataclass
class GPUBaselineResult:
    n_qubits: int
    n_gates: int
    statevector_memory_gb: float
    statevector_flops: float
    estimated_gpu_time_s: float  # A100 80GB baseline
    qpu_time_s: float           # 0.5µs gate time
    gpu_faster: bool
    crossover_n: int            # estimated crossover qubit count
```

### Utility Functions

```python
statevector_memory_gb(n) -> float       # 2^n complex128 entries
statevector_flops(n, n_gates) -> float  # 2^n * n_gates matrix-vector ops
estimate_gpu_time(n, n_gates, tflops=312.0) -> float  # A100 FP64
estimate_qpu_time(n, n_gates, gate_time_us=0.5) -> float
gate_count_xy_trotter(n, reps=10) -> int
scaling_comparison(n_values=None) -> dict  # batch scan over system sizes
```

---

## MPS Baseline (`benchmarks.mps_baseline`)

Matrix Product State memory and bond dimension requirements. Determines
whether tensor network methods can simulate the entanglement at $K_c$.

```python
from scpn_quantum_control.benchmarks.mps_baseline import mps_baseline_comparison

result = mps_baseline_comparison(n=16, K_base=0.45, omega=OMEGA_N_16)
print(f"Bond dim χ={result.required_chi}, MPS memory: {result.mps_memory_bytes/1e6:.1f}MB")
print(f"Exact memory: {result.exact_memory_bytes/1e6:.1f}MB")
print(f"MPS feasible: {result.mps_feasible}")
```

### `MPSBaselineResult`

```python
@dataclass
class MPSBaselineResult:
    n_qubits: int
    entropy: float
    required_chi: int           # bond dimension from entanglement entropy
    mps_memory_bytes: int
    exact_memory_bytes: int
    mps_feasible: bool          # MPS memory < exact memory
    advantage_n: int | None     # n where MPS fails
```

---

## Application-Quantum-Simulation Protocol (`benchmarks.appqsim_protocol`)

Measures simulation quality via order parameter error and correlator fidelity,
following the AppQSim benchmarking protocol.

```python
from scpn_quantum_control.benchmarks.appqsim_protocol import appqsim_benchmark

metrics = appqsim_benchmark(K, omega, trotter_reps=10, shots=10000)
print(f"R_error: {metrics.r_error:.4f}, correlator_fidelity: {metrics.correlator_fidelity:.4f}")
```

### `AppQSimMetrics`

```python
@dataclass
class AppQSimMetrics:
    r_error: float              # |R_exact - R_measured|
    correlator_fidelity: float  # Tr(C_exact @ C_measured) / norms
    circuit_depth: int
    n_qubits: int
    trotter_reps: int
```
