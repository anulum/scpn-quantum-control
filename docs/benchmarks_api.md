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
