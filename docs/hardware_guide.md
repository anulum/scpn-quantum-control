# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Execution Guide

# Hardware Execution Guide

The `hardware` package provides the full stack from circuit compilation
to QPU execution, noise modelling, classical reference computation, and
multi-backend support. 16 modules covering IBM superconducting, trapped
ion, PennyLane, Cirq, GPU acceleration, and circuit cutting.

## Architecture

```
Experiment Definition (experiments.py)
    │
    ├── HardwareRunner (runner.py)
    │   ├── connect() → IBM Runtime / AerSimulator
    │   ├── transpile() → native gate set
    │   ├── run_circuit() → JobResult
    │   └── run_with_zne() → ZNE-mitigated result
    │
    ├── Noise Model (noise_model.py)
    │   └── heron_r2_noise_model() → NoiseModel (thermal + depolarizing)
    │
    ├── Classical Reference (classical.py)
    │   ├── classical_kuramoto_reference() → Euler integration
    │   ├── classical_exact_diag() → full eigendecomposition
    │   ├── classical_exact_evolution() → matrix expm
    │   └── classical_brute_mpc() → brute-force MPC
    │
    ├── Multi-Backend
    │   ├── PennyLane (pennylane_adapter.py)
    │   ├── Cirq (cirq_adapter.py)
    │   ├── Trapped Ion (trapped_ion.py)
    │   ├── GPU (gpu_accel.py, jax_accel.py)
    │   └── Plugin Registry (plugin_registry.py)
    │
    └── Circuit Tools
        ├── Circuit Cutting (circuit_cutting.py, cutting_runner.py)
        ├── QASM Export (qasm_export.py)
        ├── Circuit Export (circuit_export.py)
        └── QCVV (qcvv.py)
```

## Prerequisites

### IBM Quantum

1. Account: https://quantum.cloud.ibm.com
2. Credentials (use `ibm_cloud` channel, NOT deprecated `ibm_quantum`):
   ```bash
   export IBM_QUANTUM_TOKEN="your-token-here"
   export IBM_QUANTUM_CRN="your-crn-instance-id"
   ```
3. Install IBM runtime:
   ```bash
   pip install -e ".[ibm]"
   ```

## HardwareRunner (`runner.py`)

The primary execution interface. Handles authentication, backend selection,
transpilation, job submission, and result collection.

### Connection

```python
from scpn_quantum_control.hardware import HardwareRunner

# Real hardware
runner = HardwareRunner(use_simulator=False)
runner.connect()  # Authenticates with IBM_QUANTUM_TOKEN env var

# Local simulator (default)
runner = HardwareRunner(use_simulator=True, results_dir="results/")
runner.connect()
```

When `use_simulator=True`, uses `AerSimulator` with the Heron r2 noise
model for realistic local testing without QPU budget consumption.

### Transpilation

```python
transpiled = runner.transpile(circuit, optimization_level=3)
```

Uses Qiskit's preset pass manager with Heron r2 target. Optimization
level 3 performs heavy gate cancellation and routing.

### Execution

```python
result = runner.run_circuit(
    circuit,
    experiment_name="kuramoto_4osc",
    shots=10000,
)
# result: JobResult with counts, wall_time_s, metadata
```

### ZNE Execution

```python
result = runner.run_with_zne(
    circuit,
    experiment_name="kuramoto_zne",
    noise_scales=[1, 3, 5],
    shots=10000,
)
```

Internally calls `gate_fold_circuit` from the mitigation package.

### `JobResult`

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | str | IBM job ID or "simulator" |
| `backend_name` | str | Backend identifier |
| `experiment_name` | str | User-specified experiment name |
| `counts` | dict or None | Measurement counts |
| `expectation_values` | ndarray or None | Computed expectations |
| `metadata` | dict | Arbitrary metadata |
| `wall_time_s` | float | Total execution time |
| `timestamp` | str | ISO timestamp |

Results are serialised to JSON via `to_dict()` and saved to `results_dir`.

---

## Noise Model (`noise_model.py`)

IBM Heron r2 calibration (ibm_fez, February 2026 median):

| Parameter | Value | Description |
|-----------|-------|-------------|
| T1 | 300 us | Longitudinal relaxation |
| T2 | 200 us | Transverse relaxation |
| CZ error | 0.5% | Two-qubit gate error rate |
| Readout error | 0.2% | Measurement error rate |
| Single-gate time | 0.06 us | SX/X/RZ duration |
| Two-gate time | 0.66 us | CZ/ECR duration |

### `heron_r2_noise_model(t1_us, t2_us, cz_error, readout_error)`

Constructs a Qiskit-Aer `NoiseModel`:
- Single-qubit gates: thermal relaxation only
- Two-qubit gates (ECR/CZ): thermal relaxation + depolarizing
- Readout: symmetric bit-flip error

```python
from scpn_quantum_control.hardware import heron_r2_noise_model

model = heron_r2_noise_model()
# Use with AerSimulator:
from qiskit_aer import AerSimulator
backend = AerSimulator(noise_model=model)
```

---

## Classical Reference (`classical.py`)

Exact classical computations for hardware experiment comparison. Every
quantum result should be compared against these references.

### `classical_kuramoto_reference(n_osc, t_max, dt, K=None, omega=None)`

Euler integration of the classical Kuramoto model:

```
d(theta_i)/dt = omega_i + sum_j K[i,j] * sin(theta_j - theta_i)
```

Returns `{times, theta, R}` — phase trajectories and order parameter.

**Rust acceleration**: `scpn_quantum_engine.kuramoto_euler()` at 33x
speedup for n >= 8.

### `classical_exact_diag(n, K=None, omega=None)`

Full eigendecomposition of the XY Hamiltonian. Returns eigenvalues,
eigenvectors, ground state, and ground energy.

For n <= 14: direct dense diagonalisation via `numpy.linalg.eigh`.
For n > 14: sparse ARPACK via `scipy.sparse.linalg.eigsh`.

### `classical_exact_evolution(n, t_max, dt, K=None, omega=None)`

Matrix exponential evolution: psi(t+dt) = exp(-iHdt) psi(t).

Returns time series of R(t) and energy E(t) for direct comparison
with Trotter evolution on quantum hardware.

### `classical_brute_mpc(K, omega, horizon, theta_init)`

Brute-force model predictive control: enumerate all 2^horizon action
sequences and select the one maximising R(t_final).

**Rust acceleration**: `scpn_quantum_engine.brute_mpc()` with rayon
parallel enumeration at 5-50x speedup.

---

## Experiments (`experiments.py`)

Pre-defined experiment configurations for systematic QPU characterisation.

### `ALL_EXPERIMENTS`

Registry of all 19 experiment functions:

| Experiment | Qubits | Description |
|------------|--------|-------------|
| `kuramoto_4osc` | 4 | Basic Trotter evolution, R(t) |
| `kuramoto_4osc_trotter2` | 4 | Suzuki-Trotter 2nd order |
| `kuramoto_4osc_zne` | 4 | ZNE-mitigated Kuramoto |
| `kuramoto_8osc` | 8 | 8-qubit Kuramoto dynamics |
| `kuramoto_8osc_zne` | 8 | ZNE-mitigated 8-qubit |
| `vqe_4q` | 4 | VQE ground state search |
| `vqe_8q` | 8 | VQE with physics-informed ansatz |
| `vqe_8q_hardware` | 8 | VQE targeting real QPU |
| `vqe_landscape` | 4 | Energy landscape scan |
| `qaoa_mpc_4` | 4 | QAOA-based MPC |
| `upde_16_snapshot` | 16 | Full 16-qubit UPDE state snapshot |
| `upde_16_dd` | 16 | UPDE with dynamical decoupling |
| `noise_baseline` | 4 | Noise characterisation baseline |
| `ansatz_comparison_hw` | 4 | Compare ansatz architectures |
| `sync_threshold` | 4 | Synchronisation threshold detection |
| `decoherence_scaling` | 4 | Depth vs fidelity scaling |
| `zne_higher_order` | 4 | Higher-order ZNE extrapolation |
| `bell_test_4q` | 4 | CHSH Bell test on hardware |
| `correlator_4q` | 4 | XY correlator measurement |

Each experiment function returns a dict with `circuit`, `shots`,
`n_qubits`, and experiment-specific metadata.

### QPU Budget

Free tier: 10 minutes/month on ibm_fez (Heron r2, 156 qubits).

| Experiment | Circuits | Shots | QPU Seconds |
|------------|----------|-------|-------------|
| kuramoto_4osc (1 step) | 3 | 10k | ~15 |
| vqe_4q (100 COBYLA iter) | ~100 | 10k | ~15 |
| qaoa_mpc_4 (p=1) | ~30 | 10k | ~100 |
| upde_16 snapshot | 3 | 20k | ~60 |

---

## Multi-Backend Support

### PennyLane Adapter (`pennylane_adapter.py`)

```python
from scpn_quantum_control.hardware.pennylane_adapter import PennyLaneRunner

runner = PennyLaneRunner(K, omega, device="default.qubit")
result = runner.run_trotter(t=0.5, reps=2)
# result: PennyLaneResult(energy, order_parameter, n_qubits, device_name, statevector)
```

VQE via PennyLane optimisers:
```python
result = runner.run_vqe(ansatz_depth=1, maxiter=5, seed=42)
```

Requires: `pip install pennylane`

### Cirq Adapter (`cirq_adapter.py`)

```python
from scpn_quantum_control.hardware.cirq_adapter import CirqRunner

runner = CirqRunner(K, omega)
result = runner.run_trotter(t=0.5, reps=2)
```

Enables targeting Google Sycamore/Weber QPUs via Cirq.

Requires: `pip install cirq-core`

### Trapped Ion (`trapped_ion.py`)

```python
from scpn_quantum_control.hardware import transpile_for_trapped_ion, trapped_ion_noise_model

ion_circuit = transpile_for_trapped_ion(circuit, n_qubits=4)
model = trapped_ion_noise_model()
```

Target: IonQ Forte / Quantinuum H-series. Native gate set: MS (Molmer-Sorensen),
Rz, Ry.

### GPU Acceleration (`gpu_accel.py`)

cuQuantum integration for large-scale statevector simulation. Falls back
to CPU when CUDA is not available.

### JAX Acceleration (`jax_accel.py`)

JAX-based compilation for VQE parameter optimisation. Enables automatic
differentiation of quantum circuits.

### Plugin Registry (`plugin_registry.py`)

Dynamic backend registration. Third-party backends register via:

```python
from scpn_quantum_control.hardware.plugin_registry import register_backend

register_backend("my_backend", MyBackendClass)
```

---

## Circuit Tools

### Circuit Cutting (`circuit_cutting.py`, `cutting_runner.py`)

Decomposes large circuits (> available qubits) into subcircuits connected
by classical communication. Enables running n-qubit circuits on n/2-qubit
hardware with polynomial overhead.

```python
from scpn_quantum_control.hardware.circuit_cutting import partition_circuit
from scpn_quantum_control.hardware.cutting_runner import CuttingRunner

subcircuits = partition_circuit(circuit, max_partition_size=8)
runner = CuttingRunner(backend)
result = runner.run_partitioned(subcircuits)
```

### QASM Export (`qasm_export.py`)

Export circuits to OpenQASM 2.0/3.0 for platform-independent storage
and submission to third-party systems.

### Circuit Export (`circuit_export.py`)

Export circuits to JSON, LaTeX (Qiskit drawer), and SVG formats for
documentation and publication.

### QCVV (`qcvv.py`)

Quantum Characterisation, Verification, and Validation protocols.
Randomised benchmarking and gate set tomography for hardware qualification.

---

## Decoherence Reference

| Depth Range | Expected Error | Recommendation |
|-------------|---------------|----------------|
| < 50 | < 5% | Publishable as-is |
| 50-150 | 5-15% | Publishable with error bars |
| 150-250 | 15-25% | Apply ZNE mitigation |
| 250-400 | 25-40% | Qualitative trends only |
| > 400 | > 40% | Do not trust individual values |

## Native Gate Set (Heron r2)

| Gate | Description | Duration |
|------|-------------|----------|
| CZ | Two-qubit entangling (native) | 0.66 us |
| RZ(theta) | Z rotation (virtual) | 0 us |
| SX | sqrt(X) | 0.06 us |
| X | Pauli-X | 0.06 us |
| ID | Identity (delay) | 0.06 us |

Transpilation from Qiskit standard gates increases depth. Typical
expansion: 1 CNOT → 2 SX + 1 CZ + RZ gates.

## Rust Acceleration

The `classical.py` module transparently uses Rust via `scpn_quantum_engine`
when available:

| Python Function | Rust Function | Speedup | Method |
|-----------------|---------------|---------|--------|
| `classical_kuramoto_reference` | `kuramoto_euler`, `kuramoto_trajectory` | 33x | rayon parallel Euler steps |
| `_expectation_pauli` | `expectation_pauli_fast` | 3-10x | Bitwise Pauli ops |
| `classical_brute_mpc` | `brute_mpc` | 5-50x | rayon parallel 2^horizon enumeration |
| `_state_order_param` | `state_order_param_sparse` | 2-5x | SIMD-friendly inner loop |
| `_order_parameter` (Floquet) | `all_xy_expectations` | 5-20x | Batch bitwise, single FFI call |

All Rust functions accept split real/imaginary arrays (no complex64 across FFI).
Python fallback always available when the Rust crate is not installed.

## Interpreting Results

Order parameter R from qubit expectation values:

```
R = (1/N) × |sum_i (<X_i> + i×<Y_i>)|
```

where `<X_i> = 2×P(|0>)_x - 1` from X-basis measurement. Requires 3
measurement bases (X, Y, Z) for full reconstruction.

Compare `hw_R` against `exact_R` (from `classical_kuramoto_reference` or
`classical_exact_evolution`) to quantify hardware error.

## Testing

72 tests across 8 test files:

- `test_runner.py` — HardwareRunner lifecycle, simulator mode, job serialisation
- `test_noise_model.py` — NoiseModel construction, error rates, parameter overrides
- `test_classical.py` — Kuramoto reference, exact diag, evolution parity
- `test_experiments.py` — All 19 experiment definitions, circuit validity
- `test_pennylane_adapter.py` — PennyLane Trotter, VQE, device selection
- `test_cirq_adapter.py` — Cirq Trotter, simulator parity
- `test_circuit_cutting.py` — Partitioning, recombination, overhead bounds
- `test_qcvv.py` — RB, gate set tomography, fidelity extraction

## Pipeline Performance

Measured on ML350 Gen8 (128 GB RAM, Xeon E5-2620v2):

| Operation | System | Wall Time |
|-----------|--------|-----------|
| `HardwareRunner.connect` (simulator) | — | 50 ms |
| `runner.transpile` (opt level 3) | 4 qubits | 120 ms |
| `runner.run_circuit` (simulator) | 4 qubits, 10k shots | 800 ms |
| `classical_kuramoto_reference` (Rust) | 8 oscillators | 0.3 ms |
| `classical_kuramoto_reference` (Python) | 8 oscillators | 12 ms |
| `classical_exact_diag` | 8 qubits | 15 ms |
| `classical_exact_evolution` | 8 qubits | 120 ms |
| `heron_r2_noise_model` | — | 5 ms |
| `PennyLaneRunner.run_trotter` | 3 qubits | 50 ms |
| `partition_circuit` | 16 → 2×8 qubits | 25 ms |
