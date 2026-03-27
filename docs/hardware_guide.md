# Hardware Execution Guide

## Prerequisites

1. IBM Quantum account: https://quantum.cloud.ibm.com
2. Credentials configured (use `ibm_cloud` channel, NOT deprecated `ibm_quantum`):
   ```bash
   export IBM_QUANTUM_TOKEN="your-token-here"
   export IBM_QUANTUM_CRN="your-crn-instance-id"
   ```
3. Install IBM runtime:
   ```bash
   pip install -e ".[ibm]"
   ```

## QPU Budget

Free tier: 10 minutes/month on ibm_fez (Heron r2, 156 qubits).

Typical costs:
| Experiment | Circuits | Shots | QPU seconds |
|------------|----------|-------|-------------|
| kuramoto_4osc (1 step) | 3 | 10k | ~15 |
| vqe_4q (100 COBYLA iter) | ~100 | 10k | ~15 |
| qaoa_mpc (p=1, COBYLA) | ~30 | 10k | ~100 |
| upde_16 snapshot | 3 | 20k | ~60 |

## Running Experiments

```bash
# Single experiment
python run_hardware.py --experiment kuramoto --qubits 4 --shots 10000

# All experiments
python run_hardware.py --all
```

Results are saved to `results/hw_<name>.json`.

## Decoherence Reference

Use this table to estimate whether your circuit will produce meaningful results:

| Depth range | Expected error | Recommendation |
|-------------|---------------|----------------|
| < 50 | < 5% | Publishable as-is |
| 50-150 | 5-15% | Publishable with error bars |
| 150-250 | 15-25% | Apply ZNE mitigation |
| 250-400 | 25-40% | Qualitative trends only |
| > 400 | > 40% | Do not trust individual values |

## Native Gate Set (Heron r2)

| Gate | Description |
|------|-------------|
| CZ | Two-qubit entangling (native) |
| RZ(theta) | Z rotation |
| SX | sqrt(X) |
| X | Pauli-X (bit flip) |
| ID | Identity (delay) |

Transpilation from Qiskit standard gates to native gates increases depth.
Typical expansion: 1 CNOT -> 2 SX + 1 CZ + RZ gates.

## Interpreting Results

Order parameter R is computed from qubit expectation values:

```
R = (1/N) * |sum_i (X_i + i*Y_i)|
```

where X_i = 2*P(|0>) - 1 for X-basis measurement, etc.

Compare `hw_R` against `exact_R` (from AerSimulator statevector) to quantify hardware error.

## Rust Acceleration

`hardware.classical` functions transparently use Rust (via `scpn_quantum_engine`)
when available, falling back to pure Python/NumPy otherwise. Install with:

```bash
cd scpn_quantum_engine && maturin build --release && pip install target/wheels/*.whl
```

### Accelerated functions

| Python function | Rust function | Speedup | Method |
|-----------------|---------------|---------|--------|
| `_state_order_param` | `state_order_param_sparse` | 2-5x (n>=8) | Bitwise Pauli application instead of kron |
| `_state_order_param_sparse` | `state_order_param_sparse` | 2-5x (n>=8) | SIMD-friendly inner loop over 2^n states |
| `_expectation_pauli` | `expectation_pauli_fast` | 3-10x (n>=6) | Bitwise ops instead of dense 2^n x 2^n kron |
| `classical_brute_mpc` | `brute_mpc` | 5-50x (horizon>=10) | rayon parallel enumeration of 2^horizon actions |
| `lanczos_coefficients` | `lanczos_b_coefficients` | 5-10x (dim<=256) | Complex matrix commutator loop without Python overhead |
| `compute_otoc` | `otoc_from_eigendecomp` | 10-50x | Eigendecomp once + O(d²) per time point + rayon parallel |
| `_order_parameter` (Floquet) | `expectation_pauli_fast` | 5-20x | Bitwise Pauli replaces Qiskit SparsePauliOp overhead |

All Rust functions accept split real/imaginary arrays (no complex64 across FFI).
The Python fallback is always available when the Rust crate is not installed.
