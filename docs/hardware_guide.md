# Hardware Execution Guide

## Prerequisites

1. IBM Quantum account: https://quantum.ibm.com
2. Credentials configured:
   ```bash
   export QISKIT_IBM_TOKEN="your-token-here"
   # or save to ~/.qiskit/qiskit-ibm.json
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
