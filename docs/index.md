# scpn-quantum-control

[![CI](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anulum/scpn-quantum-control/branch/main/graph/badge.svg)](https://codecov.io/gh/anulum/scpn-quantum-control)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/anulum/scpn-quantum-control/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0%2B-6929C4.svg)](https://qiskit.org)

Quantum simulation of coupled oscillators on IBM superconducting hardware.

## What this does

The **Self-Consistent Phenomenological Network (SCPN)** models hierarchical
dynamics as 16 coupled Kuramoto oscillators with a coupling matrix K_nm.
The Kuramoto model maps directly to the XY spin Hamiltonian — superconducting
qubits simulate it natively via Trotterized time evolution.

This package compiles SCPN coupling parameters into Qiskit circuits and
runs them on IBM Heron r2 hardware (156 qubits).

## Key results

| Result | Value |
|--------|-------|
| VQE ground-state error | **0.05%** (4-qubit, ibm_fez) |
| 16-layer UPDE snapshot | 46% error at depth 770 (NISQ-consistent) |
| Decoherence curve | 12 points, depth 5→770 |
| Coherence wall | depth 250-400 (Heron r2) |
| Test suite | 442 passing |

## Modules

| Module | Purpose |
|--------|---------|
| `bridge` | K_nm → Hamiltonian, ansatz, circuit converters |
| `phase` | Kuramoto XY solver, VQE, UPDE-16, Trotter |
| `control` | QAOA-MPC, VQLS Grad-Shafranov, Petri nets, disruption classifier |
| `qsnn` | Quantum spiking neural networks (LIF, STDP, synapses) |
| `crypto` | Topology-authenticated QKD, Bell tests, key rates, percolation |
| `qec` | Toric code + MWPM decoder with K_nm-weighted distances |
| `mitigation` | ZNE (unitary folding) + dynamical decoupling (XY4, X2, CPMG) |
| `hardware` | IBM Quantum runner, 20 pre-built experiments, classical references |

## Quick example

```python
from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import QuantumKuramotoSolver

K = build_knm_paper27(L=4)
omega = OMEGA_N_16[:4]
solver = QuantumKuramotoSolver(4, K, omega)
result = solver.run(t_max=0.5, dt=0.1, trotter_per_step=2)
print(f"R(t): {result['R']}")
```

## Next steps

- [Installation](installation.md) — pip install + dev setup
- [Quickstart](quickstart.md) — first experiment in 5 minutes
- [Orchestrator Integration](orchestrator_integration.md) — fusion-defined Kuramoto/UPDE specs into quantum lanes
- [API Reference](api.md) — full module documentation
- [Hardware Guide](hardware_guide.md) — IBM Quantum setup
