# scpn-quantum-control

[![CI](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://github.com/anulum/scpn-quantum-control/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0%2B-6929C4.svg)](https://qiskit.org)

NISQ quantum simulation of coupled Kuramoto oscillator networks on IBM superconducting hardware.

## What this does

The Kuramoto model for coupled oscillators maps directly to the quantum
XY spin Hamiltonian — superconducting qubits simulate it natively via
Trotterized time evolution.

Supply any coupling matrix K and natural frequencies omega; this package
compiles them into Qiskit circuits and runs them on statevector simulation
or IBM Heron r2 hardware. Ships with the SCPN 16-oscillator network as
a built-in example.

## Key results

| Result | Value |
|--------|-------|
| VQE ground-state error | **0.05%** (4-qubit, ibm_fez) |
| 16-layer UPDE snapshot | 46% error at depth 770 (NISQ-consistent) |
| Decoherence curve | 12 points, depth 5→770 |
| Coherence wall | depth 250–400 (Heron r2) |
| Test suite | 627+ passing, 100% coverage |

## Modules

| Module | Purpose |
|--------|---------|
| `bridge` | K_nm → Hamiltonian, SNN adapter (sc-neurocore), SSGF adapter, orchestrator bridge |
| `phase` | Kuramoto XY solver, VQE, UPDE-16, Trotter |
| `control` | QAOA-MPC, VQLS Grad-Shafranov, Petri nets, ITER disruption classifier |
| `qsnn` | Quantum spiking neural networks (LIF, STDP, synapses, training) |
| `crypto` | Topology-authenticated QKD, Bell tests, key rates, percolation |
| `qec` | Toric code + MWPM decoder, fault-tolerant UPDE (repetition code) |
| `mitigation` | ZNE (unitary folding), PEC (Pauli twirling), dynamical decoupling |
| `hardware` | IBM Quantum runner, trapped-ion backend, 20 pre-built experiments |
| `identity` | VQE attractor basin, coherence budget, entanglement witness, fingerprint, binding spec |
| `benchmarks` | Classical vs quantum scaling, crossover extrapolation |

## Quick example

Any coupling topology works — bring your own K and omega:

```python
from scpn_quantum_control import QuantumKuramotoSolver, build_kuramoto_ring

K, omega = build_kuramoto_ring(6, coupling=0.5, rng_seed=42)
solver = QuantumKuramotoSolver(6, K, omega)
result = solver.run(t_max=1.0, dt=0.1, trotter_per_step=2)
print(f"R(t): {result['R']}")
```

Or use the built-in SCPN network:

```python
from scpn_quantum_control import QuantumKuramotoSolver, build_knm_paper27, OMEGA_N_16

K = build_knm_paper27(L=4)
solver = QuantumKuramotoSolver(4, K, OMEGA_N_16[:4])
result = solver.run(t_max=0.5, dt=0.1, trotter_per_step=2)
```

## Limitations

- **NISQ benchmarking only.** Circuit depths >400 hit the coherence wall; cloud QPUs cannot provide the <1 ms deterministic latency required for real tokamak control.
- **SCPN is an unpublished model.** The K_nm parameterisation comes from a 2025 working paper with no external citations. The Kuramoto→XY mapping is standard; the specific coupling structure is not independently validated.
- **No quantum advantage at this scale.** At N=4–16, classical ODE solvers are faster and more accurate. Advantage requires N>>20 with error-corrected qubits.

## Next steps

- [Installation](installation.md) — pip install + dev setup
- [Quickstart](quickstart.md) — first experiment in 5 minutes
- [Equations](equations.md) — mathematical foundations
- [API Reference](api.md) — full module documentation
- [Bridges](bridges_api.md) — cross-repo integrations
- [Hardware Guide](hardware_guide.md) — IBM Quantum setup
