# scpn-quantum-control

[![CI](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anulum/scpn-quantum-control/branch/main/graph/badge.svg)](https://codecov.io/gh/anulum/scpn-quantum-control)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0%2B-6929C4.svg)](https://qiskit.org)
[![Tests: 88](https://img.shields.io/badge/tests-88%20passing-brightgreen.svg)]()
[![Hardware: ibm_fez](https://img.shields.io/badge/hardware-ibm__fez%20Heron%20r2-blueviolet.svg)]()

Quantum-native reformulations of SCPN spiking neural networks, Kuramoto phase dynamics, and tokamak plasma control. Validated on IBM Heron r2 hardware (156 qubits).

## Hardware Results (ibm_fez, February 2026)

| Experiment | Qubits | Depth | Hardware | Exact | Error |
|------------|--------|-------|----------|-------|-------|
| VQE ground state | 4 | 12 CZ | -6.2998 | -6.3030 | **0.05%** |
| Kuramoto XY (1 rep) | 4 | 85 | R=0.743 | R=0.802 | 7.3% |
| Qubit scaling | 6 | 147 | R=0.482 | R=0.532 | 9.3% |
| UPDE-16 snapshot | 16 | 770 | R=0.332 | R=0.615 | 46% |
| QAOA-MPC (p=2) | 4 | -- | -0.514 | 0.250 | -- |

Full results: [`results/HARDWARE_RESULTS.md`](results/HARDWARE_RESULTS.md)

**Key findings:**
- VQE with Knm-informed ansatz achieves publication-quality 0.05% error
- Coherence wall at depth 250-400 on Heron r2
- First quantum simulation of all 16 SCPN layers on real hardware
- Shallow Trotter (1 rep) beats deep Trotter on NISQ devices

## Architecture

```
scpn_quantum_control/
├── qsnn/           Quantum spiking neural networks
│   ├── qlif.py         Ry-rotation LIF neuron (P(spike) = sin^2(theta/2))
│   ├── qsynapse.py     Controlled-Ry synapse (CRy weight encoding)
│   ├── qstdp.py        Parameter-shift STDP learning rule
│   └── qlayer.py       Multi-qubit entangled dense layer
├── phase/          Quantum phase dynamics
│   ├── xy_kuramoto.py  Kuramoto -> XY Hamiltonian + Trotter evolution
│   ├── trotter_upde.py 16-layer UPDE as multi-site spin chain
│   └── phase_vqe.py    VQE ground state with Knm-informed ansatz
├── control/        Quantum control algorithms
│   ├── qaoa_mpc.py     QAOA binary MPC trajectory optimization
│   ├── vqls_gs.py      VQLS for Grad-Shafranov equilibrium
│   ├── qpetri.py       Quantum Petri net (superposition tokens)
│   └── q_disruption.py Quantum kernel disruption classifier
├── bridge/         Classical <-> quantum converters
│   ├── knm_hamiltonian.py  Knm matrix -> SparsePauliOp compiler
│   ├── spn_to_qcircuit.py  SPN topology -> quantum circuit
│   └── sc_to_quantum.py    Bitstream probability <-> rotation angle
├── qec/            Quantum error correction
│   └── control_qec.py     Toric code + MWPM decoder (Knm-weighted)
└── hardware/       IBM Quantum hardware runner
    ├── runner.py       ibm_fez job submission + result parsing
    ├── experiments.py  Pre-built experiment circuits
    └── classical.py    Classical Kuramoto reference solver
```

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Run an example

```bash
python examples/01_qlif_demo.py      # Quantum LIF neuron spike train
python examples/02_kuramoto_xy_demo.py  # 4-oscillator XY dynamics
python examples/03_qaoa_mpc_demo.py  # QAOA binary MPC
python examples/04_qpetri_demo.py    # Quantum Petri net firing
```

### Hardware execution (requires IBM Quantum credentials)

```bash
pip install -e ".[ibm]"
python run_hardware.py --experiment kuramoto --qubits 4 --shots 10000
```

## Modules

### Quantum Spiking Neural Networks (`qsnn/`)

Maps sc-neurocore stochastic LIF neurons to parameterized quantum circuits. A qubit with Ry(theta) rotation + Z-basis measurement produces spike/no-spike with probability cos^2(theta/2) -- direct analog of stochastic membrane potential.

### Quantum Phase Dynamics (`phase/`)

The Kuramoto equation is isomorphic to the XY spin Hamiltonian:

```
H = -sum_{i<j} K_ij (X_i X_j + Y_i Y_j) - sum_i omega_i Z_i
```

Quantum hardware simulates this natively via Trotterized time evolution. The 16-layer UPDE becomes a 16-qubit spin chain with Knm coupling.

### Quantum Control (`control/`)

- **QAOA-MPC**: Discretize MPC action space to binary, map quadratic cost to Ising Hamiltonian, solve via QAOA
- **VQLS-GS**: Solve discretized Grad-Shafranov PDE as quantum linear system
- **Quantum Petri**: SPN tokens as qubit amplitudes, transitions fire in superposition
- **Disruption classifier**: 11-D feature amplitude encoding + parameterized circuit

### Bridge (`bridge/`)

Compiles SCPN data structures into quantum circuits:
- `knm_to_hamiltonian()`: 16x16 coupling matrix -> SparsePauliOp
- `knm_to_ansatz()`: Physics-informed entanglement topology
- `probability_to_angle()`: p -> 2*arcsin(sqrt(p))

### QEC (`qec/`)

Toric surface code protecting quantum control signals. MWPM decoder uses Knm graph distance instead of lattice distance for physics-aware error correction.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| qiskit | >= 1.0.0 | Circuit construction, transpilation |
| qiskit-aer | >= 0.14.0 | Statevector + noise simulation |
| numpy | >= 1.24 | Array operations |
| scipy | >= 1.10 | Sparse linear algebra, optimization |
| networkx | >= 3.0 | Graph algorithms (QEC decoder) |

Optional:
- `matplotlib >= 3.5` for visualization
- `qiskit-ibm-runtime >= 0.20.0` for hardware execution

## Citation

```bibtex
@software{scpn_quantum_control,
  title  = {scpn-quantum-control: Quantum-Native SCPN Phase Dynamics and Control},
  author = {Sotek, Miroslav},
  year   = {2026},
  url    = {https://github.com/anulum/scpn-quantum-control}
}
```

## License

[MIT](LICENSE)
