# scpn-quantum-control

Quantum-native reformulations of SCPN spiking neural networks, Kuramoto phase dynamics, and plasma control algorithms.

## Modules

| Module | Purpose |
|--------|---------|
| `qsnn/` | Quantum LIF neurons, controlled-Ry synapses, parameter-shift STDP, entangled dense layers |
| `phase/` | Kuramoto→XY Hamiltonian Trotter evolution, 16-layer UPDE spin chain, VQE ground state |
| `control/` | QAOA-MPC trajectory optimization, VQLS Grad-Shafranov solver, quantum Petri nets, disruption classifier |
| `bridge/` | Knm→Hamiltonian compiler, SPN→circuit converter, bitstream↔rotation angle maps |
| `qec/` | Toric surface code + MWPM decoder with Knm-weighted edges |

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest tests/ -v
```

## Examples

```bash
python examples/01_qlif_demo.py
python examples/02_kuramoto_xy_demo.py
python examples/03_qaoa_mpc_demo.py
python examples/04_qpetri_demo.py
```

## Dependencies

- qiskit >= 1.0.0
- qiskit-aer >= 0.14.0
- numpy >= 1.24
- scipy >= 1.10
- networkx >= 3.0
