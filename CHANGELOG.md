# Changelog

All notable changes to scpn-quantum-control are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-02-28

### Added

- **qsnn/**: Quantum LIF neuron (`qlif.py`), controlled-Ry synapse (`qsynapse.py`), parameter-shift STDP (`qstdp.py`), entangled dense layer (`qlayer.py`)
- **phase/**: Kuramoto XY Hamiltonian solver (`xy_kuramoto.py`), 16-layer Trotter UPDE (`trotter_upde.py`), VQE ground state finder (`phase_vqe.py`)
- **control/**: QAOA-MPC binary trajectory optimizer (`qaoa_mpc.py`), VQLS Grad-Shafranov solver (`vqls_gs.py`), quantum Petri net (`qpetri.py`), quantum disruption classifier (`q_disruption.py`)
- **bridge/**: Knm-to-Hamiltonian compiler (`knm_hamiltonian.py`), SPN-to-circuit converter (`spn_to_qcircuit.py`), bitstream-rotation bridge (`sc_to_quantum.py`)
- **qec/**: Toric surface code + MWPM decoder with Knm-weighted edges (`control_qec.py`)
- **hardware/**: IBM Quantum runner for ibm_fez Heron r2 (`runner.py`, `experiments.py`, `classical.py`)
- 88 unit tests, 4 example scripts, 19 hardware result files
- Hardware validation on ibm_fez: VQE 0.05% error, 12-point decoherence curve, first 16-layer SCPN quantum simulation
- CI workflow with Python 3.9-3.12 matrix, coverage, ruff lint
- Full documentation: architecture, API reference, hardware results
