# Examples

All examples run on statevector simulation — no IBM Quantum credentials needed.

Install first:
```bash
pip install -e ".[dev]"
```

## 01_qlif_demo.py — Quantum LIF Neuron

Constructs a quantum leaky integrate-and-fire neuron that encodes membrane
potential as an Ry rotation angle. Sweeps input current levels and compares
the quantum spike rate (from Z-basis measurement) with the classical
Bernoulli expectation sin^2(theta/2). Demonstrates the probability-to-angle
bridge.

```bash
python examples/01_qlif_demo.py
```

## 02_kuramoto_xy_demo.py — Kuramoto XY Dynamics

Builds a 4-oscillator XY Hamiltonian from the SCPN coupling matrix K_nm
and evolves it via Trotter decomposition. Prints the order parameter R(t)
at each timestep, showing how synchronization emerges from the coupling
topology. This is the core quantum-classical mapping: classical Kuramoto
ODE → quantum XY spin chain.

```bash
python examples/02_kuramoto_xy_demo.py
```

## 03_qaoa_mpc_demo.py — QAOA Binary MPC

Formulates a 4-step binary coil control problem as an Ising Hamiltonian
and solves it with QAOA (p=2 layers). Compares the QAOA-selected action
sequence against brute-force enumeration. Demonstrates mapping a control
optimization problem to a combinatorial quantum algorithm.

```bash
python examples/03_qaoa_mpc_demo.py
```

## 04_qpetri_demo.py — Quantum Petri Net

Creates a 3-place, 2-transition Petri net where tokens are encoded as
qubit amplitudes and transitions fire via controlled rotations. Shows
how superposition of markings enables parallel exploration of the
reachability graph — relevant for modeling concurrent processes in
control systems.

```bash
python examples/04_qpetri_demo.py
```

## 05_vqe_ansatz_comparison.py — VQE Ansatz Benchmark

Compares three VQE ansatze on the 4-qubit Kuramoto Hamiltonian: the
K_nm-informed ansatz (entanglement topology matches coupling graph),
a generic hardware-efficient ansatz, and EfficientSU2. Reports energy,
parameter count, and convergence iterations. The K_nm-informed ansatz
converges fastest — this is the ansatz used in the 0.05% hardware result.

```bash
python examples/05_vqe_ansatz_comparison.py
```

## 06_zne_demo.py — Zero-Noise Extrapolation

Demonstrates error mitigation on a noisy Heron r2 simulator. Applies
global unitary folding at scale factors [1, 3, 5] and uses Richardson
extrapolation to estimate the zero-noise order parameter. Shows how
ZNE can partially recover the ideal result from noisy circuit execution.

```bash
python examples/06_zne_demo.py
```
