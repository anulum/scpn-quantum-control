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

## 07_crypto_bell_test.py — CHSH Bell Test

Prepares the K_nm ground state via VQE and evaluates the CHSH S-parameter
for nearest-neighbour qubit pairs. Demonstrates entanglement certification
using the coupling topology as shared secret.

```bash
python examples/07_crypto_bell_test.py
```

## 08_dynamical_decoupling.py — DD Fidelity Comparison

Compares the order parameter R with and without XY4 dynamical decoupling
on a 4-qubit Kuramoto circuit under synthetic Heron r2 noise. Shows how
DD pulse insertion partially recovers coherence on NISQ hardware.

```bash
python examples/08_dynamical_decoupling.py
```

## 09_classical_vs_quantum_benchmark.py — Classical vs Quantum

Benchmarks classical Euler ODE, exact matrix-exponential evolution, and
quantum Trotterized XY simulation at N=4, 8, 16. Reports wall-clock time,
R(t) accuracy, and VQE ground-state energy vs exact diagonalization.

```bash
python examples/09_classical_vs_quantum_benchmark.py
```

## 10_identity_continuity_demo.py — Identity Analysis

End-to-end demo of all identity modules: VQE attractor basin, coherence
budget, entanglement witness, spectral fingerprint, and binding spec.

```bash
python examples/10_identity_continuity_demo.py
```

## 11_pec_demo.py — PEC Error Mitigation

Probabilistic error cancellation: quasi-probability coefficients and
Monte Carlo sampling at varying gate error rates.

```bash
python examples/11_pec_demo.py
```

## 12_trapped_ion_demo.py — Trapped-Ion Backend

Synthetic trapped-ion noise model with all-to-all connectivity.
Transpiles and runs Kuramoto circuits under MS gate noise.

```bash
python examples/12_trapped_ion_demo.py
```

## 13_iter_disruption_demo.py — ITER Disruption Classifier

11 physics-based ITER features, synthetic data generation, and quantum
circuit classifier benchmark.

```bash
python examples/13_iter_disruption_demo.py
```

## 14_quantum_advantage_demo.py — Quantum Advantage Scaling

Classical vs quantum wall-clock timing with crossover extrapolation.

```bash
python examples/14_quantum_advantage_demo.py
```

## 15_qsnn_training_demo.py — QSNN Training

Parameter-shift gradient descent on QuantumDenseLayer synapse angles.

```bash
python examples/15_qsnn_training_demo.py
```

## 16_fault_tolerant_demo.py — Fault-Tolerant UPDE

Repetition-code logical qubits with syndrome extraction at distance 3 and 5.

```bash
python examples/16_fault_tolerant_demo.py
```

## 17_snn_ssgf_bridges_demo.py — Cross-Repo Bridges

SNN adapter (spike trains → quantum → currents), SSGF adapter (geometry
→ Hamiltonian → roundtrip), and orchestrator phase mapping (18 ↔ 35).

```bash
python examples/17_snn_ssgf_bridges_demo.py
```

## 18_end_to_end_pipeline.py — Full Pipeline

K_nm → VQE → Trotter evolution → ZNE → PEC → classical comparison.
All 5 stages in one script.

```bash
python examples/18_end_to_end_pipeline.py
```
