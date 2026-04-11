# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quickstart

# Quickstart

All examples run on the local AerSimulator — no IBM credentials needed.

## 1. Kuramoto dynamics (4 oscillators)

```python
from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import QuantumKuramotoSolver

K = build_knm_paper27(L=4)
omega = OMEGA_N_16[:4]

solver = QuantumKuramotoSolver(4, K, omega)
result = solver.run(t_max=0.5, dt=0.1, trotter_per_step=2)

for t, R in zip(result["times"], result["R"]):
    print(f"  t={t:.1f}: R={R:.4f}")
```

The Kuramoto order parameter R measures phase synchronization: R=1 means
all oscillators are in phase, R=0 means incoherent.

## 2. VQE ground state

```python
from scpn_quantum_control.phase import PhaseVQE
from scpn_quantum_control.bridge import build_knm_paper27, OMEGA_N_16

K = build_knm_paper27(L=4)
omega = OMEGA_N_16[:4]

vqe = PhaseVQE(K, omega, ansatz_reps=2)
sol = vqe.solve(optimizer="COBYLA", maxiter=200)
print(f"VQE energy:   {sol['ground_energy']:.6f}")
print(f"Exact energy: {sol['exact_energy']:.6f}")
print(f"Error:        {sol['energy_gap']:.6f}")
```

On IBM hardware this achieves 0.05% error (4 qubits).

## 3. Run a hardware experiment on simulator

```python
from scpn_quantum_control.hardware import HardwareRunner
from scpn_quantum_control.hardware.experiments import kuramoto_4osc_experiment

runner = HardwareRunner(use_simulator=True)
runner.connect()

result = kuramoto_4osc_experiment(runner, shots=10000, n_time_steps=4, dt=0.1)
print(f"hw_R:  {result['hw_R']}")
print(f"exact: {result['classical_R']}")
```

## 4. ZNE error mitigation

```python
from scpn_quantum_control.hardware import HardwareRunner
from scpn_quantum_control.hardware.experiments import kuramoto_4osc_zne_experiment

runner = HardwareRunner(use_simulator=True)
runner.connect()

result = kuramoto_4osc_zne_experiment(runner, shots=10000, scales=[1, 3, 5])
print(f"R at scale 1: {result['R_per_scale'][0]:.4f}")
print(f"R at scale 5: {result['R_per_scale'][2]:.4f}")
print(f"ZNE R(0):     {result['zne_R']:.4f}")
print(f"Exact R:      {result['classical_R']:.4f}")
```

ZNE (zero-noise extrapolation) runs the same circuit at increasing noise
levels, then fits a polynomial to extrapolate to zero noise.

## 5. Full 16-layer UPDE

```python
from scpn_quantum_control.phase import QuantumUPDESolver

solver = QuantumUPDESolver()  # uses canonical SCPN parameters
result = solver.step(dt=0.05)
print(f"R_global: {result['R_global']:.4f}")
```

## 6. Crypto Bell test on simulator

```python
from scpn_quantum_control.hardware import HardwareRunner
from scpn_quantum_control.hardware.experiments import bell_test_4q_experiment

runner = HardwareRunner(use_simulator=True)
runner.connect()

result = bell_test_4q_experiment(runner, shots=10000, maxiter=100)
print(f"S_hw:  {result['S_hw']:.4f}")
print(f"S_sim: {result['S_sim']:.4f}")
print(f"Violates classical (S>2): {result['violates_classical_hw']}")
```

The Bell test prepares the VQE ground state of H(K_nm), measures in 4 basis
combinations (ZZ, ZX, XZ, XX), and checks whether the CHSH S-value exceeds
the classical bound of 2.

## Available experiments

20 pre-built experiments in `ALL_EXPERIMENTS`:

```python
from scpn_quantum_control.hardware.experiments import ALL_EXPERIMENTS
for name in sorted(ALL_EXPERIMENTS):
    print(name)
```

See [Experiment Roadmap](EXPERIMENT_ROADMAP.md) for the full plan.

## GUESS error mitigation in 5 lines (added April 2026)

For any XY Hamiltonian run on hardware, the conserved total magnetisation
$\sum Z_i$ is a free guide observable. GUESS (Oliva del Moral *et al.*,
arXiv:2603.13060) uses its decay under noise to correct the target:

```python
from scpn_quantum_control.mitigation.symmetry_decay import (
    learn_symmetry_decay, guess_extrapolate, xy_magnetisation_ideal,
)

s_ideal = xy_magnetisation_ideal(n_qubits=4, initial_state="ground")  # = +4
model = learn_symmetry_decay(s_ideal,
                             noisy_symmetry_values=[3.92, 3.65, 3.10],
                             noise_scales=[1, 3, 5])
mitigated = guess_extrapolate(target_noisy_value=0.45,
                              symmetry_noisy_value=3.92,
                              decay_model=model).mitigated_value
```

See [`symmetry_decay_guess.md`](symmetry_decay_guess.md) for the full
theory, the Phase 1 ibm_kingston worked example, and a comparison with
generic Mitiq ZNE.

## Running examples

```bash
python examples/01_qlif_demo.py           # Quantum LIF neuron
python examples/02_kuramoto_xy_demo.py    # Kuramoto XY dynamics
python examples/05_vqe_ansatz_comparison.py  # Ansatz benchmark
python examples/06_zne_demo.py            # ZNE demo
```
