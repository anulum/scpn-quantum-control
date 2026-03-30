# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase Evolution API Reference

# Phase Evolution API Reference

*14 modules implementing variational, adiabatic, Floquet, and time-evolution algorithms
for the Kuramoto-XY Hamiltonian on quantum hardware.*

These modules answer the question: given a coupling matrix $K_{nm}$ and natural
frequencies $\omega_i$, how do we prepare, evolve, and optimise quantum states that
encode the synchronization dynamics?

---

## Core Solvers

### `xy_kuramoto` — Trotterised Time Evolution

The workhorse: Lie-Trotter decomposition of the XY Hamiltonian for stroboscopic
simulation of Kuramoto dynamics on a gate-based quantum computer.

```python
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver
```

**`QuantumKuramotoSolver(n_oscillators, K_coupling, omega_natural, backend=None)`**

| Method | Description |
|--------|-------------|
| `.build_hamiltonian()` | Construct $H_{XY}$ as `SparsePauliOp` |
| `.evolve(time, trotter_steps=1)` | Build Trotter circuit $U(t) \approx [e^{-iH_{XY}\Delta t} e^{-iH_Z\Delta t}]^{t/\Delta t}$ |
| `.measure_order_parameter(statevector)` | Extract $(R, \psi)$ from state |
| `.run(t_max, dt, trotter_per_step=5)` | Full simulation: returns `{times, R}` |

The Trotter decomposition splits the Hamiltonian into mutually commuting XX+YY coupling
terms and single-qubit Z rotations. The Trotter error scales as $O(\Delta t^2)$ per step
for this particular decomposition. For better accuracy at longer times, use the
second-order Suzuki-Trotter variant or the QSVT-based evolution (see below).

Think of this solver as a stroboscopic camera pointed at the quantum oscillators. At
each time step, the camera captures a snapshot: how synchronised are the oscillators
right now? The Trotter decomposition is the mechanism by which we "advance the clock"
on the quantum computer — we approximate continuous time evolution by a sequence of
discrete, implementable quantum gates.

### `trotter_upde` — Full 16-Layer UPDE Solver

Extends the Kuramoto solver to the full SCPN 16-layer hierarchy using the canonical
$K_{nm}$ matrix from Paper 27.

```python
from scpn_quantum_control.phase.trotter_upde import QuantumUPDESolver
```

**`QuantumUPDESolver(n_layers=16, knm=None, omega=None)`**

| Method | Description |
|--------|-------------|
| `.build_hamiltonian()` | 16-qubit XY Hamiltonian from $K_{nm}$ |
| `.step(dt, shots=10000)` | Single Trotter step, measure $R$ per layer |
| `.run(n_steps, dt, shots=10000)` | Multi-step simulation with per-layer tracking |

Returns `per_layer_R` (order parameter per SCPN layer) and `global_R` (system-wide
synchronization). This is the quantum digital twin of the UPDE — the master equation
of the SCPN framework running on real quantum hardware.

### `trotter_error` — Error Analysis

Quantifies Trotter error by comparing the approximate evolution operator with the exact
matrix exponential.

```python
from scpn_quantum_control.phase.trotter_error import (
    trotter_fidelity,
    optimal_trotter_steps,
)
```

---

## Variational Methods

### `phase_vqe` — Variational Quantum Eigensolver

Ground state preparation via parametric circuit optimisation.

```python
from scpn_quantum_control.phase.phase_vqe import PhaseVQE
```

**`PhaseVQE(K, omega, ansatz_reps=2, threshold=0.01)`**

| Method | Description |
|--------|-------------|
| `.solve(optimizer="COBYLA", maxiter=200, seed=None)` | Returns ground energy, exact energy, relative error, convergence status |
| `.ground_state()` | `Statevector` of the optimised state (or `None` if not yet solved) |

The ansatz is physics-informed: entangling gates (CZ) connect only qubit pairs where
$K_{ij} > \varepsilon$, respecting the coupling topology. This eliminates barren
plateaus caused by unnecessary entanglement in the ansatz and reduces the parameter
count.

The VQE is the gateway to all downstream analysis: once you have the ground state, you
can measure witnesses, compute QFI, extract entanglement spectra, and evaluate every
probe in the analysis API. For systems larger than ~6 qubits, VQE on quantum hardware
becomes necessary because exact diagonalisation is classically intractable.

### `adapt_vqe` — Gradient-Driven Operator Selection

ADAPT-VQE (Grimsley et al., Nat. Commun. **10**, 3007, 2019) builds the ansatz
dynamically by selecting, at each step, the operator with the largest energy gradient.
This avoids the fixed-ansatz depth problem of standard VQE.

```python
from scpn_quantum_control.phase.adapt_vqe import (
    adapt_vqe_solve,
    ADAPTResult,
)
```

`adapt_vqe_solve(K, omega, operator_pool=None, max_layers=20, grad_threshold=1e-3)` →
`ADAPTResult` with: `energy`, `exact_energy`, `n_layers`, `selected_operators`,
`gradient_norms`.

The operator pool defaults to single-excitation and double-excitation Pauli operators
drawn from the DLA of the XY Hamiltonian. Since $\dim(\mathrm{DLA}) = 2^{2N-1} - 2$
(Gem 11), the pool spans exactly the reachable subspace — no wasted operators.

### `varqite` — Variational Quantum Imaginary Time Evolution

Guaranteed convergence to the ground state via imaginary time evolution
$|\psi(\tau)\rangle \propto e^{-H\tau}|\psi(0)\rangle$, implemented variationally
via McLachlan's principle.

```python
from scpn_quantum_control.phase.varqite import (
    varqite_solve,
    VarQITEResult,
)
```

`varqite_solve(K, omega, tau_max=5.0, dt=0.1, reps=2)` → `VarQITEResult` with:
`tau_values`, `energies`, `final_energy`, `exact_energy`, `params_history`.

Imaginary time evolution is the quantum physicist's gradient descent: it suppresses
excited-state components exponentially in $\tau$, so the state inevitably relaxes to the
ground state. The variational implementation avoids the non-unitary evolution problem
by projecting the dynamics onto a parametric circuit manifold.

### `avqds` — Adaptive Variational Quantum Dynamics Simulation

McLachlan variational principle for *real*-time dynamics. Circuit depth is independent of
simulation time $t$ (unlike Trotter, where depth $\propto t/\Delta t$).

```python
from scpn_quantum_control.phase.avqds import (
    avqds_evolve,
    AVQDSResult,
)
```

`avqds_evolve(K, omega, t_max=5.0, dt=0.1, reps=2)` → `AVQDSResult` with:
`times`, `energies`, `R_values`, `params_history`.

---

## Transfer Learning and Ansatz Design

### `cross_domain_transfer` — VQE Parameter Warm-Starting

Optimal VQE parameters from one Kuramoto-XY system used to warm-start optimisation on
a different system. Provides 2–5× convergence speedup for systems sharing the same
coupling topology class.

```python
from scpn_quantum_control.phase.cross_domain_transfer import (
    PhysicalSystem,
    TransferResult,
    transfer_experiment,
    build_systems,
    run_transfer_matrix,
    summarize_transfer,
)
```

**`PhysicalSystem(name, K, omega)`** — defines a physical system by its coupling matrix
and frequencies.

**`transfer_experiment(source, target, reps=2, maxiter=200, seed=42) → TransferResult`**

Runs VQE on the source system, transfers parameters to the target, and compares against
random initialisation on the target. `TransferResult` fields: `source_system`,
`target_system`, `random_init_energy`, `transfer_init_energy`, `speedup`,
`energy_improvement`, `exact_energy`.

**`build_systems(n_qubits=4) → list[PhysicalSystem]`**

Default set of physical systems for benchmarking: ring, chain, star, complete graph.

**`run_transfer_matrix(systems, ...) → list[TransferResult]`**

All-pairs transfer experiments.

The mechanism behind cross-domain transfer: systems with the same coupling graph have
the same DLA (Gem 11). The variational landscape is determined by the DLA, so optimal
parameters for one system land in a good basin of attraction for another system with the
same algebra. Systems with different topologies have different DLAs, and transfer fails.

### `ansatz_methodology` — Topology-Aware Ansatz Construction

Analysis of different ansatz strategies: hardware-efficient, chemistry-inspired,
and $K_{nm}$-topology-informed.

### `ansatz_bench` — Ansatz Benchmarking

Systematic comparison of ansatz expressibility, trainability, and convergence rate.

---

## Advanced Evolution Algorithms

### `qsvt_evolution` — Quantum Singular Value Transformation

QSVT resource estimation for Hamiltonian simulation (Gilyén et al., STOC 2019).
Achieves optimal query complexity $O(t + \log(1/\varepsilon))$, a 260× estimated
speedup over first-order Trotter at target fidelity $\varepsilon = 10^{-3}$.

```python
from scpn_quantum_control.phase.qsvt_evolution import (
    qsvt_resource_estimate,
    QSVTResourceEstimate,
)
```

`qsvt_resource_estimate(K, omega, t_max, epsilon=1e-3)` → `QSVTResourceEstimate` with:
`n_queries`, `trotter_depth_equivalent`, `speedup_factor`, `block_encoding_cost`.

This module provides resource estimates, not executable circuits. QSVT circuits require
block encoding of the Hamiltonian, which demands ancilla qubits and multi-controlled
gates that exceed current hardware capabilities. The estimates inform hardware roadmap
planning.

### `adiabatic_preparation` — Ground State via Adiabatic Path

Linearly interpolates from a trivial Hamiltonian (all-Z field) to the full XY
Hamiltonian: $H(s) = (1-s)H_0 + s \cdot H_{XY}$, with $s \in [0, 1]$.

```python
from scpn_quantum_control.phase.adiabatic_preparation import (
    adiabatic_prepare,
    AdiabaticResult,
)
```

The adiabatic theorem guarantees convergence if the spectral gap never closes along
the path. At the BKT transition ($K \approx K_c$), the gap closes exponentially (Gem 23),
making adiabatic preparation exponentially slow. This module is most useful for
$K \neq K_c$, where the gap remains open and adiabatic preparation is efficient.

**Rust acceleration:** Hamiltonian construction at each adiabatic step via
`build_xy_hamiltonian_dense` (Qiskit-free).

---

## Periodically Driven Systems

### `floquet_kuramoto` — Discrete Time Crystal

Periodically driven Kuramoto-XY with heterogeneous frequencies: the first
Floquet-Kuramoto DTC.

```python
from scpn_quantum_control.phase.floquet_kuramoto import (
    floquet_evolve,
    scan_drive_amplitude,
    FloquetResult,
)
```

**`floquet_evolve(K_topology, omega, K_base, drive_amplitude, drive_frequency, n_periods=10, steps_per_period=20) → FloquetResult`**

Evolves $|\psi(t)\rangle$ under $K(t) = K_{\mathrm{base}}(1 + \delta\cos(\Omega t))
\cdot K_{\mathrm{topology}}$ using piecewise-constant Trotter steps within each driving
period. Returns the time series of $R(t)$ and the subharmonic ratio (power at
$\Omega/2$ divided by power at $\Omega$, computed via FFT).

`FloquetResult` fields: `times`, `R_values`, `drive_signal`, `subharmonic_ratio`,
`is_dtc_candidate` (True when `subharmonic_ratio > 1`).

**Rust acceleration:** Hamiltonian construction via `build_xy_hamiltonian_dense` (Qiskit-free).
Order parameter R computed via `all_xy_expectations` (batch bitwise Pauli, 1 FFI call instead
of 2n Qiskit SparsePauliOp evaluations).

**`scan_drive_amplitude(K_topology, omega, K_base, drive_frequency, amplitudes=None, ...) → dict`**

Scans drive amplitude $\delta$ to map the DTC phase boundary.

A discrete time crystal is a phase of matter that spontaneously breaks the discrete
time-translation symmetry of its periodic drive. If the system is driven at frequency
$\Omega$, a DTC responds at $\Omega/2$ — it oscillates at half the driving frequency,
even though nothing in the Hamiltonian has that period. This is the temporal analogue of
a spatial crystal breaking continuous translational symmetry.

The Kuramoto-XY version is novel because every oscillator has a different natural
frequency $\omega_i$. In conventional DTCs, all spins are identical. The heterogeneous
frequencies act as effective disorder, which may stabilise the DTC via many-body
localisation (MBL). This is an open question — the module provides the tools to
investigate it.
