# Phase Evolution API Reference

*14 modules implementing variational, adiabatic, Floquet, and time-evolution algorithms
for the Kuramoto-XY Hamiltonian on quantum hardware.*

These modules answer the question: given a coupling matrix $K_{nm}$ and natural
frequencies $\omega_i$, how do we prepare, evolve, and optimise quantum states that
encode the synchronization dynamics?

This is an advanced module reference. Start with
[Stable Facades API](stable_facades_api.md) and
[Kuramoto Core Facade](kuramoto_core_facade.md) for stable user workflows, then
use this page when direct solver classes or evolution algorithms are required.

---

## Core Solvers

### `xy_kuramoto` — Trotterised Time Evolution

The workhorse: Lie-Trotter decomposition of the XY Hamiltonian for stroboscopic
simulation of Kuramoto dynamics on a gate-based quantum computer.

```python
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver
```

**`QuantumKuramotoSolver(n_oscillators, K_coupling, omega_natural, trotter_order=1)`**

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

Input validation is intentionally strict for this gate-model XY solver:
`K_coupling` must be finite, square, symmetric, and shaped
`(n_oscillators, n_oscillators)`; `omega_natural` must be finite and shaped
`(n_oscillators,)`; both numeric inputs must already be real numeric arrays or
array-like numeric sequences rather than strings, booleans, objects, or complex
values that NumPy could coerce silently; `trotter_order` is limited to 1 or 2. Directed or
non-reciprocal Kuramoto variants should use a dedicated model rather than being
silently coerced into this symmetric XY mapping.

Think of this solver as a stroboscopic camera pointed at the quantum oscillators. At
each time step, the camera captures a snapshot: how synchronised are the oscillators
right now? The Trotter decomposition is the mechanism by which we "advance the clock"
on the quantum computer — we approximate continuous time evolution by a sequence of
discrete, implementable quantum gates.

### `kuramoto_variants` — Higher-Order, Monitored, and PT-Symmetric Trajectories

Reusable trajectory surfaces for Kuramoto extensions beyond the symmetric
gate-model XY Hamiltonian.

```python
from scpn_quantum_control.phase.kuramoto_variants import (
    HigherOrderKuramotoSpec,
    MonitoredKuramotoSpec,
    PTSymmetricKuramotoSpec,
    build_triadic_ring_terms,
    simulate_higher_order_kuramoto,
    simulate_monitored_kuramoto,
    simulate_pt_symmetric_kuramoto,
)
```

| Surface | Description |
|---------|-------------|
| `HigherOrderKuramotoSpec` | Pairwise \(K_{ij}\) plus anchored triadic terms \(B_{ijk}\sin(\theta_j+\theta_k-2\theta_i)\). |
| `MonitoredKuramotoSpec` | Deterministic order-parameter readout and feedback closure around a target \(R_\star\). |
| `PTSymmetricKuramotoSpec` | Balanced gain/loss complex oscillator evolution with \(\sum_i\gamma_i=0\). |
| `simulate_*` | Return `KuramotoVariantResult` with `times`, `r_values`, `backend`, and diagnostics. |

The preferred backend is Rust PyO3 (`higher_order_kuramoto_trajectory`,
`monitored_kuramoto_trajectory`, `pt_symmetric_kuramoto_trajectory`) with NumPy
fallbacks using the same equations. Stable-facade users can call
`simulate_variant_trajectory(problem, variant, ...)`.

### `trotter_upde` — Full 16-Layer UPDE Solver

Extends the Kuramoto solver to the full SCPN 16-layer hierarchy using the canonical
$K_{nm}$ matrix from Paper 27.

```python
from scpn_quantum_control.phase.trotter_upde import QuantumUPDESolver
```

**`QuantumUPDESolver(K=None, omega=None, trotter_order=1)`**

| Method | Description |
|--------|-------------|
| `.hamiltonian()` | Compiled XY Hamiltonian from $K_{nm}$ |
| `.step(dt=0.1, trotter_steps=5)` | Single Trotter step returning `UPDEStepResult` with `R_global`, `psi`, and `dt` |
| `.run(n_steps=50, dt=0.1, trotter_per_step=5)` | Multi-step simulation returning `UPDETrajectoryResult` with `times`, `R`, and `n_layers` |
| `.reset()` | Reset the cached statevector before the next step |

The trajectory result records the global order parameter over the requested time
grid. Hardware claims must be cited through the hardware ledger and committed raw
artifacts.

### `trotter_error` — Error Analysis

Quantifies Trotter error by comparing the two-group (`H_XY` then `H_Z`) product
formula with the exact matrix exponential. The empirical measurement and the
analytical bound use the **same** operator splitting and the **same** spectral
(induced 2-) norm, so `trotter_error_bound` rigorously upper-bounds
`trotter_error_norm` of the matching order. The spectral norm is the
algorithm-error norm in which the Childs et al. (PRX 11, 011020, 2021)
product-formula bounds are stated; the Frobenius norm of the same difference can
exceed these commutator estimates by up to a factor `√(2^n)` and is therefore
not the reported quantity.

```python
from scpn_quantum_control.phase.trotter_error import (
    optimal_dt,
    trotter_error_bound,
    trotter_error_norm,
    trotter_error_sweep,
)
```

`order=1` measures the Lie-Trotter product `(e^{-iH_XY τ} e^{-iH_Z τ})^r`;
`order=2` measures the symmetric Suzuki-Trotter product
`(e^{-iH_XY τ/2} e^{-iH_Z τ} e^{-iH_XY τ/2})^r`. Dense exact comparison APIs
accept `max_dense_gib`:
`trotter_error_norm(K, omega, t, reps, order=1, *, max_dense_gib=None)` and
`trotter_error_sweep(K, omega, t_values, reps_values, order=1, *, max_dense_gib=None)`.
Second-order analytical bounds also forward the budget through
`nested_commutator_norm_bound`, `trotter_error_bound`, and `optimal_dt` when
the exact small-system nested commutator path is selected.

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

### `adapt_vqe` — Adaptive layered VQE

Grows a variational ansatz from a physics-motivated operator pool (Grimsley et
al., Nat. Commun. **10**, 3007, 2019) until the energy converges, returning the
variational ground-state estimate.

The original ADAPT selection rule grows the ansatz by the operator gradient
`⟨ψ|[H, A_k]|ψ⟩`. For the real-symmetric Kuramoto-XY Hamiltonian that rule is
ill-conditioned: every **real** state gives identically zero pool gradients (the
`i(X_iX_j+Y_iY_j)` exchange generators because `[H,G]` is real-antisymmetric, the
`iY_i` generators because `H` has no single-spin-flip terms), and the real
reference `|0…0⟩` is itself an eigenstate. Gradient-selection therefore stalls at
zero operators and reports the energy of an *excited* eigenstate as if converged.
The implementation removes this by the symmetric reference `|+⟩^{⊗n}`, random
non-zero angle initialisation with restarts, and growth by full pool layers; the
variational optimum then reaches the exact ground state for diagonalisable sizes.

```python
from scpn_quantum_control.phase.adapt_vqe import adapt_vqe, ADAPTResult
```

`adapt_vqe(K, omega, max_iterations=20, gradient_threshold=1e-3, maxiter_opt=200,
seed=None, *, n_restarts=4, max_dense_gib=None)` → `ADAPTResult` with: `energy`,
`n_iterations` (ansatz layers), `n_parameters`, `gradient_norms` (per-layer
optimiser gradient norm), `energies` (reference energy then best per layer),
`selected_operators` (pool indices used), `converged`.

The pool holds `i(X_iX_j + Y_iY_j)` exchange generators for each coupled pair and
`iY_i` single-qubit generators; over a few layers it spans the reachable subspace
and the optimiser recovers the exact ground-state energy.

### `varqite` — Variational Quantum Imaginary Time Evolution

Guaranteed convergence to the ground state via imaginary time evolution
$|\psi(\tau)\rangle \propto e^{-H\tau}|\psi(0)\rangle$, implemented variationally
via McLachlan's principle.

```python
from scpn_quantum_control.phase.varqite import (
    varqite_ground_state,
    VarQITEResult,
)
```

`varqite_ground_state(K, omega, tau_max=5.0, dt=0.1, reps=2)` → `VarQITEResult` with:
`tau_values`, `energies`, `final_energy`, `exact_energy`, `params_history`.

Imaginary time evolution is the quantum physicist's gradient descent: it suppresses
excited-state components exponentially in $\tau$, so the state inevitably relaxes to the
ground state. The variational implementation avoids the non-unitary evolution problem
by projecting the dynamics onto a parametric circuit manifold.

### `avqds` — Fixed-ansatz McLachlan variational real-time dynamics

McLachlan's time-dependent variational principle for *real*-time dynamics. Circuit depth is
independent of simulation time $t$ (unlike Trotter, where depth $\propto t/\Delta t$) because the
ansatz is fixed at construction. This is the non-adaptive special case of AVQDS (Yao et al., PRX
Quantum 2, 030307, 2021): it runs the McLachlan equation of motion on a fixed parameter set and does
**not** grow the ansatz from an operator pool, so `n_params` is constant across the trajectory. The
metric $M$ is the **analytic** quantum geometric tensor (see `variational_metric` below): the state
derivatives are exact, not finite-difference estimates.

```python
from scpn_quantum_control.phase.avqds import (
    avqds_simulate,
    AVQDSResult,
)
```

`avqds_simulate(K, omega, t_total=1.0, n_steps=20, ansatz_reps=2, seed=None, *, max_dense_gib=None)` →
`AVQDSResult` with: `times`, `energies`, `fidelities`,
`parameters_history`, `n_params`, `final_energy`, and `final_fidelity`.
The result arrays and parameter history use explicit `float64` array contracts;
the dense Hamiltonian and exact-reference evolution are converted through a
single sparse-compatible `complex128` matrix boundary before time stepping.

### `variational_metric` — analytic quantum geometric tensor

Shared exact linear-system assembly used by both `avqds` (real time) and `varqite`
(imaginary time). The state derivatives $\partial_k|\psi\rangle$ are computed
exactly through the π-shift identity $\partial_k|\psi(\theta)\rangle =
\tfrac{1}{2}|\psi(\theta + \pi e_k)\rangle$, valid because every ansatz parameter
drives a single Pauli-generated rotation. This removes the $O(\varepsilon^2)$ bias
and step-size hyperparameter of a finite-difference metric.

```python
from scpn_quantum_control.phase.variational_metric import (
    analytic_state_derivatives,
    assert_single_parameter_rotations,
    mclachlan_metric,
    real_time_force,
    imaginary_time_force,
)
```

- `analytic_state_derivatives(state_of, params)` → rows $\partial_k|\psi\rangle$
  from a `state_of(values) → statevector` callable (the caller supplies the
  simulator).
- `mclachlan_metric(state_derivatives)` → $G_{ij} = \mathrm{Re}\langle\partial_i\psi|\partial_j\psi\rangle$.
- `real_time_force(state_derivatives, H|ψ⟩)` → $V_i = -\mathrm{Im}\langle\partial_i\psi|H|\psi\rangle$.
- `imaginary_time_force(state_derivatives, (H-\langle H\rangle)|ψ⟩)` → $C_i = -\mathrm{Re}\langle\partial_i\psi|(H-\langle H\rangle)|\psi\rangle$.
- `assert_single_parameter_rotations(ansatz)` validates the π-shift precondition
  (each parameter in exactly one `rx`/`ry`/`rz` gate) on real circuits.

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

Default set of physical systems for local transfer experiments:
`scpn_neural`, `nearest_neighbor`, `mean_field`, and `power_law`. Each system
uses explicit `float64` coupling and frequency arrays.

**`run_transfer_matrix(n_qubits=4, reps=2, maxiter=100, seed=42) → list[TransferResult]`**

All-pairs transfer experiments over the default system set, excluding
self-transfer pairs.

The mechanism behind cross-domain transfer: systems with the same coupling graph have
the same DLA (Gem 11). The variational landscape is determined by the DLA, so optimal
parameters for one system land in a good basin of attraction for another system with the
same algebra. Systems with different topologies have different DLAs, and transfer fails.

### `ansatz_methodology` — Topology-Aware Ansatz Construction

Analysis of different ansatz strategies: hardware-efficient, chemistry-inspired,
and $K_{nm}$-topology-informed.

### `ansatz_bench` — Ansatz Benchmarking

Systematic comparison of ansatz expressibility, trainability, and convergence rate.
`benchmark_ansatz(...)` and `run_ansatz_benchmark(...)` return
`AnsatzBenchmarkRow`, a typed JSON-ready row with ansatz name, qubit count,
parameter count, energy, evaluation count, optimisation history, and repetition
metadata. These rows are local functional benchmark evidence; production
performance claims still require isolated-affinity artefacts.

---

## Advanced Evolution Algorithms

### `qsvt_evolution` — Quantum Singular Value Transformation

QSVT resource estimation for Hamiltonian simulation (Gilyén et al., STOC 2019).
The query estimate scales as
$O(\alpha t + \log(1/\varepsilon))$, where $\alpha$ is the Hamiltonian's Pauli
coefficient 1-norm. Reported step-to-query ratios depend on the supplied
Hamiltonian, time, and error budget; they are not wall-time speedup claims.

```python
from scpn_quantum_control.phase.qsvt_evolution import (
    qsvt_resource_estimate,
    QSVTResourceEstimate,
)
```

`qsvt_resource_estimate(K, omega, t=1.0, epsilon=0.01, *, max_dense_gib=None)`
returns `QSVTResourceEstimate` with `alpha`, `spectral_norm`,
`simulation_time`, `target_error`, `qsvt_queries`, `trotter1_steps`,
`trotter2_steps`, both step-to-query ratios, `n_qubits`, and the conservative
`n_ancilla_qsvt` block-encoding estimate.

Inputs are validated before any Hamiltonian construction or resource-claim
calculation: `K` must be a finite square symmetric coupling matrix, `omega`
must be a finite vector with matching dimension, and both must already contain
real numeric scalar values rather than strings, booleans, objects, or complex
values that NumPy could silently coerce. Simulation time must be finite and
non-negative, and `epsilon` must satisfy `0 < epsilon < 1`. The lower-level
query-count helpers apply the same explicit real-scalar `alpha`, time, and
error-budget checks so invalid budgets cannot be silently clamped or coerced.
For fewer than 14 qubits, `max_dense_gib` guards the two-matrix dense spectral
workspace before allocation; larger systems use sparse extremal eigensolves.

This module provides query/step estimates, not executable circuits, verified
QSP phases, or latency benchmarks. QSVT circuits require a hardware-specific
block encoding of the Hamiltonian, including ancilla and selection logic that
is not constructed here. The estimates inform hardware-roadmap planning only.

`qsp_phase_angles(degree, allow_initial_guess=True)` accepts only a
non-negative integer `degree`. The returned values are symmetric seed angles for
offline optimisation only; with `allow_initial_guess=False` the function raises
until production QSP phase synthesis and verification are wired.

### `adiabatic_preparation` — Ground State via Adiabatic Path

Linearly interpolates from a trivial Hamiltonian (all-Z field) to the full XY
Hamiltonian: $H(s) = (1-s)H_0 + s \cdot H_{XY}$, with $s \in [0, 1]$.

```python
from scpn_quantum_control.phase.adiabatic_preparation import (
    adiabatic_ramp,
    AdiabaticResult,
)
```

The adiabatic theorem guarantees convergence if the spectral gap never closes along
the path. At the BKT transition ($K \approx K_c$), the gap closes exponentially (Gem 23),
making adiabatic preparation exponentially slow. This module is most useful for
$K \neq K_c$, where the gap remains open and adiabatic preparation is efficient.
Inputs are accepted only as explicit real numeric values: `omega`,
`K_topology`, `K_target`, `T_total`, and optional scan grids must not rely on
string, boolean, object, or complex coercion before validation.
The returned `AdiabaticResult` exposes `float64` arrays for `times`,
`K_schedule`, `fidelity`, and `gap`, plus scalar `final_fidelity`, `min_gap`,
and `min_gap_K` values.

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

**`floquet_evolve(K_topology, omega, K_base, drive_amplitude, drive_frequency, n_periods=10, steps_per_period=20, *, max_dense_gib=None) → FloquetResult`**

Evolves $|\psi(t)\rangle$ under $K(t) = K_{\mathrm{base}}(1 + \delta\cos(\Omega t))
\cdot K_{\mathrm{topology}}$ using piecewise-constant Trotter steps within each driving
period. Returns the time series of $R(t)$ and the subharmonic ratio (power at
$\Omega/2$ divided by power at $\Omega$, computed via FFT).

`FloquetResult` fields: `times`, `R_values`, `drive_signal`, `subharmonic_ratio`,
`is_dtc_candidate` (True when `subharmonic_ratio > 0.1`,
the `DTC_SUBHARMONIC_THRESHOLD` used by the module).

**Rust acceleration:** Hamiltonian construction via `build_xy_hamiltonian_dense` (Qiskit-free).
Order parameter R computed via `all_xy_expectations` (batch bitwise Pauli, 1 FFI call instead
of 2n Qiskit SparsePauliOp evaluations).

**`scan_drive_amplitude(K_topology, omega, K_base, drive_frequency, amplitudes=None, ..., max_dense_gib=None) → dict`**

Scans drive amplitude $\delta$ to map the DTC phase boundary. The dense budget
is forwarded to every piecewise-constant Floquet Hamiltonian build.

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
