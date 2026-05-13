# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Analysis API Reference

# Analysis API Reference

*42 modules for probing quantum synchronization transitions, entanglement structure,
topological invariants, and computational complexity of the Kuramoto-XY Hamiltonian.*

This is an advanced module reference. Use
[Stable Facades API](stable_facades_api.md) and
[Kuramoto Core Facade](kuramoto_core_facade.md) for first-path workflows, then
drop into this page when a specific analysis probe or low-level diagnostic is
needed.

---

## Synchronization Detection

### `sync_witness` — Synchronization Witness Operators

Three Hermitian witness constructions that certify quantum synchronization from
hardware measurement counts. No state tomography required.

```python
from scpn_quantum_control.analysis.sync_witness import (
    WitnessResult,
    correlation_witness_from_counts,
    fiedler_witness_from_counts,
    fiedler_witness_from_correlator,
    topological_witness_from_correlator,
    evaluate_all_witnesses,
    calibrate_thresholds,
)
```

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `correlation_witness_from_counts` | X/Y counts, `n_qubits`, `threshold` | `WitnessResult` | Mean pairwise XY correlator vs threshold |
| `fiedler_witness_from_counts` | X/Y counts, `n_qubits`, `threshold` | `WitnessResult` | Algebraic connectivity of correlation Laplacian |
| `fiedler_witness_from_correlator` | `corr_matrix`, `threshold` | `WitnessResult` | From pre-computed correlation matrix |
| `topological_witness_from_correlator` | `corr_matrix`, `threshold`, `max_dim` | `WitnessResult` | Persistent H₁ via Vietoris-Rips (requires `ripser`) |
| `evaluate_all_witnesses` | X/Y counts, `n_qubits`, thresholds | `dict[str, WitnessResult]` | All three witnesses from one measurement set |
| `calibrate_thresholds` | `K`, `omega`, `K_base_range`, `n_samples` | `dict[str, float]` | Classical Kuramoto calibration of thresholds |

`WitnessResult` fields: `witness_name`, `expectation_value` (negative = synchronised),
`threshold`, `is_synchronized`, `raw_observable`, `n_qubits`.

Full theory and examples: [Research Gems — Gem 1](research_gems.md#gem-1-synchronization-witness-operators).

### `witness_discovery` — Automated Witness Search

Bayesian plus bandit search over Kuramoto control candidates, scored by the existing
correlation and Fiedler synchronisation witnesses.

```python
from scpn_quantum_control.analysis.witness_discovery import (
    WitnessCandidate,
    WitnessDiscoverySpec,
    discover_kuramoto_witnesses,
    score_witness_candidates,
)
```

| Function | Description |
|----------|-------------|
| `discover_kuramoto_witnesses(K_nm, omega, theta0, spec)` | Run deterministic initial design, RBF Bayesian UCB, and bandit local exploration. |
| `score_witness_candidates(K_nm, omega, candidates)` | Score fixed candidates through the same witness objective. |
| `WitnessDiscoveryResult.ranked(limit)` | Return candidates sorted by descending witness score. |

The Rust path `kuramoto_witness_candidate_features` evaluates final order
parameter, mean pairwise correlation, and final phases for candidate batches.
The Python scorer then evaluates the existing witness objects, so the discovery
loop stays connected to the hardware-measurable witness definitions.

`RLDiscoveryAgent` is a compatibility wrapper around the same production search.
It accepts only the wired objective: correlation and Fiedler observables with
`reward_function="witness_score"`, positive `n_episodes`, and no external
`runner`. Unsupported compatibility parameters fail at construction instead of
being silently ignored.

### `sync_entanglement_witness` — R as Entanglement Witness

The Kuramoto order parameter $R$ reinterpreted as an entanglement witness. For separable
states, $R \leq R_{\mathrm{sep}}$. Exceeding the separable bound certifies entanglement.

```python
from scpn_quantum_control.analysis.sync_entanglement_witness import (
    EntanglementWitnessResult,
    R_entanglement_scan,
    R_from_statevector,
    R_separable_bound,
    R_separable_bound_at_energy,
    detect_entanglement_from_R,
)
```

| Function | Description |
|----------|-------------|
| `R_from_statevector(psi, n_qubits)` | Compute $R = \|(1/N)\sum_i(\langle X_i\rangle + i\langle Y_i\rangle)\|$ |
| `R_separable_bound(n_qubits)` | Maximum $R$ achievable by any separable state (= 1.0) |
| `R_separable_bound_at_energy(K, omega, target_energy, n_samples=1000, seed=42, *, max_dense_gib=None)` | Dense exact max $R$ over sampled product states at fixed energy |
| `detect_entanglement_from_R(K, omega, n_samples=2000, seed=42, *, max_dense_gib=None)` | Ground-state witness evaluation with dense exact small-system guard |
| `R_entanglement_scan(K, omega, K_base_range=None, n_K_values=15, n_samples=500, seed=42, *, max_dense_gib=None)` | Coupling scan of $R_\mathrm{ground}$ and the energy-constrained separable bound |

The returned `entanglement_depth` is a certified lower bound from this witness:
`1` when the separable bound is not violated and `2` when entanglement is
certified. The R witness alone does not certify stronger multipartite depth;
that requires separate k-producibility bounds or a dedicated depth witness.

### `critical_concordance` — Multi-Probe $K_c$ Agreement

Runs a finite-size dense exact coupling scan and compares where the order
parameter, QFI, spectral gap, and concurrence-graph probes localise the same
coupling region.

```python
from scpn_quantum_control.analysis.critical_concordance import (
    critical_concordance,
    ConcordanceResult,
)
```

`critical_concordance(omega, K_topology, k_range=None, concurrence_threshold=1e-4, *, max_dense_gib=None)` returns `ConcordanceResult` with
fields: `k_values`, `R_values`, `qfi_values`, `gap_values`, `fiedler_values`,
`n_entangled_pairs`, `k_c_from_gap`, `k_c_from_qfi`, `k_c_from_fiedler`,
`k_c_from_R_deriv`, and `concordance_spread`.

---

## Phase Transition Probes

### `qfi_criticality` — Quantum Fisher Information at $K_c$

QFI diverges where the spectral gap closes — the synchronization transition is a
metrological sweet spot.

```python
from scpn_quantum_control.analysis.qfi_criticality import (
    qfi_vs_coupling,
    QFICriticalityResult,
)
```

`qfi_vs_coupling(K, omega, K_base_range=None, n_K=20)` → `QFICriticalityResult` with:
`K_values`, `qfi_values`, `gap_values`, `peak_K` (coupling at max QFI).

### `entanglement_percolation` — Finite-Size Entanglement Percolation

Compares the concurrence-graph percolation point with a selected finite-size
order-parameter threshold. This is a dense exact diagnostic, not a standalone
thermodynamic-limit proof.

```python
from scpn_quantum_control.analysis.entanglement_percolation import (
    percolation_scan,
    PercolationScanResult,
)
```

`percolation_scan(omega, K_topology, k_range=None, concurrence_threshold=1e-4, R_threshold=0.5, *, max_dense_gib=None)` →
`PercolationScanResult` with: `k_values`, `fiedler_values`,
`max_concurrence`, `mean_concurrence`, `n_entangled_pairs`, `R_values`,
`k_percolation`, and `k_sync`.

### `berry_phase` — Berry Connection and Fidelity Susceptibility

Finite-size dense exact scan of ground-state overlaps. On the one-dimensional
open coupling path, the accumulated Berry connection is gauge-dependent; the
fidelity and fidelity susceptibility are the gauge-invariant diagnostics.

```python
from scpn_quantum_control.analysis.berry_phase import (
    berry_phase_scan,
    BerryPhaseResult,
)
```

`berry_phase_scan(omega, K_topology, k_range=None, *, max_dense_gib=None)` →
`BerryPhaseResult` with: `k_values`, `berry_connection`, `berry_curvature`,
`accumulated_phase`, `fidelity`, `fidelity_susceptibility`, `spectral_gap`,
and `curvature_peak_k`.

### `finite_size_scaling` — Finite-Size Gap-Minimum Scaling

Fits finite-size gap-minimum estimates to a BKT-motivated
$K_c(N) = K_c(\infty) + a/(\ln N)^2$ ansatz and a power-law comparison model.

```python
from scpn_quantum_control.analysis.finite_size_scaling import (
    finite_size_scaling,
    FSSResult,
)
```

`finite_size_scaling(system_sizes=None, k_range=None, *, max_dense_gib=None)` → `FSSResult`
with: `system_sizes`, `k_c_values`, `gap_min_values`,
`k_c_extrapolated_bkt`, and `k_c_extrapolated_power`.

### `adiabatic_preparation` — Adiabatic State Preparation

Finite-size dense exact adiabatic path from a weak-coupling initial ground
state to the target XY Hamiltonian. Computes instantaneous gap and fidelity
along the selected schedule.

```python
from scpn_quantum_control.phase.adiabatic_preparation import (
    adiabatic_ramp,
    AdiabaticResult,
)
```

`adiabatic_ramp(omega, K_topology, K_target, T_total=10.0, n_steps=50, *, max_dense_gib=None)` →
`AdiabaticResult` with: `times`, `K_schedule`, `fidelity`, `gap`,
`final_fidelity`, `min_gap`, `min_gap_K`.

---

## Entanglement and Correlations

### `entanglement_entropy` — Half-Chain Entropy and Schmidt Gap

Entanglement entropy and Schmidt gap across the synchronization transition. At BKT
criticality, entropy follows CFT scaling $S \sim (c/3)\ln L$ with $c = 1$.

```python
from scpn_quantum_control.analysis.entanglement_entropy import (
    entanglement_vs_coupling,
    EntanglementScanResult,
)
```

`entanglement_vs_coupling(omega, K_topology, k_range=None)` →
`EntanglementScanResult` with: `k_values`, `entropy`, `schmidt_gap`,
`spectral_gap`, `entropy_peak_K`, `schmidt_gap_min_K`.

**Rust acceleration:** Hamiltonian construction via `build_xy_hamiltonian_dense` (Qiskit-free).

### `entanglement_spectrum` — Full Entanglement Spectrum

Computes the full entanglement spectrum (all Schmidt coefficients) and estimates the
CFT central charge from the entropy scaling.

```python
from scpn_quantum_control.analysis.entanglement_spectrum import (
    entanglement_spectrum,
    cft_central_charge,
)
```

### `pairing_correlator` — Richardson Pairing $\langle S^+_i S^-_j\rangle$

Detects Richardson pairing (the superconducting analogue of synchronization) via
spin-raising/lowering correlators. Strong pairing = synchronised phase.

```python
from scpn_quantum_control.analysis.pairing_correlator import (
    pairing_map,
    pairing_vs_anisotropy,
    PairingResult,
)
```

`pairing_map(omega, K_topology, K_base, delta=0.0, *, max_dense_gib=None)` →
`PairingResult` with the full pairing matrix, maximum/mean pairing, topology
correlation, qubit count, anisotropy, and base coupling.

`pairing_vs_anisotropy(omega, K_topology, K_base, delta_range=None, *, max_dense_gib=None)`
forwards the dense budget to every XXZ ground-state solve in the scan.

---

## Quantum Chaos and Dynamics

### `otoc` — Out-of-Time-Order Correlator

Core OTOC computation: $F(t) = \langle W^\dagger(t) V^\dagger W(t) V\rangle$.

```python
from scpn_quantum_control.analysis.otoc import (
    compute_otoc,
    OTOCResult,
)
```

`compute_otoc(K, omega, times, w_qubit=0, v_qubit=None)` → `OTOCResult` with:
`times`, `otoc_values`, `lyapunov_estimate`, `scrambling_time`.

**Rust acceleration:** When `scpn_quantum_engine` is installed, OTOC uses eigendecomposition
+ rayon-parallel time loop ($O(d^2)$ per time point vs $O(d^3)$ `scipy.expm`). Hamiltonian
construction uses `build_xy_hamiltonian_dense` (bitwise, Qiskit-free). 10-50× faster for n ≤ 8.

### `otoc_sync_probe` — OTOC Scan Across $K_c$

Scans OTOC diagnostics vs coupling strength to detect the synchronization transition
via chaos measures.

```python
from scpn_quantum_control.analysis.otoc_sync_probe import (
    otoc_sync_scan,
    OTOCSyncScanResult,
)
```

`otoc_sync_scan(K, omega, K_base_range=None, n_K_values=15, t_max=2.0)` →
`OTOCSyncScanResult` with: `K_base_values`, `lyapunov_values`, `scrambling_times`,
`otoc_final_values`, `R_classical`, `peak_scrambling_K`.

### `spectral_form_factor` — SFF and Level Statistics

Spectral Form Factor diagnoses chaos via Random Matrix Theory level statistics.

```python
from scpn_quantum_control.analysis.spectral_form_factor import (
    spectral_form_factor,
    level_spacing_ratio,
    SFFResult,
)
```

| Function | Description |
|----------|-------------|
| `spectral_form_factor(H, t_values)` | $g(t) = \|\mathrm{Tr}(e^{-iHt})\|^2 / d^2$ |
| `level_spacing_ratio(H)` | Mean ratio $\bar{r}$: Poisson ≈ 0.386, GOE ≈ 0.536 |

### `loschmidt_echo` — Loschmidt Echo and DQPT

Dynamical Quantum Phase Transitions detected via non-analyticities in the Loschmidt
return rate $\lambda(t) = -\ln\mathcal{L}(t)/N$.

```python
from scpn_quantum_control.analysis.loschmidt_echo import (
    loschmidt_echo,
    LoschmidtResult,
)
```

`loschmidt_echo(K, omega, K_i, K_f, times)` → `LoschmidtResult` with:
`times`, `echo_values`, `return_rate`, `dqpt_times` (cusp locations).

**Rust acceleration:** Hamiltonian construction via `build_xy_hamiltonian_dense` (Qiskit-free).

### `krylov_complexity` — Operator Spreading Complexity

Lanczos coefficients $b_n$ and Krylov complexity $C_K(t) = \sum_n n |\phi_n(t)|^2$.
Maximum at $K_c$.

```python
from scpn_quantum_control.analysis.krylov_complexity import (
    krylov_complexity,
    krylov_vs_coupling,
    KrylovResult,
)
```

`krylov_complexity(H, O_init, t_max=10.0, n_times=100, max_lanczos=50)` →
`KrylovResult` with Lanczos coefficients, times, complexity values, peak
complexity, and realised Krylov dimension.

`krylov_vs_coupling(omega, K_topology, k_range=None, t_max=10.0, n_times=50, *, max_dense_gib=None)`
builds the dense Hamiltonian/probe workspace under the caller's budget before
scanning peak complexity against coupling.

**Rust acceleration:** Lanczos b-coefficients computed via `lanczos_b_coefficients` (complex
matrix commutator loop in Rust, 5-10× for dim ≤ 256). Hamiltonian via `build_xy_hamiltonian_dense`.

---

## Quantum Information Measures

### `qfi` — Quantum Fisher Information Matrix

Full QFI matrix for parameter estimation precision bounds.

```python
from scpn_quantum_control.analysis.qfi import (
    quantum_fisher_information,
    spectral_gap,
    precision_bounds,
)
```

| Function | Description |
|----------|-------------|
| `quantum_fisher_information(state, generators)` | QFI matrix $F_{ij}$ |
| `spectral_gap(H)` | $E_1 - E_0$ |
| `precision_bounds(qfi_matrix)` | Cramér-Rao lower bounds $\delta\theta_i \geq 1/\sqrt{F_{ii}}$ |

`QuantumFisherInformation` is the observable-wrapper adapter for production
metrology calls. When `coupling_matrix` and `natural_frequencies` are supplied
it routes to the spectral QFI engine and validates that the coupling matrix is
square, symmetric, finite-valued, and dimension-compatible with the frequency
vector. Optional `coupling_pairs` must be distinct in-range integer index pairs,
and `n_measurements` must be a positive integer because it rescales the
Cramér-Rao precision bound. Counts-derived sync/DLA estimates are exposed only
through the explicit `allow_proxy_estimate=True` diagnostic path and are labelled
as proxy values, never as production QFI.

### `magic_nonstabilizerness` — Stabilizer Rényi Entropy

Magic $M_2 = -\log_2(\sum_P \langle P\rangle^4 / 2^N)$ peaks at $K_c$ — the critical
state is maximally non-classical.

```python
from scpn_quantum_control.analysis.magic_nonstabilizerness import (
    magic_at_coupling,
    magic_vs_coupling,
    MagicResult,
)
```

`magic_at_coupling(omega, K_topology, K_base, *, max_dense_gib=None)` computes
the dense exact ground state and Stabilizer Renyi entropy at one coupling.

`magic_vs_coupling(omega, K_topology, k_range=None, *, max_dense_gib=None)`
forwards the dense eigensolver budget to every coupling point and returns a
`MagicScanResult` with the scanned values and peak location.

### `quantum_phi` — Integrated Information (IIT)

Quantum integrated information from the Kuramoto-XY ground-state density
matrix. `compute_quantum_phi(K, omega)` computes the minimum mutual information
over bipartitions and reports the minimum-information partition.

```python
from scpn_quantum_control.analysis.quantum_phi import (
    compute_quantum_phi,
    PhiResult,
)
```

`IntegratedInformationPhi` is the dashboard-facing wrapper. When supplied with
`coupling_matrix` and `natural_frequencies`, it routes to `compute_quantum_phi`
and returns `phi`, `phi_max`, entropy, and partition metadata. Counts-only
entropy remains available only via `allow_entropy_proxy=True` and is labelled
`entropy_proxy`, never `phi`.

### `shadow_tomography` — Classical Shadow Estimation

$O(\log M)$ shots for $M$ observables via random Clifford measurements.

```python
from scpn_quantum_control.analysis.shadow_tomography import (
    random_clifford_shadow,
    estimate_observable,
    ShadowResult,
)
```

### `quantum_speed_limit` — QSL for BKT Synchronization

Mandelstam-Tamm and Margolus-Levitin speed limits: minimum time to evolve between
states across the synchronization transition.

```python
from scpn_quantum_control.analysis.quantum_speed_limit import (
    qsl_vs_coupling,
    QSLResult,
)
```

`qsl_vs_coupling(K, omega, t_target=1.0, K_base_range=None, n_K=15)` → `QSLResult`
with: `K_base`, `mt_limits` (Mandelstam-Tamm), `ml_limits` (Margolus-Levitin).

---

## Topological Analysis

### `quantum_persistent_homology` — Full PH Pipeline

Hardware counts → correlation matrix → distance → Vietoris-Rips → persistence diagram
→ $p_{H_1}$.

```python
from scpn_quantum_control.analysis.quantum_persistent_homology import (
    counts_to_persistence,
    coupling_scan_persistence,
    PersistenceResult,
    PersistenceScanResult,
)
```

| Function | Description |
|----------|-------------|
| `counts_to_persistence(x_counts, y_counts, n_qubits, max_dim=1)` | Single-point PH from hardware counts |
| `coupling_scan_persistence(K, omega, K_range, ...)` | $p_{H_1}$ vs coupling strength |

### `persistent_homology` — Classical PH Utilities

Distance matrix construction, Rips filtration, Betti number extraction.

### `h1_persistence` — Vortex Density at BKT

$H_1$ persistence as a function of coupling — the topological order parameter for the
BKT transition.

### `vortex_binding` — Kosterlitz RG Flow

Vortex-antivortex binding energy and Kosterlitz renormalization group flow equations.

---

## Algebraic Structure

### `dynamical_lie_algebra` — DLA Computation

Computes the Dynamical Lie Algebra and its dimension for the Kuramoto-XY Hamiltonian.
Result: $\dim(\mathrm{DLA}) = 2^{2N-1} - 2$ for non-degenerate frequencies.

```python
from scpn_quantum_control.analysis.dynamical_lie_algebra import (
    compute_dla,
    DLAResult,
)
```

`compute_dla(K, omega)` → `DLAResult` with: `generators` (list of Pauli strings),
`dimension`, `n_qubits`, `predicted_dim` ($2^{2N-1} - 2$).

### `dla_parity_theorem` — Z₂ Parity Proof

Formal verification that Z₂ parity is the *only* symmetry of the heterogeneous XY
Hamiltonian.

```python
from scpn_quantum_control.analysis.dla_parity_theorem import (
    verify_z2_parity,
    ParityTheoremResult,
)
```

---

## BKT Phase Analysis

### `bkt_analysis` — Core BKT Diagnostics

Fiedler eigenvalue, $T_{\mathrm{BKT}}$, $p_{H_1}$ prediction from coupling structure.

```python
from scpn_quantum_control.analysis.bkt_analysis import (
    bkt_scan,
    BKTResult,
)
```

### `bkt_universals` — 10 Candidate Expressions for $p_{H_1} = 0.72$

Systematic search for the analytical formula behind the universal $p_{H_1}$ value.

### `p_h1_derivation` — $A_{HP} \times \sqrt{2/\pi} = 0.717$

The derivation closing the $p_{H_1}$ gap to 0.5% accuracy.

### `phase_diagram` — $K_c$ vs $T_{\mathrm{eff}}$ Boundary

Full synchronization phase diagram in the coupling-temperature plane.

### `xxz_phase_diagram` — $K_c$ vs Anisotropy $\Delta$

Finite-size gap-minimum diagnostics in the $(K, \Delta)$ plane from XY-like
($\Delta=0$) to Heisenberg-like ($\Delta=1$) Hamiltonians.

```python
from scpn_quantum_control.analysis.xxz_phase_diagram import (
    anisotropy_phase_diagram,
    PhaseDiagramResult,
)
```

`anisotropy_phase_diagram(omega, K_topology, delta_range=None, k_range=None, *, max_dense_gib=None)` → `PhaseDiagramResult`
with: `delta_values`, `k_c_values`, `gap_min_values`, and `scans`.

---

## Open Quantum Systems

### `quantum_mpemba` — Quantum Mpemba Effect

Ordered states thermalize faster under amplitude damping — the quantum Mpemba effect
in synchronization dynamics.

```python
from scpn_quantum_control.analysis.quantum_mpemba import (
    mpemba_experiment,
    MpembaResult,
)
```

`mpemba_experiment(omega, K, K_base=1.0, gamma=0.1, t_max=5.0, n_steps=50)` →
`MpembaResult` with: `times`, `fidelity_ground`, `fidelity_plus` (|+⟩^N),
`mpemba_detected` (True if |+⟩ thermalizes faster).

### `lindblad_ness` — Non-Equilibrium Steady State

Lindblad NESS under amplitude damping: the long-time limit that retains
synchronization signatures.

```python
from scpn_quantum_control.analysis.lindblad_ness import (
    ness_vs_coupling,
    NESSResult,
)
```

`ness_vs_coupling(K, omega, gamma=0.1, K_base_range=None, n_K=15)` → `NESSResult`
with: `K_values`, `R_ness` (order parameter of NESS), `purity_ness`, `entropy_ness`.

---

## Reservoir Computing

### `qrc_phase_detector` — Exact QRC-Style Feature Map

The Kuramoto-XY Hamiltonian supplies exact dense ground-state Pauli features
for a ridge-regression classifier. This is a deterministic small-system
feature-map reference, not a scalable reservoir simulator.

```python
from scpn_quantum_control.analysis.qrc_phase_detector import (
    qrc_phase_detection,
    QRCPhaseResult,
)
```

`qrc_phase_detection(omega, K_topology, k_train, k_test, k_threshold, alpha=0.1, max_weight=2, *, max_dense_gib=None)` →
`QRCPhaseResult` with: `accuracy`, `n_train`, `n_test`, `n_features`,
`weights`, and `k_boundary_predicted`.

---

## Classical Simulations

### `monte_carlo_xy` — Classical XY Monte Carlo

Metropolis Monte Carlo for the classical XY model. Uses the Rust engine
(`scpn_quantum_engine`) when available; falls back to pure Python.

```python
from scpn_quantum_control.analysis.monte_carlo_xy import (
    mc_simulate,
    MCResult,
)
```

### `graph_topology_scan` — Coupling Graph Analysis

Network topology metrics (clustering, betweenness, modularity) of the $K_{nm}$ matrix.

### `koopman` — Koopman Linearisation

Koopman operator for the nonlinear Kuramoto dynamics — the BQP argument for quantum
advantage.

`build_koopman_generator_rust()` now routes to the optional
`scpn_quantum_engine.koopman_generator` kernel when that export is present and
falls back to the validated NumPy generator otherwise. Set `require_rust=True`
when a benchmark or release gate must prove that the native kernel, not the
fallback, served the dense generator.

### `hamiltonian_learning` — Recover $K_{nm}$ from Measurements

Learn the coupling matrix from measurement data using compressed sensing.

### `hamiltonian_self_consistency` — Self-Consistency Loop

Round-trip verification: $K_{nm}$ → Hamiltonian → ground state → correlators → $K_{nm}^{\mathrm{eff}}$.

```python
from scpn_quantum_control.analysis.hamiltonian_self_consistency import (
    self_consistency_check,
    correlator_shot_noise,
    SelfConsistencyResult,
)
```

### `enaqt` — Environment-Assisted Quantum Transport

Noise-enhanced transport optimisation — the Goldilocks zone where decoherence
*improves* energy transfer (relevant to FMO photosynthetic complex benchmarks).

`enaqt_scan(K, omega, gamma_range=None, t_evolve=1.0, n_steps=50, *, max_dense_gib=None)`
returns `ENAQTResult` with the optimal dephasing rate, coherent endpoint,
large-noise endpoint, and enhancement ratio. The implementation is a dense
small-system Lindblad diagnostic; `max_dense_gib` gates the Hamiltonian,
density matrix, and work buffers before allocation.

### `entanglement_enhanced_sync` — Entangled Initial-State Synchronization

`simulate_sync_trajectory(K, omega, state_type, t_max=2.0, n_steps=20, *, max_dense_gib=None)`
evolves product, Bell-pair, GHZ, or W initial states under the dense exact
Kuramoto-XY Hamiltonian and records the order-parameter trajectory. The dense
matrix exponential and statevector workspaces are budgeted before Hamiltonian
construction.

`compare_all_initial_states(K, omega, t_max=2.0, n_steps=20, *, max_dense_gib=None)`
forwards the same dense budget to every initial-state trajectory before
`entanglement_advantage(...)` compares final $R$ and convergence speed.
