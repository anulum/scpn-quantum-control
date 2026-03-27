# Analysis API Reference

*41 modules for probing quantum synchronization transitions, entanglement structure,
topological invariants, and computational complexity of the Kuramoto-XY Hamiltonian.*

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

`WitnessResult` fields: `witness_name`, `expectation_value` (negative = synchronized),
`threshold`, `is_synchronized`, `raw_observable`, `n_qubits`.

Full theory and examples: [Research Gems — Gem 1](research_gems.md#gem-1-synchronization-witness-operators).

### `sync_entanglement_witness` — R as Entanglement Witness

The Kuramoto order parameter $R$ reinterpreted as an entanglement witness. For separable
states, $R \leq R_{\mathrm{sep}}$. Exceeding the separable bound certifies entanglement.

```python
from scpn_quantum_control.analysis.sync_entanglement_witness import (
    R_from_statevector,
    R_separable_bound,
    R_separable_bound_at_energy,
    r_witness_from_statevector,
    SyncEntanglementResult,
)
```

| Function | Description |
|----------|-------------|
| `R_from_statevector(sv)` | Compute $R = \|(1/N)\sum_i(\langle X_i\rangle + i\langle Y_i\rangle)\|$ |
| `R_separable_bound(n_qubits)` | Maximum $R$ achievable by any separable state (= 1.0) |
| `R_separable_bound_at_energy(K, omega, target_energy, n_samples, seed)` | Max $R$ over product states at given energy |
| `r_witness_from_statevector(sv, K, omega)` | Full witness evaluation: $R$, separable bound, entanglement certified? |

### `critical_concordance` — Multi-Probe $K_c$ Agreement

Scans coupling strength and evaluates all probes simultaneously to verify they converge
on the same critical coupling.

```python
from scpn_quantum_control.analysis.critical_concordance import (
    critical_concordance,
    ConcordanceResult,
)
```

`critical_concordance(K, omega, n_K=20, K_max=3.0)` returns `ConcordanceResult` with
fields: `K_values`, `R_values`, `qfi_values`, `gap_values`, `fiedler_values`,
`berry_connection`.

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

### `entanglement_percolation` — Sync Threshold as Percolation

Tests whether entanglement percolation (Fiedler $\lambda_2 > 0$ of the concurrence
graph) coincides with synchronization $K_c$.

```python
from scpn_quantum_control.analysis.entanglement_percolation import (
    entanglement_percolation_scan,
    PercolationResult,
)
```

`entanglement_percolation_scan(K, omega, K_base_range=None, n_K=20)` →
`PercolationResult` with: `K_values`, `R_values`, `fiedler_values`,
`concurrence_matrices`, `percolation_K`.

### `berry_fidelity` — Berry Phase and Fidelity Susceptibility

Gauge-invariant fidelity susceptibility $\chi_F$ peaks at BKT transition.

```python
from scpn_quantum_control.analysis.berry_fidelity import (
    berry_fidelity_scan,
    BerryFidelityResult,
)
```

`berry_fidelity_scan(K, omega, K_base_range=None, n_K=20, dK=0.01)` →
`BerryFidelityResult` with: `K_values`, `fidelity_values`,
`fidelity_susceptibility`, `berry_connection`.

### `finite_size_scaling` — BKT Finite-Size Extraction

Fits $K_c(N) = K_c(\infty) + a/(\ln N)^2$ to extract thermodynamic-limit $K_c$.

```python
from scpn_quantum_control.analysis.finite_size_scaling import (
    finite_size_scaling,
    FSSResult,
)
```

`finite_size_scaling(omega_full, K_base_fn, system_sizes=None, n_K=15)` → `FSSResult`
with: `system_sizes`, `Kc_values`, `Kc_inf` (extrapolated), `fit_a`, `fit_residual`.

### `adiabatic_gap` — Adiabatic Preparation Hardness

Computes the minimum spectral gap along the adiabatic path to the BKT ground state.

```python
from scpn_quantum_control.analysis.adiabatic_gap import (
    adiabatic_gap_scan,
    AdiabaticGapResult,
)
```

`adiabatic_gap_scan(K, omega, n_points=50, s_range=(0.0, 1.0))` →
`AdiabaticGapResult` with: `s_values`, `gaps`, `min_gap`, `min_gap_s`,
`adiabatic_time_estimate`.

---

## Entanglement and Correlations

### `entanglement_entropy` — Half-Chain Entropy and Schmidt Gap

Entanglement entropy and Schmidt gap across the synchronization transition. At BKT
criticality, entropy follows CFT scaling $S \sim (c/3)\ln L$ with $c = 1$.

```python
from scpn_quantum_control.analysis.entanglement_entropy import (
    entanglement_entropy_scan,
    EntanglementEntropyResult,
)
```

`entanglement_entropy_scan(K, omega, K_base_range=None, n_K=20)` →
`EntanglementEntropyResult` with: `K_values`, `entropy_values`, `schmidt_gaps`,
`schmidt_spectra`.

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
spin-raising/lowering correlators. Strong pairing = synchronized phase.

```python
from scpn_quantum_control.analysis.pairing_correlator import (
    pairing_correlator_scan,
    PairingResult,
)
```

`pairing_correlator_scan(K, omega, delta=0.0, K_base_range=None, n_K=15)` →
`PairingResult` with: `K_values`, `mean_pairing`, `pairing_matrices`.

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
    krylov_complexity_scan,
    KrylovResult,
)
```

`krylov_complexity_scan(K, omega, operator=None, K_base_range=None, n_K=15, t_max=2.0)`
→ `KrylovResult` with: `K_values`, `lanczos_b`, `complexity_values`,
`peak_complexity_K`.

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

### `magic_nonstabilizerness` — Stabilizer Rényi Entropy

Magic $M_2 = -\log_2(\sum_P \langle P\rangle^4 / 2^N)$ peaks at $K_c$ — the critical
state is maximally non-classical.

```python
from scpn_quantum_control.analysis.magic_nonstabilizerness import (
    magic_scan,
    MagicResult,
)
```

`magic_scan(K, omega, K_base_range=None, n_K=15)` → `MagicResult` with:
`K_values`, `magic_values`, `peak_magic_K`.

### `quantum_phi` — Integrated Information (IIT)

Tononi's Φ (integrated information) from the quantum density matrix.

```python
from scpn_quantum_control.analysis.quantum_phi import (
    compute_phi,
    PhiResult,
)
```

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

Phase boundary in the $(K, \Delta)$ plane from XY ($\Delta=0$) to Heisenberg ($\Delta=1$).

```python
from scpn_quantum_control.analysis.xxz_phase_diagram import (
    xxz_phase_scan,
    XXZPhaseResult,
)
```

`xxz_phase_scan(K, omega, delta_range=None, K_base_range=None)` → `XXZPhaseResult`
with: `delta_values`, `K_values`, `R_matrix`, `Kc_vs_delta`.

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

### `qrc_phase_detector` — Self-Probing QRC

The Kuramoto-XY system uses its own Pauli observables as features for a ridge
regression classifier. The reservoir IS the system under study.

```python
from scpn_quantum_control.analysis.qrc_phase_detector import (
    qrc_phase_scan,
    QRCResult,
)
```

`qrc_phase_scan(K, omega, K_base_range=None, n_K=30, n_train_frac=0.7)` →
`QRCResult` with: `K_values`, `predictions`, `accuracy`, `feature_importance`.

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
