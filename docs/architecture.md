# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Architecture

# Architecture

## Package Statistics (v0.9.5)

| Metric | Count |
|--------|-------|
| Python modules | 201 |
| Rust crate | 1 (PyO3 0.25, **36 functions**, 20 source files including `validation.rs`, `symmetry_decay.rs`, `community.rs`, `pulse_shaping.rs`) |
| Tests | 4,828+ (97%+ coverage) |
| Lines of code | ~35,000 |
| Subpackages | 19 |
| Research gems | 35 (≈ 5 novel, ≈ 10 first-application, including GUESS symmetry-decay ZNE and DynQ topology-agnostic placement) |
| Examples | 21 |
| Notebooks | 47 |
| Doc pages | 50+ |

## Subpackage Dependency Graph

The 19 subpackages form a directed acyclic graph. `bridge/` is the foundation —
every other subpackage depends on it for Hamiltonian construction and data
conversion. `analysis/` is the largest consumer, using `phase/` for state
preparation and `bridge/` for Hamiltonian access.

```mermaid
graph TD
    bridge["bridge/ (12)\nK_nm → quantum objects"]
    phase["phase/ (26)\nTime evolution"]
    analysis["analysis/ (44)\nSync probes"]
    control["control/ (7)\nQuantum control"]
    qsnn["qsnn/ (6)\nQuantum SNN"]
    identity["identity/ (6)\nIdentity analysis"]
    hardware["hardware/ (17)\nBackends"]
    mitigation["mitigation/ (7)\nError mitigation"]
    qec["qec/ (5)\nError correction"]
    gauge["gauge/ (5)\nGauge theory"]
    apps["applications/ (11)\nBenchmarks"]
    crypto["crypto/ (6)\nQKD"]
    benchmarks["benchmarks/ (4)\nPerformance"]
    ssgf["ssgf/ (4)\nGeometry"]
    tcbo["tcbo/ (1)\nTCBO observer"]
    pgbo["pgbo/ (1)\nPGBO bridge"]
    l16["l16/ (1)\nLayer 16 director"]

    bridge --> phase
    bridge --> analysis
    bridge --> control
    bridge --> qsnn
    bridge --> identity
    bridge --> apps
    bridge --> crypto
    bridge --> ssgf
    phase --> analysis
    phase --> identity
    phase --> apps
    phase --> gauge
    analysis --> gauge
    hardware --> phase
    hardware --> apps
    mitigation --> hardware
    qec --> hardware
    benchmarks --> phase
    benchmarks --> hardware
    bridge --> tcbo
    bridge --> pgbo
    bridge --> l16
    phase --> tcbo
    analysis --> pgbo

    style bridge fill:#6929C4,color:#fff
    style analysis fill:#d4a017,color:#000
    style phase fill:#6929C4,color:#fff
    style hardware fill:#2ecc71,color:#000
```

## Hardware Execution Pipeline

Circuit depth after transpilation determines which decoherence regime applies.
The pipeline is the same for all experiments — only the circuit construction
step differs.

```mermaid
graph LR
    subgraph "Classical Side"
        A["K_nm matrix"] --> B["knm_to_hamiltonian()"]
        B --> C["Build QuantumCircuit\n(Trotter / VQE / QAOA)"]
        C --> D["Transpile to\nnative gates"]
    end

    subgraph "Quantum Side"
        D --> E["Submit via\nSamplerV2"]
        E --> F["Parse bit-string\ncounts"]
    end

    subgraph "Analysis"
        F --> G["⟨X⟩, ⟨Y⟩, ⟨Z⟩"]
        G --> H["Order parameter\nR(t)"]
        G --> I["Witnesses,\nQFI, PH, ..."]
    end

    style A fill:#6929C4,color:#fff
    style H fill:#2ecc71,color:#000
    style I fill:#d4a017,color:#000
```

**Decoherence regimes on Heron r2:**

| Transpiled depth | Regime | Accuracy | Strategy |
|:----------------:|--------|----------|----------|
| < 150 | Near-ideal | < 10% error | Publish directly |
| 150–400 | Mitigable | 10–30% error | ZNE + Z₂ post-selection |
| > 400 | Noise-dominated | > 30% error | Qualitative only |

## Module Dependency Graph (Full Detail)

```
bridge/                                    ← Foundation: K_nm → quantum objects
├── knm_hamiltonian.py                       Canonical K_nm data, XY + XXZ Hamiltonians, ansatz
├── snn_adapter.py                           sc-neurocore ArcaneNeuron bridge (optional)
├── snn_backward.py                          Parameter-shift gradient through quantum layer
├── ssgf_adapter.py                          SSGF geometry engine bridge (optional)
├── ssgf_w_adapter.py                        Correlator-weighted geometry W update
├── control_plasma_knm.py                    scpn-control plasma K_nm bridge (optional)
├── phase_artifact.py                        Shared UPDE phase artifact schema
├── orchestrator_adapter.py                  Phase-orchestrator payload adapter
├── orchestrator_feedback.py                 Advance/hold/rollback from quantum state
├── sc_to_quantum.py                         Angle/probability conversion
└── spn_to_qcircuit.py                       SPN token → circuit amplitude

analysis/                                  ← 41 modules: probes of the sync transition
├── sync_witness.py                          ★ Synchronization witnesses (Gem 1)
├── sync_entanglement_witness.py             ★ R as entanglement witness (Gem 12)
├── quantum_persistent_homology.py           ★ Full PH pipeline from counts (Gem 5)
├── persistent_homology.py                     Classical PH utilities
├── h1_persistence.py                          Vortex density at BKT
├── entanglement_enhanced_sync.py            ★ Entanglement lowers K_c (Gem 7)
├── hamiltonian_self_consistency.py           ★ K_nm round-trip verification (Gem 10)
├── hamiltonian_learning.py                    Recover K_nm from measurements
├── dynamical_lie_algebra.py                 ★ DLA dimension = 2^(2N-1)-2 (Gem 11)
├── dla_parity_theorem.py                    ★ Z₂ parity proof (Gem 14)
├── qfi_criticality.py                       ★ QFI metrological sweet spot (Gem 15)
├── qfi.py                                     Full QFI matrix computation
├── entanglement_percolation.py              ★ Percolation = sync threshold (Gem 16)
├── qrc_phase_detector.py                    ★ Self-probing reservoir (Gem 17)
├── critical_concordance.py                  ★ Multi-probe K_c agreement (Gem 19)
├── berry_fidelity.py                        ★ Berry phase / χ_F at BKT (Gem 20)
├── quantum_mpemba.py                        ★ Quantum Mpemba effect (Gem 21)
├── lindblad_ness.py                         ★ Lindblad NESS (Gem 22)
├── adiabatic_gap.py                         ★ Adiabatic preparation hardness (Gem 23)
├── pairing_correlator.py                    ★ Richardson pairing (Gem 25)
├── xxz_phase_diagram.py                     ★ K_c vs Δ crossover (Gem 26)
├── spectral_form_factor.py                  ★ SFF + level statistics (Gem 27)
├── loschmidt_echo.py                        ★ Loschmidt echo / DQPT (Gem 28)
├── entanglement_entropy.py                  ★ Half-chain entropy + Schmidt gap (Gem 29-30)
├── entanglement_spectrum.py                   Full entanglement spectrum + CFT c
├── krylov_complexity.py                     ★ Krylov complexity (Gem 31, highest novelty)
├── magic_nonstabilizerness.py               ★ Stabilizer Rényi entropy (Gem 32)
├── finite_size_scaling.py                   ★ BKT logarithmic corrections (Gem 33)
├── otoc.py                                    Core OTOC computation
├── otoc_sync_probe.py                       ★ OTOC as sync probe (Gem 9)
├── quantum_speed_limit.py                   ★ QSL for BKT sync (Gem 13)
├── quantum_phi.py                             IIT Φ from density matrix
├── shadow_tomography.py                       Classical shadow estimation
├── bkt_analysis.py                            Core BKT diagnostics
├── bkt_universals.py                          10 candidate expressions for p_H1
├── p_h1_derivation.py                         A_HP × √(2/π) = 0.717
├── phase_diagram.py                           K_c vs T_eff boundary
├── graph_topology_scan.py                     Coupling graph metrics
├── koopman.py                                 Koopman linearisation (BQP argument)
├── monte_carlo_xy.py                          Classical XY MC (Rust-accelerated)
├── vortex_binding.py                          Kosterlitz RG flow
└── enaqt.py                                   Environment-assisted quantum transport

phase/                                     ← 14 modules: time evolution + variational
├── xy_kuramoto.py                             Trotterised XY solver
├── trotter_upde.py                            Full 16-layer UPDE solver
├── trotter_error.py                           Trotter error analysis
├── phase_vqe.py                               Variational eigensolver
├── adapt_vqe.py                             ★ Gradient-driven operator selection
├── varqite.py                                 Imaginary time evolution
├── avqds.py                                   Adaptive variational dynamics
├── qsvt_evolution.py                          QSVT resource estimation (260× speedup)
├── adiabatic_preparation.py                   Adiabatic ground state prep
├── cross_domain_transfer.py                 ★ VQE parameter warm-starting (Gem 8)
├── floquet_kuramoto.py                      ★ Discrete time crystal (Gem 18)
├── coupling_topology_ansatz.py              ★ K_nm-informed ansatz (Gem 4)
├── ansatz_methodology.py                      Ansatz strategy analysis
└── ansatz_bench.py                            Ansatz benchmarking

control/                                   ← Quantum control + classification
├── qaoa_mpc.py                                QAOA model-predictive control
├── vqls_gs.py                                 VQLS Grad-Shafranov solver
├── qpetri.py                                  Quantum Petri nets
├── q_disruption.py                            Disruption classifier
└── q_disruption_iter.py                       ITER 11-feature + fusion-core adapter

qsnn/                                      ← Quantum spiking neural networks
├── qlif.py                                    Quantum LIF neuron
├── qsynapse.py                                Quantum synapse (CRy)
├── qstdp.py                                   Quantum STDP learning
├── qlayer.py                                  Dense quantum layer
└── training.py                                Parameter-shift trainer

identity/                                  ← Identity continuity analysis
├── ground_state.py                            VQE attractor basin
├── coherence_budget.py                        Heron r2 decoherence budget
├── entanglement_witness.py                    CHSH S-parameter
├── identity_key.py                            Spectral fingerprint + HMAC
├── robustness.py                              Adiabatic robustness certificate
└── binding_spec.py                            6-layer topology + orchestrator mapping

mitigation/                                ← Error mitigation
├── zne.py                                     Zero-noise extrapolation
├── pec.py                                     Probabilistic error cancellation
├── dd.py                                      Dynamical decoupling
└── symmetry_verification.py                 ★ Z₂ parity post-selection (Gem 2)

gauge/                                     ← U(1) gauge theory probes
├── wilson_loop.py                             Wilson loop measurement
├── vortex_detector.py                         BKT vortex density
├── cft_analysis.py                            CFT central charge extraction
├── universality.py                            BKT universality class check
└── confinement.py                             String tension + confinement

ssgf/                                      ← SSGF quantum integration
├── quantum_gradient.py                        dC_quantum/dz via finite differences
├── quantum_costs.py                           C_micro, C4_tcbo, C_pgbo
├── quantum_outer_cycle.py                     Variational z descent
└── quantum_spectral.py                        Fiedler via QPE resource estimation

applications/                              ← Physical system benchmarks
├── fmo_benchmark.py                           FMO photosynthetic complex (7 chromophores)
├── power_grid.py                              IEEE 5-bus power grid
├── josephson_array.py                         JJA/transmon self-simulation
├── eeg_benchmark.py                           8-channel alpha-band PLV
├── iter_benchmark.py                          8 MHD mode coupling
├── cross_domain.py                            5-system benchmark summary
├── quantum_kernel.py                          K_nm-informed classification
├── quantum_reservoir.py                       Pauli feature extraction
├── disruption_classifier.py                   Plasma stability classification
└── quantum_evs.py                             Quantum-enhanced EVS for CCW

benchmarks/                                ← Performance baselines
├── quantum_advantage.py                       Classical vs quantum scaling
├── mps_baseline.py                            MPS bond dimension + advantage threshold
├── gpu_baseline.py                            A100 FLOPS + GPU vs QPU crossover
└── appqsim_protocol.py                        Application-oriented fidelity metrics

qec/                                       ← Quantum error correction
├── control_qec.py                             Toric code + MWPM decoder
├── fault_tolerant.py                          RepetitionCodeUPDE
├── surface_code_upde.py                       Surface code resource estimation
└── error_budget.py                            3-channel Trotter+gate+logical allocation

hardware/                                  ← Backend + experiments
├── runner.py                                  IBM Quantum job submission
├── experiments.py                             20 pre-built experiments
├── trapped_ion.py                             Trapped-ion noise model
├── classical.py                               Rust-accelerated Kuramoto reference
├── gpu_accel.py                               CuPy GPU offload (opt-in)
├── circuit_cutting.py                         Partition optimiser for 32-64 oscillators
├── qasm_export.py                             OpenQASM 3.0 export
├── qcvv.py                                    State fidelity + mirror circuits + XEB
└── cirq_adapter.py                            Cirq backend adapter (optional)

crypto/                                    ← Quantum key distribution
├── qkd_bb84.py                                BB84 protocol
├── bell_test.py                               CHSH test
├── topology_auth.py                           Topology-authenticated QKD
└── percolation.py                             Key rate percolation

tcbo/                                      ← TCBO quantum observer
└── quantum_observer.py                        p_h1, TEE, string order, Betti proxies

pgbo/                                      ← PGBO quantum bridge
└── quantum_bridge.py                          Quantum geometric tensor, Berry curvature

l16/                                       ← Layer 16 quantum director
└── quantum_director.py                        Loschmidt echo, stability score

scpn_quantum_engine/                       ← Rust crate (PyO3 0.25, rayon parallel)
└── src/lib.rs                                 15 functions: kuramoto_euler, kuramoto_trajectory,
                                               order_parameter, build_knm, pec_coefficients,
                                               pec_sample_parallel, dla_dimension, mc_xy_simulate,
                                               state_order_param_sparse, expectation_pauli_fast,
                                               brute_mpc, lanczos_b_coefficients,
                                               otoc_from_eigendecomp,
                                               build_xy_hamiltonian_dense,
                                               all_xy_expectations
```

★ marks modules from the 33 Research Gems (Rounds 1-8, March 2026).

## Classical-to-Quantum Mapping

Each module maps a classical SCPN computation to its quantum analog:

| Classical (SCPN) | Quantum (this repo) | Mapping |
|-------------------|---------------------|---------|
| Stochastic LIF membrane potential | Ry(theta) rotation angle | theta = pi * (v - v_rest) / (v_threshold - v_rest) |
| Bitstream AND-gate synapse | CRy(theta_w) controlled rotation | P(out) = P(pre) * sin^2(theta_w/2) |
| STDP correlation learning | Parameter-shift gradient rule | dw = lr * pre * d<Z>/d(theta) |
| Kuramoto ODE (dtheta/dt) | XY Hamiltonian Trotter evolution | H = -K_ij(XX + YY) - omega_i Z_i |
| 16-layer UPDE coupling | 16-qubit spin chain | Knm -> J_ij entangling gates |
| MPC quadratic cost | QAOA Ising Hamiltonian | ||state - target||^2 -> ZZ + Z terms |
| Grad-Shafranov PDE | VQLS linear system | Laplacian A, source b -> A|x> ~ |b> |
| SPN token probability | Qubit amplitude | p -> amplitude encoding |
| Disruption feature vector | Amplitude-encoded state | 11-D -> 16-D zero-padded |

## Cross-Repository Integration

This package is one node in a five-repository ecosystem. Each bridge adapter
converts between the data representations of the two repositories it connects.

```mermaid
graph LR
    SC["sc-neurocore\n(SNN engine)"] -->|"snn_adapter\nmembrane → Ry angle"| QC["scpn-quantum-\ncontrol"]
    SSGF["SSGF geometry\nengine"] -->|"ssgf_adapter\nW → H_XY"| QC
    PO["scpn-phase-\norchestrator"] <-->|"orchestrator_adapter\npayload ↔ artifact"| QC
    FC["scpn-fusion-core\n+ scpn-control"] -->|"control_plasma_knm\nplasma K_nm"| QC
    QC -->|"phase_artifact\nUPDE state"| PO

    style QC fill:#6929C4,color:#fff
    style SC fill:#2ecc71,color:#000
    style PO fill:#d4a017,color:#000
    style FC fill:#e67e22,color:#000
```

| Bridge | Source repo | Data in | Data out |
|--------|-----------|---------|----------|
| `snn_adapter` | sc-neurocore | ArcaneNeuron membrane $v$ | $R_y(\theta)$ angle |
| `ssgf_adapter` | SSGF engine | Geometry matrix $W$ | XY Hamiltonian |
| `orchestrator_adapter` | scpn-phase-orchestrator | State payload (regime, phases) | UPDEPhaseArtifact |
| `orchestrator_feedback` | scpn-phase-orchestrator | Quantum $R$, fidelity | Advance/hold/rollback |
| `control_plasma_knm` | scpn-control | Plasma-native $K_{nm}$ | Standard $K_{nm}$ array |
| `snn_backward` | sc-neurocore | Loss gradient | Parameter-shift $\nabla\theta$ |

## Data Flow: Knm → Hamiltonian → Circuit → Measurement → R

```python
from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16, build_knm_paper27, knm_to_hamiltonian,
)
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

K = build_knm_paper27()
omega = OMEGA_N_16[:4]
solver = QuantumKuramotoSolver(4, K[:4, :4], omega)
result = solver.run(t_max=0.4, dt=0.1)
# result["R_trajectory"] -> [0.80, 0.78, 0.76, 0.73]
```
