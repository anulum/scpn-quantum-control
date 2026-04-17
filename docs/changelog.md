# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Changelog

# Changelog

All notable changes to scpn-quantum-control are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Full detailed changelog: [CHANGELOG.md](https://github.com/anulum/scpn-quantum-control/blob/main/CHANGELOG.md)

## [0.9.5] - 2026-03-29 / 2026-04-11

**Phase 1 IBM hardware confirmation of the DLA parity asymmetry +
five strategic tweaks (GUESS, DynQ, ICI, hypergeometric, FFI hardening)
+ repository hygiene (gitleaks + custom secret scanner).**

### Phase 1 IBM ibm_kingston campaign (2026-04-10)

- **First publishable hardware confirmation** of the DLA parity
  asymmetry on IBM ibm_kingston (Heron r2, 156 qubits). 348 circuits
  with up to 21 reps per (depth, sector) point at $n = 4$. Mean
  asymmetry $+10.8\,\%$ for Trotter depths $\ge 4$, peak $+17.48\,\%$
  at depth 6. Welch's t-test 7/8 depths at $p < 0.05$, Fisher's
  combined $p \ll 10^{-16}$. Consistent with the $4.5\text{–}9.6\,\%$
  apriori simulator prediction.
- Full statistical analysis script with error bars, Welch t-test, and
  matplotlib figures (`scripts/analyse_phase1_dla_parity.py`,
  `figures/phase1/`).
- 267-line short-paper draft for *Quantum Science and Technology* /
  *Physical Review Research* in `paper/`.

### Strategic tweaks from Gemini Deep Research report (2026-04-08)

- **GUESS symmetry-decay ZNE** (`mitigation/symmetry_decay.py`,
  `scpn_quantum_engine/src/symmetry_decay.rs`) — physics-informed
  zero-noise extrapolation using $\sum Z_i$ as the guide observable.
  Oliva del Moral *et al.*, arXiv:2603.13060. 20 multi-angle tests.
- **DynQ topology-agnostic qubit mapper**
  (`hardware/qubit_mapper.py`,
  `scpn_quantum_engine/src/community.rs`) — Louvain community
  detection on calibration-weighted QPU graphs. Liu *et al.*,
  arXiv:2601.19635. 17 multi-angle tests.
- **PMP / ICI pulse sequences + (α,β)-hypergeometric pulse shaping**
  (`phase/pulse_shaping.py`,
  `scpn_quantum_engine/src/pulse_shaping.rs`) — Liu *et al.* (2023)
  and Ventura Meinersen *et al.*, arXiv:2504.08031. Rust paths give
  44× hypergeometric envelope speedup and 1,665× ICI three-level
  evolution speedup, both verified to machine precision.
- **FFI boundary hardening** — every `#[pyfunction]` returns
  `PyResult<T>` and validates inputs via `validation.rs`. 16 unit
  tests.
- Long-form docs ≥ 567 lines: `docs/symmetry_decay_guess.md` (891) and
  `docs/dynq_qubit_mapping.md` (878).

### Repository hygiene (2026-04-10)

- **`tools/check_secrets.py`** — custom vault-pattern secret scanner
  with Shannon-entropy filter and keyword-based password detection.
- **gitleaks v8.21.2** pre-commit hook for generic secret detection.
- Incident report
  (`.coordination/incidents/INCIDENT_2026-04-10T2336_...`) for the
  prevented FTP-credentials leak that motivated the new scanners.
- Tests collected: 2,813 → **4,828** (97%+ coverage).
- Python modules 165 → **201**, subpackages 17 → **19**, Rust
  functions 22 → **36** across 20 source files.

### Earlier v0.9.5 work (2026-03-29 — 2026-04-07)

- Multi-Scale QEC, Free Energy Principle, Ψ-field lattice gauge theory.
- 10X Strange Loop co-evolution engine, DynamicCouplingEngine,
  TopologicalCouplingOptimizer, BiologicalSurfaceCode,
  LindbladSyncEngine (MCWF), StructuredAnsatz.
- 27 FIM experiment notebooks (NB14–47), 81 FIM tests, IBM hardware v2.
- 19 scientific discoveries + 6 honest negative results.

## [0.9.4] - 2026-03-29

**Coverage 98%, OpenSSF badge, 2715 tests.**

- 81 new tests, PennyLane/JAX/ripser mock guards
- OpenSSF Best Practices badge (100%)
- 3 benchmark API docs

## [0.9.3] - 2026-03-28

**Rust engine 15 functions, IBM 20/20 experiments, 16 figures.**

- 4 new Rust functions (11→15): lanczos, OTOC, dense Hamiltonian, batch Pauli
- IBM hardware: 22 jobs on ibm_fez, CHSH S=2.165, 16q UPDE
- JAX GPU backend, PyPI Rust wheel CI, BKT universality

## [0.9.2] - 2026-03-26

**Runner + experiment test coverage.**

- 38 runner + 22 experiment coverage tests
- requirements.txt pinned versions

## [0.9.1] - 2026-03-25

**33 research gems, 56 new modules, 9,772 lines.**

- 33 research modules (Rounds 1–8): witnesses, PH, OTOC, ADAPT-VQE, VarQITE, AVQDS, Floquet DTC, BKT, DLA parity
- 14 analysis modules, 9 phase modules, 6 hardware/bridge modules
- IBM hardware campaign: 9 jobs on ibm_fez

## [0.9.0] - 2026-03-22

**SCPN-native quantum control.**

- Analysis: shadow tomography, Koopman, DLA, entanglement spectrum, QFI
- Hardware: GPU offload, circuit cutting, trapped-ion, PennyLane/Cirq
- Identity: VQE attractor, coherence budget, entanglement witness
- Gauge: U(1) Wilson loops, vortex detection, CFT, universality
- 1789 tests, 100% coverage

## [0.8.0] - 2026-03-15

**Cross-repo bridges, crypto, QEC expansion.**

- Bridge: SSGF, SPN-to-circuit, SNN adapter, orchestrator
- Crypto: BB84, Bell tests, topology QKD
- QEC: fault-tolerant UPDE, surface code, error budget

## [0.7.0] - 2026-03-02

**Packaging, RNG hygiene, CI, exports hardening.**

- Fix crypto `__all__` (callable symbols, not module names)
- PEP 561 `py.typed` marker
- Seeded RNG in `QuantumDenseLayer`, `inf` → `nan` in PhaseVQE
- Named constants in percolation, pip cache in CI, dependency upper bounds
- 456 tests, 99%+ coverage

## [0.6.4] - 2026-03-01

**Docs/metadata hardening.**

- Fix stale test counts (411/424 → 442) in docs pages
- Fix header generator version string (v0.5.1 → v0.6.3)
- Complete README architecture tree (3 missing files)
- Add scpn-phase-orchestrator to Related Repositories
- Update SECURITY.md supported versions (0.6.x)
- Add pip ecosystem to dependabot
- Version bump 0.6.2 → 0.6.4

## [0.6.3] - 2026-03-01

**Coverage gate, mitigation API docs, notebook table.**

- Coverage gate in CI, mitigation API docs, notebook summary table, ruff fix
- Test count: 424 → 442

## [0.6.2] - 2026-03-01

**Notebook fixes + Knm heatmap figure.**

- Notebooks 01/03/04: `classical_kuramoto_ode` → `classical_kuramoto_reference`
- Notebook 03: ZNE scales [1,2,3,4,5] → [1,3,5,7,9] (odd required by gate_fold_circuit)
- Notebook 04: rewrite to 8-qubit Trotter + 16-layer classical (16-qubit statevector intractable on laptop)
- `figures/generate_knm_heatmap.py` + `figures/knm_heatmap.png` (16×16 K_nm coupling matrix)
- Knm heatmap figure in README with annotated calibration anchors
- All 4 notebooks executed with embedded outputs
- Remove misplaced docs/SESSION_LOG and docs/HANDOVER (duplicates of .coordination/)

## [0.6.1] - 2026-03-01

**mypy + Zenodo metadata fixes.**

- mypy errors in bridge module: remove FloatArray type alias (incompatible with Python 3.9), fix Path(None) in control_plasma_knm.py
- Zenodo metadata enriched (.zenodo.json, CITATION.cff)

## [0.6.0] - 2026-03-01

**Hardening + high-level API.**

- Input validation guards on all public API constructors — prevents div-by-zero in qlif, qsynapse, qstdp, qaoa_mpc, classical_kuramoto_reference; bounds-checks on bell_inequality_test, best_entanglement_path
- `PhaseVQE.solve()` now returns `exact_energy`, `energy_gap`, `relative_error_pct`, `n_params`
- Notebook 02 fixed to use enriched solve() dict
- 13 validation tests (424 total)

## [0.5.1] - 2026-03-01

**Version alignment.** Fixes `__version__` mismatch from v0.5.0 tag timing.

## [0.5.0] - 2026-03-01

**Quantum cryptography hardware experiments.**

- 3 new crypto experiments: Bell test (CHSH), ZZ correlator, QKD QBER
- `devetak_winter_rate()` key rate computation
- 20 experiments in registry, 411 tests

## [0.4.0] - 2026-02-28

**Docs, notebooks, and test depth.**

- GitHub Pages MkDocs Material site (7 pages, auto-deploy)
- 4 Jupyter notebooks: Kuramoto XY, VQE ground state, ZNE mitigation, UPDE-16
- 10 new hardware experiments (noise baseline, 8-osc ZNE, Trotter order-2, etc.)
- 14 property-based tests (hypothesis), 8 edge-case tests, 13 coverage-gap tests
- 4 integration tests, 7 regression tests
- Test count: 208 → 254

## [0.3.0] - 2026-02-28

**README rewrite and validation hardening.**

- Motivating abstract: SCPN→XY isomorphism, Kuramoto-to-Hamiltonian derivation, 4 figures
- mypy expanded to 30 source files (8 module paths), zero errors
- VALIDATION.md test count 88 → 199
- Paper 27 citation, examples/README.md walkthrough

## [0.2.0–0.2.7] - 2026-02-28

**Error mitigation, QEC fixes, classical references.**

- ZNE (unitary folding + Richardson), DD (XY4, X2) for idle qubits
- Heron r2 noise model factory
- MWPM decoder: 3 bug fixes (dual edges, seam-crossing, d=5 > d=3)
- Classical endianness fix verified against Statevector to 1e-6
- QAOA Ising encoding fix, Petri net multi-input AND gating
- Second-order Trotter, energy tracking
- 20 classical reference tests, parametrized quantum-vs-classical validation
- Test count: 88 → 208

## [0.1.0] - 2026-02-28

**Initial release.**

- qsnn/: Quantum LIF neuron, CRy synapse, parameter-shift STDP, dense layer
- phase/: Kuramoto XY solver, 16-layer Trotter UPDE, VQE ground state
- control/: QAOA-MPC, VQLS Grad-Shafranov, quantum Petri net, disruption classifier
- bridge/: Knm→Hamiltonian compiler, SPN→circuit, bitstream→rotation
- qec/: Toric surface code + MWPM decoder (Knm-weighted)
- hardware/: IBM Quantum runner for ibm_fez Heron r2
- 88 tests, 4 examples, 19 hardware result files
- Hardware: VQE 0.05% error, 12-point decoherence curve, 16-layer UPDE snapshot
