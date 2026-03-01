# Changelog

All notable changes to scpn-quantum-control are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Full detailed changelog: [CHANGELOG.md](https://github.com/anulum/scpn-quantum-control/blob/main/CHANGELOG.md)

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
