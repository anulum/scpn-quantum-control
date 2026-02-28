# Roadmap

## Completed

### v0.1.0 (February 2026)
- Core modules: qsnn, phase, control, bridge, qec, hardware
- 88 tests, 6 examples, hardware validation on ibm_fez
- VQE 0.05% error, 12-point decoherence curve, 16-layer UPDE snapshot

### v0.2.0 (February 2026)
- ZNE + dynamical decoupling error mitigation
- Heron r2 noise model, Trotter error analysis, ansatz benchmarking
- 10 hardware experiments added (total 17)

### v0.3.0 (February 2026)
- README rewrite with Kuramoto-XY derivation
- Expanded mypy coverage (27 → 30 files)
- GitHub Pages docs (MkDocs Material)

### v0.4.0 (February 2026)
- 4 Jupyter notebooks, 14 property-based tests (hypothesis)
- Integration + regression test suites
- Test count: 208 → 254 → 408

### v0.5.0 (March 2026) — current
- 3 crypto hardware experiments: Bell test, correlator, QKD QBER
- 20 total experiments in registry, 411 tests

## v0.6.0 (Q1 2026)

- ZNE error mitigation on kuramoto_4osc (target: halve depth-85 error)
- VQE 8-qubit on hardware (56 CZ gates, within coherence)
- DD on full 16-layer UPDE
- Noise baseline repeat for calibration drift tracking

## v0.7.0 (Q2 2026)

- Kuramoto synchronization bifurcation on hardware (sweep K_base)
- Hardware ansatz comparison (Knm-informed vs TwoLocal vs EfficientSU2)
- Higher-order ZNE (5-point polynomial extrapolation)
- Per-gate depolarization rate extraction from qubit scaling data

## v0.8.0 (Q3 2026)

- PEC (probabilistic error cancellation) integration
- Trapped-ion backend support (IonQ / Quantinuum)
- Quantum advantage benchmark: quantum Kuramoto vs classical ODE at N=20+
- arXiv preprint submission

## Future

- Fault-tolerant UPDE simulation (surface code logical qubits)
- QSNN training loop on hardware (parameter-shift STDP)
- Quantum disruption classifier trained on ITER disruption database
- Integration with SCPN SSGF geometry engine
