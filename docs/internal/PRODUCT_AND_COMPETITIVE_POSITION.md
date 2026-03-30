# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Product Description & Competitive Position

**Last updated:** 2026-03-26

## What It Is

A Python library (+ Rust acceleration via PyO3) that compiles SCPN Kuramoto coupling matrices (K_nm) into Qiskit quantum circuits and validates them on IBM Quantum hardware (Heron r2, 156 qubits).

## Core Idea

The classical Kuramoto model of coupled oscillators maps exactly onto the XY spin Hamiltonian. This library implements, validates, and exploits that isomorphism on real quantum hardware.

## What It Does

1. **Bridges** — Takes a coupling matrix K_nm from any SCPN domain (neural, plasma, power grid, cardiac, EEG, Josephson arrays, FMO photosynthesis) and converts it to a quantum Hamiltonian (`SparsePauliOp`)
2. **Simulates** — Trotter-Suzuki evolution, VQE ground state, QAOA optimal control, VarQITE imaginary-time evolution, ADAPT-VQE, QSVT
3. **Mitigates errors** — ZNE (zero-noise extrapolation), PEC (probabilistic error cancellation), dynamical decoupling (XY4, X2, CPMG), symmetry verification, CPDR
4. **Runs on hardware** — IBM Heron r2 via Qiskit Runtime, plus trapped-ion adapter (IonQ/Quantinuum)
5. **Measures** — Identity robustness (VQE energy gap), coherence budgets, entanglement witnesses (CHSH), quantum fingerprints (K_nm-authenticated QKD), BKT phase transitions, gauge theory observables (Wilson loops, vortex density, CFT central charge), DLA structure, Monte Carlo XY sampling
6. **Controls** — QAOA-MPC for plasma disruption avoidance, quantum Petri nets, VQLS for Grad-Shafranov equilibrium, ITER disruption classifier

## By The Numbers

- 155 Python modules + 1 Rust crate (PyO3)
- 33 research modules (~4 novel constructions, ~8 first-application, rest standard diagnostics)
- 18 subpackages: analysis, applications, benchmarks, bridge, control, crypto, gauge, hardware, identity, l16, mitigation, pgbo, phase, qec, qsnn, ssgf, tcbo
- 2,813 tests, 98% coverage
- 18 example scripts, 13 Jupyter notebooks
- 0.05% VQE ground-state error on 4-qubit XY Hamiltonian (measured)
- 20 hardware experiments implemented, tested on AerSimulator, queued for IBM Heron r2 QPU

## Competitive Position

**No direct competitor exists.** The specific mapping (Kuramoto K_nm -> XY Hamiltonian -> NISQ circuits -> hardware validation) is novel. Nobody has published this vertical stack.

### Nearest Competitors

| Project | What it does | Gap vs scpn-quantum-control |
|---------|-------------|---------------------------|
| Qiskit Nature | Molecular Hamiltonians -> VQE | Chemistry-only, no Kuramoto/oscillator networks, no coupled-phase physics |
| PennyLane | Differentiable quantum circuits | Framework/toolkit, no domain physics, no oscillator mapping |
| Mitiq | Error mitigation toolkit | Mitigation only, no Hamiltonian construction, no domain bridge |
| QuTiP | Open quantum systems simulation | Classical simulation only, no NISQ hardware path, no circuit compilation |
| Cirq | Google quantum circuits | Low-level framework, no coupled-oscillator mapping, no domain physics |
| OpenFermion | Fermionic Hamiltonians | Chemistry/materials focus, no Kuramoto model, no synchronization physics |
| Qiskit Dynamics | Pulse-level simulation | Pulse engineering, not Hamiltonian physics, different abstraction layer |

### What Nobody Else Has

1. **The Kuramoto-XY isomorphism as a software pipeline** — proved, implemented, validated on hardware
2. **Domain-agnostic oscillator-to-circuit compilation** — same engine handles neural, plasma, power grid, cardiac, photosynthetic systems
3. **Identity continuity characterization** — VQE energy gap as quantitative measure of attractor basin robustness (novel concept)
4. **K_nm-authenticated QKD** — coupling topology as shared secret for quantum key distribution
5. **Cross-domain validation** — same Hamiltonian tested against 7 physical systems with domain-specific observables
6. **Error mitigation comparison** on Kuramoto dynamics — ZNE, PEC, DD, CPDR, symmetry verification all implemented and benchmarked on the same system

### Honest Limitations

- At current NISQ scale (4-16 qubits), **no quantum advantage** over classical ODE solvers for Kuramoto dynamics
- The quantum layer doesn't replace the classical engine — it **characterizes the boundaries** of what the classical engine can sustain (decoherence budgets, fidelity decay, identity robustness)
- Hardware campaign complete: 33 jobs on ibm_fez, 176K+ shots (as of v0.9.5)
- The SCPN model itself is unpublished — the quantum validation assumes the classical K_nm framework is correct
- QSP phase angles are placeholder (simplified, not full Haah 2018 optimization)
- QSNN training is O(samples * params * epochs) with no parallelism — research prototype, not production

### Strategic Value

The library's value is not in beating classical simulators today. It is in:

1. **Proving the mapping works on real hardware** — the 0.05% VQE error validates the theoretical bridge
2. **Establishing priority** — ~4 novel constructions + ~8 first-applications, documented and tested
3. **Building the measurement toolkit** — coherence budgets, identity robustness, entanglement structure are quantities that only quantum measurement can provide for quantum-mechanical systems
4. **Positioning for fault-tolerant era** — when logical qubits scale to N=20+, the same Hamiltonian pipeline becomes genuinely advantageous for real-time oscillator control (estimated crossover: 20-40 qubits)
5. **Cross-ecosystem integration** — bridges to sc-neurocore (SNN), scpn-phase-orchestrator (Kuramoto engine), scpn-fusion-core (plasma), scpn-control (classical control)
