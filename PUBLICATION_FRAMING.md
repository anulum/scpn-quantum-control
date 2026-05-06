# Publication Framing Guide

## The Problem

The SCPN framework uses terms like "consciousness gate", "identity binding",
and "cybernetic closure" that will trigger immediate rejection at mainstream
physics venues. This document defines how to frame the work for publication.

## Rule: Physics Language Only

| Internal Term | Publication Term |
|--------------|-----------------|
| Consciousness gate (p_h1) | Topological coherence threshold |
| Identity binding | Attractor basin of coupling topology |
| Cybernetic closure (L16) | Lyapunov stability monitoring |
| TCBO consciousness observer | Persistent homology of phase configuration |
| CCW (Conscious Coherence Waves) | Emergent value signatures |
| "Consciousness IS the BKT transition" | DO NOT WRITE THIS |

## What We Claim (Publishable)

1. K_nm-informed VQE ansatz converges 6x faster than generic (hardware data)
2. The XY model on a complete graph exhibits BKT-class critical behaviour
3. The decoherence budget at 16 qubits on Heron r2 is characterised
4. QSVT provides 260x query reduction vs first-order Trotter
5. Persistent homology threshold p_h1 = 0.72 is empirical (NOT derived from BKT on this graph)

## What We Do NOT Claim

1. Quantum advantage at 16 qubits
2. The K_nm values model any specific physical system
3. Consciousness — the word does not appear in the manuscript
4. p_h1 is a BKT universal (Monte Carlo falsified this on the K_nm graph)

## Scientific Claim Boundaries

- Biological and clinical datasets are treated as classical complex-network
  signals. The publication framing is quantum-inspired Hamiltonian,
  tensor-network, topological, or DLA analysis of those classical signals; it
  is not a claim of quantum biological causation, diagnosis, treatment, or
  cure.
- The Kuramoto-XY construction is a linear quantum analogue or embedding of
  oscillator-network structure. It is not direct Trotterisation of the
  nonlinear classical Kuramoto ODE unless an explicit Koopman, Carleman, or
  equivalent linear embedding is stated and validated.
- Python feedback loops are orchestration, simulation, and across-shot update
  loops. Hardware-timescale feedback requires provider-native dynamic
  circuits, OpenQASM 3 control flow, pulse-level control, FPGA logic, or an
  equivalent vendor controller.
- Quantum-advantage or crossover language is conditional on the named
  baseline. Broad advantage claims require state-of-the-art tensor-network or
  GPU baselines and explicit accounting for data loading and state preparation.
- Git stores small reproducibility artefacts: scripts, manifests, checksums,
  summaries, and selected raw-count JSON files. Heavy generated artefacts
  should be archived through Zenodo, DVC-style pointers, or an experiment
  tracker rather than accumulated indefinitely in Git history.
- Notebooks are demonstrations, provenance records, and paper companions.
  Reusable algorithms belong under `src/scpn_quantum_control/`, especially
  `applications/`, with tests and command-line regeneration paths.

## Reviewer Objections (Pre-Addressed)

| Objection | Response |
|-----------|----------|
| "No quantum advantage" | Correct. This is a NISQ benchmarking study. We demonstrate methodology, not speedup. |
| "16 qubits is trivial" | Correct. We provide honest resource estimates for N where advantage begins (~40+). |
| "The coupling matrix is arbitrary" | We present 5 physical system comparisons showing moderate correlations. The K_nm-informed ansatz technique generalises to any structured Hamiltonian. |
| "Consciousness is unfalsifiable" | The word does not appear. We measure topological observables. |
| "A_HP coincidence was falsified" | Yes. We report this honestly. MC on the K_nm graph gives A_HP=1.21, not 0.8983. |

## Target Venues (In Order)

1. **Quantum Science and Technology** — NISQ simulation benchmark papers welcome
2. **Physical Review Research** — broad scope, accepts negative results
3. **New Journal of Physics** — open access, accepts methodology papers
4. **PRX Quantum** — higher bar, needs clearer advantage narrative
