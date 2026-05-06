# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control - QPU System Capability Dossier

# QPU System Capability Dossier

This dossier states what data the SCPN quantum-control stack needs,
what computations it can route to quantum hardware, how the results are
used, and what the system can realistically deliver. It is written for
engineering execution and future resource-allocation proposals.

## System objective

The objective is a provider-neutral quantum compute fabric for
oscillator-network problems:

```text
real-world source -> source artifact -> oscillator artifact
                  -> QPU/CPU/GPU/Rust compute request
                  -> result artifact -> memory / controller / interface
```

The QPU is one compute unit in this fabric. It is selected when the
kernel benefits from quantum hardware, when hardware-measured evidence
is required, or when a provider-specific modality is the natural match
for the problem.

## Data inventory

The system needs six classes of data.

| Data class | Examples | Owner | Use |
|------------|----------|-------|-----|
| Source data | EEG windows, connectomes, plasma mode measurements, grid frequency traces, biological networks, recorded campaign inputs. | SC-NeuroCore or domain source repository. | Defines the real-world state to be compiled. |
| Oscillator artifact | `K_nm`, `omega`, optional `theta0`, layer labels, provenance, hashes. | Phase Orchestrator emits; Quantum Control validates. | Canonical input for QPU/CPU/GPU kernels. |
| Stream deltas | Changed phases, frequencies, couplings, control settings, event times, deadlines. | Live bridge and runtime broker. | Batch or near-real-time updates without recompiling a whole experiment from scratch. Hardware-timescale feedback needs provider-native dynamic circuits, pulse control, FPGA logic, or equivalent controllers. |
| QPU node data | Backend topology, native gates, qubit count, queue status, calibration, cost units, availability. | Provider adapter. | Routing, cost control, and compilation. |
| Compute request | Kernel, backend policy, budget, circuit limits, mitigation plan, idempotency key. | Quantum Control scheduler. | Auditable admission control before compute. |
| Result artifact | Job ids, counts, observable values, confidence, mitigation metadata, result hashes. | Quantum Control runtime. | Downstream memory, analysis, publication, and control. |

No campaign path may replace missing source data with unlabelled random
matrices. Synthetic data is allowed only for smoke tests and must be
labelled as synthetic, simulation, or fixture.

## Compute possibilities

The stack should support these compute classes.

### Synchronisation diagnostics

Input:

- oscillator artifact
- backend policy
- sync-witness kernel

Computation:

- compile Kuramoto-XY or native analog representation
- measure bitstring marginals and phase-order proxies
- return synchronisation order and uncertainty metadata

Use:

- detect coherent, incoherent, fragmented, and near-transition regimes
- compare source states over time
- feed controller decisions

Possible delivery:

- result artifact with `sync_order`, marginals, confidence, backend,
  mitigation, and source hash

### DLA and symmetry diagnostics

Input:

- oscillator artifact or compiled circuit
- DLA/parity kernel request

Computation:

- measure parity-sector robustness or related symmetry witnesses
- compare odd/even or other declared dynamical subspaces

Use:

- identify subspaces that survive noise or perturbation
- support hardware-facing quantum-control claims
- guide error-mitigation and ansatz design

Possible delivery:

- DLA witness artifact with measured/proxy classification, confidence,
  and calibration context

### Scrambling and instability propagation

Input:

- Hamiltonian/circuit schedule
- OTOC or scrambling kernel request

Computation:

- estimate short-time information spreading
- compare coupling regimes or topology changes

Use:

- early-warning indicators for cascading failures
- instability propagation diagnostics
- memory-loss or disturbance-spread analysis

Possible delivery:

- scrambling curve or short-time proxy with explicit shot/error metadata

### Quantum feature generation

Input:

- source or oscillator artifact
- feature-map kernel
- downstream model contract

Computation:

- encode state into quantum feature map
- sample hardware/simulator features or kernel entries

Use:

- forecasting and classification for high-dimensional dynamical systems
- comparison against classical feature baselines

Possible delivery:

- feature artifact for downstream ML with provider, seed/job id,
  mitigation, and feature-version metadata

### Variational and control search

Input:

- oscillator artifact
- objective function
- budget and backend policy

Computation:

- simulator-first search
- optional hardware micro-probes
- candidate control/action ranking

Use:

- propose interventions for plasma, grid, neural, or engineered systems
- tune ansatze or mitigation settings

Possible delivery:

- ranked control candidates with cost trace, objective trace, and
  replayable request/result hashes

### Analog Hamiltonian simulation

Input:

- geometry-aware oscillator artifact
- neutral-atom or analog provider descriptor

Computation:

- map graph geometry and interaction strengths to native analog controls
- run analog evolution or emulator preflight

Use:

- direct oscillator-network experiments where digital Trotterisation is
  not the natural representation
- Rydberg/neutral-atom graph dynamics

Possible delivery:

- analog result artifact with register geometry, pulse/control schedule,
  measured observables, and emulator comparison

### Optimisation and scheduling

Input:

- intervention/routing/scheduling objective
- QUBO, Ising, or constrained model

Computation:

- route to annealer, hybrid solver, CPU/Rust optimiser, or simulator

Use:

- choose interventions under constraints
- allocate QPU jobs
- optimise graph partitions, schedules, or resource routing

Possible delivery:

- ranked solution artifact with solver metadata and constraint
  satisfaction report

### Calibration and backend selection

Input:

- provider node descriptor
- calibration probe request

Computation:

- run small reference circuits or provider syntax/emulator checks
- compare measured error indicators with declared thresholds

Use:

- decide whether a backend is fit for a given kernel
- prevent expensive runs on unsuitable hardware

Possible delivery:

- calibration-readiness artifact: accepted, degraded, rejected, or
  simulator-only

## Application mapping

| Real-world domain | Source artifact | QPU kernel | Expected practical output |
|-------------------|-----------------|------------|---------------------------|
| Neuroscience / connectomes | Connectome graph, EEG phase windows, neural stream deltas. | Synchronisation, feature map, scrambling, DLA diagnostics. | Regime classification, coherence/fragmentation tracking, tipping indicators, replayable state comparison. |
| Plasma / fusion | Mode-coupling graph, frequency traces, control actuator metadata. | Synchronisation, scrambling, variational control, analog simulation. | Disruption precursors, instability propagation ranking, candidate control settings. |
| Power grid | Grid node phases, admittance-like couplings, frequency deviations. | Synchronisation, tipping diagnostics, optimisation. | Loss-of-synchrony warning, cascade-risk ranking, intervention prioritisation. |
| Biological networks | Protein, cellular, cardiac, or photosynthetic coupling graphs. | Synchronisation, analog simulation, feature generation. | Robustness maps, perturbation sensitivity, regime comparison. |
| Quantum hardware control | Backend calibration and QPU result history. | DLA parity, calibration probes, mitigation search. | Backend routing, mitigation selection, protected-subspace diagnostics. |
| ML forecasting | Time-series artifacts and labels where available. | Quantum feature map, kernel sampling, simulator/QPU comparison. | Feature artifacts for classical predictors with hardware/simulator provenance. |
| Scheduling and resource allocation | Job queues, intervention candidates, constraints. | Annealing/hybrid optimisation or classical optimiser. | Ranked schedules, feasible intervention sets, budget-aware routing. |

The QPU does not replace the classical stack. It adds selected quantum
measurements or sampling features where those outputs are useful to the
controller, memory, or scientific analysis.

## Expected system behaviour

The system should:

1. Accept source or replay data only through declared artifacts.
2. Validate matrix invariants and provenance before compute.
3. Route each request to a suitable compute unit: Rust, GPU, simulator,
   QPU, analog emulator, annealer, or hybrid solver.
4. Estimate cost before hardware submission.
5. Persist job ids immediately after submission.
6. Separate queued, running, failed, skipped, simulated, and completed
   states.
7. Retrieve hardware results later without rewriting history.
8. Classify every observable as measured, mitigated, simulated,
   emulator-derived, or proxy.
9. Feed only result artifacts, not terminal logs, into memory and
   controllers.
10. Allow multiple isolated instances to run simultaneously.

## What the system may deliver

Near-term deliverables:

- simulator-only dry run across all three repositories
- provider-neutral QPU request/result schemas
- backend capability descriptors
- stream-delta and fusion artifacts for live and multi-node modes
- cost and budget guard
- hardware micro-probe workflow
- result artifacts for synchronisation and DLA kernels
- cross-provider readiness matrix for allocation proposals

Medium-term deliverables:

- multi-provider broker for IBM, Braket, Azure, IonQ, Quantinuum,
  Pasqal, D-Wave, and local simulators where credentials/access exist
- QPU/simulator shadow execution
- application-specific pipelines for connectome, plasma, power-grid,
  and biological systems

Longer-term deliverables:

- real-time or near-real-time quantum-assisted control loops where
  backend latency permits
- routed multi-QPU compute fabric
- persistent memory of source artifacts, compute requests, results, and
  decisions
- allocation-ready dossiers per provider and per scientific objective

## What the system should not claim prematurely

The system must not claim:

- real-time control when the backend is batch queued
- hardware evidence when the result came from simulator or emulator
- measured observables when the value is a model proxy
- publication-grade provenance from synthetic smoke fixtures
- provider availability without checking the account-visible target
- quantum advantage without a classical baseline and cost accounting

The most valuable claim is narrower and stronger: the stack can express
real-world dynamical states as auditable artifacts, route appropriate
kernels to quantum/classical compute units, and return traceable result
artifacts that can be replayed, compared, and used by controllers.

## Allocation argument

For a QPU allocation request, the project can state:

1. We have a provider-neutral artifact contract.
2. We have simulator preflight and syntax/emulator gates.
3. We know which kernels map to which hardware modalities.
4. We have budget and idempotency controls.
5. We have a result-artifact format that preserves job ids, counts,
   mitigation metadata, and observable classification.
6. We can run the same scientific objective across several providers
   and compare results without changing the source artifact.
7. We can separate smoke tests, simulations, hardware micro-probes, and
   publication-grade runs.

That is the practical basis for asking for resources: the allocation is
not exploratory time for loose scripts; it is a controlled compute path
with data, routing, validation, and expected outputs already specified.

## Minimum next build

The first build now implements:

```text
QPUDataArtifact
-> QPUComputeRequest
-> simulator execution
-> QPUComputeResult
-> QPUNodeDescriptor / QPUStreamDelta / QPUFusionResult primitives
```

Runnable command:

```bash
python -m scpn_quantum_control.qpu_compute run-simulator \
  --artifact artifact.json \
  --request-out request.json \
  --result-out result.json
```

Then extend to:

```text
SC-NeuroCore bridge
-> Phase Orchestrator artifact
-> Quantum Control simulator result
```

Only after that path is stable should hardware submission be enabled by
policy.
