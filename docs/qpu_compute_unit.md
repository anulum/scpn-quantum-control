# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control - QPU Compute Unit Architecture

# QPU Compute Unit Architecture

The QPU is not only a validation target for isolated tests. In the SCPN
stack it should become a scheduled compute unit: a probabilistic
accelerator that accepts typed oscillator artifacts, runs selected
quantum kernels, and returns auditable result artifacts to the wider
pipeline.

This document defines the intended architecture. It does not claim that
every kernel is already production-ready.

## Core position

The QPU sits beside CPU, Rust, Julia, Go, Mojo, GPU, and simulator
paths. It is selected when the requested computation is naturally
quantum or when hardware-measured observables are the point of the
experiment.

```text
source/replay -> oscillator artifact -> kernel selection
                                      -> CPU/Rust path
                                      -> GPU/simulator path
                                      -> QPU compute unit
                                      -> result artifact
                                      -> controller / memory / analysis
```

The QPU must never be a silent fallback. A run is either explicitly
hardware, explicitly simulator, or explicitly skipped.

## What the QPU computes

A QPU compute unit should expose kernels, not arbitrary scripts.

| Kernel family | Input | Output | Use |
|---------------|-------|--------|-----|
| Synchronisation witness | `K_nm`, `omega`, optional `theta0` | order parameters and bitstring marginals | Phase-state measurement for oscillator networks. |
| DLA parity witness | compiled circuit or artifact | odd/even robustness metrics | Symmetry and protection diagnostics. |
| OTOC / scrambling probe | Hamiltonian/circuit schedule | scrambling curve or short-time proxy | Chaos, instability, and memory-loss diagnostics. |
| Quantum feature map | artifact plus feature encoding | kernel entries or sampled features | ML and forecasting interfaces. |
| Variational control | artifact plus objective | candidate parameters and energy/cost trace | Optimisation and control search. |
| Noise-characterisation probe | reference circuits | calibration-aware error model updates | Runtime mitigation and backend selection. |

The common contract is that each kernel returns either raw counts,
observable estimates, or both. Any proxy observable must be labelled as
model-derived rather than hardware-measured.

## Request artifact

Every hardware or simulator job should be derived from a compute request
that contains:

| Field | Meaning |
|-------|---------|
| `request_schema` | Versioned request schema. |
| `qpu_data_artifact_sha256` | Hash of the input oscillator artifact. |
| `kernel` | Named compute kernel, e.g. `sync_witness` or `dla_parity`. |
| `backend_policy` | Hardware, simulator, or hybrid selection rules. |
| `budget` | Shot cap, job cap, wall-time cap, and monetary/cost-class cap. |
| `circuit_limits` | Qubit, depth, two-qubit-gate, and measurement limits. |
| `mitigation` | Declared mitigation plan, e.g. readout, DD, ZNE, or none. |
| `idempotency_key` | Stable key preventing duplicate paid submissions. |
| `output_dir` | Isolated result location for this instance. |

This request layer is what allows many QPU-using pipelines to run
simultaneously without corrupting each other.

## Result artifact

Every completed compute unit call should emit a result artifact with:

- request hash
- input artifact hash
- backend name and backend family
- simulator seed or QPU job id
- queue timestamps and execution timestamps
- shot count and measurement register name
- circuit depth, width, and two-qubit gate count after transpilation
- mitigation actually applied
- counts hash and optional counts payload path
- observable values with measured/proxy classification
- failure state, retry state, or skip reason

The result artifact is the only object downstream controllers should
consume. Terminal output is not a result.

## Scheduler model

The scheduler is a gate between the pipeline and paid/queued hardware.
It must implement:

1. **Admission control** - reject jobs without publication-safe or
   explicitly synthetic provenance.
2. **Budget control** - enforce shot, job, time, and backend limits.
3. **Idempotency** - avoid duplicate hardware jobs for the same request.
4. **Queue awareness** - submit, persist job ids, and retrieve later.
5. **Concurrency limits** - run multiple instances without exceeding
   backend or account limits.
6. **Backend policy** - choose local simulator, QPU, or skip according
   to declared policy.
7. **Result finalisation** - reprocess counts into observables only
   after real results are retrieved.

This keeps hardware computation scientific rather than opportunistic.

## Networked multi-QPU mode

The current hardware path is mostly batch-oriented: submit a circuit,
persist a job id, retrieve counts later. That must remain supported, but
it must not become the only mental model. If several QPUs become
available at the same time, the compute unit becomes a network of
specialised quantum workers.

```text
artifact stream -> broker -> QPU A: sync witness
                        -> QPU B: DLA/parity probe
                        -> QPU C: feature-map kernel
                        -> simulator/Rust shadow path
                        -> result fusion -> controller / memory
```

Each QPU node needs its own capability descriptor:

| Field | Meaning |
|-------|---------|
| `node_id` | Stable local name for the compute node. |
| `provider` | Backend provider or local simulator family. |
| `topology` | Coupling graph, native gates, qubit count, and connectivity. |
| `latency_class` | Batch, near-real-time, or real-time. |
| `kernel_capabilities` | Which kernels can run on this node. |
| `calibration_state` | Last calibration timestamp and quality summary. |
| `budget_state` | Remaining queue, shot, or credit budget. |
| `routing_policy` | When this node should receive work. |

The broker routes requests by capability, not by hard-coded backend
name. A synchronisation witness and an OTOC probe may go to different
nodes if their topologies, noise, or latency make that the better
choice.

## Multi-instance execution

The stack should support simultaneous specialised instances:

| Instance | QPU role | Example |
|----------|----------|---------|
| Replay instance | Recompute a recorded artifact exactly. | Re-run a saved connectome artifact on simulator and QPU. |
| Calibration instance | Estimate backend noise or mitigation parameters. | Probe DD/ZNE settings before a hardware campaign. |
| Controller instance | Feed measurements into a next-step control policy. | Near-real-time plasma or network stability loop. |
| Memory instance | Store and compare result artifacts over time. | Detect drift in synchronisation or DLA metrics. |
| Interface instance | Serve a user/tool request over a stable API. | Ask for a kernel result without knowing Qiskit details. |

The opening is substantial: the ecosystem becomes a compute fabric.
Each repo can run its specialised role while the QPU layer acts as a
bounded accelerator with persistent memory of what it computed.

## Integration with memory and interfaces

The memory layer should not store loose numbers. It should store
artifact references:

```text
source_artifact_sha256
qpu_request_sha256
qpu_result_sha256
observable_version
analysis_version
decision_or_policy_version
```

Interfaces should expose verbs over artifacts:

- `compile` - source/domain data to QPU data artifact
- `submit` - artifact plus kernel to compute request
- `retrieve` - job id to result artifact
- `analyse` - counts to observable artifact
- `decide` - observable artifact to controller action
- `replay` - artifact hashes to reproducible rerun

This creates a durable working memory for experiments, controllers, and
user-facing tools. The system can ask "what did the QPU actually
compute for this source state?" and retrieve a verifiable answer.

## Real-time stream semantics

Real-time QPU access changes the data model. A static artifact is still
needed for provenance, but the live system also needs a stream of state
deltas:

```text
state_t -> delta_t -> compile/update -> quantum kernel
        -> measured delta -> controller action -> state_t+1
```

The stream should carry:

| Stream field | Purpose |
|--------------|---------|
| `stream_id` | Stable identity of the live source. |
| `sequence_id` | Monotonic event counter. |
| `event_time` | Time at the source. |
| `ingest_time` | Time observed by the pipeline. |
| `state_delta` | Changed phases, frequencies, couplings, or controls. |
| `artifact_base_sha256` | Hash of the base artifact this delta modifies. |
| `deadline` | Latest useful response time. |
| `control_window` | Time interval during which the response can affect the system. |
| `confidence` | Source-side confidence or measurement quality. |

This is different from the current batch form. Instead of compiling a
complete new circuit for every observation, the runtime should prefer
incremental updates when the kernel and hardware support them:

- update only changed frequencies or coupling weights
- reuse a compiled layout when topology is unchanged
- maintain rolling calibration and mitigation state
- return partial observables before full analysis is complete
- drop stale requests whose deadline has passed

The controller must know whether a result is fresh enough to act on. A
high-confidence result that arrives outside the control window is
operationally useless.

## Result fusion across QPUs

Networked QPU output needs a fusion layer. It combines several
probabilistic measurements into one decision-grade result without
pretending they are identical.

Fusion metadata should include:

- contributing node ids
- kernel and observable versions per node
- backend calibration state per node
- shot counts and confidence intervals
- agreement/disagreement metrics
- weighting rule used for the fused result
- stale or failed nodes excluded from the decision

For real-world control, disagreement is information. If one QPU reports
loss of synchronisation while another reports a stable feature-map
classification, the controller should surface that conflict rather than
averaging it away.

## Design constraints

The QPU path has hard constraints:

- It is probabilistic. Results need confidence intervals or shot-error
  metadata.
- It is queued. Runtime code must support asynchronous submission and
  later retrieval.
- It is expensive. Budgets are part of the request, not an operator
  afterthought.
- It is calibration-sensitive. Result artifacts must record backend and
  mitigation metadata.
- It is not always the correct backend. Rust, GPU, or simulator paths
  remain first-class compute units.

The scheduler should select QPU only when hardware data is required or
when the kernel is designed to extract a quantum observable unavailable
from the classical path.

## Minimal implementation path

1. Keep the current QPU data artifact as the source-side input.
2. Add a `QPUComputeRequest` schema with kernel, backend policy, budget,
   idempotency, and output directory.
3. Add a `QPUComputeResult` schema with job ids, counts hashes,
   mitigation metadata, and observable classifications.
4. Add a simulator-only dry-run command that consumes a QPU data
   artifact and emits a result artifact.
5. Add a hardware-submission command that writes queued result artifacts
   without pretending counts exist.
6. Add a retrieval command that finalises queued artifacts once counts
   are available.
7. Add an inter-repository health check:

```text
SC-NeuroCore bridge -> Phase Orchestrator artifact
-> Quantum Control simulator request -> result artifact
```

Only after that path is green should the same request be allowed to
target hardware.

The first executable slice is available as:

```bash
python -m scpn_quantum_control.qpu_compute run-simulator \
  --artifact artifact.json \
  --request-out request.json \
  --result-out result.json \
  --kernel sync_dla \
  --shots 1024
```

For smoke artifacts, add `--allow-synthetic`. Without that flag,
synthetic, simulation, and fixture source modes are rejected by the
publication gate.

The current runner:

- reads a validated QPU data artifact
- creates a `QPUComputeRequest`
- builds a `StructuredAnsatz`
- runs exact local statevector simulation
- converts probabilities to deterministic counts
- computes synchronisation and DLA observables from those counts
- writes a `QPUComputeResult`

The output is explicitly classified as simulated, not hardware
evidence.

The same module also now implements the infrastructure records needed
for networked and real-time operation:

- `QPUNodeDescriptor` for routable IBM, Braket, Azure, IonQ,
  Quantinuum, Pasqal, D-Wave, research-programme, and local nodes.
- `QPUStreamDelta` for live source updates against a base artifact.
- `QPUFusionResult` plus `fuse_compute_results(...)` for combining
  measurements from several compute nodes without losing disagreement
  metrics.

The remaining later layers are provider adapters, broker admission
control, hardware submit/retrieve commands, and cross-repo health checks.

Provider-specific access planning is tracked in
`docs/qpu_provider_readiness.md`. That matrix must be refreshed from
official provider documentation before any allocation request or paid
hardware campaign.

## Strategic consequence

Treating the QPU as a compute unit changes the project from a set of
quantum experiments into an operating model:

- QPU work becomes schedulable and replayable.
- Controllers can call quantum kernels as tools.
- Memory systems can compare verified quantum outputs over time.
- Multiple domain pipelines can share one hardware budget safely.
- Real hardware evidence is separated from simulation and proxy values.
- Multiple QPUs can become a routed compute network rather than a
  manually selected backend list.
- Real-time streams can drive quantum kernels as part of control loops,
  provided deadlines, freshness, and provenance are explicit.

This is the architecture needed for SC-NeuroCore, Phase Orchestrator,
and Quantum Control to be alive as one system rather than three adjacent
repositories.
