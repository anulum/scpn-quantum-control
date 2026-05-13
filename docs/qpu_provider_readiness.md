# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control - QPU Provider Readiness Matrix

# QPU Provider Readiness Matrix

This document prepares the SCPN stack for allocation requests across
known accessible QPU families. The goal is not to hard-code one vendor;
the goal is to make every provider an instance of the same compute-unit
contract:

```text
QPU data artifact -> compute request -> provider adapter
                  -> result artifact -> fusion / memory / controller
```

Provider availability changes. Before any submitted proposal or paid run,
refresh the provider status from the official console or documentation.

## Provider classes

| Class | Examples | Best fit in SCPN pipeline |
|-------|----------|---------------------------|
| Superconducting gate-model | IBM, Rigetti, IQM, QCI preview | Wide circuits, hardware-native graph layouts, fast queue experiments, DLA and synchronisation witnesses. |
| Trapped-ion gate-model | IonQ, Quantinuum, AQT | All-to-all connectivity, deeper small-N circuits, mid-circuit measurement where available. |
| Neutral-atom analog/digital | QuEra, Pasqal | Direct oscillator-network and analog Hamiltonian experiments, graph geometry, Rydberg dynamics. |
| Annealing / hybrid optimisation | D-Wave Leap | QUBO/Ising scheduling, routing, intervention ranking, coarse optimisation subproblems. |
| Photonic / continuous-variable | Quandela, Xanadu-style platforms where accessible | Bosonic/CV synchronisation and sampling-oriented feature generation. |
| Simulator / emulator / resource estimator | Braket simulators, Azure provider simulators, Quantinuum emulators, local Aer, Rust paths | Preflight, shadow execution, calibration, cost avoidance, reproducibility. |

## Access matrix

| Access route | Publicly documented providers | Adapter priority | Notes |
|--------------|------------------------------|------------------|-------|
| IBM Quantum | IBM superconducting QPUs including Eagle and Heron families. | Existing first-class path. | Query available backends per account; do not assume a named backend is always available. |
| Amazon Braket | AQT, IonQ, IQM, QuEra, Rigetti, plus managed simulators. | High. | Single cloud surface for several modalities; useful for multi-provider routing. |
| Azure Quantum | IonQ, Pasqal, Quantinuum, Rigetti, plus simulators and resource estimation. | High. | Good proposal surface because provider credits and target availability are documented by workspace/region. |
| IQM Resonance / direct IQM server | IQM superconducting QPUs and Qiskit-on-IQM fake/facade backends. | High. | Direct European superconducting replication route for IBM-sensitive circuit witnesses. |
| IonQ Quantum Cloud | IonQ direct access to simulators and trapped-ion QPUs. | High. | Direct route for Aria/Forte-family work and research-credit applications. |
| Quantinuum Systems / Nexus | Quantinuum hardware, emulators, syntax checkers. | High. | Strong fit for mid-circuit measurement, all-to-all trapped-ion circuits, and hardware-credit accounting. |
| Pasqal Cloud | Neutral-atom QPU and emulators. | Medium-high. | Strong fit for analog oscillator-network experiments and Rydberg geometry. |
| D-Wave Leap | Advantage annealer and hybrid solvers. | Medium. | Not a gate-model backend; route only optimisation kernels, not circuit witnesses. |
| Other direct research programmes | Provider-specific previews and national lab allocations. | Case-by-case. | Must satisfy the same node descriptor and result artifact contract before use. |

## Node descriptor

Every accessible backend becomes a `QPUNodeDescriptor`:

| Field | Required meaning |
|-------|------------------|
| `node_id` | Stable internal name, e.g. `aws.ionq.forte_1` or `ibm.heron_r2`. |
| `access_route` | `ibm`, `aws_braket`, `azure_quantum`, `ionq_cloud`, `quantinuum`, `pasqal`, `dwave`, or `local`. |
| `provider` | Hardware provider. |
| `modality` | `superconducting`, `trapped_ion`, `neutral_atom`, `annealer`, `photonic`, `simulator`. |
| `execution_model` | `gate_model`, `analog_hamiltonian`, `annealing`, `hybrid_solver`, `emulator`. |
| `latency_class` | `batch`, `near_real_time`, or `real_time_candidate`. |
| `qubit_or_variable_limit` | Current account-visible limit. |
| `native_features` | Connectivity, native gates, analog controls, mid-circuit measurement, reset, pulse control. |
| `cost_model` | Shots, credits, task price, QPU time, HQC/AQT, or provider-specific unit. |
| `queue_model` | Immediate, reserved, windowed, commercial period, or batch queue. |
| `kernel_capabilities` | SCPN kernels allowed on the node. |
| `calibration_snapshot` | Timestamped backend quality metadata when available. |
| `verification_status` | `unverified`, `simulator_green`, `syntax_green`, `hardware_probe_green`, or `production_ready`. |

The broker must route by descriptor fields, not by hard-coded backend
names.

The descriptor schema is implemented in
`scpn_quantum_control.qpu_compute.QPUNodeDescriptor`. Provider adapters
must emit that object before any job can be considered routable.

## Kernel routing

| SCPN kernel | Preferred modalities | Reason |
|-------------|----------------------|--------|
| Synchronisation witness | Superconducting, trapped-ion, neutral-atom | Direct circuit or analog measurement of phase/coherence structure. |
| DLA parity witness | Superconducting, trapped-ion | Needs controlled circuit structure and reliable parity readout. |
| OTOC / scrambling | Trapped-ion, superconducting, high-fidelity simulator | Deeper coherent circuits benefit from all-to-all or high-quality transpilation. |
| Quantum feature map | Superconducting, trapped-ion, simulator | Classical ML interface can tolerate batched samples and compare across providers. |
| Analog Kuramoto / Rydberg graph | Neutral-atom | Native geometry and interaction physics align with oscillator-network structure. |
| Variational control | Simulator first, then superconducting/trapped-ion | Requires many iterations; hardware only after simulator and budget gates pass. |
| Intervention ranking / scheduling | D-Wave hybrid, CPU/Rust optimiser, simulator | Optimisation kernel, not a gate-model observable. |
| Backend calibration probe | Every provider used | Required before trusting a hardware campaign. |

## Adapter responsibilities

Each provider adapter must implement the same stages:

1. `capabilities()` - return a `QPUNodeDescriptor`.
2. `validate_request(request)` - reject incompatible kernels before cost
   is incurred.
3. `compile(request)` - translate the SCPN kernel into provider-native
   representation.
4. `estimate_cost(compiled)` - return provider-unit and project-budget
   estimates.
5. `submit(compiled)` - submit only after budget and idempotency gates.
6. `retrieve(job_ref)` - retrieve counts, samples, energies, or analog
   observables.
7. `normalise_result(raw)` - emit the common result artifact.
8. `classify_observables(result)` - mark each value as measured,
   mitigated, simulated, emulator-derived, or model proxy.

No adapter may hide an unavailable provider by silently switching to a
different backend. A fallback is allowed only when the compute request
declares it.

## Proposal-ready allocation package

When applying for resources, the dossier should include:

- scientific objective and real-world application
- kernel family requested
- input artifact schema and example artifact hash
- expected circuit/analog workload shape
- requested provider class and why that modality is appropriate
- simulator preflight evidence
- cost estimate and budget guard
- data-management plan for job ids, counts, and result artifacts
- risk plan for queue delays, calibration drift, and failed jobs
- publication-safe provenance statement
- ethical/safety statement for real-world control use

The strongest allocation argument is not "we want QPU time." It is:

```text
We have a provider-neutral pipeline, simulator preflight, typed
artifacts, budget controls, and a precise kernel that maps to this
provider's hardware advantage.
```

## Provider-specific fit

### IBM Quantum

Use for superconducting gate-model experiments, DLA parity, structured
Kuramoto circuits, Qiskit Runtime integration, and Heron/Eagle-family
hardware comparison. Strong current fit because Quantum Control already
has Qiskit/IBM runtime paths.

Readiness requirement:

- account-visible backend descriptor
- calibration snapshot
- transpiled depth/two-qubit-gate budget
- simulator or Aer preflight
- hardware micro-probe before full campaign

### Amazon Braket

Use as a multi-provider broker surface. Braket currently documents QPU
access for AQT, IonQ, IQM, QuEra, and Rigetti, plus simulators. This is
important because one access route can cover trapped-ion,
superconducting, and neutral-atom modalities.

Readiness requirement:

- ARN-based node descriptor
- task batching policy
- S3/output artifact mapping
- provider-specific result normalisation

### Azure Quantum

Use for provider diversity, proposal credits, QIR/Q# interoperability,
and access to IonQ, Pasqal, Quantinuum, and Rigetti targets where
available by region/account.

Readiness requirement:

- workspace/provider/target descriptor
- pricing unit mapping
- region availability check
- simulator/syntax-checker preflight where available

### IonQ Cloud

Use for trapped-ion all-to-all circuits, smaller but deeper feature maps,
and native-gate experiments. IonQ documents direct cloud access, SDKs,
API keys, job dashboards, simulators, and Aria/Forte-family systems.

Readiness requirement:

- target and algorithmic-qubit descriptor
- native-gate or Qiskit compilation path
- quota and research-credit tracking
- all-to-all routing policy

### Quantinuum

Use for high-fidelity trapped-ion circuits, mid-circuit measurement,
conditional logic, emulators, and syntax checkers. Quantinuum's
hardware-credit model should be represented explicitly in the budget.

Readiness requirement:

- syntax-checker pass before hardware
- emulator comparison
- HQC/eHQC cost estimate
- mid-circuit measurement capability flags

### Pasqal and QuEra

Use for neutral-atom and Rydberg-native oscillator-network experiments.
These are the natural candidates for analog Kuramoto, graph geometry,
and Hamiltonian simulation variants where Trotterisation is the wrong
abstraction.

Readiness requirement:

- geometry/register descriptor
- analog Hamiltonian parameter bounds
- emulator preflight
- measured observable mapping into the common result artifact

### IQM

Use for superconducting diversity beyond IBM while keeping the Qiskit
circuit path close to the existing Kuramoto-XY, DLA parity, and FIM
workflows. The built-in `iqm` adapter uses Qiskit-on-IQM and is
approval-gated for remote execution. Local IQM fake backends are suitable
for syntax and topology preflight, but they are not hardware evidence.

Readiness requirement:

- `iqm-client[qiskit]` installed in an isolated runner environment such as
  `.venv-iqm`; current IQM client releases pin Qiskit below the main
  IBM/Qiskit environment used by this repository
- fake/facade backend preflight for the selected circuit family
- explicit IQM server URL and quantum-computer name for remote runs
- topology/native-gate translation recorded in the artefact ledger
- calibration snapshot or provider metadata captured where available
- micro-probe counts before any full witness campaign

### Rigetti

Use for superconducting diversity beyond IBM, especially when routing,
native-gate sets, low-latency hybrid workflows, or access through Braket
is strategically important.

Readiness requirement:

- topology descriptor
- native-gate translation
- calibration snapshot
- cross-provider witness comparison plan

### D-Wave Leap

Use only for optimisation kernels: scheduling, intervention ranking,
layout/routing, and QUBO/Ising formulations. Do not route gate-model
observables to D-Wave.

Readiness requirement:

- QUBO/Ising mapping
- variable-count estimate
- hybrid solver selection
- result conversion into ranked interventions or schedules

## Funding-readiness state machine

Every provider starts at `unverified` and advances only by evidence:

```text
unverified
-> docs_mapped
-> simulator_green
-> syntax_green
-> micro_probe_green
-> allocation_ready
-> production_ready
```

Definitions:

- `docs_mapped`: public provider docs mapped into a descriptor.
- `simulator_green`: local/provider simulator executes our kernel.
- `syntax_green`: provider compiler/syntax checker accepts the job.
- `micro_probe_green`: tiny hardware job returns valid result artifact.
- `allocation_ready`: proposal dossier has objective, budget, preflight,
  risk, and provenance.
- `production_ready`: repeated hardware runs are stable enough for the
  declared application.

## Official documentation references

Provider details must be refreshed before submission from official
sources:

- IBM Quantum hardware and QPU information:
  <https://www.ibm.com/quantum/hardware>,
  <https://docs.quantum.ibm.com/guides/qpu-information>
- Amazon Braket hardware providers and devices:
  <https://aws.amazon.com/braket/hardware-providers/>,
  <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>
- Azure Quantum providers, targets, and availability:
  <https://learn.microsoft.com/en-us/azure/quantum/provider-global-availability>,
  <https://learn.microsoft.com/en-us/azure/quantum/qc-target-list>
- IonQ Cloud documentation:
  <https://docs.ionq.com/>,
  <https://www.ionq.com/quantum-cloud>
- Quantinuum Systems documentation:
  <https://docs.quantinuum.com/h-series/>
- IQM Qiskit integration and fake backends:
  <https://docs.iqm.tech/iqm-client/user_guide_qiskit.html>,
  <https://docs.iqm.tech/iqm-client/api/iqm.qiskit_iqm.fake_backends.html>
- Pasqal Cloud documentation:
  <https://docs.pasqal.com/cloud/>
- D-Wave Leap:
  <https://www.dwavequantum.com/quantum-cloud-services/>
