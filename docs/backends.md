# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Backends & Plugin Registry Documentation

# Backends & Plugin Registry

Two modules for runtime backend management:

1. **Backend dispatch** (`backend_dispatch.py`) — switch between numpy,
   JAX, and PyTorch for array operations
2. **Plugin registry** (`hardware/plugin_registry.py`) — legacy runner
   registration for direct adapter construction
3. **Provider-neutral backend registry** (`hardware/backends.py`) —
   production routing descriptors for IBM Runtime, local Aer, Cirq,
   Amazon Braket, PennyLane, analogue, and hybrid compiler paths
4. **Hardware abstraction layer** (`hardware/hal.py`) — provider-neutral
   workload, job, result, approval, and adapter protocol across cloud and
   simulator routes

---

## Part 1: Backend Dispatch

`scpn_quantum_control.backend_dispatch`

Runtime array backend selection, inspired by TensorCircuit's
`tc.set_backend()`. All array operations in downstream code use the
selected backend.

### API Reference

```python
from scpn_quantum_control.backend_dispatch import (
    set_backend,
    get_backend,
    get_array_module,
    to_numpy,
    from_numpy,
    available_backends,
)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `set_backend(name)` | `str → None` | Set active backend: `"numpy"`, `"jax"`, `"torch"` |
| `get_backend()` | `() → str` | Current backend name |
| `get_array_module()` | `() → module` | Active array module (`np`, `jnp`, or `torch`) |
| `to_numpy(arr)` | `Any → ndarray` | Convert any backend array to numpy |
| `from_numpy(arr)` | `ndarray → Any` | Convert numpy to current backend |
| `available_backends()` | `() → list[str]` | List installed backends |

### Example

```python
from scpn_quantum_control.backend_dispatch import (
    set_backend, get_backend, available_backends,
    get_array_module, to_numpy, from_numpy
)
import numpy as np

# Check what's available
print(available_backends())  # ['numpy', 'jax', 'torch'] (if installed)

# Default is numpy
assert get_backend() == "numpy"

# Switch to JAX
set_backend("jax")
xp = get_array_module()  # jax.numpy
arr = from_numpy(np.array([1.0, 2.0, 3.0]))
print(type(arr))  # jaxlib.xla_extension.ArrayImpl

# Convert back
arr_np = to_numpy(arr)
print(type(arr_np))  # numpy.ndarray

# Switch to PyTorch
set_backend("torch")
xp = get_array_module()  # torch
arr_t = from_numpy(np.array([1.0, 2.0]))
print(type(arr_t))  # torch.Tensor

# Reset to numpy
set_backend("numpy")
```

---

## Part 2: Plugin Registry

`scpn_quantum_control.hardware.plugin_registry`

Extensible plugin architecture for quantum hardware backends. Register
and discover backends at runtime without hard-coding imports.

Inspired by OpenFermion's plugin system (Google Quantum AI).

### Built-In Backends

The registry includes lazy loaders for three backends:

| Backend | Package | Provides |
|---------|---------|----------|
| `qiskit` | `qiskit` | Trotter circuits, IBM execution |
| `pennylane` | `pennylane` | Differentiable circuits and VQE value/gradient adapter |
| `cirq` | `cirq-core` | Google Quantum circuits |

These are loaded on first access — no import cost if unused.

### API Reference

```python
from scpn_quantum_control.hardware.plugin_registry import registry
```

#### `PluginRegistry` Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `list_backends()` | `() → list[str]` | All registered + lazy-loadable names |
| `available_backends()` | `() → list[str]` | Only importable backends |
| `is_available(name)` | `str → bool` | Check if backend is installed |
| `get_runner(name, K, omega, **kw)` | `(str, ndarray, ndarray, ...) → Runner` | Get instantiated runner |
| `register(name)` | `str → decorator` | Decorator for custom backends |
| `register_class(name, cls)` | `(str, type) → None` | Programmatic registration |

#### Runner Interface

Runners returned by `get_runner` implement:

```python
class Runner:
    def __init__(self, K, omega, **kwargs): ...
    def run_trotter(self, t: float, reps: int) -> dict: ...
    def run_vqe(self, **kwargs) -> dict: ...  # optional
```

### Example: Using Built-In Backends

```python
from scpn_quantum_control.hardware.plugin_registry import registry
import numpy as np

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# List available backends
print(registry.available_backends())

# Use Qiskit backend
if registry.is_available("qiskit"):
    runner = registry.get_runner("qiskit", K, omega)
    result = runner.run_trotter(t=0.1, reps=5)
    print(f"Qiskit circuit depth: {result['depth']}")
```

### Example: Custom Backend

```python
from scpn_quantum_control.hardware.plugin_registry import registry

@registry.register("my_simulator")
class MySimulator:
    def __init__(self, K, omega, **kwargs):
        self.K = K
        self.omega = omega

    def run_trotter(self, t=0.1, reps=5):
        # Custom simulation logic
        return {"energy": -1.23, "method": "my_simulator"}

# Now usable via registry
runner = registry.get_runner("my_simulator", K, omega)
result = runner.run_trotter(t=0.1, reps=5)
```

---

## Part 3: Provider-Neutral Quantum Backends

`scpn_quantum_control.hardware.backends`

The production registry exposes a single capability contract across the
hardware and simulator surface. Registry lookup is deliberately
non-authenticating and non-submitting: it imports at most the local SDK
module needed for availability checks and never reads credentials, opens
network sessions, or queues QPU jobs.

### Built-In Production Descriptors

| Backend | Provider | Execution mode | Submission policy |
|---------|----------|----------------|-------------------|
| `qiskit_ibm` | IBM Quantum | cloud QPU | approval required |
| `qiskit_aer` | local Qiskit Aer | local simulator | no submission |
| `cirq` | Google Cirq | local simulator/export | no submission |
| `braket` | Amazon Braket | cloud QPU or managed simulator | approval required |
| `pennylane` | PennyLane | adapter router | provider plugin decides |
| `analog_kuramoto` | internal compiler | analogue programme compiler | no registry-time submission |
| `hybrid_digital_analog` | internal compiler | hybrid compiler | no registry-time submission |

Cloud descriptors advertise that a submission interface exists, but they
also set `submit_requires_approval=True`. Production execution code must
pass through the explicit hardware approval scheduler before any live IBM
or AWS work is attempted.

### Descriptor API

```python
from scpn_quantum_control.hardware import (
    describe_backend,
    describe_hal_backend_profile,
    list_hal_backend_descriptors,
    list_quantum_backends,
)

ibm = describe_backend("qiskit_ibm")
assert ibm.provider == "ibm_quantum"
assert ibm.can_submit is True
assert ibm.submit_requires_approval is True

local = describe_backend("qiskit_aer")
assert local.can_simulate is True
assert local.can_submit is False

for descriptor in list_quantum_backends():
    print(descriptor.name, descriptor.execution_mode, descriptor.available)

quera = describe_hal_backend_profile("quera_bloqade")
assert quera.adapter_module == "scpn_quantum_control.hardware.hal_quera_bloqade"
assert quera.submit_requires_approval is True

for route in list_hal_backend_descriptors():
    print(route.name, route.provider, route.workloads)
```

Every descriptor records:

| Field | Meaning |
|-------|---------|
| `name` | registry key used by routing code |
| `provider` | provider namespace, e.g. `ibm_quantum`, `local_qiskit_aer`, `aws_braket` |
| `execution_mode` | local simulator, cloud QPU, managed simulator, or adapter router |
| `sdk_package` | Python package expected for the route |
| `adapter_module` | repository module that owns execution or export |
| `available` | import-time availability, without credentials or network calls |
| `can_simulate` / `can_submit` | whether the descriptor exposes simulation or live submission semantics |
| `submit_requires_approval` | mandatory cloud-job approval flag |
| `supports_*` | shot, statevector, mid-circuit, and pulse capability flags |
| `capabilities` / `workloads` | stable machine-readable routing tags |

`list_quantum_backends()` describes plugin-registry entries and may probe
import-time availability through each backend's `is_available()` method.
`list_hal_backend_descriptors()` describes every built-in HAL profile using
static metadata only. It is the safer selector input when an application needs
the complete IBM, Braket, Azure, IonQ, Rigetti, Quantinuum, QuEra, qBraid,
simulator, and future-profile route matrix without importing provider SDKs.

Legacy third-party plugins that only implement `name` and
`is_available()` are still accepted. `describe_backend()` gives them a
conservative descriptor with `can_submit=False` and
`submit_requires_approval=True` until they implement a real
`descriptor()` method returning `QuantumBackendDescriptor`.

---

## Part 4: Hardware Abstraction Layer

`scpn_quantum_control.hardware.hal`

The HAL is the execution contract above provider descriptors. It decouples SCPN
workloads from provider SDKs by using immutable metadata profiles and injected
adapter objects. Constructing the HAL is offline and metadata-only: it does not
import cloud SDKs, inspect credentials, open network sessions, or submit jobs.

### Built-In HAL Profiles

Built-in route families include:

| Backend id | Provider | Broker | Modality |
|------------|----------|--------|----------|
| `ibm_quantum` | IBM | direct | superconducting gate model |
| `ionq_cloud` | IonQ | direct | trapped-ion gate model |
| `aws_braket_ionq` | IonQ | AWS Braket | trapped-ion gate model |
| `aws_braket_iqm` | IQM | AWS Braket | superconducting gate model |
| `aws_braket_quera` | QuEra | AWS Braket | neutral-atom analogue |
| `aws_braket_rigetti` | Rigetti | AWS Braket | superconducting gate model |
| `aws_braket_aqt` | AQT | AWS Braket | trapped-ion gate model |
| `aws_braket_sv1` | AWS | AWS Braket | managed statevector simulator |
| `aws_braket_dm1` | AWS | AWS Braket | managed density-matrix simulator |
| `aws_braket_tn1` | AWS | AWS Braket | managed tensor-network simulator |
| `azure_quantum_quantinuum` | Quantinuum | Azure Quantum | trapped-ion gate model |
| `azure_quantum_quantinuum_emulator` | Quantinuum | Azure Quantum | trapped-ion emulator |
| `azure_quantum_ionq` | IonQ | Azure Quantum | trapped-ion gate model |
| `azure_quantum_ionq_simulator` | IonQ | Azure Quantum | managed gate-model simulator |
| `azure_quantum_rigetti` | Rigetti | Azure Quantum | superconducting gate model |
| `azure_quantum_rigetti_qvm` | Rigetti | Azure Quantum | managed QVM simulator |
| `azure_quantum_pasqal` | Pasqal | Azure Quantum | neutral-atom analogue |
| `azure_quantum_pasqal_emulator` | Pasqal | Azure Quantum | neutral-atom emulator |
| `azure_quantum_qci_preview` | Quantum Circuits | Azure Quantum | private-preview superconducting route |
| `quantinuum_cloud` | Quantinuum | direct | trapped-ion gate model |
| `rigetti_qcs` | Rigetti | direct | superconducting gate model |
| `quera_bloqade` | QuEra | direct | neutral-atom analogue |
| `iqm_cloud` | IQM | direct | superconducting gate model |
| `pasqal_cloud` | Pasqal | direct | neutral-atom analogue |
| `oqc_cloud` | OQC | direct | superconducting gate model |
| `qbraid_ionq` | IonQ | qBraid | trapped-ion gate model |
| `qbraid_runtime` | dynamic catalog | qBraid | provider-agnostic runtime |
| `quandela_cloud` | Quandela | direct | photonic gate model |
| `dwave_leap` | D-Wave | direct | quantum annealing |
| `strangeworks_compute` | dynamic catalog | Strangeworks | provider-agnostic compute |
| `local_statevector` | SCPN | local | deterministic simulator |
| `local_braket_sv` | AWS Braket SDK | local | statevector simulator |
| `local_braket_dm` | AWS Braket SDK | local | density-matrix simulator |
| `local_braket_ahs` | AWS Braket SDK | local | analogue Hamiltonian simulator |
| `local_qiskit_aer` | Qiskit Aer | local | simulator |
| `local_cirq` | Cirq | local | simulator |
| `local_pennylane` | PennyLane | local | simulator |

All cloud profiles set `submit_requires_approval=True`. A cloud workload fails
closed unless the application has registered a concrete adapter and supplied an
approval token for that submission. Provider credentials, queue selection,
region policy, pricing, and detailed target selection belong inside the
provider adapter, not the HAL registry.

### Aggregator/provider matrix

`built_in_aggregator_provider_routes()` provides the broker-facing coverage
view used by conformance tests and documentation. Each row resolves to a real
HAL profile and adapter module, so broad aggregator catalogues are first-class
without multiplying identical runtime adapters.

| Aggregator | Provider family | Executable HAL route |
| --- | --- | --- |
| Direct | IBM Quantum / Qiskit Runtime | `ibm_quantum` |
| AWS Braket | AQT, IonQ, IQM, QuEra, Rigetti, Amazon simulators | `aws_braket_*` |
| Azure Quantum | IonQ, Quantinuum, Rigetti, Pasqal, QCI preview | `azure_quantum_*` |
| qBraid | AWS Braket, Azure Quantum, Equal1, IBM Quantum, IonQ, IQM, NEC vector annealer, OQC, Pasqal, Quantinuum, QuEra, Rigetti, QIR simulator | `qbraid_ionq` or `qbraid_runtime` |
| Strangeworks | AQT, IonQ, IQM, Quantinuum, QuEra, Rigetti, IBM Quantum/Qiskit Runtime, AWS Braket, Azure Quantum, classical/HPC compute targets | `strangeworks_compute` |

Dynamic rows carry `dynamic_catalog_target` notes and still require the
runtime caller to inject the authenticated provider backend or workspace. The
matrix is metadata-only: it imports no provider SDKs, reads no credentials,
authenticates nowhere, and submits no jobs.

`resolve_aggregator_provider_route()` is the fail-closed selector above this
matrix. It returns the selected row, executable HAL profile, and backend
descriptor for a requested aggregator/provider/IR tuple, and raises
`LookupError` when no row exists or the requested IR format is unsupported.

### HAL API

```python
from scpn_quantum_control.hardware import (
    AzureQuantumHALAdapter,
    BraketLocalHALAdapter,
    CirqLocalHALAdapter,
    DWaveLeapHALAdapter,
    HardwareAbstractionLayer,
    IonQCloudHALAdapter,
    IQMHALAdapter,
    LocalDeterministicSimulator,
    OQCHALAdapter,
    PasqalPulserHALAdapter,
    PennyLaneDeviceHALAdapter,
    QbraidRuntimeHALAdapter,
    QuandelaPercevalHALAdapter,
    QuEraBloqadeHALAdapter,
    QuantinuumCloudHALAdapter,
    RigettiQCSHALAdapter,
    QiskitAerHALAdapter,
    QuantumWorkload,
    azure_openqasm3_to_workload,
    braket_circuit_to_workload,
    bloqade_ahs_workload,
    cirq_circuit_workload,
    dwave_bqm_workload,
    ionq_qis_workload,
    iqm_qiskit_workload,
    oqc_openqasm3_workload,
    pulser_sequence_workload,
    pennylane_gate_workload,
    qbraid_program_to_workload,
    quandela_perceval_workload,
    quantinuum_tket_workload,
    rigetti_quil_workload,
    qiskit_circuit_to_workload,
)

hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(LocalDeterministicSimulator(hal.profile("local_statevector")))

job = hal.submit(
    "local_statevector",
    QuantumWorkload(
        workload_id="demo",
        ir_format="mlir",
        program="module {}",
        n_qubits=4,
        shots=1024,
    ),
)
result = hal.result(job)
```

Injected adapters implement `QuantumBackend.submit`, `status`, `result`, and
`cancel`. The HAL validates workload id, IR format, qubit limits, shot count,
backend registration, and approval before delegating.

The Qiskit adapter layer provides:

- `QiskitAerHALAdapter` for local `qiskit-aer` execution through HAL.
- `QiskitRuntimeHALAdapter` for IBM Runtime Sampler execution through HAL.
- `qiskit_circuit_to_workload()` for base64 QPY payloads. This is the preferred
  high-fidelity Qiskit transport because it preserves circuit structure without
  requiring lossy text conversion.
- `qiskit_circuit_to_qasm3_workload()` for OpenQASM 3 payloads when the
  `qiskit-qasm3-import` optional importer is installed.

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))
result = hal.result(
    hal.submit(
        "local_qiskit_aer",
        qiskit_circuit_to_workload(qc, workload_id="h_sample", shots=256),
    )
)
```

The Braket adapter layer provides:

- `BraketLocalHALAdapter` for local Braket SV/DM simulator execution through
  HAL.
- `BraketAwsHALAdapter` for AWS Braket QPU or managed-simulator task
  submission through HAL with explicit approval tokens.
- `braket_circuit_to_workload()` for OpenQASM 3 payloads generated from
  `braket.circuits.Circuit`.
- `snapshot_from_braket_device()` for no-submit capability snapshots from
  injected Braket device metadata, including OpenQASM and AHS action support,
  shot limits, queue depth, online state, and calibration/update timestamp.

```python
from braket.circuits import Circuit

circuit = Circuit().h(0).cnot(0, 1)
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(BraketLocalHALAdapter(hal.profile("local_braket_sv")))
result = hal.result(
    hal.submit(
        "local_braket_sv",
        braket_circuit_to_workload(circuit, workload_id="bell", shots=256),
    )
)
```

The Azure Quantum adapter layer provides `AzureQuantumHALAdapter` and
`azure_openqasm3_to_workload()`. The adapter calls Azure `Target.submit(...)`
only after HAL approval has been supplied and only when a target object or
explicit workspace/target factory was injected.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    AzureQuantumHALAdapter(
        hal.profile("azure_quantum_ionq_simulator"),
        target=target,
    )
)
job = hal.submit(
    "azure_quantum_ionq_simulator",
    azure_openqasm3_to_workload(
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nx q[0];",
        workload_id="azure_x",
        n_qubits=1,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The direct IonQ adapter layer provides `IonQCloudHALAdapter` and
`ionq_qis_workload()`. It uses IonQ API v0.4 directly, targets a named IonQ
backend such as `simulator` or `qpu.forte-1`, submits IonQ QIS JSON circuits,
fetches sparse probability results, converts them to fixed-shot bitstring
counts, and cancels via the v0.4 job status endpoint. API keys are supplied by
constructor argument or `IONQ_API_KEY`; they are never part of `QuantumWorkload`.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    IonQCloudHALAdapter(
        hal.profile("ionq_cloud"),
        api_key=ionq_api_key,
        backend="simulator",
    )
)
job = hal.submit(
    "ionq_cloud",
    ionq_qis_workload(
        [{"gate": "h", "target": 0}, {"gate": "cnot", "control": 0, "target": 1}],
        workload_id="ionq_bell",
        n_qubits=2,
        shots=256,
    ),
    approval_id="approved-run",
)
```

The local Cirq adapter layer provides `CirqLocalHALAdapter` and
`cirq_circuit_workload()`. It runs a caller-supplied Cirq circuit
representation through an injected Cirq simulator or simulator factory,
normalises measurement histograms into HAL counts, and uses the same
`submit/status/result/cancel` lifecycle shape as cloud adapters without
requiring a cloud approval token. Automatic circuit reconstruction remains
explicit through `circuit_factory`.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    CirqLocalHALAdapter(
        hal.profile("local_cirq"),
        circuit_factory=cirq_circuit_factory,
        simulator_factory=cirq_simulator_factory,
    )
)
job = hal.submit(
    "local_cirq",
    cirq_circuit_workload(cirq_payload, workload_id="cirq_bell", n_qubits=2, shots=256),
)
```

The direct OQC adapter layer provides `OQCHALAdapter` and
`oqc_openqasm3_workload()`. It consumes OpenQASM 3 programs, validates the
program header before submission, calls an injected QCAAS-style client, normalises
provider counts into HAL counts, and keeps OQC cloud execution approval-gated.
Automatic client construction is calibration-gated.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    OQCHALAdapter(
        hal.profile("oqc_cloud"),
        client_factory=calibrated_oqc_client_factory,
        target="Lucy",
    )
)
job = hal.submit(
    "oqc_cloud",
    oqc_openqasm3_workload(openqasm3_program, workload_id="oqc_bell", n_qubits=2, shots=256),
    approval_id="approved-run",
)
```

`provider_optional_dependency_matrix()` gives an offline smoke matrix for every
built-in HAL route. It uses import-spec discovery only; it does not import
provider SDKs, read credentials, authenticate, create clients, or touch provider
networks.

`aggregator_provider_optional_dependency_matrix()` joins the
aggregator/provider route matrix to the same dependency evidence. This is the
preflight surface for broker/provider combinations: it reports the route id,
resolved HAL backend, SDK package, import names, supported IR formats, approval
gate, and whether the row is a dynamic catalogue target.

`probe_aggregator_provider_capability()` is the next no-submit layer. It accepts
an injected provider metadata probe, resolves the selected aggregator/provider
route, rejects route mismatches and non-no-submit snapshots, and returns a
ready/blocked/unknown decision based on online status, qubit count, and required
IR support. The function does not create clients or submit jobs; provider SDK
authentication remains inside the injected read-only probe.
`snapshot_from_braket_device()`, `snapshot_from_qiskit_runtime_backend()`,
`snapshot_from_qbraid_device()`, and `snapshot_from_strangeworks_backend()` are
concrete metadata adapters for this contract. They consume injected SDK
backend/device objects and record target name, qubit count, route-supported or
declared IR formats, gate basis, queue depth, shot/circuit limits, online state,
simulator state, and calibration timestamp when the provider object exposes
those fields.

The same check is exposed as `scpn-provider-smoke`. In CI or operator
preflight lanes, install `scpn-quantum-control[providers]` and run:

```bash
scpn-provider-smoke --format table
scpn-provider-smoke --format json --sdk-package qiskit-ibm-runtime --require-all
scpn-provider-smoke --aggregator-routes --aggregator qbraid --provider rigetti --ir-format quil --format json
```

The portable `providers` extra intentionally excludes the current D-Wave, IQM,
and QuEra direct SDK extras because their dependency trees are not compatible
with the shared development/application environment as one aggregate install.
Use `[dwave]`, `[iqm]`, or `[quera]` in isolated runner environments when those
direct routes are being exercised.

`isolated_provider_smoke_lanes()` and `scpn-provider-smoke --plan-isolated`
emit deterministic runner commands for those conflict-prone SDK families. Each
lane creates a provider-specific virtual environment, installs only the matching
extra, and runs a filtered no-network smoke gate:

```bash
scpn-provider-smoke --plan-isolated --format table

python -m venv .venv-provider-iqm
.venv-provider-iqm/bin/python -m pip install -U pip
.venv-provider-iqm/bin/python -m pip install -e ".[iqm]"
.venv-provider-iqm/bin/scpn-provider-smoke --backend iqm_cloud --require-all
```

The same isolated lanes are available as the manual GitHub Actions
workflow `Provider Isolated Smoke`. It is deliberately separate from the
blocking CI gate and performs offline import checks only; it does not read
credentials, authenticate, create provider clients, or submit jobs.

The direct Quandela adapter layer provides `QuandelaPercevalHALAdapter` and
`quandela_perceval_workload()`. It consumes `scpn.quandela.perceval.v1`
photonic circuit plans, validates mode count, input occupations,
beam-splitter and phase-shifter components, and postselection bounds, then
executes through an injected Perceval processor or processor/sampler factory.
Remote Quandela execution remains approval-gated and automatic processor
construction is calibration-gated.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    QuandelaPercevalHALAdapter(
        hal.profile("quandela_cloud"),
        processor_factory=calibrated_quandela_processor_factory,
        sampler_factory=calibrated_perceval_sampler_factory,
        target="ascella",
    )
)
job = hal.submit(
    "quandela_cloud",
    quandela_perceval_workload(
        photonic_plan,
        workload_id="quandela_pair",
        n_modes=2,
        shots=256,
    ),
    approval_id="approved-run",
)
```

The direct D-Wave Leap adapter layer provides `DWaveLeapHALAdapter` and
`dwave_bqm_workload()`. It consumes a `scpn.dwave.bqm.v1` binary quadratic
model payload, validates variable order, linear and quadratic biases, vartype,
and read count, submits through an injected sampler or the D-Wave system SDK,
normalises sample-set occurrences into HAL counts, and keeps Leap execution
approval-gated.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    DWaveLeapHALAdapter(
        hal.profile("dwave_leap"),
        sampler_factory=calibrated_dwave_sampler_factory,
        solver="Advantage_system",
    )
)
job = hal.submit(
    "dwave_leap",
    dwave_bqm_workload(
        linear={"0": -1.0, "1": 0.5},
        quadratic={("0", "1"): -0.25},
        workload_id="dwave_pair",
        n_variables=2,
        reads=256,
    ),
    approval_id="approved-run",
)
```

The direct IQM adapter layer provides `IQMHALAdapter` and
`iqm_qiskit_workload()`. It follows the IQM Qiskit provider path lazily,
accepts injected backend objects for tests or calibrated execution routes,
encodes circuits as QPY-backed `qiskit_qpy` workloads, normalises job status
and counts into HAL payloads, and keeps remote execution approval-gated.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    IQMHALAdapter(
        hal.profile("iqm_cloud"),
        server_url=iqm_server_url,
        quantum_computer="garnet",
    )
)
job = hal.submit(
    "iqm_cloud",
    iqm_qiskit_workload(qiskit_circuit, workload_id="iqm_bell", shots=256),
    approval_id="approved-run",
)
```

The direct Pasqal adapter layer provides `PasqalPulserHALAdapter` and
`pulser_sequence_workload()`. It consumes `pulser_sequence_plan_v1`, validates
register coordinates, Rabi envelope points, local detunings, interaction terms,
and FIM feedback terms, submits through an injected Pasqal client or client
factory, normalises result counters into HAL counts, and keeps remote execution
approval-gated.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    PasqalPulserHALAdapter(
        hal.profile("pasqal_cloud"),
        client_factory=calibrated_pasqal_client_factory,
        target="FRESNEL",
    )
)
job = hal.submit(
    "pasqal_cloud",
    pulser_sequence_workload(pulser_plan, workload_id="pasqal_pair", n_qubits=2, shots=256),
    approval_id="approved-run",
)
```

The direct QuEra/Bloqade adapter layer provides `QuEraBloqadeHALAdapter` and
`bloqade_ahs_workload()`. It consumes the repository's `bloqade_ahs_plan_v1`
neutral-atom export schema, validates atom geometry and piecewise schedules,
runs an injected Bloqade local or remote routine with `run(shots=..., name=...)`,
normalises `fetch()`/`report()` bitstrings or count mappings into HAL counts,
and cancels batches that expose `cancel()`. Automatic provider-object
construction remains calibration-gated; production callers inject the calibrated
Bloqade routine or a routine factory.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    QuEraBloqadeHALAdapter(
        hal.profile("quera_bloqade"),
        routine=calibrated_bloqade_routine,
        routine_name="aquila-approved-route",
    )
)
job = hal.submit(
    "quera_bloqade",
    bloqade_ahs_workload(
        bloqade_ahs_plan,
        workload_id="quera_rydberg_pair",
        n_qubits=2,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The direct Quantinuum adapter layer provides `QuantinuumCloudHALAdapter` and
`quantinuum_tket_workload()`. It keeps pytket and pytket-quantinuum loading lazy
until a concrete adapter is registered, then follows the documented Quantinuum
route: `get_compiled_circuit(...)`, `process_circuit(..., n_shots=...)`,
`circuit_status(...)`, `get_result(...).get_counts()`, and
`QuantinuumBackend.cancel(...)`. Direct Quantinuum execution is tket-native;
OpenQASM 3, QIR, and MLIR route entries are registry-level translation targets
and must be converted before submission.

```python
quantinuum_machine = "H1-1E"
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    QuantinuumCloudHALAdapter(
        hal.profile("quantinuum_cloud"),
        machine=quantinuum_machine,
    )
)
job = hal.submit(
    "quantinuum_cloud",
    quantinuum_tket_workload(
        {"name": "h_sample", "qubits": 1},
        workload_id="quantinuum_h",
        n_qubits=1,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The direct Rigetti adapter layer provides `RigettiQCSHALAdapter` and
`rigetti_quil_workload()`. It keeps pyQuil and QCS loading lazy until a concrete
adapter is registered, then follows the documented pyQuil route: `Program`,
`wrap_in_numshots_loop(...)`, `QuantumComputer.compile(...)`,
`QuantumComputer.run(...)`, and `ro` register readout extraction. Direct
Rigetti execution is Quil-native; OpenQASM 3 and MLIR route entries are
registry-level translation targets and must be converted before submission.

```python
rigetti_qc_name = "9q-square-qvm"
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    RigettiQCSHALAdapter(
        hal.profile("rigetti_qcs"),
        quantum_computer_name=rigetti_qc_name,
    )
)
job = hal.submit(
    "rigetti_qcs",
    rigetti_quil_workload(
        "DECLARE ro BIT[1]\nH 0\nMEASURE 0 ro[0]",
        workload_id="rigetti_h",
        n_qubits=1,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The PennyLane adapter layer provides `PennyLaneDeviceHALAdapter` and
`pennylane_gate_workload()`. It executes a strict SCPN native-gate payload on a
local PennyLane device such as `default.qubit`; unsupported gate names, invalid
wire references, wrong arity, and malformed JSON are rejected before execution.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(PennyLaneDeviceHALAdapter(hal.profile("local_pennylane")))
result = hal.result(
    hal.submit(
        "local_pennylane",
        pennylane_gate_workload(
            [{"gate": "h", "wires": [0]}, {"gate": "cnot", "wires": [0, 1]}],
            workload_id="pl_bell",
            n_qubits=2,
            shots=256,
        ),
    )
)
```

The qBraid adapter layer provides `QbraidRuntimeHALAdapter` and
`qbraid_program_to_workload()`. It accepts injected qBraid runtime devices or
providers, supports provider lookup by qBraid device id, forwards the exact HAL
program payload to `device.run(...)`, and converts qBraid measurement counts
back into `QuantumJobResult`. Use `qbraid_ionq` for the named IonQ broker route
or `qbraid_runtime` for qBraid's dynamic provider catalog. Cloud submission
remains approval-gated by HAL. For pre-submit capability checks, use
`snapshot_from_qbraid_device()` with an authenticated qBraid device object; the
snapshot path reads metadata only and fails closed when the target does not
declare supported IR formats.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(QbraidRuntimeHALAdapter(hal.profile("qbraid_ionq"), device=qbraid_device))
job = hal.submit(
    "qbraid_ionq",
    qbraid_program_to_workload(
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nx q[0];",
        workload_id="qbraid_x",
        ir_format="openqasm3",
        n_qubits=1,
        shots=128,
    ),
    approval_id="approved-run",
)
```

The Strangeworks adapter layer provides `StrangeworksComputeHALAdapter` and
`strangeworks_program_to_workload()`. It mirrors the qBraid dynamic-catalog
pattern: production callers inject an authenticated Strangeworks backend,
workspace, or factory; the adapter forwards the HAL program payload to the
selected backend, records the resolved backend id, and normalises measurement
counts into `QuantumJobResult`. For pre-submit capability checks, use
`snapshot_from_strangeworks_backend()` with an injected Strangeworks backend; it
reads the backend metadata surface and feeds the route-bound no-submit readiness
decision before any workload path is eligible.

```python
hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(
    StrangeworksComputeHALAdapter(
        hal.profile("strangeworks_compute"),
        workspace=strangeworks_workspace,
        backend_id="rigetti.qvm",
    )
)
job = hal.submit(
    "strangeworks_compute",
    strangeworks_program_to_workload(
        "DECLARE ro BIT[2]",
        workload_id="sw_rigetti",
        ir_format="quil",
        n_qubits=2,
        shots=128,
    ),
    approval_id="approved-run",
)
```

---

## Comparison

| Feature | Backend Dispatch | Plugin Registry | HAL | TensorCircuit |
|---------|-----------------|-----------------|-----|---------------|
| Array backend switching | Yes | No | No | Yes |
| Hardware backend registry | No | Yes | Yes | No |
| Provider-neutral execution contract | No | No | Yes | No |
| Cloud approval gate | No | Partial | Yes | No |
| Custom backends | No | Yes (decorator) | Yes (protocol adapter) | No |
| Lazy loading | N/A | Yes | Metadata-only | No |
| JAX support | Yes | Via backends | Via adapter | Yes |
| PyTorch support | Yes | Via backends | Via adapter | Yes |

---

## References

1. Zhang, S.-X. *et al.* "TensorCircuit: An open-source cloud-oriented
   quantum computing platform." arXiv:2205.10091 (2022).
2. McClean, J. R. *et al.* "OpenFermion: The electronic structure package
   for quantum computers." *Quantum Sci. Technol.* **5**, 034014 (2020).

---

## See Also

- [Backend Selector](backend_selector.md) — auto-select based on system size
- [Multi-Platform Export](multi_platform.md) — circuit export to QASM/Quil/Cirq
- [GPU Batch VQE](gpu.md) — PyTorch GPU acceleration
