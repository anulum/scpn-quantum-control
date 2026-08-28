# Analog Oscillator Mapping Feasibility

The analog-mapping feasibility contract provides a fail-closed answer to a
deliberately narrow question:

> Can the existing internal analog compiler represent this coupling matrix,
> topology, detuning, and measurement contract in its mathematical design
> model, and what current public documentation prevents promoting that result
> to a hardware claim?

It does not submit a job, construct a provider SDK object, inspect credentials,
or claim that a physical device implements Kuramoto dynamics.

## Evidence levels

The product separates three levels that must not be conflated:

1. `internal_compiler_model` means the local compiler reconstructs the requested
   design-unit coupling matrix. It is executable local software evidence only.
2. `capability_sketch` means a dated official source describes relevant device
   or SDK capabilities. It is not a device ticket, calibration, or executable
   mapping.
3. Hardware support remains blocked until a separately approved provider run
   supplies a current device descriptor, calibration, observable mapping, raw
   result artifact, and comparison against the same digital reference.

Every report fixes these flags to `False`:

- `hardware_submission_allowed`
- `hardware_support_claim_allowed`
- `analog_advantage_claim_allowed`

## Public surface

The `scpn_quantum_control.analog_mapping` package exposes:

- immutable `MappingRequest`, `AnalogPlatformProfile`, `MappingResult`, and
  `FeasibilityReport` contracts;
- a validated static JSON profile catalogue with source URL and verification
  timestamp;
- topology classification for ring, all-to-all, and sparse graphs;
- fail-closed topology, sign, range, local-detuning, measurement, and profile
  posture diagnostics;
- a bounded dense `N <= 6` mathematical XY-model comparison against a
  first-order Lie-Trotter reference;
- an analytic coupling-scale mean-square objective, exact derivative, and
  symmetric parameter-drift sensitivity;
- deterministic JSON evidence and Markdown rendering.

## Example

```python
import numpy as np

from scpn_quantum_control.analog_mapping import (
    MappingRequest,
    build_analog_mapping_evidence,
    write_analog_mapping_evidence,
)

k_nm = np.array(
    [
        [0.0, 0.30, 0.0, -0.20],
        [0.30, 0.0, 0.25, 0.0],
        [0.0, 0.25, 0.0, 0.15],
        [-0.20, 0.0, 0.15, 0.0],
    ]
)
omega = np.array([0.10, -0.15, 0.20, -0.05])

request = MappingRequest.from_arrays(
    k_nm,
    omega,
    topology="ring",
    measurement="phase_proxy",
    duration=0.2,
    coupling_scale=1.25,
    comparison_tolerance=5e-3,
)
bundle = build_analog_mapping_evidence(
    request,
    "scpn_circuit_qed_design_v1",
    trotter_steps=32,
)
write_analog_mapping_evidence("results/analog_mapping_evidence.json", bundle)
```

`bundle.report.supported` refers only to the internal compiler model. The three
hardware and advantage flags remain false even when that value is true.

## Static profile catalogue

The packaged `platform_profiles.v1.json` catalogue is data, not a provider
driver. These are the current rows:

| Profile | Posture | Source-grounded boundary |
| --- | --- | --- |
| `scpn_circuit_qed_design_v1` | internal compiler model | Local exchange-term schema; no calibrated device coupler map |
| `pulser_analogdevice_sketch_2026_07` | capability sketch | Pulser documents register, Rydberg-channel, geometry, duration, and run constraints; arbitrary signed pairwise Kuramoto control is not established |
| `ionq_native_gate_sketch_2026_07` | unsupported | IonQ documents native gate circuits (`GPI`, `GPI2`, `MS`, or `ZZ`), not a continuous-time oscillator-control interface |
| `iqm_native_gate_sketch_2026_07` | unsupported | IQM client documentation describes native gates, device connectivity, decomposition, and routing, not arbitrary continuous-time Kuramoto control |
| `ibm_fractional_gate_sketch_2026_07` | unsupported | Qiskit 2 removed `qiskit.pulse`; IBM pulse-level access cannot be treated as a current execution route |

The hardware-readiness ledger remains authoritative for access and ticket
status. The catalogue only links to it; it does not copy or promote ledger
state.

## Feasibility decision

`assess_mapping_feasibility` first compares the declared and observed topology,
then checks node capacity, coupling signs and bounds, local detuning,
measurement, arbitrary pairwise control, evidence posture, and ledger linkage.
Any blocker produces `supported=False` and no `MappingResult`.

Only the internal compiler-model profile can reach compilation. Its signed
coupling terms are reconstructed from magnitude and phase, compared to the
requested scaled matrix, and bound to a deterministic program digest. Passing
that check means parameter fidelity in the schema. It does not mean the schema
is physically realisable.

## Bounded analog/digital comparator

For `2 <= N <= 6`, the comparator builds the dense ideal XY Hamiltonian for the
requested matrix and the reconstructed compiler matrix. From a fixed
single-excitation state it records:

- compiler-model exact-state fidelity;
- first-order Lie-Trotter state fidelity at the declared step count;
- Trotter infidelity and compiler parameter RMSE;
- whether all model-space errors fit the declared tolerance.

This comparison contains no analog-device simulator, calibrated waveform,
noise model, provider measurement, wall-clock benchmark, or hardware result.
Its result cannot support an equivalence or advantage statement.

## Differentiable calibration objective

For native design matrix `K_native`, target matrix `K_target`, and scalar scale
`s`, the local objective is

```text
L(s) = mean_upper_triangle((s K_native - K_target)^2)
dL/ds = 2 mean_upper_triangle((s K_native - K_target) K_native)
```

The implementation returns the analytic derivative and evaluates symmetric
fractional drift around the nominal scale. This is differentiable design-unit
matrix fitting. It is not a pulse gradient, measured device response, or
closed-loop calibration.

## Relationship to S10 readiness

The earlier [analog-native readiness](analog_native_readiness.md) surface counts
native compiler primitives against a digital Trotter count and prepares
non-submitting export plans. The feasibility package adds the missing
request-level mappability contract, source-dated capability profiles,
fail-closed diagnostics, bounded model comparison, and calibration objective.
Neither surface establishes analog advantage.

## Current official source pins

- Pulser hardware specifications:
  <https://pulser.readthedocs.io/en/stable/hardware.html>
- IonQ native gate API:
  <https://docs.ionq.com/api-reference/v0.3/native-gates-api>
- IQM Cirq client device and native-gate guide:
  <https://docs.meetiqm.com/iqm-client/user_guide_cirq.html>
- IBM Qiskit 2 migration guide:
  <https://docs.quantum.ibm.com/migration-guides/qiskit-2.0>

The bundled verification timestamp is `2026-07-25T23:25:53Z`. Refresh the
catalogue from official sources before relying on a provider-facing decision.
