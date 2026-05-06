# Analog Kuramoto Backends

SPDX-License-Identifier: AGPL-3.0-or-later

The analog Kuramoto backend interface compiles a validated `K_nm, omega`
problem into native programme schemas for three hardware families:

- Neutral-atom arrays using Rydberg interaction geometry.
- Circuit-QED resonator networks using tunable exchange couplers.
- Continuous-variable modes using phase rotations and beam-splitter terms.

This layer is a compiler interface, not a cloud submission client. Provider
adapters can consume the emitted dictionaries and translate them into concrete
SDK objects while preserving the same coupling, detuning, sign, and duration
metadata.

## Usage

```python
import numpy as np

from scpn_quantum_control.hardware import compile_analog_kuramoto

K = np.array([[0.0, 0.5, -0.25], [0.5, 0.0, 0.125], [-0.25, 0.125, 0.0]])
omega = np.array([0.1, -0.2, 0.3])

program = compile_analog_kuramoto(
    K,
    omega,
    platform="circuit_qed",
    duration=1.25,
)
```

## Provider-specific exports

The generic programme can be exported into provider-specific design payloads
without submitting a job:

```python
from scpn_quantum_control.hardware import export_provider_payload

neutral_program = compile_analog_kuramoto(
    K,
    omega,
    platform="neutral_atoms",
    duration=1.25,
)

pulser_plan = export_provider_payload(neutral_program, "pulser")
bloqade_plan = export_provider_payload(neutral_program, "bloqade")
```

For circuit-QED pulse-level design studies:

```python
circuit_qed_program = compile_analog_kuramoto(
    K,
    omega,
    platform="circuit_qed",
    duration=1.25,
)

ibm_pulse_plan = export_provider_payload(circuit_qed_program, "ibm_pulse")
```

Each `ProviderAnalogPayload` records the provider target, required platform,
SDK module name, local SDK availability, serialisable payload, and limitations.
`can_submit` is always `False` in this layer. Provider exports are design
artefacts and must not be reported as analogue hardware execution.

The returned `AnalogKuramotoProgram` contains:

- `coupling_terms`: upper-triangular native couplings with magnitude and phase.
- `drive_terms`: per-oscillator detunings from `omega`.
- `feedback_terms`: optional FIM feedback terms for
  `H_FIM(lambda) = -lambda M^2 / n`.
- `payload`: platform-specific serialisable programme schema.
- `metadata`: oscillator count, coupler count, native term, scaling, and user
  metadata.

## FIM feedback compilation

The compiler can include the static collective term used in the SCPN/FIM
Hamiltonian paper:

```python
program = compile_analog_kuramoto(
    K,
    omega,
    platform="circuit_qed",
    duration=1.25,
    lambda_fim=4.0,
)
```

The Hamiltonian identity is:

```text
-lambda M^2 / n = -lambda I - (2 lambda / n) sum_{i<j} Z_i Z_j.
```

The global shift is recorded as `fim_global_energy_shift`; the pairwise
feedback terms are emitted as `feedback_terms` and mirrored into the platform
payload. This is a native-programme schema for follow-up platform work, not
evidence that a specific cloud provider accepts or faithfully executes the
term.

## Platform Schemas

Neutral atoms emit `native_ahs_v1` payloads with:

- square-grid register coordinates,
- local detunings,
- Rydberg interaction terms with equivalent radii,
- optional FIM feedback terms for an Ising-style collective interaction,
- a three-point global Rabi envelope.

Circuit-QED emits `exchange_resonator_v1` payloads with:

- mode-frequency detuning terms,
- flat-top tunable exchange couplers,
- sign encoded as phase `0` or `pi`.
- optional cross-Kerr-style FIM feedback terms.

Continuous-variable platforms emit `cv_gaussian_schedule_v1` payloads with:

- phase rotations for natural frequencies,
- beam-splitter operations for non-zero couplings,
- coupling sign encoded in the beam-splitter phase.
- optional number-feedback terms for the collective FIM proposal.

## Provider Export Schemas

Pulser exports require `neutral_atoms` programmes and emit
`pulser_sequence_plan_v1` with:

- register coordinates keyed by site,
- `rydberg_global` channel metadata,
- Rabi envelope points,
- local detunings,
- Rydberg interaction terms,
- optional FIM feedback design terms.

Bloqade exports require `neutral_atoms` programmes and emit
`bloqade_ahs_plan_v1` with:

- atom positions,
- piecewise-linear Rabi amplitude and phase schedules,
- local detunings,
- Rydberg interaction terms,
- optional FIM feedback design terms.

IBM pulse-level exports require `circuit_qed` programmes and emit
`qiskit_pulse_schedule_plan_v1` with:

- mode-frequency detuning terms,
- exchange-coupler channel plans,
- optional cross-Kerr-style FIM feedback design terms.

These provider payloads intentionally stop before SDK object construction or
cloud execution. A future executable adapter must add provider-unit calibration,
backend capability checks, and an explicit submission approval gate.

## Registry Integration

The backend is registered as `analog_kuramoto` in the
`scpn_quantum_control.backends` entry-point group and is pre-registered for
source checkouts:

```python
from scpn_quantum_control.hardware import get_backend

backend = get_backend("analog_kuramoto")
assert backend.is_available()
```

The stable Kuramoto facade also exposes `compile_analog_program(problem, ...)`
for callers that already use `KuramotoProblem`.
