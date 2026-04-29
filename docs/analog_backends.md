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

The returned `AnalogKuramotoProgram` contains:

- `coupling_terms`: upper-triangular native couplings with magnitude and phase.
- `drive_terms`: per-oscillator detunings from `omega`.
- `payload`: platform-specific serialisable programme schema.
- `metadata`: oscillator count, coupler count, native term, scaling, and user
  metadata.

## Platform Schemas

Neutral atoms emit `native_ahs_v1` payloads with:

- square-grid register coordinates,
- local detunings,
- Rydberg interaction terms with equivalent radii,
- a three-point global Rabi envelope.

Circuit-QED emits `exchange_resonator_v1` payloads with:

- mode-frequency detuning terms,
- flat-top tunable exchange couplers,
- sign encoded as phase `0` or `pi`.

Continuous-variable platforms emit `cv_gaussian_schedule_v1` payloads with:

- phase rotations for natural frequencies,
- beam-splitter operations for non-zero couplings,
- coupling sign encoded in the beam-splitter phase.

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
