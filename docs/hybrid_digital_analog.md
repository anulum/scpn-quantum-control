# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hybrid Digital-Analog Execution

# Hybrid Digital-Analog Execution

The hybrid compiler splits a validated Kuramoto-XY workload into two
coherent execution blocks:

- an analog-native block for couplings that can be realised directly on a
  neutral-atom, circuit-QED, or continuous-variable platform;
- a digital residual block for the remaining couplings, compiled as a
  Trotter circuit.

Local detunings are assigned to the analog block. The digital residual uses
zero local detunings so the same single-oscillator terms are not counted
twice when a provider adapter executes the schedule.

## Usage

```python
import numpy as np

from scpn_quantum_control.hardware.hybrid_digital_analog import (
    compile_hybrid_digital_analog,
)

K_nm = np.array(
    [
        [0.0, 0.8, -0.4, 0.1],
        [0.8, 0.0, 0.3, 0.0],
        [-0.4, 0.3, 0.0, -0.2],
        [0.1, 0.0, -0.2, 0.0],
    ],
    dtype=np.float64,
)
omega = np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float64)

program = compile_hybrid_digital_analog(
    K_nm,
    omega,
    platform="circuit_qed",
    duration=1.25,
    max_analog_couplers=2,
    trotter_steps=3,
)

print(program.n_analog_couplers, program.n_digital_couplers)
print(program.payload["schema"])
```

The same compiler is available through the public Kuramoto facade:

```python
from scpn_quantum_control.kuramoto_core import (
    build_kuramoto_problem,
    compile_hybrid_program,
)

problem = build_kuramoto_problem(K_nm, omega, metadata={"case": "demo"})
program = compile_hybrid_program(
    problem,
    platform="neutral_atoms",
    duration=0.75,
    analog_threshold=0.35,
)
```

## Coupling Split

The splitter treats the upper triangle of `K_nm` as the source of coupling
terms. Couplings with `abs(K_ij) <= zero_threshold` are ignored. Eligible
analog couplings are ranked by descending `abs(K_ij)` and then by `(i, j)`
for deterministic tie-breaking. `max_analog_couplers` limits how many of
those couplings are routed to analog execution.

`analog_threshold` can be used when a platform should only receive large
native terms. Couplings below that threshold remain in the digital residual
even when analog capacity is available.

The Rust extension exposes the same deterministic partitioner as
`scpn_quantum_engine.hybrid_coupling_partition`. The Python implementation
falls back to a NumPy path when the extension is unavailable.

## Payload

The top-level payload schema is `hybrid_digital_analog_v1`:

```python
{
    "schema": "hybrid_digital_analog_v1",
    "platform": "circuit_qed",
    "duration": 1.25,
    "digital_time": 1.25,
    "partition": {
        "n_couplings": 5,
        "n_analog_couplings": 2,
        "n_digital_couplings": 3,
        "assignments": [
            {"source": 0, "target": 1, "coefficient": 0.8, "route": "analog"},
        ],
    },
    "schedule": [
        {"route": "analog_native", "payload": {"schema": "exchange_resonator_v1"}},
        {"route": "digital_residual", "payload": {"schema": "digital_residual_qasm2_v1"}},
    ],
}
```

The analog schedule item embeds the native analog payload described in the
analog backend documentation. The digital schedule item contains OpenQASM 2
for the residual Trotter circuit, plus depth, size, Trotter-step, and
residual-presence metadata.

## Backend Registry

The built-in backend registry exposes the compiler as
`hybrid_digital_analog`:

```python
from scpn_quantum_control.hardware import get_backend

backend = get_backend("hybrid_digital_analog")
program = backend.compile(problem, duration=1.0, max_analog_couplers=1)
```

Third-party provider adapters can consume `HybridDigitalAnalogProgram` and
translate the two schedule blocks into provider-native analog controls and
digital circuit submissions.
