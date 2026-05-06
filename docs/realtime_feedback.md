# Real-Time Feedback

SPDX-License-Identifier: AGPL-3.0-or-later

The real-time feedback controller closes the Kuramoto-XY loop in software
while keeping the circuit shape compatible with dynamic-circuit execution.
It is not a hardware latency claim: the shipped path is a deterministic
statevector/live-shot simulator plus an exportable monitored circuit template.
Python execution is suitable for simulation, batch orchestration, and
across-shot policy updates. Sub-microsecond or within-coherence-window feedback
must be implemented with provider-native dynamic circuits, OpenQASM 3 control
flow, pulse-level control, FPGA logic, or an equivalent hardware controller.

## Control Surface

`RealtimeSyncFeedbackController` maintains the current state, evolves it with
the scaled Kuramoto-XY Hamiltonian, samples finite-shot X/Y observables, and
updates the next coupling scale from the measured order parameter. The policy
uses a Rust PyO3 kernel when `scpn_quantum_engine.feedback_policy_batch` is
available and falls back to the identical NumPy implementation otherwise.

```python
import numpy as np

from scpn_quantum_control.control import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
)

K = np.array([[0.0, 0.35], [0.35, 0.0]])
omega = np.array([0.1, 0.6])
config = RealtimeFeedbackConfig(target_r=0.72, measurement_shots=128)

controller = RealtimeSyncFeedbackController(K, omega, config=config)
history = controller.run(4, seed=42)
```

Each `FeedbackStep` records the finite-shot order parameter, the exact
statevector order parameter, controller action, applied coupling scale, next
coupling scale, phase-correction angle, and sampled readout counts.

## Dynamic Circuit Template

`build_monitored_feedback_circuit` emits a monitored circuit with one ancilla,
one monitor classical bit per round, and final system readout. Every round
contains:

- Kuramoto-XY Trotter evolution for the current schedule point.
- A monitor-ancilla interaction with the system register.
- Mid-circuit monitor measurement.
- Conditional reset of the monitor ancilla when the monitor bit is one.
- Conditional system correction driven by the same monitor bit.

```python
from scpn_quantum_control.control import build_monitored_feedback_circuit

qc = build_monitored_feedback_circuit(K, omega, n_rounds=3)
```

The dynamic circuit is suitable for provider capability checks and QASM export
where conditional blocks are supported. For current benchmarks, controller
latency is measured locally around the simulator path rather than inferred from
cloud queue or backend timing.
