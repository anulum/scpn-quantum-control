# Orchestrator Integration

This page documents how `scpn-quantum-control` now interoperates with
`scpn-phase-orchestrator` and fusion-specific Kuramoto/UPDE specifications.

## Why this exists

Fusion control needs **cause-specific** phase semantics:

- oscillator definitions (what each phase means physically)
- hierarchy layers
- coupling (`Knm`), phase lags (`alpha`), drivers (`zeta`, `Psi`)
- objectives (`R_good`, `R_bad`)
- boundary and regime behavior

Those semantics are owned by the orchestrator/domain spec layer.
`scpn-quantum-control` should execute quantum mappings from that spec, not
redefine the semantics independently.

## Non-collision policy (with `scpn-control`)

`scpn-control` can continue evolving its plasma Knm builder independently.
This repo does **not** override it.

Instead, we added:

- a shared phase artifact schema (`UPDEPhaseArtifact`)
- an orchestrator adapter (`PhaseOrchestratorAdapter`)
- parity tests that detect drift between implementations

This keeps ownership boundaries clear while enforcing consistency.

## scpn-control plasma Knm compatibility

`scpn-control` now provides a plasma-native Knm builder:

- `build_knm_plasma(...)`
- `build_knm_plasma_from_config(...)`
- `plasma_omega(...)`

`scpn-quantum-control` integrates this through
`scpn_quantum_control.bridge.control_plasma_knm` with lazy imports, so we
can consume the latest plasma coupling logic without taking a hard runtime
dependency.

Use `repo_src` when working in a multi-repo workspace:

```python
from pathlib import Path
from scpn_quantum_control.bridge import build_knm_plasma, plasma_omega

repo_src = Path(\"../scpn-control/src\")
K = build_knm_plasma(mode=\"ntm\", repo_src=repo_src)
omega = plasma_omega(L=8, repo_src=repo_src)
```

## New bridge components

### 1. Shared phase artifact schema

Module:
`scpn_quantum_control.bridge.phase_artifact`

Key types:

- `LockSignatureArtifact`
- `LayerStateArtifact`
- `UPDEPhaseArtifact`

These support validated `dict`/JSON roundtrips for backend-independent phase
state exchange.

### 2. Orchestrator adapter

Module:
`scpn_quantum_control.bridge.orchestrator_adapter`

Key entry points:

- `from_orchestrator_state(state)`:
  converts orchestrator payloads (dataclass or dict) into `UPDEPhaseArtifact`
- `to_scpn_control_telemetry(artifact)`:
  exports control-compatible telemetry layout
- `build_knm_from_binding_spec(binding_spec)`:
  derives `Knm` from orchestrator coupling contract
- `build_omega_from_binding_spec(binding_spec)`:
  derives per-oscillator `omega`

## Fusion-defined spec -> quantum execution flow

1. Define/validate fusion binding spec in `scpn-phase-orchestrator`.
2. Build `Knm`/`omega` from that spec using adapter functions.
3. Compile `Knm` to XY Hamiltonian via `knm_to_hamiltonian`.
4. Run quantum phase solver / VQE / hardware lanes.
5. Persist phase state as `UPDEPhaseArtifact`.

## Example

```python
from scpn_quantum_control.bridge import (
    PhaseOrchestratorAdapter,
    knm_to_hamiltonian,
)

binding_spec = {
    "layers": [
        {"name": "macro", "index": 0, "oscillator_ids": ["m0", "m1"], "natural_frequency": 1.4},
        {"name": "edge", "index": 1, "oscillator_ids": ["e0"]},
    ],
    "coupling": {"base_strength": 0.45, "decay_alpha": 0.3, "templates": {}},
}

K = PhaseOrchestratorAdapter.build_knm_from_binding_spec(binding_spec)
omega = PhaseOrchestratorAdapter.build_omega_from_binding_spec(binding_spec, default_omega=1.0)
H = knm_to_hamiltonian(K, omega)
```

## Drift guardrails

Parity tests now cover:

- quantum Knm parity vs `scpn-control` Paper-27 builder
- quantum plasma Knm parity vs `scpn-control` plasma-native builders
- orchestrator exponential-decay kernel parity
- quantum base-kernel invariance on untouched edges

See:
- `tests/test_knm_parity.py`
- `tests/test_phase_artifact.py`
- `tests/test_orchestrator_adapter.py`
