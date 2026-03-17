# Bridges API Reference

Cross-repo integration adapters for sc-neurocore, SSGF, scpn-phase-orchestrator, and scpn-fusion-core.

## SNN Adapter (`bridge.snn_adapter`)

### `ArcaneNeuronBridge`

Bidirectional bridge between sc-neurocore `ArcaneNeuron` and quantum layer.

```python
from scpn_quantum_control.bridge.snn_adapter import ArcaneNeuronBridge

bridge = ArcaneNeuronBridge(n_neurons=2, n_inputs=3, seed=42)
result = bridge.step(np.array([1.0, 0.5, 0.0]))
# result["spikes"]: binary spike vector
# result["output_currents"]: quantum layer output
# result["v_deep"]: identity state (persists across reset)
# result["confidence"]: neuron confidence
```

Requires: `pip install sc-neurocore`

| Method | Returns |
|--------|---------|
| `step(currents)` | dict with spikes, output_currents, v_deep, confidence |
| `step_neurons(currents)` | binary spike vector |
| `quantum_forward()` | output currents from quantum layer |
| `reset()` | reset neurons (v_deep persists) |

### `SNNQuantumBridge`

Pure-numpy bridge (no sc-neurocore dependency).

```python
bridge = SNNQuantumBridge(n_neurons=2, n_inputs=3, seed=42)
out = bridge.forward(spike_history)  # (timesteps, n_inputs) -> (n_neurons,)
```

### Standalone Functions

```python
spike_train_to_rotations(spikes, window=10) -> np.ndarray  # firing rate * pi
quantum_measurement_to_current(probs, scale=1.0) -> np.ndarray
```

## SSGF Adapter (`bridge.ssgf_adapter`)

### `SSGFQuantumLoop`

Quantum-in-the-loop wrapper for SSGFEngine. Each step reads W and theta,
runs Trotter evolution on statevector, writes theta back.

```python
from scpn_quantum_control.bridge.ssgf_adapter import SSGFQuantumLoop

loop = SSGFQuantumLoop(engine, dt=0.1, trotter_reps=3)
result = loop.quantum_step()
# result["theta"]: updated phases
# result["R_global"]: order parameter
```

Requires: SCPN-CODEBASE `optimizations.ssgf.SSGFEngine` on sys.path.

### Standalone Functions

```python
ssgf_w_to_hamiltonian(W, omega) -> SparsePauliOp
ssgf_state_to_quantum({"theta": [...]}) -> QuantumCircuit
quantum_to_ssgf_state(statevector, n_osc) -> {"theta": [...], "R_global": float}
```

## Orchestrator Phase Mapping (`identity.binding_spec`)

Bidirectional mapping between 18 quantum oscillators and 35 orchestrator
domainpack oscillators.

```python
from scpn_quantum_control.identity.binding_spec import (
    quantum_to_orchestrator_phases,
    orchestrator_to_quantum_phases,
    ORCHESTRATOR_MAPPING,
)

# Quantum (18) -> Orchestrator (35): broadcast
orch = quantum_to_orchestrator_phases(theta_18)

# Orchestrator (35) -> Quantum (18): circular mean
theta = orchestrator_to_quantum_phases(orch_dict)
```

## Fusion-Core Disruption Adapter (`control.q_disruption_iter`)

### `from_fusion_core_shot`

Converts scpn-fusion-core NPZ archive shot data to ITER 11-feature vector.

```python
from scpn_quantum_control.control.q_disruption_iter import from_fusion_core_shot

shot = np.load("disruption_shot_001.npz", allow_pickle=True)
features, label = from_fusion_core_shot(dict(shot))
# features: (11,) normalized to [0, 1]
# label: 0 (safe) or 1 (disruption)
```

Supported keys: `Ip_MA`, `q95`, `ne_1e19`, `beta_N`, `locked_mode_amp`.
Missing keys default to ITER operational point centers.
