# Mitigation API Reference

Error mitigation modules for noise reduction on NISQ hardware.

## Zero-Noise Extrapolation (`mitigation.zne`)

### `ZNEResult`

```python
@dataclass
class ZNEResult:
    noise_scales: list[int]
    expectation_values: list[float]
    zero_noise_estimate: float
    fit_residual: float
```

Fields:

| Field | Type | Description |
|-------|------|-------------|
| `noise_scales` | `list[int]` | Odd integers used as folding factors (e.g. `[1, 3, 5]`) |
| `expectation_values` | `list[float]` | Measured expectation at each noise scale |
| `zero_noise_estimate` | `float` | Richardson-extrapolated value at zero noise |
| `fit_residual` | `float` | L2 residual of the polynomial fit |

### `gate_fold_circuit`

```python
gate_fold_circuit(circuit: QuantumCircuit, scale: int) -> QuantumCircuit
```

Global unitary folding: `G → G (G†G)^((scale-1)/2)`.

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `circuit` | `QuantumCircuit` | Input circuit (measurements stripped before folding, re-appended) |
| `scale` | `int` | Odd positive integer folding factor (1 = no folding, 3 = one fold, etc.) |

**Returns:** New `QuantumCircuit` with folded unitaries and measurements re-appended.

**Raises:** `ValueError` if `scale` is not a positive odd integer.

### `zne_extrapolate`

```python
zne_extrapolate(noise_scales: list[int], expectation_values: list[float], order: int = 1) -> ZNEResult
```

Richardson extrapolation to zero noise.

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `noise_scales` | `list[int]` | — | Noise amplification factors (must match `expectation_values` length) |
| `expectation_values` | `list[float]` | — | Measured expectations at each scale |
| `order` | `int` | `1` | Polynomial degree: 1 = linear, 2 = quadratic |

**Returns:** `ZNEResult` with extrapolated zero-noise estimate and fit residual.

## Dynamical Decoupling (`mitigation.dd`)

### `DDSequence`

```python
class DDSequence(Enum):
    XY4 = "XY4"
    X2 = "X2"
    CPMG = "CPMG"
```

Supported pulse sequences:

| Sequence | Gates | Reference |
|----------|-------|-----------|
| `XY4` | X-Y-X-Y | Viola et al. 1999 |
| `X2` | X-X | Spin echo |
| `CPMG` | Y-Y | Carr-Purcell-Meiboom-Gill |

### `insert_dd_sequence`

```python
insert_dd_sequence(
    circuit: QuantumCircuit,
    idle_qubits: list[int],
    sequence: DDSequence = DDSequence.XY4,
) -> QuantumCircuit
```

Insert DD pulses on idle qubits after existing gates.

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `circuit` | `QuantumCircuit` | — | Circuit with idle periods to fill |
| `idle_qubits` | `list[int]` | — | Qubit indices to apply DD pulses |
| `sequence` | `DDSequence` | `DDSequence.XY4` | Pulse sequence type |

**Returns:** New `QuantumCircuit` with DD pulses inserted.

For transpiler-level insertion on hardware, use `HardwareRunner.transpile_with_dd()` which integrates with Qiskit's `PadDynamicalDecoupling` pass.
