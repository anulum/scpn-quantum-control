# QEC Decoder Boundary

This page records the shipped quantum-error-correction decoder surfaces and the
decoder families that are intentionally not claimed.

## Shipped Decoders

### Toric Control Decoder

`scpn_quantum_control.qec.control_qec.MWPMDecoder` implements minimum-weight
perfect matching for the toric surface-code model used by `ControlQEC`.

Production path:

```python
import numpy as np

from scpn_quantum_control.qec import ControlQEC

qec = ControlQEC(distance=3)
err_x, err_z = qec.simulate_errors(0.005, rng=np.random.default_rng(7))
syn_z, syn_x = qec.get_syndrome(err_x, err_z)
success = qec.decode_and_correct(err_x, err_z)
```

The primal path decodes vertex syndromes for X-error correction. The dual path
decodes plaquette syndromes for Z-error correction. Residual syndromes and
non-trivial toric homology cycles are rejected by `decode_and_correct`.

Optional K_nm weighting rescales the toric Manhattan distance used by matching.
It is a weighting heuristic for the toric decoder, not a new decoder family.

### Biological Graph Decoder

`scpn_quantum_control.qec.biological_surface_code.BiologicalMWPMDecoder` maps
edge-local Z errors on a biological coupling graph to X-stabiliser syndromes and
matches graph defects by weighted shortest paths.

Production path:

```python
import numpy as np

from scpn_quantum_control.qec import BiologicalMWPMDecoder, BiologicalSurfaceCode

K = np.array(
    [
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ],
    dtype=float,
)
code = BiologicalSurfaceCode(K)
decoder = BiologicalMWPMDecoder(code)
z_errors = np.zeros(code.num_data, dtype=np.int8)
correction, residual = decoder.decode_and_apply(z_errors)
```

When the Rust extension is available, bounded defect sets use
`biological_decode_z_errors` for exact MWPM. Larger defect sets fall back to the
Python NetworkX path and expose the selected backend through
`last_decoder_backend`.

## Explicit Non-Claims

- No union-find decoder is implemented or exported.
- No MWPM decoder with explicit rough-boundary absorption is implemented for
  the biological graph code.
- No hardware syndrome-extraction controller, lattice-surgery planner, or
  provider-native QEC runtime is implemented.
- No claim is made that biological systems perform literal quantum error
  correction.

The biological graph decoder therefore fails closed when any connected
component has odd syndrome parity. That error means the current graph decoder
cannot match all defects without a boundary model; it is not silently converted
into a correction.

## Related Surfaces

- [`API Overview`](api.md#qec)
- [`Multi-Scale Quantum Error Correction`](multiscale_qec.md)
- [`DLA-Protected Logical Synchronisation`](dla_protected_subspace.md)
- [`Rust Engine`](rust_engine.md)
