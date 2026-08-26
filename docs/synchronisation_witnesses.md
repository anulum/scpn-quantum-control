# Synchronisation witnesses

`scpn_quantum_control.phase.synchronisation_witness` evaluates bounded phase
clouds with two complementary signals: harmonic Kuramoto order parameters and
exact Vietoris–Rips persistence over circular geodesic distances. The default
suite contains deterministic synchronised, uniformly desynchronised, and
three-cluster reference regimes.

## What the suite certifies

The first harmonic measures global phase alignment. The second harmonic can
reveal anti-phase or two-cluster structure that the first harmonic cancels.
Persistent homology supplies component counts and loop lifetimes across an
explicit filtration grid. Bootstrap perturbations estimate local uncertainty
in the first-harmonic order parameter.

The implementation performs exact boundary-matrix reduction over `GF(2)` for
small phase clouds. It supports homology dimensions zero and one. It is not an
accelerated persistence kernel or a general high-dimensional topology engine.

## Reference suite

```python
from scpn_quantum_control.phase import run_sync_witness_suite


suite = run_sync_witness_suite()
assert suite.passed
for record in suite.records:
    print(
        record.case_id,
        record.regime,
        record.order_parameter,
        record.persistent_component_count,
        record.dominant_h1_persistence,
    )
```

Every result carries an evidence classification and a claim boundary. Boundary
rows keep high-dimensional manifold inference and provider-backed hardware
phase tomography explicitly closed.

## Evidence artefacts

The evidence writer serialises the reference records, boundary rows,
environment metadata, schema, and promotion flags to matching JSON and
Markdown files:

```bash
PYTHONPATH=src:. python scripts/export_sync_witness_evidence.py
```

The committed examples are
`data/differentiable_phase_qnode/sync_witness_evidence_20260709.json` and
`.md`. They are deterministic local synthetic evidence. They do not establish
device performance, provider execution, phase tomography, isolated timing, or
production eligibility.

For exact signatures and fields, see the
[Synchronisation witness API](api/synchronisation_witnesses.md).
