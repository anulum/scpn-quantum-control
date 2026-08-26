# Synchronisation witness API

Modules:

- `scpn_quantum_control.phase.synchronisation_witness`
- `scpn_quantum_control.benchmarks.sync_witness_evidence`

Importing either module performs no provider call or filesystem write.

## Numerical functions

- `harmonic_order_parameter(phases, *, harmonic=1) -> float` returns a Daido
  order-parameter magnitude.
- `geodesic_phase_distance_matrix(phases) -> numpy.ndarray` returns pairwise
  circular arc distances.
- `vietoris_rips_persistence(distance, *, max_dimension=1) -> dict` returns
  birth/death pairs for dimensions zero and, when requested, one.
- `betti_curve(persistence_pairs, thresholds) -> numpy.ndarray` counts classes
  alive at each threshold.
- `phase_cloud_synchronisation_witness(...) -> SyncWitnessRecord` combines the
  numerical measures and checks caller-supplied regime bounds.

All array inputs are validated for shape and finite values. Thresholds must be
non-negative and strictly increasing. Distance matrices must be square,
symmetric, non-negative, and zero-diagonal.

## Evidence records

`SyncWitnessCase` stores a deterministic input and its acceptance bounds.
`SyncWitnessRecord` stores the computed order parameters, uncertainty, Betti
curves, persistence diagrams, component count, dominant loop lifetime, and
verdict. `SyncWitnessBoundaryRow` records a deliberately unsupported route.
`SyncWitnessSuiteResult` aggregates records and exposes `passed`,
`records_for_regime()`, and `to_dict()`.

`default_sync_witness_cases()` returns the three reference regimes.
`run_sync_witness_suite(cases=None)` evaluates either those defaults or a
non-empty caller-supplied sequence.

## Artefact writer

`sync_witness_evidence_payload()` returns the JSON-oriented evidence object.
`render_sync_witness_evidence_markdown()` renders its record table and, when
present, its boundary table. `write_sync_witness_evidence_artifact()` writes
matching `.json` and `.md` destinations and returns
`SyncWitnessEvidenceArtifact` metadata.

## Full autodoc

::: scpn_quantum_control.phase.synchronisation_witness
    options:
      show_root_heading: false
      show_source: false
      members_order: source

::: scpn_quantum_control.benchmarks.sync_witness_evidence
    options:
      show_root_heading: false
      show_source: false
      members_order: source
