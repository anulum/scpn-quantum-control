# Decisive classical-run API

`scpn_quantum_control.benchmarks.decisive_run_harness` executes the classical
side of the preregistered decisive-advantage protocol. It assembles comparable
rows for exact statevector evolution, a classical phase ODE, and an optional
matrix-product-state evolution, then delegates validation and the final verdict
to the protocol owner.

The harness never creates a quantum result. Without a real QPU row the default
decision remains `inconclusive`, even when all classical rows are valid.

## Configuration and output records

`DecisiveRunConfig` bounds all execution controls:

- `t_max` and `dt` must be finite and positive, with `dt <= t_max`;
- `mps_bond_dim` must be positive;
- `reserved_core` must be a non-negative logical-core index; and
- `include_mps=False` produces a reasoned skipped row rather than deleting the
  required baseline from the evidence table.

`DecisiveRunArtifact` is an immutable, JSON-ready record containing the protocol
identifier, decision size, reference observable, timing grade, three classical
rows, delegated schema validation, fail-closed decision, provenance, host
readiness, claim boundary, and run metadata. `to_dict()` returns fresh mappings
and lists suitable for serialisation.

## Row builders

`dense_reference_row(...)` evolves the exact statevector, extracts final
synchronisation order parameter `R`, and records zero reference error. It also
records the exact ground energy supplied by the existing diagonalisation path.

`ode_row(...)` executes the classical phase ODE and compares its final `R` with
the dense reference. A missing final observable produces infinite error and
therefore cannot pass the accuracy gate.

`mps_row(...)` maps the tensor-network result into the same schema. Missing
tensor-network support becomes an explicit `skipped` row with its dependency or
size reason; an available run without a final observable cannot pass accuracy.

All rows carry the command, machine, dependency versions, exact Git commit when
available, wall-clock measurement, and a documented analytic memory estimate.

## Complete execution

```python
from scpn_quantum_control.benchmarks.decisive_run_harness import (
    DecisiveRunConfig,
    run_decisive_benchmark,
)

artifact = run_decisive_benchmark(
    config=DecisiveRunConfig(t_max=1.0, dt=0.1),
)
print(artifact.decision["label"])
print(artifact.timing_grade)
```

When no protocol is supplied, `run_decisive_benchmark()` uses the frozen
decisive-advantage protocol. When no host verdict is supplied, it captures the
live isolation state for `reserved_core`. Passing a pre-captured
`HostReadiness` is intended for controlled orchestration and deterministic
tests; it does not alter numerical results.

## Provenance helpers

`command_line()` records the current argument vector and uses `python` only for
an empty vector. `dependency_versions()` records Python plus the tracked
scientific packages, representing absent optional packages as `not installed`.
`git_commit()` resolves an actual Git executable and returns `unknown` when the
repository identity cannot be read; it never fabricates a commit.

## Evidence boundary

Wall-clock values are measurements of one execution environment. They are
labelled `isolated_measured` only when the host-readiness contract passes;
otherwise they are `advisory_shared_host`. Memory values are analytic models,
not process-resident-set measurements. Neither timing grade can establish
quantum advantage without a valid, matched-budget QPU row.

## API reference

::: scpn_quantum_control.benchmarks.decisive_run_harness
    options:
      show_root_heading: true
      members_order: source
