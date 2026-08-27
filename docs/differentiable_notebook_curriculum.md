# Differentiable notebook curriculum

Versioned **core-six onboarding curriculum** under `notebooks/differentiable/`
with mandatory `hardware_execution: false` honesty.

Module: `scpn_quantum_control.differentiable_notebook_curriculum`

This page documents a bounded catalogue and filesystem probe for the existing
core-six curriculum. It does not execute notebooks, install optional
frameworks, convert the historical archive, or certify the nbclient matrix.

## Contract discovery

| Function | Contract |
|---|---|
| `list_curriculum_notebook_ids()` | Returns all six stable ids in curriculum order. |
| `get_curriculum_notebook(notebook_id)` | Resolves one exact row; blank and unknown ids raise `ValueError`. |
| `iter_curriculum_notebooks(...)` | Returns all rows or filters by runtime class. |
| `map_differentiable_curriculum_public_surfaces()` | Describes the owning registry module and curriculum directory. |

Discovery is static and local. It reads no notebook body and performs no
provider, hardware, credential, kernel, or package operation.

## Public value objects

- `CurriculumNotebookRow` records the stable id, title, relative path, runtime
  class, package declarations, order, and `hardware_execution=False` boundary.
- `PathEligibilityDecision` records an allowed/refused outcome, reason, ordered
  blockers, and the shared non-promotional claim boundary.
- `MaterialisedCurriculumProbe` records ordered ids, counts, the default id,
  hardware honesty, and the number of missing notebook paths.

All records are immutable slot-backed dataclasses with validated construction
and JSON-ready `to_dict()` mappings. A zero missing-path count proves only that
the six files exist under the selected root, not that any cell executed.

## Rules

| Rule | Behaviour |
|---|---|
| Registry schema | `differentiable_notebook_curriculum.v2` |
| Default notebook | `01_parameter_shift_kuramoto_xy` |
| Hardware execution | Always false |
| Live QPU notebooks | Refused |
| Full archive conversion | Refused |
| Blank/unknown id | Fail closed |

## Eligibility decisions

`decide_differentiable_curriculum_path()` permits only the bounded core-six CPU
curriculum. It returns a structured refusal for live hardware execution or
full historical-archive conversion. When both are requested, blockers are
de-duplicated in first-seen order.

An allowed result is a catalogue-path decision only. It is not provider
authority, QPU access, package compatibility, or notebook execution evidence.

Claim boundary:

> Differentiable notebook curriculum registry only; catalogues the
> core-six onboarding curriculum under notebooks/differentiable with
> hardware_execution=false honesty; materialised manifest/probe only;
> refuses invent-green live QPU notebooks and full archive conversion;
> does not claim a full nbclient CI matrix green

## Public API

```python
from scpn_quantum_control.differentiable_notebook_curriculum import (
    assert_differentiable_curriculum_integrity,
    build_differentiable_curriculum_registry,
    decide_differentiable_curriculum_path,
    list_curriculum_notebook_ids,
    materialise_curriculum_probe,
)

assert len(list_curriculum_notebook_ids()) == 6
reg = assert_differentiable_curriculum_integrity(
    build_differentiable_curriculum_registry()
)
probe = materialise_curriculum_probe()
assert probe.hardware_execution_any is False
assert probe.missing_path_count == 0  # when run from repo root

refused = decide_differentiable_curriculum_path(request_hardware_execution=True)
assert refused.allowed is False
```

## Directory resolution and materialised probe

`resolve_curriculum_directory()` returns the absolute
`notebooks/differentiable` path beneath an explicit repository root or the
package-derived default root. It does not create the directory.

`materialise_curriculum_probe()` first applies the eligibility policy, then
checks each canonical row with `Path.is_file()`. It reports all ordered ids,
the exact row count, the default id, aggregate hardware flag, and missing-path
count. It does not parse JSON, trust notebook metadata, run a kernel, or mutate
any file.

## Core six

| Order | ID |
|---:|---|
| 1 | `01_parameter_shift_kuramoto_xy` |
| 2 | `02_gradient_tape_simulator` |
| 3 | `03_jax_batched_quantum_gradients` |
| 4 | `04_pytorch_quantum_layer` |
| 5 | `05_fail_closed_boundaries` |
| 6 | `06_witnesses_challenge_fixture` |

## Registry integrity

`build_differentiable_curriculum_registry()` emits schema
`differentiable_notebook_curriculum.v2`, the full core-six catalogue, public surface
map, default id, directory, counts, policy note, and claim boundary.

Always validate transported or stored payloads through
`assert_differentiable_curriculum_integrity()`. It rejects:

- stale schemas or altered claim boundaries;
- missing, empty, non-list, non-mapping, blank, duplicate, missing, or extra rows;
- unknown runtime classes or missing relative paths;
- any `hardware_execution=True` row or relaxed registry policy;
- loss of the default first notebook; and
- `blank_entry_count` or `notebook_count` drift.

## Failure handling and operational non-effects

Treat `ValueError` as a caller-contract, path-policy, or transported registry
failure. Treat `RuntimeError` from catalogue construction as repository
corruption.

This registry performs no network access, credential lookup, provider or QPU
discovery, notebook execution, kernel launch, package installation, archive
conversion, file creation, notebook rewrite, metadata mutation, result
promotion, or evidence publication.

## Bounded status

Shipped: curriculum map, directory and manifest registry, public API map, and
fail-closed refusal of QPU execution and full-archive conversion.

Outside this bounded registry: long-form notebook body rewrites, a complete
nbclient execution matrix, and PennyLane or Qiskit companion expansion.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
