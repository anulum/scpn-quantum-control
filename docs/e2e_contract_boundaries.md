# E2E contract boundaries

This page records the current end-to-end and contract-test boundary
coverage used for release-safety review.  The audit is intentionally
conservative: a boundary is marked covered only when a pytest module
name maps to that boundary, and static notebook/example checks are not
treated as executed scientific validation.

## Boundary categories

| Boundary | Current contract |
| --- | --- |
| Hardware/QPU | Hardware runner, backend, QPU data, and simulator guard tests cover the execution boundary without requiring live QPU access. |
| Bridge | Bridge property and coverage tests cover Hamiltonian, adapter, and spectral bridge surfaces. |
| SC-NeuroCore | SNN adapter, training, validation, and E2E bridge tests cover the local integration boundary. |
| Phase Orchestrator | Adapter, helper, error, and feedback tests cover orchestrator payload conversion and telemetry contracts. |
| Notebook workflows | Static contract verifies committed notebooks are valid nbformat-4 JSON artefacts with cells, metadata, recognised cell types, and notebook-compatible source fields. |
| Example workflows | Static contract verifies every example script parses, exposes `main()`, uses a main guard, and is listed in `examples/README.md`. |

## Audit command

```bash
./.venv-linux/bin/python tools/audit_e2e_contract_boundaries.py --tests-root tests --fail-on-missing
```

The command must report `covered: 6` and `missing: 0` before this
roadmap item is considered closed.

## Claim boundary

The notebook and example contracts are repository-hygiene checks.  They
do not execute long-running notebooks, optional-dataset workflows, IBM
jobs, or optimisation-heavy demonstrations.  Any scientific result used
in a paper must still be regenerated from its dedicated committed
script and artefact package.
