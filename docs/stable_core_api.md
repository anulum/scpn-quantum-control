<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- stable core API -->

# Stable Core API

The stable core API defines the first durable contracts for production
workflows:

- `Problem`: typed Kuramoto/XY domain descriptor.
- `Backend`: backend capability and hardware-submission boundary.
- `Experiment`: problem, backend, objective, seed, shot, and metadata contract.
- `Result`: status, observables, artifact, blocker, and metadata contract.
- `problem_from_kuramoto` / `problem_to_kuramoto`: adapters to the existing
  Kuramoto core facade.
- standard backend profiles for classical reference, hardware replay, Qiskit,
  QuTiP, PennyLane, and Pulser-surrogate targets.
- `backend_capability_matrix()` for deterministic audit/reporting of declared
  backend capability profiles.
- generated backend capability artefacts:
  `data/stable_core/backend_capability_matrix.json` and
  `docs/stable_core_backend_capability_matrix.md`.

These contracts are intentionally smaller than the full research surface. They
are the integration point for future Qiskit, Qiskit Dynamics, QuTiP,
PennyLane, Pulser-surrogate, classical-reference, benchmark, and hardware-replay
adapters.

## Claim boundary

The stable core API is a contract layer. It does not run circuits, submit
hardware jobs, solve Lindblad dynamics, or perform optimal control by itself.
Backends must declare capabilities before an `Experiment` can target an
objective, and hardware-submission backends require preregistration metadata.

## Kuramoto facade bridge

The stable core layer is wired to the existing Kuramoto core facade through
lossless problem adapters:

- `problem_from_kuramoto()` converts a validated `KuramotoProblem` into a
  stable `Problem` with a required `problem_id`.
- `problem_to_kuramoto()` converts a stable `Problem` back into the existing
  `KuramotoProblem` facade for Hamiltonian, circuit, analog, hybrid, and
  trajectory compilation.

This bridge keeps the stable API useful without forcing a package layout split
or changing the lower-level compiler path.

## Backend profiles

Stable backend helpers encode the bridge-first strategy without claiming that
every backend is natively implemented:

- `classical_reference_backend()` for no-QPU reference solvers.
- `hardware_replay_backend()` for offline hardware result-pack replay.
- `qiskit_backend()` for Qiskit circuit/runtime paths; hardware submission is
  disabled by default and still requires experiment preregistration metadata.
- `qutip_backend()` for open-system and Hamiltonian-dynamics adapters.
- `pennylane_backend()` for autodiff and hybrid-control adapters.
- `pulser_surrogate_backend()` for analog-surrogate and pulse-schedule adapters.

These helpers declare capability profiles only. Adapter implementations must
still satisfy their own evidence, dependency, and claim-boundary gates.

`backend_capability_matrix()` returns the same profile set as a deterministic
tuple of dictionaries. Use it for release audits, documentation generators, and
multi-backend compiler preflight checks before claiming adapter support.

## Stable-core capability gate

Capability claims for the stable core are bound to a one-command gate:

```bash
scpn-bench stable-core-capability-gate
```

The gate regenerates the committed stable-core capability payload and markdown
artifacts and then compares them to committed references. This keeps stable-core
capability profiles reproducible and avoids manual promotion of adapter support.

## Stable-core release/repro gate

Stable-core releases and reproducibility checks now use a bundled gate:

```bash
scpn-bench stable-core-release-gate
```

This command is the preferred no-QPU entry point when stable-core documentation,
API surfaces, or release notes touch stable-core claims. It runs both component
gates as a single bundle:

- `scpn-bench stable-core-capability-gate`
- `scpn-bench stable-core-contract-gate`

Component commands remain available for focused checks when only one contract or
capability path changes.

## Stable-core contract gate

Contract drift for the stable-core first-path surfaces is controlled by:

```bash
scpn-bench stable-core-contract-gate
```

The gate is the offline fixture check for the `stable_core` contracts:

- `Problem`
- `Backend`
- `Experiment`
- `Result`
- Kuramoto adaptor helpers

It is designed for no-hardware runs and is intended to be run when contract
signatures, validation rules, or adaptor mappings change before docs or API
surface edits are published.

Regenerate the committed artifacts with:

```bash
PYTHONPATH=src python scripts/export_stable_core_capability_matrix.py
```

::: scpn_quantum_control.stable_core
    options:
      members:
        - Problem
        - Backend
        - Experiment
        - Result
        - backend_capability_matrix
        - stable_core_capability_payload
        - stable_core_capability_markdown
        - write_stable_core_capability_artifacts
        - build_problem
        - build_backend
        - build_experiment
        - build_result
        - classical_reference_backend
        - hardware_replay_backend
        - qiskit_backend
        - qutip_backend
        - pennylane_backend
        - pulser_surrogate_backend
        - problem_from_kuramoto
        - problem_to_kuramoto
      show_root_heading: true
      show_source: false
