<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Metriq Submission Readiness -->

# Metriq Submission Readiness

Date checked: 2026-05-06

This note records the readiness gate for submitting `scpn-quantum-control`
results to Metriq.

## Current Metriq Workflow

The current Metriq platform describes benchmark contribution through the
`metriq-gym` command-line workflow. The public FAQ states that new benchmark
results should be added with the `upload` command from the `metriq-gym` CLI.

The Metriq-Gym documentation describes benchmark execution as:

```bash
mgym job dispatch <benchmark-config.json> --provider <provider> --device <device>
mgym job poll <job-id>
```

and documents a suite of benchmark definitions such as mirror circuits, EPLG,
BSEQ, and WIT.

## Readiness Decision

No Metriq submission was made in this pass.

Reason:

- the committed SCPN benchmark artefacts are valid for the project papers, but
  they are not Metriq-Gym result artefacts;
- Metriq expects benchmark outputs generated through its own schemas and upload
  path;
- submitting project-specific Rust/VQE/DLA/FIM benchmark tables as Metriq
  results would blur benchmark definitions and violate the repository's
  bounded-claim discipline.

## Local Installation State

`metriq-gym` was installed into an isolated user-level virtual environment:

```text
/home/anulum/.venvs/scpn-metriq
```

The command-line entry point is:

```text
/home/anulum/.venvs/scpn-metriq/bin/mgym
```

Validation:

```bash
/home/anulum/.venvs/scpn-metriq/bin/python -m pip check
/home/anulum/.venvs/scpn-metriq/bin/mgym --help
./.venv-linux/bin/python -m pip check
```

Both the isolated Metriq environment and the project environment reported no
broken Python requirements after the final repair.

The installation is intentionally not inside the repository and not inside
`.venv-linux`. A first attempt to install `metriq-gym` directly into
`.venv-linux` introduced incompatible dependency pins around Qiskit, NumPy, and
Click. Those conflicts were removed from `.venv-linux`, and the project
environment was rechecked before proceeding.

## Valid Submission Gate

A future Metriq submission is valid only after one of these paths is completed:

1. Run an existing Metriq-Gym benchmark on a supported local simulator or QPU
   backend and upload the resulting Metriq artefact through the documented
   `mgym` path.
2. Propose an SCPN/Kuramoto-XY benchmark definition upstream to Metriq-Gym,
   wait for review/acceptance, then run and upload results using that accepted
   schema.

The first path is lower risk and should be preferred for an initial submission.

## Suggested Initial Benchmark

A no-QPU readiness run should use a local simulator first:

```bash
mgym job dispatch metriq_gym/schemas/examples/<benchmark>.example.json \
  --provider local \
  --device aer_simulator
mgym job poll latest
```

Only after local schema, environment, and upload behaviour are understood should
any hardware benchmark be considered. Hardware execution must remain explicitly
approved and budget-gated.

## Claim Boundary

This note does not claim:

- any Metriq result submission;
- any accepted Metriq benchmark;
- any Metriq score;
- any QPU execution;
- any performance comparison beyond the existing project-specific benchmark
  artefacts.

It records the submission gate required to keep Metriq results scientifically
and procedurally valid.
