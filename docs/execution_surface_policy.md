# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — execution-surface policy

# Execution-Surface Policy

`scpn-bench` executes fixed repository scripts with local user privileges.
Each harness therefore carries an explicit `ExecutionSurfacePolicy`:

| Field | Meaning |
|-------|---------|
| `classification` | trust class for the harness |
| `network_allowed` | whether the harness may open network connections |
| `credential_allowed` | whether it may read tokens, keys, CRNs, or API credentials |
| `hardware_submission_allowed` | whether it may submit live QPU/cloud jobs |
| `allowed_write_roots` | repository-relative artefact roots the harness may update |
| `subprocess_allowed` | whether `scpn-bench` may launch the fixed script |
| `ci_blocking` | whether policy violations should block CI |

The default benchmark harness policy is `trusted_offline_executable`:
network access, credential reads, and live hardware submission are all
disallowed. Before `scpn-bench` launches a harness, it checks that the
script path stays inside the repository, that the script exists, that the
policy is executable, and that declared write roots do not escape the
repository.

`scpn-bench` also resolves the Python interpreter and the post-run `git diff`
command to absolute executable file paths before process launch. Missing or
non-executable tools fail closed with a non-zero CLI status instead of running
partially resolved commands from ambient `PATH` state.

Notebook and publication scripts are scanned without execution by
`scpn_quantum_control.execution_surface.scan_execution_surface_path`.
The scanner reports machine-readable findings for shell magic,
subprocess use, network access, credential reads, hardware submission
surfaces, and external publication commands such as Kaggle pushes.

The scanner is intentionally conservative. A finding is not a claim that
the notebook has been executed in CI or that a cloud job was submitted;
it is a review boundary that must be classified before the file can be
treated as trusted executable material.

`docs/execution_surface_manifest.toml` is the blocking review manifest for
known notebook and publication-script surfaces. Each `[[surface]]` entry
declares the repository-relative path, one of the approved classifications
(`trusted_static`, `trusted_offline_executable`, `external_publication`,
`hardware_gated`, or `untrusted_user`), the exact scanner rules allowed for
that file, and whether the entry blocks CI-style checks.

`evaluate_execution_surface_manifest()` scans every CI-blocking manifest
entry and returns machine-readable violations for missing paths,
repository-escape attempts, or scanner findings that are not explicitly
allowed by the manifest. A trusted notebook therefore cannot gain network,
credential, hardware-submission, shell, subprocess, or publication behaviour
without a manifest and test update.

The current blocking inventory additionally requires every high-risk
finding in repository notebooks and publication scripts to be manifested.
High-risk rules are `credential_read`, `hardware_submission`,
`network_access`, and `external_publication`. Shell and subprocess-only
Colab setup cells remain visible scanner findings, but are not yet forced
into the high-risk manifest unless they appear on a path already classified
for external publication or hardware-gated use.
