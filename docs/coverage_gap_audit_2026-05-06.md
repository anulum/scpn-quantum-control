<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Coverage Gap Audit -->

# Coverage Gap Audit

Date added: 2026-05-06

This note records the coverage-gap audit slice of the broader
coverage/test-quality release task. It does not claim that total line coverage
has reached 100 percent.

## Tool

The deterministic audit helper is:

```bash
./.venv-linux/bin/python tools/audit_coverage_gaps.py
```

It inventories Python package files under `src/scpn_quantum_control/`, parses a
`coverage.xml` report produced by `pytest-cov`, and classifies each source file
as:

| Status | Meaning |
|--------|---------|
| `ok` | Present in the XML report and at or above the configured per-file threshold. |
| `below_threshold` | Present in the report but below the configured per-file threshold. |
| `missing_from_report` | Present in source but absent from the coverage report. |

The machine-readable form is:

```bash
./.venv-linux/bin/python tools/audit_coverage_gaps.py --json
```

The release-gating form is:

```bash
./.venv-linux/bin/python tools/audit_coverage_gaps.py --fail-on-gap
```

## Intended Coverage Workflow

Generate a fresh XML report first:

```bash
./.venv-linux/bin/pytest --cov=scpn_quantum_control --cov-report=xml
```

Then run the audit:

```bash
./.venv-linux/bin/python tools/audit_coverage_gaps.py --fail-on-gap
```

The tool is intentionally read-only. It does not run tests, mutate coverage
settings, or infer scientific validation from coverage percentage alone.

## CI Integration

The main CI workflow runs the audit after the Python 3.12 coverage job creates
`coverage.xml`:

```bash
python tools/audit_coverage_gaps.py --json > coverage-gap-audit.json
```

CI uploads `coverage-gap-audit.json` as a 30-day artifact named
`coverage-gap-audit-3.12`. This is intentionally an observation step, not a hard
per-file gate. The existing CI gate remains `--cov-fail-under=95`; the stricter
`--fail-on-gap` mode is reserved for release closure once uncovered files have
been closed with behavioural tests or explicitly justified exclusions.

## Claim Boundary

This closes only the release-safety need for a reproducible coverage-gap
inventory. The broader roadmap item remains open until uncovered files are
closed with behavioural tests or explicitly justified exclusions.
