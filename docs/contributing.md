# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Contributing

# Contributing

## Setup

```bash
git clone https://github.com/anulum/scpn-quantum-control.git
cd scpn-quantum-control
pip install -e ".[dev]"
pre-commit install
pytest tests/ -x -q
```

## Code Quality Gates

All contributions must pass the full CI pipeline. The pre-commit hooks
mirror the CI gates so you catch issues locally before push:

```bash
gitleaks                                            # generic secret scan (v8.21.2)
python tools/check_secrets.py                       # vault-pattern + keyword scan
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest tests/ -v --ignore=tests/test_hardware_runner.py
python scripts/check_version_consistency.py
```

The pre-commit configuration in `.pre-commit-config.yaml` runs all of
the above on every `git commit`. The pre-push hook additionally runs
`tools/preflight.py` which executes the same gates plus the full
test+coverage matrix.

**Secret hygiene (Tier 0 rule).** Never write credentials, API tokens,
or passwords into any committed file — including documentation,
internal notes, and CLAUDE.md files. Read credentials from
`agentic-shared/CREDENTIALS.md` at runtime or from environment
variables. The `tools/check_secrets.py` hook will block any commit
containing an inline credential keyword (`password:`, `token:`,
`api_key:` ...) with a non-placeholder value, or any high-entropy
substring extracted from the local credentials vault. See
`.coordination/incidents/INCIDENT_2026-04-10T2336_ftp_creds_in_webmaster_context.md`
for the post-mortem of the leak that motivated these scanners.

## Testing Requirements

Every new module needs a corresponding `tests/test_<module>.py` with:

- At least one physics verification (quantum result matches classical reference)
- At least one circuit validity check (transpiles on AerSimulator without error)
- At least one edge case (zero input, identity coupling, etc.)

Statistical tests should use `n_shots >= 1000` for any assertion on measurement
outcomes.

## Hardware Experiments

Hardware runs require IBM Quantum credentials and available QPU budget. Save
results as JSON in `results/` with IBM job ID for reproducibility.

## Pull Request Checklist

- [ ] Tests pass on Python 3.10–3.13
- [ ] Lint and type-check clean
- [ ] No new dependencies without justification
- [ ] CHANGELOG.md updated for user-facing changes
- [ ] Hardware experiments documented with JSON + job ID

## Contact

Questions or proposals: [protoscience@anulum.li](mailto:protoscience@anulum.li) |
[GitHub Discussions](https://github.com/anulum/scpn-quantum-control/discussions)
