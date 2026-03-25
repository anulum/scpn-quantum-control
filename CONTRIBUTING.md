# Contributing to scpn-quantum-control

© 1996–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li

Thank you for your interest in scpn-quantum-control. Contributions are welcome under the following guidelines.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork and create a feature branch:
   ```bash
   git clone https://github.com/<your-user>/scpn-quantum-control.git
   cd scpn-quantum-control
   git checkout -b feature/your-feature
   ```
3. **Install dev dependencies and git hooks:**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```
4. **Build** the Rust engine (optional, for Rust work):
   ```bash
   cd scpn_quantum_engine && maturin develop --release && cd ..
   ```
5. **Run tests** to verify your setup:
   ```bash
   pytest tests/ -v --ignore=tests/test_hardware_runner.py
   ```

## Preflight Gate

Every push is guarded by pre-commit hooks that run the same checks as CI:

| Gate | What it checks |
|------|----------------|
| **ruff check** | Code quality and import hygiene |
| **ruff format** | Python formatting (`src/` and `tests/`) |
| **mypy** | Type checking (public API boundaries) |
| **version-sync** | Version consistency across pyproject.toml, CITATION.cff, .zenodo.json |
| **preflight** | Full CI mirror (lint + version-sync + tests) |

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest tests/ -v --ignore=tests/test_hardware_runner.py
```

The pre-push hook runs the full preflight automatically before every `git push`.

## Development Guidelines

### Code Style

- **Python**: `ruff format` + `ruff check` (both enforced in CI). Use type hints on public APIs only.
- **Rust**: `cargo fmt` before committing.
- **SPDX header**: Every `.py`, `.rs`, `.yml` file must start with `# SPDX-License-Identifier: AGPL-3.0-or-later`.

### Quantum-Specific Rules

- Every quantum circuit must transpile on `AerSimulator` without error
- Statistical tests compare quantum vs classical outputs (not exact match)
- Use `n_shots >= 1000` for any statistical assertion
- Never hardcode backend names; accept backend as parameter
- Hamiltonian construction must preserve Hermiticity
- All angle computations use radians (never degrees)

### Testing

- All new modules must have pytest coverage in `tests/test_<module>.py`
- Each test file needs: one physics verification, one circuit validity check, one edge case
- Coverage gate: 95% minimum, 100% target
- Hardware tests go in `tests/test_hardware_runner.py` (skipped by default)

### Commit Messages

Follow conventional commit format:
```
feat(scope): short description
fix(scope): short description
docs(scope): short description
```

Examples:
```
feat(analysis): add Krylov complexity probe for sync transition
fix(bridge): correct XXZ Hamiltonian sign convention
docs(tutorials): add Floquet time crystal tutorial
```

## What to Contribute

**High-value contributions:**
- New synchronization probes (analysis modules)
- VQE ansatz strategies for the Kuramoto-XY Hamiltonian
- Error mitigation techniques beyond ZNE/PEC
- Jupyter notebook tutorials
- Hardware benchmarks on new IBM/IonQ/Rigetti backends
- Bug reports with reproducible test cases

**Please discuss first** (open an issue) before:
- Changing the public Python API
- Modifying the bridge layer contracts
- Adding new package dependencies

## Submitting a Pull Request

1. Run the full preflight — all gates must pass
2. Add a changelog entry if the change is user-visible
3. Ensure SPDX headers are present on new files
4. Open a PR against `main` with a clear description
5. Reference any related issues

## CI Pipeline

| Job | What it checks |
|-----|----------------|
| `lint` | ruff check + ruff format |
| `test` | pytest on Python 3.9–3.13 |
| `type-check` | mypy |
| `wheel-check` | Build + install + smoke test |
| `security` | bandit + pip-audit |
| `docker` | Full test suite in Docker container |
| `hardware-smoke` | Simulator-only hardware tests |
| `integration-optional` | Cross-repo bridge tests (sc-neurocore) |

## Licence

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](LICENSE). For commercial licensing enquiries, contact protoscience@anulum.li.
