# Contributing

## Setup

```bash
git clone https://github.com/anulum/scpn-quantum-control.git
cd scpn-quantum-control
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Standards

### Anti-Slop Policy

This project enforces the anti-slop policy from the root CLAUDE.md. Key rules:

- No narration comments ("Let's use...", "We will...")
- No comments restating what code does
- Every magic number references its source (paper, equation, calibration)
- No buzzword names (call it what it does, not what it aspires to be)
- No trivial wrappers around standard library calls
- Commit messages: imperative, under 72 chars, no filler words

### Quantum-Specific Rules

- Every quantum circuit must transpile on `AerSimulator` without error
- Statistical tests compare quantum vs classical outputs (not exact match)
- Use `n_shots >= 1000` for any statistical assertion
- Never hardcode backend names; accept backend as parameter
- Hamiltonian construction must preserve Hermiticity
- All angle computations use radians (never degrees)

### Type Annotations

Required on public API boundaries (function signatures users call).
Not required on internal/private helpers.

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=scpn_quantum_control --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

Every new module needs a corresponding `tests/test_<module>.py` with:
- At least one physics verification (quantum result matches classical reference)
- At least one circuit validity check (transpiles without error)
- At least one edge case (zero input, identity coupling, etc.)

### Pull Request Checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Lint clean: `ruff check src/ tests/`
- [ ] Type check clean: `mypy src/`
- [ ] No new magic numbers without source citation
- [ ] CHANGELOG.md updated for user-facing changes
- [ ] Hardware experiments documented in `results/` with JSON + job ID

## Architecture

```
qsnn/     -> sc-neurocore quantum analogs
phase/    -> Kuramoto/UPDE quantum simulation
control/  -> QAOA/VQLS/QPetri quantum optimization
bridge/   -> Classical <-> quantum data format converters
qec/      -> Error correction for control signals
hardware/ -> IBM Quantum job runner + result parser
```

Each module maps to a classical SCPN counterpart. The bridge/ layer handles all format conversion so module code stays clean.

## Hardware Experiments

Hardware runs require:
1. IBM Quantum account with `ibm_fez` access
2. `QISKIT_IBM_TOKEN` env var or `~/.qiskit/qiskit-ibm.json`
3. Available QPU budget (10 min/month on free tier)

Save results as JSON in `results/` with the naming convention:
```
hw_<experiment>_<params>.json
sim_<experiment>_<params>.json
```

Include IBM job ID in every result file for reproducibility.
