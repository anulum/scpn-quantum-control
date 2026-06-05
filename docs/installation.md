# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Installation

# Installation

For a non-technical orientation before installing, read
[Onboarding](onboarding.md). For the fastest working run after installation,
continue with [Quickstart](quickstart.md).

## From PyPI

```bash
pip install scpn-quantum-control
```

## Installation posture

The installation is intentionally tiered:

- **Base path** (`pip install scpn-quantum-control`) for standard simulator workflows.
- **Developer and extension path** (`pip install -e ".[dev]"`) for local testing and
  contribution.
- **Operational path** with selected extras only: install exactly the integration
  surface your workflow needs.

The production value is that teams can keep environments lean and evidence
deterministic: each additional capability is explicit at install time rather than
implicitly enabling unsupported routes.

## From source (development)

```bash
git clone https://github.com/anulum/scpn-quantum-control.git
cd scpn-quantum-control
pip install -e ".[dev]"
```

This installs pytest, ruff, and pytest-cov for development.

## Optional dependencies

```bash
# Visualisation (matplotlib)
pip install -e ".[viz]"

# IBM Quantum hardware execution
# Pulls in qiskit-ibm-runtime (>=0.40, <1.0). The current pinned working
# version on the dev machine is 0.46.x. Note that 0.46+ changed the
# DataBin classical-register name handling — runner.py was updated to
# handle both legacy 'meas' and per-circuit names ('c', 'cr', 'c0').
pip install -e ".[ibm]"

# Portable provider bundle for offline SDK smoke coverage.
# This installs IBM, Braket, Azure Quantum, IonQ REST, OQC, Pasqal,
# Quandela, Quantinuum, Rigetti, qBraid, Cirq, Qiskit Aer, and PennyLane
# routes without credentials or live provider calls.
pip install -e ".[providers]"
scpn-provider-smoke --format table
scpn-provider-smoke --format json --sdk-package qiskit-ibm-runtime --require-all

# Provider-specific isolated extras. These are intentionally outside
# [providers] because their current SDK dependency trees conflict with
# common development or application extras on the shared Qiskit 2.x stack.
pip install -e ".[dwave]"   # D-Wave Leap; requires click >=8.2 via dwave-cloud-client
pip install -e ".[iqm]"     # IQM direct client; Qiskit bridge remains isolated
pip install -e ".[quera]"   # QuEra Bloqade; current Bloqade meta-package resolves separately
pip install -e ".[strangeworks]"  # Strangeworks Compute; Python >=3.11 SDK lane

# Generate the deterministic isolated-lane plan.
scpn-provider-smoke --plan-isolated --format table

# Run one isolated lane without touching the shared development environment.
python -m venv .venv-provider-dwave
.venv-provider-dwave/bin/python -m pip install -U pip
.venv-provider-dwave/bin/python -m pip install -e ".[dwave]"
.venv-provider-dwave/bin/scpn-provider-smoke --backend dwave_leap --require-all

# The manual "Provider Isolated Smoke" GitHub Actions workflow runs
# D-Wave, IQM, QuEra, and Strangeworks lanes in separate virtual environments.
# It is offline: no credentials, authentication, provider clients, or job submission.

# Rust acceleration (158–5,401× faster Hamiltonian construction;
# 1,665× faster ICI three-level evolution; 44× faster (α,β)-hypergeometric
# envelope; 2–10× across Pauli expectations and OTOC)
pip install scpn-quantum-engine

# Or build from source (requires Rust toolchain):
cd scpn_quantum_engine && maturin develop --release && cd ..

# Unified configuration (pydantic-settings → SCPNConfig)
pip install -e ".[config]"

# Structured logging (structlog → configure_logging + get_logger)
pip install -e ".[logging]"

# Julia acceleration tier (juliacall → accel/julia/order_parameter.jl)
# First call pays a one-off ~20 s JIT boot cost; subsequent calls are
# steady-state. Rust tier is measured faster on every N we have
# benchmarked (see docs/pipeline_performance.md §"Multi-language accel
# chain"), so Julia is a secondary tier — install when you need a
# second independent solver for cross-validation.
pip install -e ".[julia]"

# Cross-validation (QuTiP + Dynamiqs-JAX for XY Hamiltonian diff checks)
pip install -e ".[xvalidate]"

# Application plugin extras for raw-domain adapters
pip install -e ".[app-eeg]"         # EEG/MEG readers and MNE pipelines
pip install -e ".[app-plasma]"      # HDF5/tabular tokamak diagnostics
pip install -e ".[app-power-grid]"  # power-system case readers
pip install -e ".[app-fep]"         # predictive-coding workflow config

# Portable optional surface — excludes CUDA/JAX wheels that need a matching accelerator stack
pip install -e ".[all]"

# Accelerator extras — install only on machines with the matching CUDA stack
pip install -e ".[accelerated]"
```

## Rust Acceleration

The optional `scpn-quantum-engine` package provides 55 Rust-accelerated PyO3 bindings
via PyO3. When installed, all analysis modules transparently use the Rust fast
paths. When not installed, everything works via pure Python/NumPy.

Pre-built wheels are available for Linux (x86_64, aarch64), macOS (x86_64, ARM),
and Windows (x64). See [Rust Engine docs](rust_engine.md) for the full API and
benchmark results.

## Requirements

- Python 3.10+
- Qiskit 2.2+
- qiskit-aer 0.15+
- NumPy 1.24+
- SciPy 1.10+
- NetworkX 3.0+

`pyproject.toml` is the canonical dependency source. `requirements.txt`
is kept only as a pip-compatible mirror for users who cannot consume
project metadata directly.

Before release, verify that mirror with:

```bash
python tools/check_dependency_drift.py
```

## Verify installation

```bash
python -c "import scpn_quantum_control; print('OK')"
pytest tests/ -x -q  # full suite should pass
```

For documentation contributors:

```bash
pip install -e ".[docs]"
mkdocs build --strict
```

For release candidates, pair the docs build with the release-readiness audit in
[Release Readiness Gate](release_readiness.md).

## IBM Quantum setup (optional)

Only needed for real hardware execution. See [Hardware Guide](hardware_guide.md).

```python
from scpn_quantum_control.hardware import HardwareRunner

# One-time: save your API token
HardwareRunner.save_token("your-ibm-quantum-token")

# Connect to hardware
runner = HardwareRunner()
runner.connect()
print(f"Backend: {runner.backend_name}")
```

Free tier: 10 minutes QPU time per month on ibm_fez (Heron r2, 156 qubits).
